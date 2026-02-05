import os
import sys
import json
import argparse
import cv2
from pathlib import Path
import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.model import build_stage1_model, build_stage2_model
from src.utils import load_config, get_device


class TemporalSmoother:
    def __init__(self, alpha=0.6):
        self.alpha = alpha
        self.prev = None

    def apply(self, prob):
        if self.prev is None:
            self.prev = prob
            return prob
        blended = self.alpha * prob + (1 - self.alpha) * self.prev
        self.prev = blended
        return blended


def _stage1_probs(model, img_rgb, device, image_size):
    resized = cv2.resize(img_rgb, (image_size, image_size))
    tensor = torch.from_numpy(resized.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    tensor = tensor.to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    probs = np.stack(
        [
            cv2.resize(probs[0], (img_rgb.shape[1], img_rgb.shape[0])),
            cv2.resize(probs[1], (img_rgb.shape[1], img_rgb.shape[0])),
            cv2.resize(probs[2], (img_rgb.shape[1], img_rgb.shape[0])),
        ],
        axis=0,
    )
    return probs


def _refine_instrument(stage2_model, img_rgb, inst_prob, device, proposal_thresh=0.30, padding=20):
    h, w = img_rgb.shape[:2]
    refined = inst_prob.copy()
    cand = inst_prob > proposal_thresh
    if cand.sum() == 0:
        return refined
    num_labels, labels = cv2.connectedComponents(cand.astype(np.uint8))
    for lbl in range(1, num_labels):
        ys, xs = np.where(labels == lbl)
        if ys.size == 0:
            continue
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w - 1, x2 + padding)
        y2 = min(h - 1, y2 + padding)
        crop = img_rgb[y1 : y2 + 1, x1 : x2 + 1]
        crop_resized = cv2.resize(crop, (256, 256))
        tensor = torch.from_numpy(crop_resized.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        tensor = tensor.to(device)
        with torch.no_grad():
            logits = stage2_model(tensor)
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
        prob = cv2.resize(prob, (x2 - x1 + 1, y2 - y1 + 1))
        refined[y1 : y2 + 1, x1 : x2 + 1] = np.maximum(refined[y1 : y2 + 1, x1 : x2 + 1], prob)
    return refined


def _clean_instrument(mask):
    num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
    cleaned = np.zeros_like(mask, dtype=np.uint8)
    for lbl in range(1, num_labels):
        ys, xs = np.where(labels == lbl)
        if ys.size < 20:
            continue
        cleaned[labels == lbl] = 1
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.erode(cleaned, kernel, iterations=1)
    return cleaned


def predict_image(img_bgr, stage1_model, stage2_model, config, smoother=None):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    probs = _stage1_probs(stage1_model, img_rgb, stage1_model.device, config["training"]["image_size"])
    inst_prob = probs[2]
    if stage2_model is not None:
        inst_prob = _refine_instrument(stage2_model, img_rgb, inst_prob, stage1_model.device)
    if smoother:
        inst_prob = smoother.apply(inst_prob)

    inst_thresh = config["inference"].get("instrument_threshold", 0.70)
    inst_mask = (inst_prob >= inst_thresh).astype(np.uint8)
    inst_mask = _clean_instrument(inst_mask)

    liver_prob = probs[1]
    bg_prob = probs[0]
    liver_thresh = config["inference"].get("liver_threshold", 0.50)
    liver_margin = config["inference"].get("liver_bg_margin", 0.0)
    liver_mask = ((liver_prob >= liver_thresh) & (liver_prob >= (bg_prob + liver_margin))).astype(np.uint8)
    final_mask = np.zeros((img_rgb.shape[0], img_rgb.shape[1]), dtype=np.uint8)
    final_mask[liver_mask == 1] = 1
    final_mask[inst_mask == 1] = 2
    return final_mask, liver_mask, inst_mask, probs


def load_models(config, device, encoder_name):
    stage1 = build_stage1_model(encoder_name=encoder_name).to(device)
    ckpts = config.get("logging", {}).get("encoder_checkpoints", {})
    if encoder_name in ckpts:
        stage1_path = ckpts[encoder_name]
    elif encoder_name.lower() == "resnet101":
        stage1_path = config["logging"]["model_path"]
    elif encoder_name.lower() == "resnet50":
        stage1_path = "models/best_precision_r50.pth"
    else:
        stage1_path = config["logging"]["model_path"]
    if not os.path.exists(stage1_path):
        raise RuntimeError(f"Checkpoint not found for {encoder_name}: {stage1_path}")
    stage1.load_state_dict(torch.load(stage1_path, map_location=device, weights_only=True))
    stage1.eval()
    stage1.device = device

    stage2_path = "models/stage2_best_precision.pth"
    stage2 = None
    if os.path.exists(stage2_path):
        stage2 = build_stage2_model().to(device)
        stage2.load_state_dict(torch.load(stage2_path, map_location=device, weights_only=True))
        stage2.eval()
    return stage1, stage2


def _metrics_from_masks(liver_mask, inst_mask, probs, inst_threshold):
    stage1_inst = (probs[2] >= inst_threshold).astype(np.uint8)
    stage1_liver = (probs[1] > 0.6).astype(np.uint8)
    def dice(a, b):
        inter = int((a & b).sum())
        total = int(a.sum() + b.sum())
        return (2 * inter + 1e-6) / (total + 1e-6)
    def prec_recall(pred, gt):
        tp = int((pred & gt).sum())
        fp = int((pred & (1 - gt)).sum())
        fn = int(((1 - pred) & gt).sum())
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        return precision, recall, fp
    inst_precision, inst_recall, inst_fp = prec_recall(inst_mask, stage1_inst)
    return {
        "liver_dice": float(dice(liver_mask, stage1_liver)),
        "instrument_dice": float(dice(inst_mask, stage1_inst)),
        "instrument_precision": float(inst_precision),
        "instrument_recall": float(inst_recall),
        "instrument_false_positive": int(inst_fp),
        "pixel_area_liver": int(liver_mask.sum()),
        "pixel_area_instrument": int(inst_mask.sum()),
    }


def infer_single_image(image_path, config_path, encoder_name="resnet101"):
    config = load_config(config_path)
    device = get_device()
    stage1, stage2 = load_models(config, device, encoder_name)
    if not os.path.exists(image_path):
        candidates = []
        uploads_dir = Path(os.path.join(os.path.dirname(__file__), "..", "uploads"))
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            candidates.extend(sorted(uploads_dir.glob(ext)))
        if not candidates:
            seg_root = Path(config["data"]["cholecseg8k"])
            candidates.extend(sorted(seg_root.rglob("*_endo.png")))
        if not candidates:
            inst_img_root = Path(config["data"].get("cholecinstanceseg_images", config["data"]["cholecinstanceseg"]))
            for ext in ("*.png", "*.jpg", "*.jpeg"):
                candidates.extend(sorted(inst_img_root.rglob(ext)))
        if not candidates:
            raise RuntimeError("No images found in uploads/ or dataset roots.")
        image_path = str(candidates[0])
    print(f"Using image: {image_path}")
    img = cv2.imread(image_path)
    final_mask, liver_mask, inst_mask, probs = predict_image(img, stage1, stage2, config)
    out = img.copy()
    out[final_mask == 1] = (0, 255, 0)
    out[final_mask == 2] = (0, 165, 255)
    out_path = f"models/infer_overlay_{encoder_name}.png"
    masks_path = f"models/infer_masks_{encoder_name}.npz"
    metrics_path = f"models/infer_metrics_{encoder_name}.json"
    cv2.imwrite(out_path, out)
    np.savez_compressed(masks_path, liver_mask=liver_mask, instrument_mask=inst_mask)
    metrics = _metrics_from_masks(liver_mask, inst_mask, probs, config["inference"]["instrument_threshold"])
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=r"C:\Users\Public\liversegnet\uploads\sample.png")
    parser.add_argument("--config", default=r"C:\Users\Public\liversegnet\configs\config.yaml")
    parser.add_argument("--encoder", default="resnet101")
    args = parser.parse_args()
    infer_single_image(args.image, args.config, args.encoder)
