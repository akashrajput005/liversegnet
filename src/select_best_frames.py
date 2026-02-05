import os
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch

from src.utils import load_config, get_device
from src.infer import load_models, predict_image


def map_seg8k_mask(mask):
    mapped = np.zeros_like(mask, dtype=np.uint8)
    mapped[mask == 21] = 1
    mapped[mask == 50] = 2
    return mapped


def dice(a, b):
    inter = int((a & b).sum())
    total = int(a.sum() + b.sum())
    return (2 * inter + 1e-6) / (total + 1e-6)


def precision_recall(pred, gt):
    tp = int((pred & gt).sum())
    fp = int((pred & (1 - gt)).sum())
    fn = int(((1 - pred) & gt).sum())
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    return precision, recall, fp


def score_sample(metrics):
    return (
        0.5 * metrics["instrument_precision"]
        + 0.3 * metrics["instrument_dice"]
        + 0.2 * metrics["liver_dice"]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=r"C:\Users\Public\liversegnet\configs\config.yaml")
    parser.add_argument("--encoder", default="resnet101")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="logs/best_frames")
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device()
    stage1, stage2 = load_models(config, device, args.encoder)

    seg_root = Path(config["data"]["cholecseg8k"])
    mask_paths = list(seg_root.rglob("*_endo_mask.png"))
    if not mask_paths:
        raise RuntimeError(f"No seg8k masks found under {seg_root}")

    rng = np.random.default_rng(args.seed)
    if args.max_samples and args.max_samples > 0 and len(mask_paths) > args.max_samples:
        mask_paths = rng.choice(mask_paths, size=args.max_samples, replace=False).tolist()

    results = []
    for mpath in mask_paths:
        ipath = str(mpath).replace("_mask.png", ".png")
        if not os.path.exists(ipath):
            continue
        img = cv2.imread(ipath)
        if img is None:
            continue
        gt_mask_raw = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE)
        if gt_mask_raw is None:
            continue
        gt_mask = map_seg8k_mask(gt_mask_raw)
        pred_mask, pred_liver, pred_inst, _ = predict_image(img, stage1, stage2, config)

        gt_liver = (gt_mask == 1).astype(np.uint8)
        gt_inst = (gt_mask == 2).astype(np.uint8)
        liver_dice = dice(pred_liver, gt_liver)
        inst_dice = dice(pred_inst, gt_inst)
        inst_precision, inst_recall, inst_fp = precision_recall(pred_inst, gt_inst)
        metrics = {
            "liver_dice": float(liver_dice),
            "instrument_dice": float(inst_dice),
            "instrument_precision": float(inst_precision),
            "instrument_recall": float(inst_recall),
            "instrument_false_positive": int(inst_fp),
            "image_path": ipath,
            "mask_path": str(mpath),
        }
        metrics["score"] = float(score_sample(metrics))
        results.append(metrics)

    if not results:
        raise RuntimeError("No valid samples scored. Check dataset paths.")

    results.sort(key=lambda x: x["score"], reverse=True)
    top = results[: args.top_k]

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, item in enumerate(top):
        img = cv2.imread(item["image_path"])
        pred_mask, _, _, _ = predict_image(img, stage1, stage2, config)
        overlay = img.copy()
        overlay[pred_mask == 1] = (0, 255, 0)
        overlay[pred_mask == 2] = (0, 0, 255)
        fname = f"{idx+1:02d}_score_{item['score']:.3f}.png"
        cv2.imwrite(str(out_dir / fname), overlay)

    csv_path = out_dir / "best_frames_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "score",
                "liver_dice",
                "instrument_dice",
                "instrument_precision",
                "instrument_recall",
                "instrument_false_positive",
                "image_path",
                "mask_path",
            ],
        )
        writer.writeheader()
        for item in top:
            writer.writerow(item)

    print(f"Saved {len(top)} best overlays to {out_dir}")
    print(f"Summary CSV: {csv_path}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
