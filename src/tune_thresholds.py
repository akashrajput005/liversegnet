import argparse
from pathlib import Path
import json
import os
import sys

import cv2
import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.utils import load_config, get_device, save_config
from src.infer import load_models, predict_image


def map_seg8k_mask(mask):
    mapped = np.zeros_like(mask, dtype=np.uint8)
    mapped[mask == 21] = 1
    mapped[mask == 50] = 2
    return mapped


def get_max_video(root):
    max_vid = 0
    for p in Path(root).iterdir():
        if p.is_dir() and p.name.lower().startswith("video"):
            try:
                vid = int(p.name.lower().replace("video", ""))
                max_vid = max(max_vid, vid)
            except ValueError:
                continue
    return max_vid


def is_seg8k_split(path, split, max_vid):
    path = path.lower().replace("\\", "/")
    import re

    match = re.search(r"video(\d+)", path)
    if match:
        vid_num = int(match.group(1))
        train_max = 60
        if max_vid and max_vid <= 60:
            train_max = max(1, int(round(max_vid * 0.8)))
            if train_max >= max_vid:
                train_max = max_vid - 1 if max_vid > 1 else 1
        if split == "train":
            return vid_num <= train_max
        if split in ["val", "validation", "test"]:
            return vid_num > train_max
    return True


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


def score(metrics):
    return 0.6 * metrics["instrument_precision"] + 0.2 * metrics["instrument_dice"] + 0.2 * metrics["liver_dice"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=r"C:\Users\Public\liversegnet\configs\config.yaml")
    parser.add_argument("--encoder", default="resnet101")
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Loading config and models...", flush=True)
    config = load_config(args.config)
    device = get_device()
    stage1, stage2 = load_models(config, device, args.encoder)

    seg_root = Path(config["data"]["cholecseg8k"])
    print(f"Scanning masks under {seg_root} ...", flush=True)
    max_vid = get_max_video(seg_root)
    val_paths = []
    for root, _, files in os.walk(seg_root):
        for name in files:
            if not name.endswith("_endo_mask.png"):
                continue
            p = Path(root) / name
            if is_seg8k_split(str(p), "val", max_vid):
                val_paths.append(p)
                if args.max_samples and len(val_paths) >= args.max_samples:
                    break
        if args.max_samples and len(val_paths) >= args.max_samples:
            break

    if not val_paths:
        raise RuntimeError(f"No seg8k masks found under {seg_root}")

    if args.max_samples and len(val_paths) > args.max_samples:
        rng = np.random.default_rng(args.seed)
        val_paths = rng.choice(val_paths, size=args.max_samples, replace=False).tolist()
    print(f"Using {len(val_paths)} validation masks for sweep", flush=True)

    liver_thresholds = [0.45, 0.50, 0.55, 0.60, 0.65]
    liver_margins = [0.0, 0.02, 0.05, 0.08]
    inst_thresholds = [0.60, 0.65, 0.70, 0.75, 0.80]

    best = None
    results = []

    for liver_t in liver_thresholds:
        for margin in liver_margins:
            for inst_t in inst_thresholds:
                config["inference"]["liver_threshold"] = liver_t
                config["inference"]["liver_bg_margin"] = margin
                config["inference"]["instrument_threshold"] = inst_t
                metrics_acc = []
                for idx, mpath in enumerate(val_paths, 1):
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
                    _, pred_liver, pred_inst, _ = predict_image(img, stage1, stage2, config)
                    gt_liver = (gt_mask == 1).astype(np.uint8)
                    gt_inst = (gt_mask == 2).astype(np.uint8)
                    liver_d = dice(pred_liver, gt_liver)
                    inst_d = dice(pred_inst, gt_inst)
                    inst_p, inst_r, inst_fp = precision_recall(pred_inst, gt_inst)
                    metrics_acc.append(
                        {
                            "liver_dice": liver_d,
                            "instrument_dice": inst_d,
                            "instrument_precision": inst_p,
                            "instrument_recall": inst_r,
                            "instrument_false_positive": inst_fp,
                        }
                    )
                    if idx % 20 == 0:
                        print(f"  processed {idx}/{len(val_paths)}", flush=True)
                if not metrics_acc:
                    continue
                avg = {
                    "liver_dice": float(np.mean([m["liver_dice"] for m in metrics_acc])),
                    "instrument_dice": float(np.mean([m["instrument_dice"] for m in metrics_acc])),
                    "instrument_precision": float(np.mean([m["instrument_precision"] for m in metrics_acc])),
                    "instrument_recall": float(np.mean([m["instrument_recall"] for m in metrics_acc])),
                }
                avg["score"] = float(score(avg))
                avg["liver_threshold"] = liver_t
                avg["liver_bg_margin"] = margin
                avg["instrument_threshold"] = inst_t
                results.append(avg)
                if best is None or avg["score"] > best["score"]:
                    best = avg

    if best is None:
        raise RuntimeError("No threshold results computed.")

    config["inference"]["liver_threshold"] = best["liver_threshold"]
    config["inference"]["liver_bg_margin"] = best["liver_bg_margin"]
    config["inference"]["instrument_threshold"] = best["instrument_threshold"]
    save_config(args.config, config)

    out_path = Path("logs/threshold_sweep.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"best": best, "all": results}, f, indent=2)

    print("Best thresholds:", best)
    print(f"Saved sweep to {out_path}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
