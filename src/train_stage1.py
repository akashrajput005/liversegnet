import os
import sys
import argparse
import multiprocessing as mp
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.dataset import Stage1Dataset
from src.model import build_stage1_model
from src.loss import Stage1Loss
from src.metrics import compute_metrics
from src.utils import load_config, ensure_dir, get_device, set_seed


def main():
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=r"C:\Users\Public\liversegnet\configs\config.yaml")
    parser.add_argument("--encoder", default="resnet101")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(42)
    device = get_device()

    ensure_dir(os.path.dirname(config["logging"]["model_path"]) or "models")
    save_path = config["logging"]["model_path"]
    if args.encoder.lower() == "resnet50":
        save_path = "models/best_precision_r50.pth"

    train_ds = Stage1Dataset(config, split="train", include_instances=True)
    val_ds = Stage1Dataset(config, split="validation", include_instances=False)

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError("Dataset is empty. Check dataset paths.")

    train_loader = DataLoader(
        train_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    model = build_stage1_model(encoder_name=args.encoder).to(device)
    criterion = Stage1Loss(
        focal_gamma=config["loss"]["focal_gamma"],
        boundary_weight=config["loss"]["boundary_weight"],
        dice_weight_bg=config["loss"].get("dice_weight_bg", 0.0),
        dice_weight_liver=config["loss"].get("dice_weight_liver", 1.0),
        dice_weight_inst=config["loss"].get("dice_weight_inst", 1.0),
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["training"]["epochs"])
    scaler = torch.cuda.amp.GradScaler(enabled=config["training"]["mixed_precision"])

    best_precision = 0.0
    patience = config["training"]["early_stopping_patience"]
    stale = 0

    for epoch in range(config["training"]["epochs"]):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Stage1 Epoch {epoch+1}/{config['training']['epochs']}")
        for step, (images, masks, sources) in enumerate(pbar, 1):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            sources = sources.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=config["training"]["mixed_precision"]):
                outputs = model(images)
                loss = criterion(outputs, masks, sources) / config["training"]["accumulation_steps"]

            scaler.scale(loss).backward()

            if step % config["training"]["accumulation_steps"] == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item()
            pbar.set_postfix({"loss": running_loss / step})

        model.eval()
        metric_sum = {"instrument_precision": 0.0, "instrument_recall": 0.0, "dice_liver": 0.0, "dice_instrument": 0.0}
        steps = 0
        with torch.no_grad():
            for images, masks, _ in val_loader:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                outputs = model(images)
                metrics = compute_metrics(outputs, masks)
                for k in metric_sum:
                    metric_sum[k] += metrics[k]
                steps += 1

        for k in metric_sum:
            metric_sum[k] /= max(steps, 1)

        scheduler.step()

        precision = metric_sum["instrument_precision"]
        if precision > best_precision:
            best_precision = precision
            stale = 0
            torch.save(model.state_dict(), save_path)
        else:
            stale += 1
            if stale >= patience:
                break


if __name__ == "__main__":
    main()
