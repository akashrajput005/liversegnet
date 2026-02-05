import os
import sys
import multiprocessing as mp
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.dataset import Stage2Dataset
from src.model import build_stage2_model
from src.loss import Stage2Loss
from src.utils import load_config, ensure_dir, get_device, set_seed


def instrument_precision_binary(logits, target):
    probs = torch.sigmoid(logits)
    preds = probs > 0.5
    tp = (preds & (target > 0.5)).sum().item()
    fp = (preds & (target <= 0.5)).sum().item()
    return tp / (tp + fp + 1e-6)


def main():
    mp.set_start_method("spawn", force=True)
    config = load_config(r"C:\Users\Public\liversegnet\configs\config.yaml")
    set_seed(42)
    device = get_device()

    ensure_dir("models")

    print("Stage2: building datasets...", flush=True)
    train_ds = Stage2Dataset(config, split="train")
    val_ds = Stage2Dataset(config, split="validation")
    print(
        f"Stage2 train: jsons={train_ds.total_jsons}, images={train_ds.resolved_images}, valid_masks={train_ds.valid_masks}, samples={len(train_ds)}"
    )
    if train_ds.example_json and train_ds.example_image:
        print(f"Stage2 example mapping: {train_ds.example_json} -> {train_ds.example_image}")
    print(
        f"Stage2 val: jsons={val_ds.total_jsons}, images={val_ds.resolved_images}, valid_masks={val_ds.valid_masks}, samples={len(val_ds)}"
    )
    if train_ds.valid_masks < 1:
        raise RuntimeError("Stage2 dataset has no valid instrument masks. Check cholecinstanceseg images/labels.")

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)

    model = build_stage2_model().to(device)
    criterion = Stage2Loss(
        alpha=config["loss"]["tversky_alpha"],
        beta=config["loss"]["tversky_beta"],
    )
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=config["training"]["mixed_precision"])

    best_precision = 0.0
    patience = config["training"]["early_stopping_patience"]
    stale = 0

    max_epochs = 20
    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Stage2 Epoch {epoch+1}/{max_epochs}")
        for step, (images, masks) in enumerate(pbar, 1):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=config["training"]["mixed_precision"]):
                logits = model(images)
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            pbar.set_postfix({"loss": running_loss / step})

        model.eval()
        val_precision = 0.0
        steps = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                logits = model(images)
                val_precision += instrument_precision_binary(logits, masks)
                steps += 1
        val_precision = val_precision / max(steps, 1)

        if val_precision > best_precision:
            best_precision = val_precision
            stale = 0
            torch.save(model.state_dict(), "models/stage2_best_precision.pth")
        else:
            stale += 1
            if stale >= patience:
                break


if __name__ == "__main__":
    main()
