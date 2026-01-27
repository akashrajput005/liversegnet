import os
import csv
import random
from datetime import datetime

import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, ConcatDataset
try:
    from cholec_dataset import CholecSeg8kDataset, get_transforms
    from instance_dataset import CholecInstancePolygonDataset
    from model import get_model, HybridLoss
except ImportError:
    from src.cholec_dataset import CholecSeg8kDataset, get_transforms
    from src.instance_dataset import CholecInstancePolygonDataset
    from src.model import get_model, HybridLoss
from tqdm import tqdm
import torch.nn.functional as F


def _dice_score(pred: torch.Tensor, target: torch.Tensor, cls: int, eps: float = 1e-6, ignore_index: int = 255) -> float:
    valid = (target != ignore_index).float()
    pred_c = ((pred == cls).float() * valid)
    target_c = ((target == cls).float() * valid)
    inter = (pred_c * target_c).sum()
    denom = pred_c.sum() + target_c.sum()
    return float((2.0 * inter + eps) / (denom + eps))


class _EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def update(self, model: torch.nn.Module):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if k not in self.shadow:
                    self.shadow[k] = v.detach().clone()
                else:
                    src = v.detach()
                    dst = self.shadow[k]
                    # Only apply EMA to floating tensors. For integer/bool tensors,
                    # copy directly (e.g., BatchNorm num_batches_tracked is int64).
                    if torch.is_floating_point(dst) and torch.is_floating_point(src):
                        dst.mul_(self.decay).add_(src, alpha=1.0 - self.decay)
                    else:
                        self.shadow[k] = src.clone()

    def apply_to(self, model: torch.nn.Module):
        model.load_state_dict(self.shadow, strict=True)


def _split_by_video(ds: CholecSeg8kDataset, val_ratio: float, seed: int):
    video_to_indices = {}
    for i in range(len(ds)):
        vid = ds.get_video_id(i)
        video_to_indices.setdefault(vid, []).append(i)

    videos = sorted(video_to_indices.keys())
    rnd = random.Random(seed)
    rnd.shuffle(videos)

    target_val = max(1, int(len(videos) * float(val_ratio)))
    val_videos = set(videos[:target_val])

    train_idx = []
    val_idx = []
    for vid, idxs in video_to_indices.items():
        (val_idx if vid in val_videos else train_idx).extend(idxs)

    return train_idx, val_idx, sorted(val_videos)


def _run_epoch(model, loader, optimizer, scheduler, loss_fn, device, scaler, ema, is_train: bool, amp_enabled: bool, stage1: bool):
    if is_train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    dice_liver = 0.0
    dice_inst = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc="Training" if is_train else "Validation")
    for images, masks in pbar:
        if device.type == 'cuda':
            images = images.contiguous(memory_format=torch.channels_last)
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        if stage1:
            masks = (masks == 1).long()

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.no_grad() if not is_train else torch.enable_grad():
            if amp_enabled:
                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(images)
                    loss = loss_fn(outputs, masks)
            else:
                outputs = model(images)
                loss = loss_fn(outputs, masks)

            if is_train:
                if amp_enabled:
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                if scheduler is not None:
                    scheduler.step()

                if ema is not None:
                    ema.update(model)

        running_loss += float(loss.item())
        n_batches += 1

        with torch.no_grad():
            pred = torch.argmax(outputs, dim=1)
            if stage1:
                dice_liver += _dice_score(pred, masks, cls=1)
            else:
                dice_liver += _dice_score(pred, masks, cls=1)
                dice_inst += _dice_score(pred, masks, cls=2)

        if stage1:
            pbar.set_postfix({'loss': float(loss.item()), 'dice_liver': dice_liver / n_batches})
        else:
            pbar.set_postfix({'loss': float(loss.item()), 'dice_liver': dice_liver / n_batches, 'dice_inst': dice_inst / n_batches})

    avg_loss = running_loss / max(1, n_batches)
    avg_dice_liver = dice_liver / max(1, n_batches)
    avg_dice_inst = dice_inst / max(1, n_batches)
    return avg_loss, avg_dice_liver, avg_dice_inst


def main():
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
         
    seed = int(config.get('seed', 42))
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
     
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        # TF32 is a big win on RTX (Ampere+) for conv/matmul throughput.
        # It slightly reduces precision but is usually fine for segmentation.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass
        print("cuDNN benchmark enabled for maximum GPU throughput.")
    
    amp_enabled = device.type == 'cuda'

    scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)

    os.makedirs('models', exist_ok=True)

    dataset_root = config.get('cholecseg8k_path') or config.get('unified_dataset_path')
    if not dataset_root or not os.path.exists(dataset_root):
        raise FileNotFoundError(f"Dataset path not found: {dataset_root}")

    full_ds = CholecSeg8kDataset(
        root_dir=dataset_root,
        transform=get_transforms(is_train=True),
        target_size=tuple(config.get('input_size', [512, 512])),
        max_samples=None,
    )

    val_ratio = float(config.get('val_ratio', 0.15))
    train_idx, val_idx, val_videos = _split_by_video(full_ds, val_ratio=val_ratio, seed=seed)
    train_ds_seg8k = Subset(full_ds, train_idx)
    val_ds = Subset(full_ds, val_idx)

    batch_size = int(config.get('batch_size', 4))
    num_workers = int(config.get('num_workers', 4))

    print(f"CholecSeg8k total: {len(full_ds)}")
    print(f"Train (video-wise): {len(train_ds_seg8k)}")
    print(f"Val (video-wise): {len(val_ds)}")
    print(f"Val videos: {val_videos}")

    # Polygon dataset (instrument-only). Only used for 3-class training.
    poly_ann_root = config.get('cholecinstanceseg_path')
    poly_img_root = config.get('reference_images_path')
    poly_ds = None
    if poly_ann_root and poly_img_root and os.path.exists(poly_ann_root) and os.path.exists(poly_img_root):
        poly_ds = CholecInstancePolygonDataset(
            ann_root=poly_ann_root,
            img_root=poly_img_root,
            split='train',
            target_size=tuple(config.get('input_size', [512, 512])),
        )

    # Stage1 uses only seg8k (needs liver labels)
    dl_common = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': (device.type == 'cuda'),
        'persistent_workers': False,
        'drop_last': True,
    }
    if num_workers > 0:
        dl_common['prefetch_factor'] = 4

    train_loader_s1 = DataLoader(
        train_ds_seg8k,
        shuffle=True,
        **dl_common,
    )

    # Stage2/U-Net uses seg8k + polygons (extra instrument supervision)
    if poly_ds is not None and len(poly_ds) > 0:
        train_ds_s2 = ConcatDataset([train_ds_seg8k, poly_ds])
        print(f"Polygon samples (train): {len(poly_ds)}")
        print(f"Stage2/U-Net train total (seg8k+poly): {len(train_ds_s2)}")
    else:
        train_ds_s2 = train_ds_seg8k

    train_loader_s2 = DataLoader(
        train_ds_s2,
        shuffle=True,
        **dl_common,
    )

    val_loader = DataLoader(
        val_ds,
        **{**dl_common, 'shuffle': False},
    )

    log_dir = os.path.join('logs', datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(log_dir, exist_ok=True)
    metrics_csv = os.path.join(log_dir, 'metrics.csv')
    with open(metrics_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['stage', 'epoch', 'train_loss', 'val_loss', 'val_dice_liver', 'val_dice_inst', 'lr'])

    # ---------------- Stage 1: Liver Only (2-class) ----------------
    stage1_path = os.path.join('models', 'deeplabv3plus_resnet50_stage1.pth')
    print(f"\n--- Stage 1: Training Liver-only model -> {stage1_path} ---")
    s1_epochs = int(config.get('stage1', {}).get('epochs', 10))
    s1_lr = float(config.get('stage1', {}).get('lr', 1e-4))

    model_s1 = get_model(architecture='deeplabv3plus', encoder='resnet50', num_classes=2).to(device)
    if device.type == 'cuda':
        model_s1 = model_s1.to(memory_format=torch.channels_last)
    opt_s1 = optim.AdamW(model_s1.parameters(), lr=s1_lr, weight_decay=1e-4)
    loss_s1 = HybridLoss(num_classes=2)
    steps_per_epoch_s1 = max(1, len(train_loader_s1))
    sched_s1 = optim.lr_scheduler.OneCycleLR(opt_s1, max_lr=s1_lr, epochs=s1_epochs, steps_per_epoch=steps_per_epoch_s1)
    ema_s1 = _EMA(model_s1, decay=0.999)
    best_s1 = -1.0
    for epoch in range(s1_epochs):
        print(f"Epoch {epoch+1}/{s1_epochs}")
        train_loss, _, _ = _run_epoch(model_s1, train_loader_s1, opt_s1, sched_s1, loss_s1, device, scaler, ema_s1, is_train=True, amp_enabled=amp_enabled, stage1=True)

        # validate using EMA weights
        model_tmp = get_model(architecture='deeplabv3plus', encoder='resnet50', num_classes=2).to(device)
        model_tmp.load_state_dict(model_s1.state_dict(), strict=True)
        ema_s1.apply_to(model_tmp)
        val_loss, val_dice_liver, _ = _run_epoch(model_tmp, val_loader, opt_s1, None, loss_s1, device, scaler, None, is_train=False, amp_enabled=amp_enabled, stage1=True)
        if val_dice_liver > best_s1:
            best_s1 = val_dice_liver
            torch.save(model_s1.state_dict(), stage1_path)
            print(f"Saved best Stage1 checkpoint: dice_liver={best_s1:.4f}")

        with open(metrics_csv, 'a', newline='') as f:
            w = csv.writer(f)
            w.writerow(['stage1', epoch + 1, train_loss, val_loss, val_dice_liver, 0.0, opt_s1.param_groups[0]['lr']])

    # ---------------- Stage 2: Full 3-class DeepLabV3+ ----------------
    stage2_path = os.path.join('models', 'deeplabv3plus_resnet50.pth')
    print(f"\n--- Stage 2: Training 3-class DeepLabV3+ -> {stage2_path} ---")
    s2_epochs = int(config.get('stage2', {}).get('epochs', 20))
    s2_lr = float(config.get('stage2', {}).get('lr', 5e-5))

    model_s2 = get_model(architecture='deeplabv3plus', encoder='resnet50', num_classes=3).to(device)
    if os.path.exists(stage1_path):
        state_dict = torch.load(stage1_path, map_location=device)
        state_dict = {k: v for k, v in state_dict.items() if 'segmentation_head' not in k}
        model_s2.load_state_dict(state_dict, strict=False)
    opt_s2 = optim.AdamW(model_s2.parameters(), lr=s2_lr, weight_decay=1e-4)
    loss_s2 = HybridLoss(num_classes=3)
    steps_per_epoch_s2 = max(1, len(train_loader_s2))
    sched_s2 = optim.lr_scheduler.OneCycleLR(opt_s2, max_lr=s2_lr, epochs=s2_epochs, steps_per_epoch=steps_per_epoch_s2)
    ema_s2 = _EMA(model_s2, decay=0.999)
    best_s2 = -1.0
    for epoch in range(s2_epochs):
        print(f"Epoch {epoch+1}/{s2_epochs}")
        train_loss, _, _ = _run_epoch(model_s2, train_loader_s2, opt_s2, sched_s2, loss_s2, device, scaler, ema_s2, is_train=True, amp_enabled=amp_enabled, stage1=False)

        model_tmp = get_model(architecture='deeplabv3plus', encoder='resnet50', num_classes=3).to(device)
        model_tmp.load_state_dict(model_s2.state_dict(), strict=True)
        ema_s2.apply_to(model_tmp)
        val_loss, val_dice_liver, val_dice_inst = _run_epoch(model_tmp, val_loader, opt_s2, None, loss_s2, device, scaler, None, is_train=False, amp_enabled=amp_enabled, stage1=False)
        avg_dice = (val_dice_liver + val_dice_inst) / 2.0
        if avg_dice > best_s2:
            best_s2 = avg_dice
            torch.save(model_s2.state_dict(), stage2_path)
            print(f"Saved best Stage2 checkpoint: avg_dice={best_s2:.4f}")

        with open(metrics_csv, 'a', newline='') as f:
            w = csv.writer(f)
            w.writerow(['stage2', epoch + 1, train_loss, val_loss, val_dice_liver, val_dice_inst, opt_s2.param_groups[0]['lr']])

    # ---------------- Baseline: U-Net ResNet34 (3-class) ----------------
    unet_path = os.path.join('models', 'unet_resnet34.pth')
    print(f"\n--- Baseline: Training 3-class U-Net -> {unet_path} ---")
    unet_epochs = max(5, int(s2_epochs // 2))
    unet_lr = 1e-4
    model_unet = get_model(architecture='unet', encoder='resnet34', num_classes=3).to(device)
    opt_unet = optim.AdamW(model_unet.parameters(), lr=unet_lr, weight_decay=1e-4)
    loss_unet = HybridLoss(num_classes=3)
    steps_per_epoch_unet = max(1, len(train_loader_s2))
    sched_unet = optim.lr_scheduler.OneCycleLR(opt_unet, max_lr=unet_lr, epochs=unet_epochs, steps_per_epoch=steps_per_epoch_unet)
    ema_unet = _EMA(model_unet, decay=0.999)
    best_unet = -1.0
    for epoch in range(unet_epochs):
        print(f"Epoch {epoch+1}/{unet_epochs}")
        train_loss, _, _ = _run_epoch(model_unet, train_loader_s2, opt_unet, sched_unet, loss_unet, device, scaler, ema_unet, is_train=True, amp_enabled=amp_enabled, stage1=False)

        model_tmp = get_model(architecture='unet', encoder='resnet34', num_classes=3).to(device)
        model_tmp.load_state_dict(model_unet.state_dict(), strict=True)
        ema_unet.apply_to(model_tmp)
        val_loss, val_dice_liver, val_dice_inst = _run_epoch(model_tmp, val_loader, opt_unet, None, loss_unet, device, scaler, None, is_train=False, amp_enabled=amp_enabled, stage1=False)
        avg_dice = (val_dice_liver + val_dice_inst) / 2.0
        if avg_dice > best_unet:
            best_unet = avg_dice
            torch.save(model_unet.state_dict(), unet_path)
            print(f"Saved best U-Net checkpoint: avg_dice={best_unet:.4f}")

        with open(metrics_csv, 'a', newline='') as f:
            w = csv.writer(f)
            w.writerow(['unet', epoch + 1, train_loss, val_loss, val_dice_liver, val_dice_inst, opt_unet.param_groups[0]['lr']])

    print("\nTraining complete.")
    print(f"Stage1: {stage1_path}")
    print(f"Stage2: {stage2_path}")
    print(f"U-Net:  {unet_path}")

if __name__ == "__main__":
    main()
