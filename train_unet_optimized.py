import os
import csv
import random
from datetime import datetime

import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, ConcatDataset
try:
    from src.cholec_dataset import CholecSeg8kDataset, get_transforms
    from src.instance_dataset import CholecInstancePolygonDataset
    from src.model import get_model, HybridLoss
except ImportError:
    from cholec_dataset import CholecSeg8kDataset, get_transforms
    from instance_dataset import CholecInstancePolygonDataset
    from model import get_model, HybridLoss
from tqdm import tqdm
import torch.nn.functional as F


def _dice_score(pred: torch.Tensor, target: torch.Tensor, cls: int, eps: float = 1e-6, ignore_index: int = 255) -> float:
    valid = (target != ignore_index).float()
    pred_c = ((pred == cls).float() * valid)
    target_c = ((target == cls).float() * valid)
    
    if target_c.sum() == 0:
        return 0.0
    
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
                    if torch.is_floating_point(dst) and torch.is_floating_point(src):
                        dst.mul_(self.decay).add_(src, alpha=1.0 - self.decay)
                    else:
                        self.shadow[k] = src.clone()

    def apply_to(self, model: torch.nn.Module):
        model.load_state_dict(self.shadow, strict=True)


class AdaptiveHybridLoss(torch.nn.Module):
    """Enhanced loss function with adaptive weighting and liver emphasis"""
    def __init__(self, num_classes=3, ignore_index=255):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        
        # Base loss components
        self.dice_loss = torch.nn.MSELoss()  # Simplified for stability
        self.ce_loss = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        
        # Adaptive weights (will be updated during training)
        self.register_buffer('liver_weight', torch.tensor(1.0))
        self.register_buffer('inst_weight', torch.tensor(1.0))
        
    def update_weights(self, epoch, liver_dice, inst_dice):
        """Dynamically adjust weights based on performance"""
        # Boost liver weight if it's struggling
        if liver_dice < 0.3 and epoch > 5:
            self.liver_weight = min(3.0, 1.0 + (0.3 - liver_dice) * 5)
        else:
            self.liver_weight = 1.0
            
        # Reduce instrument weight if it's dominating
        if inst_dice > 0.6 and liver_dice < 0.2:
            self.inst_weight = 0.5
        else:
            self.inst_weight = 1.0
    
    def forward(self, y_pred, y_true):
        # Standard cross-entropy
        ce_loss = self.ce_loss(y_pred, y_true)
        
        # Class-specific dice losses
        y_true_onehot = F.one_hot(y_true, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        
        # Liver dice
        liver_pred = y_pred[:, 1].sigmoid()
        liver_target = y_true_onehot[:, 1]
        liver_dice = 1 - (2.0 * (liver_pred * liver_target).sum() + 1e-6) / (
            liver_pred.sum() + liver_target.sum() + 1e-6)
        
        # Instrument dice
        inst_pred = y_pred[:, 2].sigmoid()
        inst_target = y_true_onehot[:, 2]
        inst_dice = 1 - (2.0 * (inst_pred * inst_target).sum() + 1e-6) / (
            inst_pred.sum() + inst_target.sum() + 1e-6)
        
        # Weighted combination
        total_loss = (0.5 * ce_loss + 
                      0.25 * self.liver_weight * liver_dice + 
                      0.25 * self.inst_weight * inst_dice)
        
        return total_loss, liver_dice.item(), inst_dice.item()


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


def _run_epoch_optimized(model, loader, optimizer, scheduler, loss_fn, device, scaler, ema, is_train: bool, amp_enabled: bool, epoch: int):
    if is_train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    dice_liver = 0.0
    dice_inst = 0.0
    n_batches = 0
    inst_samples = 0
    
    # Track individual losses for adaptive weighting
    liver_losses = []
    inst_losses = []

    pbar = tqdm(loader, desc="Training" if is_train else "Validation")
    for images, masks in pbar:
        if device.type == 'cuda':
            images = images.contiguous(memory_format=torch.channels_last)
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        # Enhanced validation
        masks = torch.clamp(masks, 0, 2).long()

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.no_grad() if not is_train else torch.enable_grad():
            outputs = model(images)
            
            # Get loss and individual dice scores
            loss, liver_dice_batch, inst_dice_batch = loss_fn(outputs, masks)
            
            # Store for adaptive weighting
            if is_train:
                liver_losses.append(liver_dice_batch)
                inst_losses.append(inst_dice_batch)
            
            # Robust loss validation
            if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 50.0:
                print(f"🚨 Bad loss detected: {loss.item()}, batch skipped")
                continue
            
            # Enhanced gradient checking
            if torch.any(torch.isnan(outputs)).item() or torch.any(torch.isinf(outputs)).item():
                print(f"🚨 NaN/Inf in model outputs, batch skipped")
                continue
            
            if is_train:
                # Gradient clipping with adaptive norm
                loss.backward()
                
                # Check gradient norms
                total_grad_norm = 0.0
                for param in model.parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.data.norm(2)
                        total_grad_norm += grad_norm.item() ** 2
                        if torch.isnan(param.grad) or torch.isinf(param.grad):
                            print(f"🚨 NaN/Inf gradient detected, batch skipped")
                            optimizer.zero_grad()
                            continue
                
                total_grad_norm = total_grad_norm ** (1. / 2)
                
                # Adaptive gradient clipping
                max_norm = 1.0 if epoch < 10 else 0.5  # More permissive early
                if total_grad_norm > max_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
                
                optimizer.step()

                if scheduler is not None and not hasattr(scheduler, 'mode'):
                    scheduler.step()

                if ema is not None:
                    ema.update(model)

        running_loss += float(loss.item())
        n_batches += 1

        with torch.no_grad():
            pred = torch.argmax(outputs, dim=1)
            dice_liver += _dice_score(pred, masks, cls=1)
            dice_inst += _dice_score(pred, masks, cls=2)
            
            if (masks == 2).sum() > 0:
                inst_samples += 1

        # Enhanced progress reporting
        if is_train and len(liver_losses) > 0:
            avg_liver_loss = sum(liver_losses[-10:]) / min(10, len(liver_losses))
            avg_inst_loss = sum(inst_losses[-10:]) / min(10, len(inst_losses))
            pbar.set_postfix({
                'loss': float(loss.item()), 
                'liver_dice': dice_liver / n_batches,
                'inst_dice': dice_inst / n_batches,
                'liver_loss': avg_liver_loss,
                'inst_loss': avg_inst_loss
            })
        else:
            pbar.set_postfix({
                'loss': float(loss.item()), 
                'liver_dice': dice_liver / n_batches,
                'inst_dice': dice_inst / n_batches
            })

    avg_loss = running_loss / max(1, n_batches)
    avg_dice_liver = dice_liver / max(1, n_batches)
    avg_dice_inst = dice_inst / max(1, n_batches)
    
    return avg_loss, avg_dice_liver, avg_dice_inst, inst_samples


def main():
    print("🚀 OPTIMIZED U-NET TRAINING - BEST OF BEST SETUP")
    print("=" * 60)
    
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
         
    seed = int(config.get('seed', 42))
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
     
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass

    amp_enabled = False
    scaler = None

    os.makedirs('models', exist_ok=True)

    # Dataset setup
    dataset_root = config.get('cholecseg8k_path') or config.get('unified_dataset_path')
    if not dataset_root or not os.path.exists(dataset_root):
        raise FileNotFoundError(f"Dataset path not found: {dataset_root}")

    full_ds = CholecSeg8kDataset(
        root_dir=dataset_root,
        transform=get_transforms(is_train=True),
        target_size=tuple(config.get('input_size', [256, 256])),
        max_samples=None,
    )

    val_ratio = float(config.get('val_ratio', 0.15))
    train_idx, val_idx, val_videos = _split_by_video(full_ds, val_ratio=val_ratio, seed=seed)
    train_ds_seg8k = Subset(full_ds, train_idx)
    val_ds = Subset(full_ds, val_idx)

    # Polygon dataset for mixed supervision
    poly_ann_root = config.get('cholecinstanceseg_path')
    poly_img_root = config.get('reference_images_path')
    poly_ds = None
    if poly_ann_root and poly_img_root and os.path.exists(poly_ann_root) and os.path.exists(poly_img_root):
        poly_ds = CholecInstancePolygonDataset(
            ann_root=poly_ann_root,
            img_root=poly_img_root,
            split='train',
            target_size=tuple(config.get('input_size', [256, 256])),
        )

    # Optimized batch size
    if torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if gpu_memory_gb >= 6:
            batch_size = 6
        elif gpu_memory_gb >= 4:
            batch_size = 4
        else:
            batch_size = 2
    else:
        batch_size = 2
    
    num_workers = min(4, os.cpu_count())

    dl_common = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': (device.type == 'cuda'),
        'persistent_workers': False,
        'drop_last': True,
    }
    if num_workers > 0:
        dl_common['prefetch_factor'] = 4

    # Combined dataset
    if poly_ds is not None and len(poly_ds) > 0:
        train_ds_s2 = ConcatDataset([train_ds_seg8k, poly_ds])
        print(f"Polygon samples (train): {len(poly_ds)}")
        print(f"U-Net train total (seg8k+poly): {len(train_ds_s2)}")
    else:
        train_ds_s2 = train_ds_seg8k

    train_loader = DataLoader(train_ds_s2, shuffle=True, **dl_common)
    val_loader = DataLoader(val_ds, **{**dl_common, 'shuffle': False})

    # Logging setup
    log_dir = os.path.join('logs', datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(log_dir, exist_ok=True)
    metrics_csv = os.path.join(log_dir, 'metrics.csv')
    with open(metrics_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['stage', 'epoch', 'train_loss', 'val_loss', 'val_dice_liver', 'val_dice_inst', 'lr', 'liver_weight', 'inst_weight'])

    # ---------------- OPTIMIZED U-NET TRAINING ----------------
    unet_path = os.path.join('models', 'unet_optimized_best.pth')
    print(f"\n--- Optimized U-Net Training -> {unet_path} ---")
    
    # Best hyperparameters based on analysis
    unet_epochs = 60  # Same as DeepLabV3+
    unet_lr = 5e-5    # Same as DeepLabV3+
    
    print(f"Optimized U-Net settings:")
    print(f"  Epochs: {unet_epochs}")
    print(f"  Learning Rate: {unet_lr}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Architecture: U-Net ResNet50 (upgraded from ResNet34)")
    
    # Use better encoder
    model_unet = get_model(architecture='unet', encoder='resnet50', num_classes=3).to(device)
    
    # Optimizer with weight decay
    opt_unet = optim.AdamW(model_unet.parameters(), lr=unet_lr, weight_decay=1e-4)
    
    # Adaptive loss function
    loss_unet = AdaptiveHybridLoss(num_classes=3)
    
    # Better scheduler (same as DeepLabV3+)
    steps_per_epoch = max(1, len(train_loader))
    sched_unet = optim.lr_scheduler.ReduceLROnPlateau(
        opt_unet, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # EMA for stability
    ema_unet = _EMA(model_unet, decay=0.999)
    
    # Training metrics
    best_unet = -1.0
    RESET_EPOCH = 30  # Reset metrics halfway through
    
    print(f"\n🎯 TRAINING STRATEGY:")
    print(f"  • Pre-trained from ImageNet (ResNet50)")
    print(f"  • Adaptive loss weighting for liver emphasis")
    print(f"  • ReduceLROnPlateau scheduler")
    print(f"  • Metric reset at epoch {RESET_EPOCH}")
    print(f"  • Enhanced gradient clipping")
    print(f"  • EMA for stable validation")
    
    for epoch in range(unet_epochs):
        print(f"\nEpoch {epoch+1}/{unet_epochs}")
        
        # Reset best metric halfway through (like DeepLabV3+)
        if epoch + 1 == RESET_EPOCH:
            print(f"🔄 Resetting best_unet metric at epoch {RESET_EPOCH}")
            best_unet = -1.0
        
        # Training
        train_loss, _, _, _ = _run_epoch_optimized(
            model_unet, train_loader, opt_unet, sched_unet, loss_unet, 
            device, scaler, ema_unet, is_train=True, amp_enabled=amp_enabled, epoch=epoch
        )

        # Validation with EMA
        model_tmp = get_model(architecture='unet', encoder='resnet50', num_classes=3).to(device)
        model_tmp.load_state_dict(model_unet.state_dict(), strict=True)
        ema_unet.apply_to(model_tmp)
        
        val_loss, val_dice_liver, val_dice_inst, val_inst_samples = _run_epoch_optimized(
            model_tmp, val_loader, opt_unet, None, loss_unet, 
            device, scaler, None, is_train=False, amp_enabled=amp_enabled, epoch=epoch
        )
        
        # Update adaptive loss weights
        loss_unet.update_weights(epoch, val_dice_liver, val_dice_inst)
        
        avg_dice = (val_dice_liver + val_dice_inst) / 2.0
        if avg_dice > best_unet:
            best_unet = avg_dice
            torch.save(model_unet.state_dict(), unet_path)
            print(f"🏆 NEW BEST U-Net checkpoint: avg_dice={best_unet:.4f}")
            print(f"   Liver: {val_dice_liver:.4f}, Instrument: {val_dice_inst:.4f}")
        
        # Step scheduler
        sched_unet.step(avg_dice)
        
        # Enhanced reporting
        print(f"📊 Epoch {epoch+1} Results:")
        print(f"   Train Loss: {train_loss:.4f}")
        print(f"   Val Loss: {val_loss:.4f}")
        print(f"   Liver Dice: {val_dice_liver:.4f}")
        print(f"   Instrument Dice: {val_dice_inst:.4f}")
        print(f"   Average Dice: {avg_dice:.4f}")
        print(f"   Learning Rate: {opt_unet.param_groups[0]['lr']:.2e}")
        print(f"   Liver Weight: {loss_unet.liver_weight.item():.2f}")
        print(f"   Instrument Weight: {loss_unet.inst_weight.item():.2f}")

        # Log metrics
        with open(metrics_csv, 'a', newline='') as f:
            w = csv.writer(f)
            w.writerow([
                'unet_optimized', epoch + 1, train_loss, val_loss, 
                val_dice_liver, val_dice_inst, opt_unet.param_groups[0]['lr'],
                loss_unet.liver_weight.item(), loss_unet.inst_weight.item()
            ])

    print(f"\n🎉 OPTIMIZED U-NET TRAINING COMPLETE!")
    print(f"Best checkpoint: {unet_path}")
    print(f"Best average dice: {best_unet:.4f}")
    print(f"Training log: {metrics_csv}")

if __name__ == "__main__":
    main()
