import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime
import yaml
from tqdm import tqdm
import pandas as pd

import cv2
import numpy as np
from dataset import PinnacleSurgicalDataset, get_pinnacle_transforms
from model import get_model, AdvancedSurgicalLoss

# EMA Implementation for Weight Stability
class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {name: param.data.clone() for name, param in model.named_parameters() if param.requires_grad}

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]

    def apply_to(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name])

class MetricsTracker:
    """Computes and stores surgical metrics (Dice, IoU) for analytics."""
    def __init__(self, log_path="logs/training_log.csv"):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.history = []

    def calculate_metrics(self, pred, target, num_classes=3):
        pred = torch.argmax(pred, dim=1)
        metrics = {}
        
        for cls in range(1, num_classes):
            p = (pred == cls).float()
            t = (target == cls).float()
            
            intersection = (p * t).sum().item()
            p_sum = p.sum().item()
            t_sum = t.sum().item()
            
            # Use 0 if the class is completely absent
            dice = (2. * intersection + 1e-6) / (p_sum + t_sum + 1e-6)
            iou = (intersection + 1e-6) / (p_sum + t_sum - intersection + 1e-6)
            precision = (intersection + 1e-6) / (p_sum + 1e-6)
            recall = (intersection + 1e-6) / (t_sum + 1e-6)
            
            name = "Liver" if cls == 1 else "Instrument"
            metrics[f"{name}_Dice"] = dice
            metrics[f"{name}_IoU"] = iou
            metrics[f"{name}_Prec"] = precision
            metrics[f"{name}_Recall"] = recall
            
        return metrics

    def log_epoch(self, epoch, train_loss, val_loss, val_metrics):
        row = {"Epoch": epoch, "Train_Loss": train_loss, "Val_Loss": val_loss}
        row.update(val_metrics)
        self.history.append(row)
        pd.DataFrame(self.history).to_csv(self.log_path, index=False)

def train_one_epoch(model, loader, optimizer, loss_fn, device, scaler, ema=None):
    model.train()
    total_loss = 0
    # Use leave=True to ensure the bar stays visible after the epoch finishes
    pbar = tqdm(loader, desc="🚀 Training", leave=True)
    
    for i, (images, masks) in enumerate(pbar):
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        if ema:
            ema.update()
            
        total_loss += loss.item()
        
        # Real-time feedback: loss updates every batch
        avg_loss = total_loss / (pbar.n + 1)
        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            avg=f"{avg_loss:.4f}",
            vram=f"{torch.cuda.memory_allocated()/1024**2:.0f}MB",
            refresh=False
        )
        
        # Periodic Heartbeat for Terminal Visibility
        if i % 50 == 0:
            print(f" > Batch {i}/{len(loader)} | Loss: {loss.item():.4f} | Avg: {avg_loss:.4f}")
        
    return total_loss / len(loader)

def validate(model, loader, loss_fn, device, tracker=None):
    model.eval()
    total_loss = 0
    all_metrics = []
    pbar = tqdm(loader, desc="🔍 Validating", leave=False)
    
    with torch.no_grad():
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            total_loss += loss.item()
            
            if tracker:
                batch_metrics = tracker.calculate_metrics(outputs, masks)
                all_metrics.append(batch_metrics)
            
            pbar.set_postfix(loss=f"{loss.item():.4f}")
                
    avg_loss = total_loss / len(loader)
    
    if all_metrics:
        avg_metrics = {key: sum(m[key] for m in all_metrics) / len(all_metrics) for key in all_metrics[0]}
        return avg_loss, avg_metrics
    return avg_loss, {}

def main():
    # Load Config
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using Device: {device}")
    
    # GPU Sweet Spot Optimization
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("⚡ GPU Sweet Spot Enabled: TF32 + cuDNN Benchmark")
    
    # 1. Dataset Setup (Top of Pinnacle Unified Sources)
    roots = {
        'cholecseg8k': 'C:\\Users\\akash\\Downloads\\cholecseg8k',
        'cholecinstanceseg': 'C:\\Users\\akash\\Downloads\\cholecinstanceseg',
        'reference_set': 'C:\\Users\\akash\\Downloads\\cholecinstance_seg_reference_image_set'
    }
    
    img_size = config.get('input_size', [256, 256])
    batch_size = 8
    num_workers = 4
    
    from dataset import PinnacleSurgicalDataset
    train_ds = PinnacleSurgicalDataset(
        roots_dict=roots,
        split='train',
        transform=get_pinnacle_transforms(img_size, is_train=True),
        target_size=img_size
    )
    val_ds = PinnacleSurgicalDataset(
        roots_dict=roots,
        split='val',
        transform=get_pinnacle_transforms(img_size, is_train=False),
        target_size=img_size
    )
    # 2. Optimized Dataloaders (Standard Shuffling)
    # Statistical Audit: 99.7% of frames have tools. 
    # WeightedRandomSampler is mathematically redundant at image level.
    # We rely on Pixel Intensification (Loss Weight 5.0) instead.
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    # 2. Model & Loss Setup
    model = get_model(architecture='deeplabv3plus', encoder='resnet101', num_classes=3).to(device)
    loss_fn = AdvancedSurgicalLoss(num_classes=3).to(device)
    scaler = torch.amp.GradScaler('cuda')
    ema = EMA(model)
    
    # 3. Training Loop (Stage 3 - Pixel-Intensive Optimization)
    epochs = 80 # Deep convergence for 2.99% tool class
    lr = float(config.get('stage2', {}).get('lr', 1e-4))
    
    tracker = MetricsTracker()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    # Smoother 3x multiplier for medical stability
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr*3, epochs=epochs, steps_per_epoch=len(train_loader))
    
    best_val_loss = float('inf')
    model_save_path = "models/pinnacle_deeplab_r101.pth"
    os.makedirs('models', exist_ok=True)
    
    print(f"🔥 Starting Pinnacle Training ({epochs} epochs)...")
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, scaler, ema)
        val_loss, val_metrics = validate(model, val_loader, loss_fn, device, tracker)
        scheduler.step()
        
        # Professional Terminal Logging
        print(f"\n[Epoch {epoch+1:02d}] " +
              f"Loss: {train_loss:.4f} | Val: {val_loss:.4f} | " +
              f"T-Dice: {val_metrics.get('Instrument_Dice', 0):.3f} | T-Prec: {val_metrics.get('Instrument_Prec', 0):.3f} | T-Rec: {val_metrics.get('Instrument_Recall', 0):.3f}")
        
        tracker.log_epoch(epoch+1, train_loss, val_loss, val_metrics)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ema.apply_to(model)
            torch.save(model.state_dict(), model_save_path)
            print(f"⭐ BEST MODEL SAVED: {model_save_path}")
            
    # Repeat for U-Net
    print("\n🏗️ Switching to U-Net Pinnacle Training...")
    model_unet = get_model(architecture='unet', encoder='efficientnet-b4', num_classes=3).to(device)
    ema_unet = EMA(model_unet)
    optimizer_unet = optim.AdamW(model_unet.parameters(), lr=lr, weight_decay=1e-3)
    # Smoother 3x multiplier
    scheduler_unet = optim.lr_scheduler.OneCycleLR(optimizer_unet, max_lr=lr*3, epochs=epochs, steps_per_epoch=len(train_loader))
    model_save_path_unet = "models/pinnacle_unet_eb4.pth"
    tracker_unet = MetricsTracker(log_path="logs/training_log_unet.csv")
    
    best_val_loss = float('inf')
    for epoch in range(epochs):
        train_loss = train_one_epoch(model_unet, train_loader, optimizer_unet, loss_fn, device, scaler, ema_unet)
        val_loss, val_metrics = validate(model_unet, val_loader, loss_fn, device, tracker_unet)
        scheduler_unet.step()
        
        print(f"\n[U-Net {epoch+1:02d}] " +
              f"Loss: {train_loss:.4f} | Val: {val_loss:.4f} | " +
              f"L-Dice: {val_metrics.get('Liver_Dice', 0):.3f} | L-IoU: {val_metrics.get('Liver_IoU', 0):.3f} | " +
              f"T-Dice: {val_metrics.get('Instrument_Dice', 0):.3f} | T-IoU: {val_metrics.get('Instrument_IoU', 0):.3f}")
        
        tracker_unet.log_epoch(epoch+1, train_loss, val_loss, val_metrics)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ema_unet.apply_to(model_unet)
            torch.save(model_unet.state_dict(), model_save_path_unet)
            print(f"⭐ BEST U-NET SAVED: {model_save_path_unet}")

if __name__ == "__main__":
    main()
