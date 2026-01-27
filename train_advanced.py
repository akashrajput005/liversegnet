import os
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, random_split
from src.cholec_dataset import CholecSeg8kDataset, get_transforms
from src.model import get_model, HybridLoss
from tqdm import tqdm
import torch.cuda.amp as amp
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import time

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()

class MetricsCalculator:
    def __init__(self, num_classes=3):
        self.num_classes = num_classes
        self.class_names = ['background', 'liver', 'instrument']
        
    def calculate_dice_coefficient(self, pred, target, smooth=1e-6):
        """Calculate Dice coefficient for each class"""
        dice_scores = []
        for cls in range(self.num_classes):
            pred_cls = (pred == cls).float()
            target_cls = (target == cls).float()
            
            intersection = (pred_cls * target_cls).sum()
            union = pred_cls.sum() + target_cls.sum()
            
            dice = (2. * intersection + smooth) / (union + smooth)
            dice_scores.append(dice.item())
            
        return dice_scores
    
    def calculate_iou(self, pred, target, smooth=1e-6):
        """Calculate IoU for each class"""
        iou_scores = []
        for cls in range(self.num_classes):
            pred_cls = (pred == cls).float()
            target_cls = (target == cls).float()
            
            intersection = (pred_cls * target_cls).sum()
            union = pred_cls.sum() + target_cls.sum() - intersection
            
            iou = (intersection + smooth) / (union + smooth)
            iou_scores.append(iou.item())
            
        return iou_scores
    
    def calculate_pixel_accuracy(self, pred, target):
        """Calculate overall pixel accuracy"""
        correct = (pred == target).sum().item()
        total = target.numel()
        return correct / total
    
    def calculate_class_accuracy(self, pred, target):
        """Calculate accuracy for each class"""
        class_acc = []
        for cls in range(self.num_classes):
            pred_cls = (pred == cls)
            target_cls = (target == cls)
            
            if target_cls.sum() > 0:  # Avoid division by zero
                correct = (pred_cls & target_cls).sum().item()
                total = target_cls.sum().item()
                class_acc.append(correct / total)
            else:
                class_acc.append(0.0)
                
        return class_acc

def train_one_epoch(model, loader, optimizer, criterion, device, scaler, metrics_calc, use_focal=False):
    model.train()
    running_loss = 0.0
    all_metrics = []
    
    pbar = tqdm(loader, desc="Training")
    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        
        with amp.autocast():
            outputs = model(images)
            if use_focal:
                loss = criterion(outputs, masks)
            else:
                loss = criterion(outputs, masks)
            
        scaler.scale(loss).backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        
        # Calculate metrics for this batch
        with torch.no_grad():
            pred = torch.argmax(outputs, dim=1)
            batch_metrics = {
                'dice': metrics_calc.calculate_dice_coefficient(pred, masks),
                'iou': metrics_calc.calculate_iou(pred, masks),
                'pixel_acc': metrics_calc.calculate_pixel_accuracy(pred, masks),
                'class_acc': metrics_calc.calculate_class_accuracy(pred, masks)
            }
            all_metrics.append(batch_metrics)
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dice_liver': f'{batch_metrics["dice"][1]:.3f}',
            'dice_inst': f'{batch_metrics["dice"][2]:.3f}'
        })
        
    # Average metrics across all batches
    avg_metrics = {}
    for key in all_metrics[0].keys():
        if key == 'dice' or key == 'iou' or key == 'class_acc':
            avg_metrics[key] = np.mean([m[key] for m in all_metrics], axis=0).tolist()
        else:
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])
    
    return running_loss / len(loader), avg_metrics

def validate_model(model, loader, criterion, device, metrics_calc):
    model.eval()
    running_loss = 0.0
    all_metrics = []
    
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validation"):
            images, masks = images.to(device), masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            running_loss += loss.item()
            
            pred = torch.argmax(outputs, dim=1)
            batch_metrics = {
                'dice': metrics_calc.calculate_dice_coefficient(pred, masks),
                'iou': metrics_calc.calculate_iou(pred, masks),
                'pixel_acc': metrics_calc.calculate_pixel_accuracy(pred, masks),
                'class_acc': metrics_calc.calculate_class_accuracy(pred, masks)
            }
            all_metrics.append(batch_metrics)
    
    # Average metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        if key == 'dice' or key == 'iou' or key == 'class_acc':
            avg_metrics[key] = np.mean([m[key] for m in all_metrics], axis=0).tolist()
        else:
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])
    
    return running_loss / len(loader), avg_metrics

def create_unified_dataset(config):
    """Create dataset with train/val split"""
    datasets = []
    
    if os.path.exists(config['cholecseg8k_path']):
        print(f"Loading CholecSeg8k from {config['cholecseg8k_path']}")
        cholec_dataset = CholecSeg8kDataset(
            root_dir=config['cholecseg8k_path'],
            transform=get_transforms(is_train=True),
            max_samples=1000
        )
        datasets.append(cholec_dataset)
    
    if not datasets:
        raise ValueError("No valid datasets found!")
    
    unified_dataset = ConcatDataset(datasets)
    
    # Create train/val split
    val_size = int(0.2 * len(unified_dataset))
    train_size = len(unified_dataset) - val_size
    train_dataset, val_dataset = random_split(unified_dataset, [train_size, val_size])
    
    print(f"Total samples: {len(unified_dataset)}")
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    return train_dataset, val_dataset

def print_metrics(metrics, prefix=""):
    """Print metrics in a formatted way"""
    class_names = ['background', 'liver', 'instrument']
    
    print(f"\n{prefix} Metrics:")
    print(f"  Pixel Accuracy: {metrics['pixel_acc']:.4f}")
    
    print(f"  Dice Coefficients:")
    for i, name in enumerate(class_names):
        print(f"    {name}: {metrics['dice'][i]:.4f}")
    
    print(f"  IoU Scores:")
    for i, name in enumerate(class_names):
        print(f"    {name}: {metrics['iou'][i]:.4f}")
    
    print(f"  Class Accuracy:")
    for i, name in enumerate(class_names):
        print(f"    {name}: {metrics['class_acc'][i]:.4f}")

def main():
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        print("cuDNN benchmark enabled for maximum GPU throughput.")
    
    scaler = amp.GradScaler()
    metrics_calc = MetricsCalculator(num_classes=3)
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Create datasets
    train_dataset, val_dataset = create_unified_dataset(config)
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    # ---------------- Model 1: U-Net ResNet34 Baseline ----------------
    print("\n" + "="*60)
    print("🏥 TRAINING U-NET RESNET34 BASELINE")
    print("="*60)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=config['num_workers'],
        pin_memory=True,
        persistent_workers=(config['num_workers'] > 0)
    )
    
    unet_model = get_model(architecture='unet', encoder='resnet34', num_classes=3).to(device)
    optimizer = optim.AdamW(unet_model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)
    criterion = HybridLoss(num_classes=3)
    focal_criterion = FocalLoss(alpha=1, gamma=2)
    
    best_val_dice = 0.0
    patience = 10
    patience_counter = 0
    
    epochs = 40
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Training
        train_loss, train_metrics = train_one_epoch(
            unet_model, train_loader, optimizer, focal_criterion, device, scaler, metrics_calc, use_focal=True
        )
        
        # Validation
        val_loss, val_metrics = validate_model(unet_model, val_loader, focal_criterion, device, metrics_calc)
        
        # Learning rate scheduling
        scheduler.step()
        
        # Print metrics
        print_metrics(train_metrics, "Train")
        print_metrics(val_metrics, "Val")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Save best model
        avg_dice_liver = val_metrics['dice'][1]
        avg_dice_instrument = val_metrics['dice'][2]
        avg_dice = (avg_dice_liver + avg_dice_instrument) / 2
        
        if avg_dice > best_val_dice:
            best_val_dice = avg_dice
            torch.save(unet_model.state_dict(), 'models/unet_resnet34.pth')
            print(f"✅ New best model saved! Avg Dice: {avg_dice:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    print(f"✅ U-Net training completed! Best Val Dice: {best_val_dice:.4f}")
    
    # ---------------- Model 2: DeepLabV3+ ResNet50 Advanced ----------------
    print("\n" + "="*60)
    print("🔬 TRAINING DEEPLABV3+ RESNET50 ADVANCED")
    print("="*60)
    
    deeplab_model = get_model(architecture='deeplabv3plus', encoder='resnet50', num_classes=3).to(device)
    optimizer = optim.AdamW(deeplab_model.parameters(), lr=5e-5, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=35, eta_min=1e-6)
    criterion = HybridLoss(num_classes=3)
    focal_criterion = FocalLoss(alpha=1, gamma=2)
    
    best_val_dice = 0.0
    patience_counter = 0
    
    epochs = 45
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        train_loss, train_metrics = train_one_epoch(
            deeplab_model, train_loader, optimizer, focal_criterion, device, scaler, metrics_calc, use_focal=True
        )
        
        val_loss, val_metrics = validate_model(deeplab_model, val_loader, focal_criterion, device, metrics_calc)
        scheduler.step()
        
        print_metrics(train_metrics, "Train")
        print_metrics(val_metrics, "Val")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
        
        avg_dice_liver = val_metrics['dice'][1]
        avg_dice_instrument = val_metrics['dice'][2]
        avg_dice = (avg_dice_liver + avg_dice_instrument) / 2
        
        if avg_dice > best_val_dice:
            best_val_dice = avg_dice
            torch.save(deeplab_model.state_dict(), 'models/deeplabv3plus_resnet50.pth')
            print(f"✅ New best model saved! Avg Dice: {avg_dice:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    print(f"✅ DeepLabV3+ training completed! Best Val Dice: {best_val_dice:.4f}")
    
    # ---------------- Model 3: Stage 1 Anatomy (2-class) ----------------
    print("\n" + "="*60)
    print("🎯 TRAINING STAGE 1 ANATOMY MODEL (2-CLASS)")
    print("="*60)
    
    # Create 2-class dataset (background + liver only)
    anatomy_model = get_model(architecture='deeplabv3plus', encoder='resnet50', num_classes=2).to(device)
    optimizer = optim.AdamW(anatomy_model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-6)
    criterion = HybridLoss(num_classes=2)
    focal_criterion = FocalLoss(alpha=1, gamma=2)
    
    # Update metrics calculator for 2 classes
    anatomy_metrics = MetricsCalculator(num_classes=2)
    
    best_val_dice = 0.0
    patience_counter = 0
    
    epochs = 30
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        train_loss, train_metrics = train_one_epoch(
            anatomy_model, train_loader, optimizer, focal_criterion, device, scaler, anatomy_metrics, use_focal=True
        )
        
        val_loss, val_metrics = validate_model(anatomy_model, val_loader, focal_criterion, device, anatomy_metrics)
        scheduler.step()
        
        print_metrics(train_metrics, "Train")
        print_metrics(val_metrics, "Val")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
        
        # For 2-class, focus on liver dice
        liver_dice = val_metrics['dice'][1]
        if liver_dice > best_val_dice:
            best_val_dice = liver_dice
            torch.save(anatomy_model.state_dict(), 'models/deeplabv3plus_resnet50_stage1.pth')
            print(f"✅ New best anatomy model saved! Liver Dice: {liver_dice:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    print(f"✅ Stage 1 anatomy training completed! Best Liver Dice: {best_val_dice:.4f}")
    
    print("\n" + "="*60)
    print("🎉 ALL MODELS TRAINED SUCCESSFULLY!")
    print("="*60)
    print("Models created:")
    print("📁 U-Net ResNet34: models/unet_resnet34.pth")
    print("📁 DeepLabV3+ ResNet50: models/deeplabv3plus_resnet50.pth")
    print("📁 Stage 1 Anatomy: models/deeplabv3plus_resnet50_stage1.pth")
    print("\nReady for surgical segmentation! 🏥")

if __name__ == "__main__":
    main()
