import os
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from src.cholec_dataset import CholecSeg8kDataset, get_transforms
from src.model import get_model, HybridLoss
from tqdm import tqdm
import torch.cuda.amp as amp
import random

def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    running_loss = 0.0
    pbar = tqdm(loader, desc="Training")
    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        
        with amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, masks)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
        
    return running_loss / len(loader)

def create_unified_dataset(config):
    """Create dataset from all available sources"""
    datasets = []
    
    # CholecSeg8k dataset
    if os.path.exists(config['cholecseg8k_path']):
        print(f"Loading CholecSeg8k from {config['cholecseg8k_path']}")
        cholec_dataset = CholecSeg8kDataset(
            root_dir=config['cholecseg8k_path'],
            transform=get_transforms(is_train=True),
            max_samples=800
        )
        datasets.append(cholec_dataset)
    
    # TODO: Add other datasets when their structure is known
    # CholecInstanceSeg dataset - need to check structure
    # Reference image set - need to check structure
    
    if not datasets:
        raise ValueError("No valid datasets found!")
    
    # Combine all datasets
    unified_dataset = ConcatDataset(datasets)
    print(f"Total samples: {len(unified_dataset)}")
    return unified_dataset

def main():
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        print("cuDNN benchmark enabled for maximum GPU throughput.")
    
    scaler = amp.GradScaler()
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # ---------------- Stage 1: U-Net ResNet34 Baseline ----------------
    print("\n--- Training U-Net ResNet34 Baseline ---")
    unet_path = 'models/unet_resnet34.pth'
    
    train_dataset = create_unified_dataset(config)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=config['num_workers'],
        pin_memory=True,
        persistent_workers=(config['num_workers'] > 0)
    )
    
    # Train U-Net baseline
    unet_model = get_model(architecture='unet', encoder='resnet34', num_classes=3).to(device)
    optimizer = optim.Adam(unet_model.parameters(), lr=1e-4)
    criterion = HybridLoss(num_classes=3)
    
    epochs = 20
    print(f"Training U-Net for {epochs} epochs...")
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        loss = train_one_epoch(unet_model, train_loader, optimizer, criterion, device, scaler)
        print(f"U-Net epoch {epoch+1} loss: {loss:.4f}")
        
    torch.save(unet_model.state_dict(), unet_path)
    print("✅ U-Net baseline model saved.")

    # ---------------- Stage 2: DeepLabV3+ ResNet50 Advanced ----------------
    print("\n--- Training DeepLabV3+ ResNet50 Advanced ---")
    deeplab_path = 'models/deeplabv3plus_resnet50.pth'
    
    # Train DeepLabV3+ advanced
    deeplab_model = get_model(architecture='deeplabv3plus', encoder='resnet50', num_classes=3).to(device)
    optimizer = optim.Adam(deeplab_model.parameters(), lr=5e-5)
    criterion = HybridLoss(num_classes=3)
    
    epochs = 25
    print(f"Training DeepLabV3+ for {epochs} epochs...")
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        loss = train_one_epoch(deeplab_model, train_loader, optimizer, criterion, device, scaler)
        print(f"DeepLabV3+ epoch {epoch+1} loss: {loss:.4f}")
        
    torch.save(deeplab_model.state_dict(), deeplab_path)
    print("✅ DeepLabV3+ advanced model saved.")
    
    # ---------------- Stage 3: Create Stage 1 Anatomy Model ----------------
    print("\n--- Creating Stage 1 Anatomy Model (2-class) ---")
    stage1_path = 'models/deeplabv3plus_resnet50_stage1.pth'
    
    # Create 2-class model for anatomy-only
    anatomy_model = get_model(architecture='deeplabv3plus', encoder='resnet50', num_classes=2).to(device)
    optimizer = optim.Adam(anatomy_model.parameters(), lr=1e-4)
    criterion = HybridLoss(num_classes=2)
    
    epochs = 15
    print(f"Training Stage 1 anatomy model for {epochs} epochs...")
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        loss = train_one_epoch(anatomy_model, train_loader, optimizer, criterion, device, scaler)
        print(f"Stage 1 epoch {epoch+1} loss: {loss:.4f}")
        
    torch.save(anatomy_model.state_dict(), stage1_path)
    print("✅ Stage 1 anatomy model saved.")
    
    print("\n🎉 All models trained successfully!")
    print("Models created:")
    print("- U-Net ResNet34: models/unet_resnet34.pth")
    print("- DeepLabV3+ ResNet50: models/deeplabv3plus_resnet50.pth")
    print("- Stage 1 Anatomy: models/deeplabv3plus_resnet50_stage1.pth")

if __name__ == "__main__":
    main()
