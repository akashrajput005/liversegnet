import os
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
try:
    from dataset import LiverInstrumentDataset, get_transforms
    from model import get_model, HybridLoss
except ImportError:
    from src.dataset import LiverInstrumentDataset, get_transforms
    from src.model import get_model, HybridLoss
from tqdm import tqdm
import torch.cuda.amp as amp

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

def main():
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        print("cuDNN benchmark enabled for maximum GPU throughput.")
    
    scaler = amp.GradScaler()
    
    # ---------------- Stage 1: Liver Only ----------------
    stage1_path = 'models/deeplab_resnet50_stage1.pth'
    if os.path.exists(stage1_path):
        print(f"\n--- Stage 1 checkpoint found at {stage1_path}. Skipping Stage 1. ---")
    else:
        print("\n--- Starting Stage 1: Advanced (Liver Only) ---")
        train_dataset = LiverInstrumentDataset(
            unified_root=config['unified_dataset_path'],
            split='train',
            transform=get_transforms(is_train=True)
        )
        # Filter for liver-only images (based on filename prefix)
        train_dataset.filenames = [f for f in train_dataset.filenames if f.startswith('liver_')]
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config['batch_size'], 
            shuffle=True, 
            num_workers=config['num_workers'],
            pin_memory=True,
            persistent_workers=(config['num_workers'] > 0)
        )
        
        model = get_model(architecture=config['active_model'], encoder=config['active_encoder'], num_classes=2).to(device)
        optimizer = optim.Adam(model.parameters(), lr=float(config['stage1']['lr']))
        criterion = HybridLoss(num_classes=2)
        
        for epoch in range(config['stage1']['epochs']):
            print(f"Epoch {epoch+1}/{config['stage1']['epochs']}")
            loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
            print(f"Stage 1 epoch {epoch+1} loss: {loss:.4f}")
            
        torch.save(model.state_dict(), stage1_path)
        print("Stage 1 model saved.")

    # ---------------- Stage 2: Fine-tuning ----------------
    print("\n--- Starting Stage 2: Advanced Combined Fine-tuning ---")
    train_dataset = LiverInstrumentDataset(
        unified_root=config['unified_dataset_path'],
        split='train',
        transform=get_transforms(is_train=True)
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=config['num_workers'],
        pin_memory=True,
        persistent_workers=(config['num_workers'] > 0)
    )
    
    model_stage2 = get_model(architecture=config['active_model'], encoder=config['active_encoder'], num_classes=3).to(device)
    
    # Load compatible weights from Stage 1
    state_dict = torch.load('models/deeplab_resnet50_stage1.pth')
    state_dict = {k: v for k, v in state_dict.items() if 'segmentation_head' not in k}
    model_stage2.load_state_dict(state_dict, strict=False)
    
    optimizer = optim.Adam(model_stage2.parameters(), lr=float(config['stage2']['lr']))
    criterion = HybridLoss(num_classes=3)
    
    for epoch in range(config['stage2']['epochs']):
        print(f"Epoch {epoch+1}/{config['stage2']['epochs']}")
        loss = train_one_epoch(model_stage2, train_loader, optimizer, criterion, device, scaler)
        print(f"Stage 2 epoch {epoch+1} loss: {loss:.4f}")
        
    torch.save(model_stage2.state_dict(), 'models/deeplab_resnet50.pth')
    print("Final Advanced model saved.")

if __name__ == "__main__":
    main()
