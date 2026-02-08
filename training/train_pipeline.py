import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.deeplab_liver import LiverSegModelA
from models.unet_tools import UNetResNet34
from datasets.choleseg8k import CholeSeg8kDataset
from datasets.cholec_instance import CholecInstanceSegDataset
from utils.transforms import Compose, Resize, RandomHorizontalFlip, RandomRotation, ColorJitter, ToTensor, Normalize
from utils.metrics import calculate_boundary_precision
from training.losses import SurgicalHybridLoss
import logging
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device, save_path, accumulation_steps=4, recovery_mode=False):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.save_path = save_path
        self.accumulation_steps = accumulation_steps
        self.recovery_mode = recovery_mode
        self.model.to(device)
        self.model.to(device)
        self.scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
        self.scheduler = None

    def train_epoch(self, epoch):
        self.model.train()
        
        # --- V2.0.6: CLINICAL RECOVERY FREEZE ---
        # If in recovery_mode, backbone remains frozen permanently
        if self.recovery_mode or epoch < 5:
            if hasattr(self.model, 'model'): # Model A (DeepLab)
                for param in self.model.model.backbone.parameters():
                    param.requires_grad = False
            elif hasattr(self.model, 'encoder'): # Model B (UNet)
                for param in self.model.encoder.parameters():
                    param.requires_grad = False
        else:
            # Unfreeze for fine-tuning (non-recovery mode)
            for param in self.model.parameters():
                param.requires_grad = True

        running_loss = 0.0
        self.optimizer.zero_grad(set_to_none=True)
        
        for i, (images, masks) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)
            
            with torch.amp.autocast('cuda', enabled=(self.scaler is not None)):
                outputs = self.model(images)
                if isinstance(outputs, dict): outputs = outputs['out']
                loss = self.criterion(outputs, masks.long() if len(masks.shape)==3 else masks)
                # Scale loss for accumulation
                loss = loss / self.accumulation_steps
            
            if self.scaler:
                self.scaler.scale(loss).backward()
                if (i + 1) % self.accumulation_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.scheduler: self.scheduler.step()
            else:
                loss.backward()
                if (i + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.scheduler: self.scheduler.step()
            
            running_loss += loss.item() * self.accumulation_steps
            if (i + 1) % 100 == 0:
                logging.info(f"Step [{i+1}/{len(self.train_loader)}] - Loss: {loss.item() * self.accumulation_steps:.4f}")
            
            # --- V2.0.8: INTERIM CHECKPOINTING ---
            if (i+1) % 500 == 0:
                checkpoint_dir = os.path.dirname(self.save_path)
                os.makedirs(checkpoint_dir, exist_ok=True)
                torch.save(self.model.state_dict(), self.save_path)
                logging.info(f"Interim Checkpoint Saved at Step {i+1}")
                
        return running_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        boundary_precisions = []
        
        with torch.no_grad():
            for images, masks in self.val_loader:
                images, masks = images.to(self.device), masks.to(self.device)
                outputs = self.model(images)
                if isinstance(outputs, dict): outputs = outputs['out']
                
                loss = self.criterion(outputs, masks.long() if len(masks.shape)==3 else masks)
                running_loss += loss.item()
                
                # Clinical Metrics
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1).cpu().numpy()
                gt_numpy = masks.cpu().numpy()
                
                for i in range(preds.shape[0]):
                    bp = calculate_boundary_precision(preds[i], gt_numpy[i])
                    boundary_precisions.append(bp)
                    
        avg_bp = np.mean(boundary_precisions) if boundary_precisions else 0.0
        logging.info(f"Clinical Validation - Avg Boundary Precision: {avg_bp:.4f}")
        return running_loss / len(self.val_loader)

    def run(self, num_epochs, dry_run=False):
        best_val_loss = float('inf')
        max_epochs = 1 if dry_run else num_epochs
        
        # --- V2.0.3: OneCycleLR Scheduler ---
        if not dry_run:
            steps_per_epoch = len(self.train_loader) // self.accumulation_steps
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=1e-4, 
                steps_per_epoch=steps_per_epoch, 
                epochs=max_epochs
            )
            
        for epoch in range(max_epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            logging.info(f"Epoch {epoch+1}/{max_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if dry_run:
                logging.info("Dry Run Complete. Hardware Integrity Verified.")
                return

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_dir = os.path.dirname(self.save_path)
                os.makedirs(checkpoint_dir, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': val_loss,
                }, self.save_path)

def start_training(model_type='A', stage=1, root_dirs={}, dry_run=False, recovery_mode=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # SOTA Transform Pipeline (Spatial + Color)
    transform = Compose([
        Resize((256, 256)),
        RandomHorizontalFlip(),
        RandomRotation(degrees=15),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if model_type == 'A':
        model = LiverSegModelA(num_classes=2 if stage==1 else 5, pretrained=not dry_run)
        if stage == 2 and not dry_run:
            s1_path = "./checkpoints/model_A_stage_1/best_model.pth"
            if os.path.exists(s1_path):
                model.load_stage1_weights(s1_path)
    else:
        model = UNetResNet34(num_classes=3 if stage==1 else 5, pretrained=not dry_run)
        if stage == 2 and not dry_run:
            s1_path = "./checkpoints/model_B_stage_1/best_model.pth"
            if os.path.exists(s1_path):
                model.load_stage1_weights(s1_path)
                
    dataset = CholeSeg8kDataset(root_dirs['choleseg8k'], stage=stage, transform=transform) if model_type=='A' else \
              CholecInstanceSegDataset(root_dirs['cholecinstanceseg'], root_dirs['reference'], stage=stage, transform=transform)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(dataset=train_ds, batch_size=2 if dry_run else 2, shuffle=True)
    val_loader = DataLoader(dataset=val_ds, batch_size=2 if dry_run else 2, shuffle=False)
    
    # SOTA Loss Alignment (Extreme Priority for B)
    if model_type == 'B':
        # High alpha for Recall (reducing tool misses)
        criterion = SurgicalHybridLoss(alpha=0.8, beta=0.2, gamma=4/3, ce_weight=0.2)
    else:
        # Balanced Anatomical Extraction
        criterion = SurgicalHybridLoss(alpha=0.7, beta=0.3, gamma=4/3, ce_weight=0.5)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    
    # V2.0.6: Dedicated Recovery Directory
    suffix = "_recovery" if recovery_mode else ""
    save_path = f"./checkpoints_V2{suffix}/model_{model_type}_stage_{stage}/best_model.pth"
    
    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, device, save_path, recovery_mode=recovery_mode)
    trainer.run(num_epochs=15 if recovery_mode else 50, dry_run=dry_run)
