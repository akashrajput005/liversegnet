import torch
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
try:
    from model import get_model
    from analytics import get_overlay, calculate_occlusion, calculate_min_distance
except ImportError:
    from src.model import get_model
    from src.analytics import get_overlay, calculate_occlusion, calculate_min_distance
import yaml
import os

class InferenceEngine:
    def __init__(self, model_path, architecture='unet', encoder='resnet34', device='cuda', img_size=(512, 512), num_classes=3):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.img_size = img_size
        
        # 1. Main Model
        self.model = get_model(architecture=architecture, encoder=encoder, num_classes=num_classes)
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            # Handle checkpoint with metadata wrapper
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
        self.model.to(self.device).eval()

        # 2. Anatomical Anchor (U-Net 3-class)
        self.anchor_model = None
        anchor_path = os.path.join(os.path.dirname(model_path), 'unet_resnet34.pth')
        if architecture == 'deeplabv3plus' and os.path.exists(anchor_path):
            self.anchor_model = get_model(architecture='unet', encoder='resnet34', num_classes=3)
            anchor_checkpoint = torch.load(anchor_path, map_location=self.device)
            if 'model_state_dict' in anchor_checkpoint:
                self.anchor_model.load_state_dict(anchor_checkpoint['model_state_dict'])
            else:
                self.anchor_model.load_state_dict(anchor_checkpoint)
            self.anchor_model.to(self.device).eval()
            print("Clinical Ensemble: U-Net Anchor Active.")

        # 3. Precision Anatomy (Stage 1 2-class) - THE ULTIMATE BACKUP
        self.anatomy_model = None
        s1_path = os.path.join(os.path.dirname(model_path), 'deeplabv3plus_resnet50_stage1.pth')
        if os.path.exists(s1_path):
            self.anatomy_model = get_model(architecture='deeplabv3plus', encoder='resnet50', num_classes=2)
            s1_checkpoint = torch.load(s1_path, map_location=self.device)
            if 'model_state_dict' in s1_checkpoint:
                self.anatomy_model.load_state_dict(s1_checkpoint['model_state_dict'])
            else:
                self.anatomy_model.load_state_dict(s1_checkpoint)
            self.anatomy_model.to(self.device).eval()
            print("Clinical Ensemble: Stage 1 Anatomy Anchor Active.")
        
        self.transform = A.Compose([
            A.Resize(img_size[0], img_size[1]),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    def predict_image_logits(self, image_bgr):
        h, w = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(image=image_rgb)['image'].unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(input_tensor) # (1, 3, H, W)
            probs = torch.softmax(output, dim=1).squeeze(0).cpu().numpy()
        return probs

    def predict_image(self, image_bgr):
        h, w = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # Preprocess
        input_tensor = self.transform(image=image_rgb)['image'].unsqueeze(0).to(self.device)
        
        # 1. Primary Inference (Instruments)
        with torch.no_grad():
            output_main = self.model(input_tensor)
            mask_main = torch.argmax(output_main, dim=1).squeeze(0).cpu().numpy()
            
        # 2. Anatomical Anchors
        mask_anchor = None
        if self.anchor_model:
            with torch.no_grad():
                output_anchor = self.anchor_model(input_tensor)
                mask_anchor = torch.argmax(output_anchor, dim=1).squeeze(0).cpu().numpy()

        mask_anatomy = None
        if self.anatomy_model:
            with torch.no_grad():
                output_anatomy = self.anatomy_model(input_tensor)
                mask_anatomy = torch.argmax(output_anatomy, dim=1).squeeze(0).cpu().numpy()

        # Resize all to original
        mask_main = cv2.resize(mask_main.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        if mask_anchor is not None:
            mask_anchor = cv2.resize(mask_anchor.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        if mask_anatomy is not None:
            mask_anatomy = cv2.resize(mask_anatomy.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        
        # --- ENSEMBLE RECONSTRUCTION WITH PROBABILITY FUSION ---
        # Accumulate probabilities from all models
        liver_prob = np.zeros((h, w), dtype=np.float32)
        inst_prob = np.zeros((h, w), dtype=np.float32)
        total_weight = 0.0
        
        # Model weights based on reliability
        if mask_anatomy is not None:
            # Stage 1 is liver specialist - high confidence on liver
            liver_prob += (mask_anatomy == 1).astype(np.float32) * 2.0
            total_weight += 2.0
        
        if mask_anchor is not None:
            # U-Net balanced model
            liver_prob += (mask_anchor == 1).astype(np.float32) * 1.5
            inst_prob += (mask_anchor == 2).astype(np.float32) * 1.5
            total_weight += 1.5
        
        # Main model always contributes
        liver_prob += (mask_main == 1).astype(np.float32) * 1.0
        inst_prob += (mask_main == 2).astype(np.float32) * 1.2 # Slighly more weight on tools for main model
        total_weight += 1.0
        
        # Normalize and SMOOTH probabilities to reduce noise
        if total_weight > 0:
            liver_prob /= total_weight
            inst_prob /= total_weight
            
        # Apply slight Gaussian Blur to probabilities for smoother boundaries
        liver_prob = cv2.GaussianBlur(liver_prob, (5, 5), 0)
        inst_prob = cv2.GaussianBlur(inst_prob, (3, 3), 0)
        
        # Adaptive Thresholding for cleaner detection
        liver_mask = (liver_prob > 0.45).astype(np.uint8)
        inst_mask = (inst_prob > 0.4).astype(np.uint8)

        # Post-Processing: Advanced Island Removal
        
        # 1. LIVER: Usually one large mass. Keep only the largest component + significant fragments
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(liver_mask, connectivity=8)
        if num_labels > 1:
            liver_mask_clean = np.zeros_like(liver_mask)
            # Find largest component index (excluding background)
            largest_idx = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
            max_area = stats[largest_idx, cv2.CC_STAT_AREA]
            
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                # Keep largest OR anything > 10% of largest area (but min 500px)
                if i == largest_idx or (area > max_area * 0.1 and area > 500):
                    liver_mask_clean[labels == i] = 1
            liver_mask = liver_mask_clean

        # 2. INSTRUMENTS: Multiple small tools possible. Remove tiny noise blobs.
        # Morphological Closing to connect tool parts
        kernel_tool = np.ones((5,5), np.uint8)
        inst_mask = cv2.morphologyEx(inst_mask, cv2.MORPH_CLOSE, kernel_tool)
        
        num_labels_i, labels_i, stats_i, _ = cv2.connectedComponentsWithStats(inst_mask, connectivity=8)
        inst_mask_clean = np.zeros_like(inst_mask)
        for i in range(1, num_labels_i):
            # Tools should be at least 150 pixels to be considered detection vs noise
            if stats_i[i, cv2.CC_STAT_AREA] > 150:
                inst_mask_clean[labels_i == i] = 1
        inst_mask = inst_mask_clean
        
        # Final refinement: Ensure tools don't blend with liver unless occluded
        # If pixels are marked as both, prefer instrument (occlusion)
        liver_mask[inst_mask == 1] = 0
        
        # Detection Flags with noise-resistant thresholds
        liver_present = bool(np.sum(liver_mask) > 500)
        inst_present = bool(np.sum(inst_mask) > 200)

        # Reconstruct final mask
        final_mask = np.zeros((h, w), dtype=np.uint8)
        final_mask[liver_mask > 0] = 1
        final_mask[inst_mask > 0] = 2
        
        # Analytics
        occlusion = calculate_occlusion(liver_mask, inst_mask)
        distance = calculate_min_distance(liver_mask, inst_mask)
        overlay = get_overlay(image_bgr, final_mask)
        
        return final_mask, overlay, occlusion, distance, liver_present, inst_present

    def predict_video(self, video_path, output_path):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if fps == 0 or w == 0 or h == 0:
             return False

        # Use mp4v codec with .avi extension for better browser support
        # Change extension to .avi if it's .mp4
        if output_path.endswith('.mp4'):
            output_path = output_path[:-4] + '.avi'
            
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            _, overlay, _, _, _, _ = self.predict_image(frame)
            out.write(overlay)
            
        cap.release()
        out.release()
        return True
