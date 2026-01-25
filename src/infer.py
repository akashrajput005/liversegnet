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
    def __init__(self, model_path, architecture='unet', encoder='resnet34', device='cuda', img_size=(512, 512)):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.img_size = img_size
        
        # 1. Main Model (3-class)
        self.model = get_model(architecture=architecture, encoder=encoder, num_classes=3)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device).eval()

        # 2. Anatomical Anchor (U-Net 3-class)
        self.anchor_model = None
        anchor_path = os.path.join(os.path.dirname(model_path), 'unet_resnet34.pth')
        if architecture == 'deeplabv3plus' and os.path.exists(anchor_path):
            self.anchor_model = get_model(architecture='unet', encoder='resnet34', num_classes=3)
            self.anchor_model.load_state_dict(torch.load(anchor_path, map_location=self.device))
            self.anchor_model.to(self.device).eval()
            print("Clinical Ensemble: U-Net Anchor Active.")

        # 3. Precision Anatomy (Stage 1 2-class) - THE ULTIMATE BACKUP
        self.anatomy_model = None
        s1_path = os.path.join(os.path.dirname(model_path), 'deeplabv3plus_resnet50_stage1.pth')
        if os.path.exists(s1_path):
            self.anatomy_model = get_model(architecture='deeplabv3plus', encoder='resnet50', num_classes=2)
            self.anatomy_model.load_state_dict(torch.load(s1_path, map_location=self.device))
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
        
        # --- ENSEMBLE RECONSTRUCTION ---
        # Class 1: Liver Priority (Stage 1 > U-Net > Stage 2)
        liver_mask = np.zeros((h, w), dtype=np.uint8)
        if mask_anatomy is not None:
            liver_mask = (mask_anatomy == 1).astype(np.uint8) # S1 Liver is class 1
        elif mask_anchor is not None:
            liver_mask = (mask_anchor == 1).astype(np.uint8)
        else:
            liver_mask = (mask_main == 1).astype(np.uint8)
            
        # Class 2: Instrument (Always use Main)
        inst_mask = (mask_main == 2).astype(np.uint8)

        # Post-Processing: Refined Sensitivity
        kernel_tiny = np.ones((2,2), np.uint8)
        
        # Clean Liver (Handle split blobs > 50px)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(liver_mask, connectivity=8)
        liver_mask_clean = np.zeros_like(liver_mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] > 50:
                liver_mask_clean[labels == i] = 1
        liver_mask = liver_mask_clean

        # Clean Instruments (Gentle preservation)
        inst_mask = cv2.morphologyEx(inst_mask, cv2.MORPH_OPEN, kernel_tiny)
        
        # Detection Flags
        liver_present = bool(np.sum(liver_mask) > 50)
        inst_present = bool(np.sum(inst_mask) > 30)

        # Reconstruct final mask overlay
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
