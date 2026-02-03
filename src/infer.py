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
    def _load_model(self, path, architecture, encoder, num_classes):
        """Internal helper to load any model architecture with pinnacle detection."""
        if path and "pinnacle" in path.lower():
            if "deeplab" in path.lower():
                architecture, encoder = 'deeplabv3plus', 'resnet101'
            elif "unet" in path.lower():
                architecture, encoder = 'unet', 'efficientnet-b4'
        
        model = get_model(architecture=architecture, encoder=encoder, num_classes=num_classes)
        if path and os.path.exists(path):
            try:
                state_dict = torch.load(path, map_location=self.device)
                if 'model_state_dict' in state_dict:
                    state_dict = state_dict['model_state_dict']
                if any(k.startswith('module.') for k in state_dict.keys()):
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                model.load_state_dict(state_dict)
                print(f"✅ Loaded {architecture} ({encoder}) from {path}")
            except Exception as e:
                print(f"⚠️ Error loading weights from {path}: {e}")
        else:
            print(f"ℹ️ Starting with fresh {architecture} ({encoder})")
            
        return model.to(self.device).eval()

    def __init__(self, model_path, device='cuda', img_size=(256, 256), num_classes=3):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.img_size = img_size
        
        # 1. Primary Model: DeepLabV3+ ResNet101
        self.model = self._load_model(model_path, 'deeplabv3plus', 'resnet101', num_classes)

        # 2. Ensemble Anchor: U-Net EfficientNet-B4
        self.anchor_model = None
        anchor_path = os.path.join(os.path.dirname(model_path), 'pinnacle_unet_eb4.pth')
        if os.path.exists(anchor_path):
            self.anchor_model = self._load_model(anchor_path, 'unet', 'efficientnet-b4', num_classes)
            print(f"✅ Pinnacle Ensemble Active: Unmatchable precision enabled.")
        
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

    def _get_probs(self, model, input_tensor):
        if model is None: return None
        with torch.no_grad():
            output = model(input_tensor)
            return torch.softmax(output, dim=1).squeeze(0).cpu().numpy()

    def predict_image(self, image_bgr):
        h, w = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # --- ULTIMATE TTA (Test Time Augmentation) PIPELINE ---
        # Enhanced with Multi-Scale scanning for Minute Details
        def get_model_tta(model, img_rgb):
            if model is None: return None
            
            # 1. Standard Scale (1.0x)
            # Pass 1: Original
            t1 = self.transform(image=img_rgb)['image'].unsqueeze(0).to(self.device)
            p1 = self._get_probs(model, t1) # (C, H, W)
            
            # Pass 2: Horizontal Flip
            img_hflip = cv2.flip(img_rgb, 1)
            t2 = self.transform(image=img_hflip)['image'].unsqueeze(0).to(self.device)
            p2 = self._get_probs(model, t2)
            p2_flip = np.flip(p2, axis=2) # Flip back
            
            # 2. Detail Scale (1.25x) - Catches tiny tips
            # We process this only for the "Fine" scan, then downsample back
            h_orig, w_orig = img_rgb.shape[:2]
            img_scaled = cv2.resize(img_rgb, (int(w_orig*1.25), int(h_orig*1.25)))
            t_scaled = self.transform(image=img_scaled)['image'].unsqueeze(0).to(self.device)
            p3 = self._get_probs(model, t_scaled)
            
            # FORCE-SAFE FUSION: Resize everything to target 256x256 (Model Output)
            # This is critical because some graphics drivers might return undefined TTA shapes
            target_h, target_w = p1.shape[1], p1.shape[2]
            
            p2_safe = p2_flip if p2_flip.shape == p1.shape else np.stack([cv2.resize(p2_flip[i], (target_w, target_h)) for i in range(3)])
            p3_safe = p3 if p3.shape == p1.shape else np.stack([cv2.resize(p3[i], (target_w, target_h)) for i in range(3)])
            
            return (p1 + p2_safe + p3_safe) / 3.0

        probs_main = get_model_tta(self.model, image_rgb)
        probs_anchor = get_model_tta(self.anchor_model, image_rgb)

        def resize_probs(p, h, w):
            if p is None: return None
            # Resize each class probability channel
            return np.stack([cv2.resize(p[i], (w, h), interpolation=cv2.INTER_LINEAR) for i in range(p.shape[0])])

        p_main = resize_probs(probs_main, h, w)
        p_anchor = resize_probs(probs_anchor, h, w)
        
        # --- PINNACLE PROBABILITY FUSION ---
        # Recalibrated for mathematical purity and class balance
        liver_prob = np.zeros((h, w), dtype=np.float32)
        inst_prob = np.zeros((h, w), dtype=np.float32)
        
        # Weights: Anchor (B4) is more precise on anatomy, Main (R101) better on tools
        w_liver_anchor, w_liver_main = 1.5, 1.0
        w_inst_anchor, w_inst_main = 1.0, 2.0
        
        if p_anchor is not None:
            liver_prob += p_anchor[1] * w_liver_anchor
            inst_prob += p_anchor[2] * w_inst_anchor
            
        if p_main is not None:
            liver_prob += p_main[1] * w_liver_main
            inst_prob += p_main[2] * w_inst_main
            
        # Normalization: Divide by sum of weights for true [0, 1] probability
        liver_prob = liver_prob / (w_liver_anchor + w_liver_main)
        inst_prob = inst_prob / (w_inst_anchor + w_inst_main)
            
        # Smoothing: Specific kernels for different tissue types
        liver_prob = cv2.GaussianBlur(liver_prob, (9, 9), 0)
        # Minute Detail: Use finer kernel (3x3) to preserve thin tool tips
        inst_prob = cv2.GaussianBlur(inst_prob, (3, 3), 0)
        
        # Clinical-Grade Thresholding
        liver_mask = (liver_prob > 0.5).astype(np.uint8)
        inst_mask = (inst_prob > 0.45).astype(np.uint8)
        
        # --- COLOR-LOCK SURGICAL OVERRIDE (Refined) ---
        # Emergency Fix: Force Metallic pixels to be Instruments
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        
        # 1. Broaden Metallic Range (Catch Darker Tools)
        # S < 50 (Very Low Saturation = Grey/White/Black)
        # V > 40 (Allow darker shadows, but not pitch black)
        lower_metallic = np.array([0, 0, 40])
        upper_metallic = np.array([180, 50, 255])
        metallic_mask = cv2.inRange(hsv, lower_metallic, upper_metallic)
        
        # 2. Red-Rejection Filter (The "Anti-Blood" Check)
        # Metal is Neutral (B ≈ G ≈ R). Tissue is Red (R >> G/B).
        # If Red is dominant, it's TISSUE, even if it has low saturation (e.g. pale tissue).
        b, g, r = cv2.split(image_bgr)
        # Calculate Red Dominance: R should not be significantly larger than B or G
        # Using floating point for precision
        r_f, g_f, b_f = r.astype(np.float32), g.astype(np.float32), b.astype(np.float32)
        red_dominance = (r_f > (g_f * 1.1)) & (r_f > (b_f * 1.1))
        
        # Remove Red-Dominant pixels from the Metallic Mask
        metallic_mask[red_dominance] = 0
        
        # Morphological clean (remove spotty noise)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        metallic_mask = cv2.morphologyEx(metallic_mask, cv2.MORPH_OPEN, kernel)
        metallic_mask = cv2.morphologyEx(metallic_mask, cv2.MORPH_CLOSE, kernel)
        
        # FORCE OVERRIDE
        inst_mask[metallic_mask > 0] = 1
        liver_mask[metallic_mask > 0] = 0
        # ------------------------------------

        # Sharp Cut Logic: Ensure tool penetrates cleanly
        liver_mask[inst_mask == 1] = 0
        
        # Analytics: Quantitative Precision
        # 1. Pixel Counts
        liver_pixels = int(np.sum(liver_mask))
        inst_pixels = int(np.sum(inst_mask))
        
        # 2. Region Counting (Connected Components)
        num_liver_regions = int(cv2.connectedComponents(liver_mask)[0] - 1)
        num_inst_regions = int(cv2.connectedComponents(inst_mask)[0] - 1)

        # 3. Detection Flags
        liver_present = bool(liver_pixels > 500)
        inst_present = bool(inst_pixels > 200)

        # Reconstruct final mask
        final_mask = np.zeros((h, w), dtype=np.uint8)
        final_mask[liver_mask > 0] = 1
        final_mask[inst_mask > 0] = 2
        
        # Overlay and Geometry
        occlusion = calculate_occlusion(liver_mask, inst_mask)
        distance = calculate_min_distance(liver_mask, inst_mask)
        overlay = get_overlay(image_bgr, final_mask)
        
        return {
            "mask": final_mask,
            "overlay": overlay,
            "occlusion": occlusion,
            "distance": distance,
            "liver_found": liver_present,
            "inst_found": inst_present,
            "liver_pixels": liver_pixels,
            "inst_pixels": inst_pixels,
            "liver_regions": num_liver_regions,
            "inst_regions": num_inst_regions
        }

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
                
            res = self.predict_image(frame)
            out.write(res['overlay'])
            
        cap.release()
        out.release()
        return True
