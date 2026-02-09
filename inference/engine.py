import torch
import numpy as np
import cv2
import os
from models.deeplab_liver import LiverSegModelA
from models.unet_tools import UNetResNet34
from utils.transforms import Compose, Resize, ToTensor, Normalize
from risk.geometry_logic import GeometrySafetyLayer

class ClinicalInferenceEngine:
    def __init__(self, model_a_path, model_b_path, device='cuda'):
        """
        V2.2.2: Added support for automatic weight fetching from Hugging Face if local files are missing.
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # --- MODEL A: ANATOMY ---
        if not os.path.exists(model_a_path):
            try:
                from huggingface_hub import hf_hub_download
                print(f"Local Model A not found. Fetching from Cloud...")
                # Authoritative Repo: akashrajput005/liversegnet-v2
                model_a_path = hf_hub_download(repo_id="akashrajput005/liversegnet-v2", filename="model_A_hybrid.pth")
            except Exception as e:
                print(f"Cloud Fetch Model A FAILED: {e}")

        # --- MODEL B: INSTRUMENTS ---
        if not os.path.exists(model_b_path):
            try:
                from huggingface_hub import hf_hub_download
                print(f"Local Model B not found. Fetching from Cloud...")
                model_b_path = hf_hub_download(repo_id="akashrajput005/liversegnet-v2", filename="model_B_hybrid.pth")
            except Exception as e:
                print(f"Cloud Fetch Model B FAILED: {e}")

        # Initialize Models (V2.0.2: Stage 2 for both Anatomy and Tools)
        self.model_a = LiverSegModelA(num_classes=5, pretrained=False).to(self.device).eval()
        self.model_b = UNetResNet34(num_classes=5, pretrained=False).to(self.device).eval()
        
        # Load Weights
        checkpoint_a = torch.load(model_a_path, map_location=self.device)
        checkpoint_b = torch.load(model_b_path, map_location=self.device)
        
        self.model_a.load_state_dict(checkpoint_a['model_state_dict'] if 'model_state_dict' in checkpoint_a else checkpoint_a, strict=False)
        self.model_b.load_state_dict(checkpoint_b['model_state_dict'] if 'model_state_dict' in checkpoint_b else checkpoint_b, strict=False)
        
        # Safety Layer
        self.safety_layer = GeometrySafetyLayer()
        
        # Parallel Streams
        self.stream_a = torch.cuda.Stream() if self.device.type == 'cuda' else None
        self.stream_b = torch.cuda.Stream() if self.device.type == 'cuda' else None
        
        # Preprocessing
        self.transform = Compose([
            Resize((256, 256)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # V2: Temporal EMA State (Instrument Smoothing)
        self.ema_alpha = 0.5
        self.prev_tips = None
        self.tip_velocity = 0.0

    def preprocess(self, frame):
        from PIL import Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        input_tensor, _ = self.transform(img, img)
        return input_tensor.unsqueeze(0).to(self.device)

    def filter_largest_component(self, mask, min_size=500):
        """V2.0.4: Discards small noisy artifacts and keeps only the primary organ mass."""
        if not np.any(mask): return mask
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        if num_labels <= 1: return mask
        
        # Find largest component (excluding background at index 0)
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        
        # If largest component is too small, discard entirely
        if stats[largest_label, cv2.CC_STAT_AREA] < min_size:
            return np.zeros_like(mask)
            
        return (labels == largest_label).astype(np.uint8)

    def infer(self, frame, confidence_threshold=0.3, use_heuristics=True):
        """
        LiverSegNet Hybrid Perception Pipeline.
        Classifies signals into:
        - NEURAL: Deep learning posterior probabilities.
        - DETERMINISTIC: Geometric constraints and morphological filters.
        - HEURISTIC: Physically-informed color recovery (Kill-switchable).
        """
        # --- [1. PREPROCESSING: DETERMINISTIC] ---
        input_tensor = self.preprocess(frame)
        h_t, w_t = 256, 256
        
        # --- [2. CORE INFERENCE: NEURAL] ---
        with torch.no_grad():
            if self.device.type == 'cuda':
                with torch.cuda.stream(self.stream_a):
                    output_a = self.model_a(input_tensor)
                    if isinstance(output_a, dict): output_a = output_a['out']
                with torch.cuda.stream(self.stream_b):
                    output_b = self.model_b(input_tensor)
                torch.cuda.synchronize()
            else:
                output_a = self.model_a(input_tensor)
                if isinstance(output_a, dict): output_a = output_a['out']
                output_b = self.model_b(input_tensor)
        
        # --- [3. ANATOMY EXTRACTION: NEURAL] ---
        probs_a = torch.softmax(output_a, dim=1)
        probs_a_np = probs_a.squeeze().cpu().numpy()
        mask_a_raw = np.zeros_like(probs_a_np[0], dtype=np.uint8)
        
        # Liver (Class 1) - High Sensitivity Neural Signal
        mask_a_raw[probs_a_np[1] > 0.08] = 1
        # Other classes - Neural Signal (Argmax)
        argmax_a = torch.argmax(output_a, dim=1).squeeze().cpu().numpy()
        mask_a_raw[(argmax_a == 2) & (probs_a_np[2] > 0.20)] = 2
        mask_a_raw[(argmax_a == 3) & (probs_a_np[3] > 0.25)] = 3
        mask_a_raw[(argmax_a == 4) & (probs_a_np[4] > 0.30)] = 4
        
        # --- [4. MASS UNIFICATION: DETERMINISTIC] ---
        mask_a = np.zeros_like(mask_a_raw)
        kernel_close = np.ones((15, 15), np.uint8)
        for class_id in [1, 2]:
            class_mask = (mask_a_raw == class_id).astype(np.uint8)
            if np.any(class_mask):
                class_mask = cv2.morphologyEx(class_mask, cv2.MORPH_CLOSE, kernel_close)
                class_mask = self.filter_largest_component(class_mask, min_size=600)
                mask_a[class_mask == 1] = class_id

        # --- [5. ANATOMICAL RECOVERY: HEURISTIC / KILL-SWITCHABLE] ---
        if use_heuristics:
            # Circle FOV suppression (Deterministic Guard for Heuristics)
            fov_mask = np.zeros((h_t, w_t), dtype=np.uint8)
            cv2.circle(fov_mask, (w_t//2, h_t//2), int(h_t*0.48), 1, -1)
            
            frame_tactical = cv2.resize(frame, (w_t, h_t))
            img_bgr = frame_tactical.astype(np.float32)
            
            # Physical Color Kernels (Heuristic Discovery)
            p_liver = (img_bgr[:,:,2] > 30) & (img_bgr[:,:,2] > img_bgr[:,:,1]*1.2) & (img_bgr[:,:,2] > img_bgr[:,:,0]*1.2)
            p_gb = (img_bgr[:,:,1] > 50) & (img_bgr[:,:,1] > img_bgr[:,:,0]*1.1) & (np.abs(img_bgr[:,:,2] - img_bgr[:,:,1]) < 60)
            p_fascia = (img_bgr[:,:,2] > 150) & (img_bgr[:,:,1] > 120) & (img_bgr[:,:,0] > 80)
            
            p_liver &= (fov_mask == 1)
            p_gb &= (fov_mask == 1)
            
            # Hybrid Fusion (Neural Seeds + Heuristic Discovery)
            for cid, profile in [(2, p_gb), (1, p_liver)]:
                model_seeds = (mask_a == cid).astype(np.uint8)
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(profile.astype(np.uint8))
                for i in range(1, num_labels):
                    area = stats[i, cv2.CC_STAT_AREA]
                    intersect = np.any((labels == i) & (model_seeds == 1))
                    if intersect or (cid == 1 and area > 8000) or (cid == 2 and area > 1500):
                        mask_a[labels == i] = cid
            
            # Tag Fascia as Background/Noise (Deterministic suppression of heuristics)
            mask_a[p_fascia & (mask_a == 0)] = 4

        # --- [6. FINAL REFINEMENT: DETERMINISTIC] ---
        final_processed_mask = np.zeros_like(mask_a)
        # Clinical Size Gates (V2.2.8): Purging phantom detections
        SIZE_GATES = {
            1: 800,  # Liver must be > 800px
            2: 400   # Gallbladder must be > 400px
        }
        
        for cid in [2, 1]:
            mass_seed = (mask_a == cid).astype(np.uint8)
            if np.any(mass_seed):
                # 1. Fill holes and consolidate
                mass = cv2.dilate(mass_seed, np.ones((5,5), np.uint8), iterations=1)
                mass = cv2.morphologyEx(mass, cv2.MORPH_CLOSE, np.ones((31,31), np.uint8))
                cnts, _ = cv2.findContours(mass, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in cnts:
                    cv2.fillPoly(mass, [cnt], 1)
                
                # 2. Clinical Size Filter (Aggressive Noise Suppression)
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mass, connectivity=8)
                refined_mass = np.zeros_like(mass)
                for i in range(1, num_labels):
                    if stats[i, cv2.CC_STAT_AREA] >= SIZE_GATES.get(cid, 500):
                        refined_mass[labels == i] = 1
                
                final_processed_mask[(refined_mass == 1) & (final_processed_mask == 0)] = cid
        mask_a = final_processed_mask

        # --- [7. TOOL PERCEPTION: NEURAL + DETERMINISTIC] ---
        output_b_np = torch.argmax(output_b, dim=1).squeeze().cpu().numpy()
        tool_mask = (output_b_np > 0).astype(np.uint8)
        mask_a[tool_mask == 1] = 0 # Instrument Shield (Deterministic)

        # --- [8. SAFETY TELEMENTRY & RESOLUTION: DETERMINISTIC] ---
        h, w = frame.shape[:2]
        mask_a_final = cv2.resize(mask_a.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        mask_b_final = cv2.resize(tool_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        
        prob_liver = probs_a[0, 1].cpu().numpy()
        prob_gb = probs_a[0, 2].cpu().numpy()
        
        raw_tips = self.safety_layer.extract_tool_tips(tool_mask)
        
        # Temporal SMA (Instrument Smoothing - Deterministic)
        smoothed_tips = []
        current_velocity = 0.0
        if self.prev_tips and raw_tips:
            for tip in raw_tips:
                dist_to_prev = [np.sqrt((tip[0]-p[0])**2 + (tip[1]-p[1])**2) for p in self.prev_tips]
                if not dist_to_prev: continue
                min_idx = np.argmin(dist_to_prev)
                min_dist = dist_to_prev[min_idx]
                prev_tip = self.prev_tips[min_idx]
                smoothed_x = self.ema_alpha * tip[0] + (1 - self.ema_alpha) * prev_tip[0]
                smoothed_y = self.ema_alpha * tip[1] + (1 - self.ema_alpha) * prev_tip[1]
                smoothed_tips.append((smoothed_x, smoothed_y))
                current_velocity = max(current_velocity, min_dist)
        else:
            smoothed_tips = raw_tips
        self.prev_tips = smoothed_tips
        self.tip_velocity = current_velocity
        
        # Kinetic Risk Calculation (Deterministic)
        risk_status, min_dist, kinetic_telemetry = self.safety_layer.calculate_risk(
            (mask_a > 0).astype(np.uint8), smoothed_tips, velocity=self.tip_velocity
        )
        
        spatial_reliability = float(torch.max(probs_a[0, 1:3, :, :]).cpu().numpy())
        census = self.validate_consensus(mask_a_final, mask_b_final)

        return {
            'mask_a': mask_a_final,
            'mask_b': mask_b_final,
            'prob_liver': prob_liver,
            'prob_gb': prob_gb,
            'tips': smoothed_tips,
            'velocity': current_velocity,
            'risk_status': risk_status,
            'min_distance': min_dist,
            'kinetic_telemetry': kinetic_telemetry,
            'spatial_reliability': spatial_reliability,
            'consensus_audit': census,
            'signals': {
                'neural': True,
                'deterministic': True,
                'heuristic': use_heuristics
            }
        }

    def validate_consensus(self, mask_a, mask_b):
        """Cross-Kernel Deterministic Consensus."""
        anatomy_roi = (mask_a > 0).astype(np.uint8)
        tool_roi = (mask_b > 0).astype(np.uint8)
        overlap_px = (anatomy_roi.astype(int) + tool_roi.astype(int) == 2).sum()
        
        dist_transform = cv2.distanceTransform(1 - anatomy_roi, cv2.DIST_L2, 3)
        min_dist = np.min(dist_transform[tool_roi == 1]) if tool_roi.any() else 50.0
        
        if overlap_px > 10:
            return {'status': 'VIOLATION', 'description': f'KERNEL MISALIGNMENT: Semantic Overlap ({overlap_px} px)', 'severity': 'HIGH'}
        elif min_dist < 10:
            return {'status': 'WARNING', 'description': f'PROXIMITY: Collision Imminent ({min_dist:.1f} px)', 'severity': 'MEDIUM'}
        return {'status': 'VERIFIED', 'description': 'Safe intra-operative separation', 'severity': 'NONE'}
