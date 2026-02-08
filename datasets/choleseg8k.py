import os
import glob
import torch
import cv2
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class CholeSeg8kDataset(Dataset):
    """
    Dataset for CholecSeg8k.
    Stage 1: Liver-only (maps class 12 to 1, others to 0)
    Stage 2: Liver + Tools (specific mapping)
    """
    CLASS_MAP = {
        'liver': 12,
        'gallbladder': 13,
    }

    def __init__(self, root_dir, stage=1, transform=None):
        self.root_dir = root_dir
        self.stage = stage
        self.transform = transform
        self.samples = self._load_samples()

    def _load_samples(self):
        # Structure: root_dir/videoXX/videoXX_XXXXX/frame_X_endo.png
        samples = []
        video_dirs = glob.glob(os.path.join(self.root_dir, "video*"))
        for vdir in video_dirs:
            # Subdirectories like video01_00080
            sub_dirs = [d for d in os.listdir(vdir) if os.path.isdir(os.path.join(vdir, d))]
            for sdir in sub_dirs:
                full_sdir = os.path.join(vdir, sdir)
                images = glob.glob(os.path.join(full_sdir, "*_endo.png"))
                for img_path in images:
                    mask_path = img_path.replace("_endo.png", "_endo_mask.png")
                    if os.path.exists(mask_path):
                        samples.append((img_path, mask_path))
        return samples

    def __len__(self):
        return len(self.samples)

    def _generate_clinical_proxy(self, mask_np):
        """
        V2.0.6: Clinical Liver Proxy Transformation.
        Transforms raw dataset labels into solid clinical anatomy.
        """
        # 1. Map inclusive liver indices (V2.0.8: Surgically Precise - Index 21 Only)
        liver_indices = [21]
        proxy_mask = np.zeros_like(mask_np)
        for val in liver_indices:
            proxy_mask[mask_np == val] = 1
        
        if not np.any(proxy_mask): return proxy_mask
        
        # 2. Erosion: Remove thin connective fascia (1-pixel shell)
        kernel_erode = np.ones((3, 3), np.uint8)
        proxy_mask = cv2.erode(proxy_mask, kernel_erode, iterations=1)
        
        # 3. Contiguity: Enforce Largest Connected Component
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(proxy_mask, connectivity=8)
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            # Minimum Area Constraint: Reject segments < 800 pixels
            if stats[largest_label, cv2.CC_STAT_AREA] < 800:
                proxy_mask = np.zeros_like(proxy_mask)
            else:
                proxy_mask = (labels == largest_label).astype(np.uint8)
        
        # 4. Hole Filling: Solidify internal parenchyma
        contours, _ = cv2.findContours(proxy_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            cv2.drawContours(proxy_mask, [cnt], 0, 1, -1)
            
        return proxy_mask

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
        mask_np = np.array(mask)
        
        if self.stage == 1:
            # Stage 1: Legacy Inclusive Mapping (for Backbone stability)
            # Stage 1: Standardize mapping [21=Liver, 22=GB]
            processed_mask = np.zeros_like(mask_np)
            processed_mask[mask_np == 21] = 1 # Liver
            processed_mask[mask_np == 22] = 2 # Gallbladder
        else:
            # --- V2.1.4: SURGICAL FOV & MULTICOLOR MASTER ---
            # 1. Circle FOV suppression (Suppress noise on lens edges)
            h_t, w_t = 256, 256
            fov_mask = np.zeros((h_t, w_t), dtype=np.uint8)
            cv2.circle(fov_mask, (w_t//2, h_t//2), int(h_t*0.48), 1, -1)
            
            img_np = np.array(image.resize((w_t, h_t)))
            if len(img_np.shape) == 3: # RGB -> BGR
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            img_bgr = img_np.astype(np.float32)
            
            # 2. Refined Anatomical Kernels (Surgically Tuned)
            # Liver: Red-Dominant Maroon (Shadow resilient)
            p_liver = (img_bgr[:,:,2] > 30) & (img_bgr[:,:,2] > img_bgr[:,:,1]*1.2) & (img_bgr[:,:,2] > img_bgr[:,:,0]*1.2)
            # Gallbladder: Green-Yellow shift
            p_gb = (img_bgr[:,:,1] > 55) & (img_bgr[:,:,1] > img_bgr[:,:,0]*1.1) & (np.abs(img_bgr[:,:,2] - img_bgr[:,:,1]) < 60)
            # Fascia: High-brightness yellow/white (Exclusion candidate)
            p_fascia = (img_bgr[:,:,2] > 180) & (img_bgr[:,:,1] > 150) & (img_bgr[:,:,0] > 100)
            
            # Apply FOV Mask
            p_liver &= (fov_mask == 1)
            p_gb &= (fov_mask == 1)
            
            # Refine processed mask
            mask_resized = cv2.resize(mask_np, (w_t, h_t), interpolation=cv2.INTER_NEAREST)
            processed_mask = np.zeros_like(mask_resized)
            
            # A. Seed with Ground Truth (Always trusted)
            processed_mask[mask_resized == 21] = 1 # Liver
            processed_mask[mask_resized == 22] = 2 # GB
            processed_mask[mask_resized == 255] = 1 # Liver (Secondary)
            
            # B. Anatomical Harvest (Discovery)
            # Capture massive liver lobes from background
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((p_liver & (mask_resized == 0)).astype(np.uint8))
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] > 5000: # Significant mass
                    processed_mask[labels == i] = 1
            
            # Capture definitive Gallbladder structure
            num_labels_gb, labels_gb, stats_gb, _ = cv2.connectedComponentsWithStats((p_gb & (mask_resized == 0)).astype(np.uint8))
            for i in range(1, num_labels_gb):
                if stats_gb[i, cv2.CC_STAT_AREA] > 1000:
                    processed_mask[labels_gb == i] = 2
            
            # C. Suppression: Remove Fascia and Out-of-FOV noise
            processed_mask[p_fascia] = 0
            processed_mask[fov_mask == 0] = 0
            
            # D. Final Mass Unification (Priority Loop)
            final_p = np.zeros_like(processed_mask)
            for cid in [2, 1]: # GB Priority
                mass = (processed_mask == cid).astype(np.uint8)
                if np.any(mass):
                    # Solidify: Close + Definitive Hole Filling
                    mass = cv2.morphologyEx(mass, cv2.MORPH_CLOSE, np.ones((11,11), np.uint8))
                    cnts, _ = cv2.findContours(mass, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                    for cnt in cnts:
                        cv2.fillPoly(mass, [cnt], 1)
                    final_p[(mass == 1) & (final_p == 0)] = cid
            processed_mask = final_p
            
        mask = Image.fromarray(processed_mask.astype(np.uint8))

        if self.transform:
            image, mask = self.transform(image, mask)
        
        return image, mask
