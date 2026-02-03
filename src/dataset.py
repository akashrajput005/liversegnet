import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class PinnacleSurgicalDataset(Dataset):
    """The Ultimate Unified Surgical Dataset: Deep-crawls all provided sources."""
    def __init__(self, roots_dict, split='train', transform=None, target_size=(512, 512)):
        self.samples = []
        self.transform = transform
        self.target_size = target_size
        self.split = split
        
        # 1. Source: cholecseg8k (Split by Video ID)
        cholec8k = roots_dict.get('cholecseg8k')
        if cholec8k and os.path.exists(cholec8k):
            # Videos for validation (approx 25%)
            val_videos = ['video14', 'video15', 'video16', 'video17']
            print(f"📦 Filtering CholecSeg8k ({split}): {cholec8k}")
            for root, dirs, files in os.walk(cholec8k):
                # Determine if current root belongs to train or val video
                is_val_path = any(v in root for v in val_videos)
                if (split == 'val' and is_val_path) or (split == 'train' and not is_val_path):
                    endo_images = [f for f in files if f.endswith('_endo.png')]
                    for img_name in endo_images:
                        img_path = os.path.join(root, img_name)
                        mask_path = os.path.join(root, img_name.replace('_endo.png', '_endo_mask.png'))
                        if os.path.exists(mask_path):
                            self.samples.append((img_path, mask_path))

        # 2. Source: Reference Image Set (Standard Train/Val folders)
        ref_set = roots_dict.get('reference_set')
        if ref_set and os.path.exists(ref_set):
            ref_split = os.path.join(ref_set, split)
            if os.path.exists(ref_split):
                print(f"📦 Integrating Reference Set ({split}): {ref_split}")
                img_dir = os.path.join(ref_split, 'images')
                mask_dir = os.path.join(ref_split, 'masks')
                if os.path.exists(img_dir):
                    for f in os.listdir(img_dir):
                        if f.endswith(('.png', '.jpg')): # Support both
                            img = os.path.join(img_dir, f)
                            mask = os.path.join(mask_dir, f if f.endswith('.png') else f.rsplit('.', 1)[0] + '.png')
                            if os.path.exists(mask):
                                self.samples.append((img, mask))

        # 3. Source: CholecInstanceSeg (Standard Train/Val folders)
        inst_seg = roots_dict.get('cholecinstanceseg')
        if inst_seg and os.path.exists(inst_seg):
            # The dataset has a nested 'cholecinstanceseg/train' etc structure
            inst_path = inst_seg
            if os.path.exists(os.path.join(inst_seg, 'cholecinstanceseg')):
                inst_path = os.path.join(inst_seg, 'cholecinstanceseg')
            
            target_path = os.path.join(inst_path, split)
            if os.path.exists(target_path):
                print(f"📦 Deep-crawling CholecInstanceSeg ({split}): {target_path}")
                for root, dirs, files in os.walk(target_path):
                    masks = [f for f in files if 'mask' in f.lower() and f.endswith('.png')]
                    for m_name in masks:
                        prefix = m_name.split('_mask')[0].split('_endo')[0]
                        img_name = prefix + ".png"
                        img_path = os.path.join(root, img_name)
                        if os.path.exists(img_path):
                            self.samples.append((img_path, os.path.join(root, m_name)))
        
        print(f"🚀 SUCCESS: Unified {len(self.samples)} samples for {split}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        image = cv2.imread(img_path)
        if image is None: return self.__getitem__((idx + 1) % len(self))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None: mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # --- UNIVERSAL SURGICAL REMAPPER (Precision Layer) ---
        # Robustly handle multiple Cholec8k/InstanceSeg/Reference encodings.
        # Goal: Class 0 (BG), Class 1 (Anatomy/Liver), Class 2 (Instruments)
        
        unique_vals = np.unique(mask)
        new_mask = np.zeros_like(mask)
        
        # 🟢 CLASS 1: ANATOMY (Organ/Tissue Context)
        # Cholec8k Anatomy Labels: 1, 2, 3, 4, 6, 8, 10, 11, 12
        # Often multi-labeled as 11, 12, 13, 21, 22, 31, 32 (ID * 10 + X)
        anatomy_ids = [1, 2, 3, 4, 6, 8, 10, 11, 12, 13, 21, 22, 31, 32, 128]
        for aid in anatomy_ids:
            new_mask[mask == aid] = 1
            
        # 🟠 CLASS 2: INSTRUMENTS (Surgical Tools)
        # Cholec8k Tool Labels: 5 (Grasper), 9 (L-hook)
        # Often encoded as 50, 90, or 255/149 in other sets
        tool_ids = [5, 7, 9, 50, 90, 149, 255]
        for tid in tool_ids:
            new_mask[mask == tid] = 2
            
        # Fallback: Catch-all for unknown high-values as background or anatomy
        # This prevents "garbage labels" from corrupting the Instrument class.
        mask = new_mask
        
        # Resizing (Precision Interpolation)
        image = cv2.resize(image, (self.target_size[1], self.target_size[0]))
        mask = cv2.resize(mask, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_NEAREST)
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask'].long()
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).long()
            
        return image, mask

def get_pinnacle_transforms(img_size=(512, 512), is_train=True):
    """High-intensity surgical augmentations for robustness at the pinnacle."""
    if is_train:
        return A.Compose([
            A.Resize(img_size[0], img_size[1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),
            # Surgical Conditions Simulations:
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.GaussianBlur(blur_limit=3, p=0.2),
                A.GlassBlur(sigma=0.7, max_delta=2, p=0.1),
            ], p=0.3),
            A.OneOf([
                A.RandomGamma(gamma_limit=(80, 120), p=0.3),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.3),
                A.CLAHE(clip_limit=2.0, p=0.2),
            ], p=0.4),
            A.OneOf([
                A.ElasticTransform(alpha=1, sigma=50, p=0.2),
                A.GridDistortion(p=0.2),
                A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.2), # Smoke/Occlusion Sim
            ], p=0.2),
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3), # Essential for tool edges
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(img_size[0], img_size[1]),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

# Simple wrapper for compatibility
def get_transforms(img_size=(512, 512), is_train=True):
    return get_pinnacle_transforms(img_size, is_train)
