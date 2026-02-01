import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random

class CholecSeg8kDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_size=(512, 512), max_samples=1000):
        """
        CholecSeg8k dataset loader
        Structure: root_dir/videoXX/frame_XXXXX_YYYYY/
        """
        self.root_dir = root_dir
        self.transform = transform
        self.target_size = target_size

        # Collect all per-frame image files (NOT just one per folder)
        # Each folder typically contains multiple frames: frame_XXX_endo.png
        # samples: (image_path, mask_path, video_id)
        self.samples = []
        for video_dir in os.listdir(root_dir):
            video_path = os.path.join(root_dir, video_dir)
            if not os.path.isdir(video_path):
                continue

            for frame_folder in os.listdir(video_path):
                frame_path = os.path.join(video_path, frame_folder)
                if not os.path.isdir(frame_path):
                    continue

                for fname in os.listdir(frame_path):
                    if not fname.endswith('_endo.png'):
                        continue

                    img_path = os.path.join(frame_path, fname)

                    color_mask_path = img_path.replace('_endo.png', '_endo_color_mask.png')
                    binary_mask_path = img_path.replace('_endo.png', '_endo_mask.png')

                    if os.path.exists(color_mask_path):
                        self.samples.append((img_path, color_mask_path, video_dir))
                    elif os.path.exists(binary_mask_path):
                        self.samples.append((img_path, binary_mask_path, video_dir))

        # Limit samples for faster training
        if max_samples and len(self.samples) > max_samples:
            self.samples = random.sample(self.samples, max_samples)

        print(f"Loaded {len(self.samples)} samples from CholecSeg8k")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, _video_id = self.samples[idx]
        
        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask with debugging
        if mask_path.endswith('_endo_color_mask.png'):
            mask_bgr = cv2.imread(mask_path)
            mask = self.convert_color_mask_to_classes(mask_bgr)
            mask_type = "color"
        else:
            mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            # Convert binary to multi-class (0=bg, 1=liver, 2=instrument)
            # For binary masks, we can only distinguish bg vs liver, no instrument info
            # CRITICAL FIX: Binary masks might have values 0-255, normalize first
            if mask_gray.max() > 1:
                # Binary mask with values 0-255, normalize to 0-1
                mask = np.where(mask_gray > 127, 1, 0)  # Threshold at 127
            else:
                # Already normalized binary mask
                mask = np.where(mask_gray > 0, 1, 0)
            
            # CRITICAL FIX: Ensure only valid class values (0, 1, 2)
            mask = np.clip(mask, 0, 2)
            mask_type = "binary"
            
            # Additional validation for binary masks
            unique_binary = np.unique(mask)
            if np.any(unique_binary < 0) or np.any(unique_binary > 2):
                print(f"🚨 Invalid binary mask values: {unique_binary}")
                mask = np.clip(mask, 0, 2)
        
        # DEBUG: Check mask values for first few samples
        if idx < 5:
            unique_vals = np.unique(mask)
            print(f"[DEBUG] idx={idx} mask_shape={mask.shape} unique_vals={unique_vals} sum={mask.sum()}")
        
        # Resize
        image = cv2.resize(image, (self.target_size[1], self.target_size[0]))
        mask = cv2.resize(mask, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_NEAREST)
        
        # CRITICAL FIX: Ensure valid values after resize (interpolation can introduce issues)
        mask = np.clip(mask, 0, 2)
        
        # DEBUG: Check values after resize
        if idx < 5:
            unique_vals_after = np.unique(mask)
            print(f"[DEBUG] idx={idx} after_resize unique_vals={unique_vals_after}")
            
            # Verify no invalid values
            if np.any(unique_vals_after < 0) or np.any(unique_vals_after > 2):
                print(f"🚨 INVALID VALUES AFTER RESIZE: {unique_vals_after}")
                mask = np.clip(mask, 0, 2)  # Force valid range
                unique_vals_fixed = np.unique(mask)
                print(f"✅ Fixed after resize: {unique_vals_fixed}")
        
        # STEP 2: Preprocessing Survival Check (disabled)
        # inst_pixels = (mask == 2).sum()
        # liver_pixels = (mask == 1).sum()
        # 
        # if idx < 3:
        #     print(f"[DEBUG] liver_pixels={liver_pixels}, inst_pixels={inst_pixels}")
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask'].long()
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).long()
            
        return image, mask

    def convert_color_mask_to_classes(self, color_mask):
        """Convert color mask to 3-class indices.

        Enhanced color detection for better instrument recognition
        
        Output classes:
        0=background, 1=liver, 2=instrument
        """
        h, w = color_mask.shape[:2]
        flat = color_mask.reshape(-1, 3)
        colors, counts = np.unique(flat, axis=0, return_counts=True)

        # Skip if only background color
        if len(colors) <= 1:
            return np.zeros((h, w), dtype=np.uint8)

        # background = most frequent color
        bg_idx = int(np.argmax(counts))

        # remaining colors sorted by area
        rem = [(i, int(c)) for i, c in enumerate(counts) if i != bg_idx]
        rem.sort(key=lambda x: x[1], reverse=True)

        class_mask = np.zeros((h, w), dtype=np.uint8)

        # assign liver = largest remaining color (if any)
        if rem:
            liver_color = colors[rem[0][0]]
            liver_pixels = np.all(color_mask == liver_color, axis=2)
            class_mask[liver_pixels] = 1

            # instruments = all other remaining colors (enhanced detection)
            for i, _c in rem[1:]:
                inst_color = colors[i]
                inst_pixels = np.all(color_mask == inst_color, axis=2)
                class_mask[inst_pixels] = 2

        # CRITICAL FIX: Ensure only valid class values (0, 1, 2)
        # This prevents CUDA errors in one_hot encoding
        class_mask = np.clip(class_mask, 0, 2)
        
        # DEBUG: Verify final mask values
        unique_final = np.unique(class_mask)
        if len(unique_final) > 3 or np.any(unique_final < 0) or np.any(unique_final > 2):
            print(f"🚨 INVALID MASK VALUES AFTER CONVERSION: {unique_final}")
            class_mask = np.clip(class_mask, 0, 2)  # Force valid range
            unique_final = np.unique(class_mask)
            print(f"✅ Fixed mask values: {unique_final}")

        # Add debug info for color detection (disabled for clean terminal)
        # unique_classes = np.unique(class_mask)
        # if len(unique_classes) > 2 and np.random.random() < 0.1:  # Only 10% of color masks
        #     print(f"🎯 Color mask detected: bg={np.sum(class_mask==0)}, liver={np.sum(class_mask==1)}, inst={np.sum(class_mask==2)}")

        return class_mask

    def get_video_id(self, idx: int) -> str:
        return self.samples[idx][2]

def get_transforms(img_size=(512, 512), is_train=True):
    if is_train:
        return A.Compose([
            A.RandomRotate90(),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
            A.OneOf([
                A.GaussianBlur(),
                A.GaussNoise(),
            ], p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
