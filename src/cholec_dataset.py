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

        # Load mask
        if mask_path.endswith('_endo_color_mask.png'):
            mask_bgr = cv2.imread(mask_path)
            mask = self.convert_color_mask_to_classes(mask_bgr)
        else:
            mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            # Convert binary to multi-class (0=bg, 1=liver, 2=instrument)
            mask = np.where(mask_gray > 0, 1, 0)
        
        # Resize
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

    def convert_color_mask_to_classes(self, color_mask):
        """Convert color mask to 3-class indices.

        Observed CholecSeg8k *_endo_color_mask.png uses a fixed palette with a handful of colors.
        HSV heuristics are brittle; instead we decode per-image by palette frequency:
        - background: most frequent color
        - liver: largest remaining region color
        - instrument: union of all remaining colors

        Output classes:
        0=background, 1=liver, 2=instrument
        """
        h, w = color_mask.shape[:2]
        flat = color_mask.reshape(-1, 3)
        colors, counts = np.unique(flat, axis=0, return_counts=True)

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

            # instruments = all other remaining colors
            for i, _c in rem[1:]:
                inst_color = colors[i]
                inst_pixels = np.all(color_mask == inst_color, axis=2)
                class_mask[inst_pixels] = 2

        # background stays 0 implicitly
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
