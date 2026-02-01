import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class LiverInstrumentDataset(Dataset):
    def __init__(self, unified_root, split='train', transform=None, target_size=(512, 512)):
        self.split_dir = os.path.join(unified_root, split)
        self.img_dir = os.path.join(self.split_dir, 'images')
        self.mask_dir = os.path.join(self.split_dir, 'masks')
        self.transform = transform
        self.target_size = target_size
        
        self.filenames = [f for f in os.listdir(self.img_dir) if f.endswith('.png')]
        
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        image = cv2.imread(os.path.join(self.img_dir, fname))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(os.path.join(self.mask_dir, fname), cv2.IMREAD_GRAYSCALE)
            
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
