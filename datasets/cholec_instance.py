import os
import glob
import json
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
import numpy as np

class CholecInstanceSegDataset(Dataset):
    """
    Dataset for CholecInstanceSeg.
    Stage 1: Tools-only
    Stage 2: Tools + Liver (if applicable/requested)
    """
    def __init__(self, annotation_root, image_root, stage=1, transform=None):
        self.annotation_root = annotation_root
        self.image_root = image_root
        self.stage = stage
        self.transform = transform
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        # CholecInstanceSeg has 'train', 'val', 'test'
        # We search all available splits and rely on Trainer's split logic
        for split in ['train', 'val']:
            ann_split_dir = os.path.join(self.annotation_root, split)
            img_split_dir = os.path.join(self.image_root, split)
            
            if not os.path.exists(ann_split_dir): continue
            
            vid_dirs = os.listdir(ann_split_dir)
            for vid in vid_dirs:
                ann_path_dir = os.path.join(ann_split_dir, vid, "ann_dir")
                img_path_dir = os.path.join(img_split_dir, vid, "img_dir")
                
                if not os.path.exists(ann_path_dir): continue
                
                json_files = glob.glob(os.path.join(ann_path_dir, "*.json"))
                for json_path in json_files:
                    # Match by basename
                    base = os.path.basename(json_path).replace(".json", ".png")
                    img_path = os.path.join(img_path_dir, base)
                    if os.path.exists(img_path):
                        samples.append((img_path, json_path))
        return samples

    def __len__(self):
        return len(self.samples)

    def _rasterize_polygons(self, json_path, shape):
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        mask = Image.new('L', (shape[1], shape[0]), 0)
        draw = ImageDraw.Draw(mask)
        
        # Mapping tool labels to 5 classes for Stage 2
        label_map = {
            'grasper': 1,
            'bipolar': 1,
            'hook': 2,    # Sharp
            'scissors': 2, # Sharp
            'spatula': 3,  # Thinish
            'irrigator': 4,
            'clipper': 4
        }
        
        for shape_data in data.get('shapes', []):
            label = shape_data.get('label', '').lower()
            points = shape_data.get('points')
            if points:
                # Default to class 1 (Tool) if label not in explicit map
                fill_val = label_map.get(label, 1) if self.stage == 2 else 1
                polygon = [tuple(p) for p in points]
                draw.polygon(polygon, outline=fill_val, fill=fill_val)
        
        return np.array(mask)

    def __getitem__(self, idx):
        img_path, json_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        
        mask_np = self._rasterize_polygons(json_path, (image.size[1], image.size[0]))
        mask = Image.fromarray(mask_np.astype(np.uint8))

        if self.transform:
            image, mask = self.transform(image, mask)
        
        return image, mask
