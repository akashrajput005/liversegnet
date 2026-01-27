import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class CholecInstancePolygonDataset(Dataset):
    def __init__(self, ann_root: str, img_root: str, split: str = 'train', target_size=(512, 512)):
        self.ann_root = ann_root
        self.img_root = img_root
        self.split = split
        self.target_size = target_size
        self.samples = []  # (img_path, json_path, video_id)

        split_dir = os.path.join(ann_root, split)
        if os.path.isdir(split_dir):
            for video_dir in os.listdir(split_dir):
                ann_dir = os.path.join(split_dir, video_dir, 'ann_dir')
                if not os.path.isdir(ann_dir):
                    continue

                for fname in os.listdir(ann_dir):
                    if not fname.lower().endswith('.json'):
                        continue
                    json_path = os.path.join(ann_dir, fname)

                    # Fast path: image filename usually matches json basename
                    base = os.path.splitext(fname)[0] + '.png'
                    img_path = os.path.join(img_root, split, video_dir, 'img_dir', base)

                    if not os.path.exists(img_path):
                        # Slow path: read JSON to locate imagePath
                        try:
                            with open(json_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                        except Exception:
                            continue

                        image_rel = data.get('imagePath')
                        if not image_rel:
                            continue

                        # JSON sometimes stores "img_dir/xxx.png" or just filename
                        img_path2 = os.path.join(img_root, split, video_dir, image_rel)
                        if os.path.exists(img_path2):
                            img_path = img_path2
                        else:
                            img_path3 = os.path.join(img_root, split, video_dir, 'img_dir', os.path.basename(image_rel))
                            if os.path.exists(img_path3):
                                img_path = img_path3
                            else:
                                continue

                    self.samples.append((img_path, json_path, f"{split}:{video_dir}"))

        print(f"Loaded {len(self.samples)} polygon samples from CholecInstanceSeg ({split})")

    def __len__(self):
        return len(self.samples)

    def get_video_id(self, idx: int) -> str:
        return self.samples[idx][2]

    def __getitem__(self, idx):
        img_path, json_path, _video_id = self.samples[idx]

        image_bgr = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        h, w = image_rgb.shape[:2]

        # 255 = ignore (unknown), 2 = instrument polygons
        mask = np.full((h, w), 255, dtype=np.uint8)

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        shapes = data.get('shapes', [])
        polys = []
        for s in shapes:
            if s.get('shape_type') != 'polygon':
                continue
            pts = s.get('points')
            if not pts:
                continue
            pts_np = np.array(pts, dtype=np.int32)
            if pts_np.ndim != 2 or pts_np.shape[1] != 2:
                continue
            polys.append(pts_np)

        if polys:
            cv2.fillPoly(mask, polys, 2)

        image_rgb = cv2.resize(image_rgb, (self.target_size[1], self.target_size[0]))
        mask = cv2.resize(mask, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_NEAREST)

        image = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).long()

        return image, mask
