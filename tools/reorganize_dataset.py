import os
import json
import cv2
import numpy as np
import shutil
from tqdm import tqdm
import yaml

def rasterize_polygons(ann_path, shape):
    mask = np.zeros(shape, dtype=np.uint8)
    with open(ann_path, 'r') as f:
        data = json.load(f)
        for shape_data in data['shapes']:
            points = np.array(shape_data['points'], dtype=np.int32)
            cv2.fillPoly(mask, [points], 2) # Class 2 for Instruments
    return mask

def main():
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    output_root = "C:\\Users\\akash\\Downloads\\CholecSegUnified"
    os.makedirs(output_root, exist_ok=True)
    
    for split in ['train', 'val']:
        img_dir = os.path.join(output_root, split, 'images')
        mask_dir = os.path.join(output_root, split, 'masks')
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
        
        print(f"\nProcessing {split} split...")
        
        # 1. Process Liver (CholecSeg8k) - Just copy/symlink
        # Note: CholecSeg8k doesn't have a strict val split in the current path.
        # We'll use a portion for 'val' if it's the val loop.
        liver_root = config['cholecseg8k_path']
        videos = [d for d in os.listdir(liver_root) if os.path.isdir(os.path.join(liver_root, d))]
        
        # Simple split logic for liver (80/20)
        train_vids = videos[:int(0.8*len(videos))]
        val_vids = videos[int(0.8*len(videos)):]
        liver_split_vids = train_vids if split == 'train' else val_vids
        
        for vid in tqdm(liver_split_vids, desc=f"Liver {split}"):
            vid_path = os.path.join(liver_root, vid)
            subfolders = [d for d in os.listdir(vid_path) if os.path.isdir(os.path.join(vid_path, d))]
            for sub in subfolders:
                sub_path = os.path.join(vid_path, sub)
                frames = [f for f in os.listdir(sub_path) if f.endswith('_endo.png')]
                for f in frames:
                    src_img = os.path.join(sub_path, f)
                    src_mask = src_img.replace('_endo.png', '_endo_mask.png')
                    
                    if os.path.exists(src_mask):
                        dst_name = f"liver_{vid}_{f}"
                        shutil.copy2(src_img, os.path.join(img_dir, dst_name))
                        
                        # Load and remap mask
                        m = cv2.imread(src_mask, cv2.IMREAD_GRAYSCALE)
                        unified_m = np.zeros_like(m)
                        unified_m[m > 0] = 1 # Liver Class
                        cv2.imwrite(os.path.join(mask_dir, dst_name), unified_m)

        # 2. Process Instruments (CholecInstanceSeg)
        inst_root = config['cholecinstanceseg_path']
        ref_root = config['reference_images_path']
        
        split_dir = os.path.join(inst_root, split)
        ref_split_dir = os.path.join(ref_root, split)
        
        if os.path.exists(split_dir):
            inst_vids = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
            for vid in tqdm(inst_vids, desc=f"Inst {split}"):
                ann_dir = os.path.join(split_dir, vid, 'ann_dir')
                img_src_dir = os.path.join(ref_split_dir, vid, 'img_dir')
                
                if os.path.exists(ann_dir) and os.path.exists(img_src_dir):
                    jsons = [f for f in os.listdir(ann_dir) if f.endswith('.json')]
                    for j in jsons:
                        img_name = j.replace('.json', '.png')
                        src_img = os.path.join(img_src_dir, img_name)
                        if os.path.exists(src_img):
                            dst_name = f"inst_{vid}_{img_name}"
                            shutil.copy2(src_img, os.path.join(img_dir, dst_name))
                            
                            # Rasterize and save mask
                            img_shape = cv2.imread(src_img).shape[:2]
                            m = rasterize_polygons(os.path.join(ann_dir, j), img_shape)
                            cv2.imwrite(os.path.join(mask_dir, dst_name), m)

    print("\nDataset reorganization complete!")
    print(f"Unified data located at: {output_root}")

if __name__ == "__main__":
    main()
