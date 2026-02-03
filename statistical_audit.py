import os
import cv2
import numpy as np
from tqdm import tqdm
import yaml

def audit():
    roots = {
        'cholecseg8k': 'C:\\Users\\akash\\Downloads\\cholecseg8k',
        'cholecinstanceseg': 'C:\\Users\\akash\\Downloads\\cholecinstanceseg',
        'reference_set': 'C:\\Users\\akash\\Downloads\\cholecinstance_seg_reference_image_set'
    }
    
    # Simple list collection (matches PinnacleSurgicalDataset logic)
    mask_paths = []
    
    # 1. Cholec8k
    path = roots['cholecseg8k']
    if os.path.exists(path):
        for root, _, files in os.walk(path):
            for f in files:
                if f.endswith('_endo_mask.png'):
                    mask_paths.append(os.path.join(root, f))
                    
    # 2. Reference Set
    path = roots['reference_set']
    if os.path.exists(path):
        for split in ['train', 'val']:
            mask_dir = os.path.join(path, split, 'masks')
            if os.path.exists(mask_dir):
                for f in os.listdir(mask_dir):
                    if f.endswith('.png'):
                        mask_paths.append(os.path.join(mask_dir, f))

    print(f"🔬 Auditing {len(mask_paths)} mask samples...")
    
    total_pixels = 0
    class_counts = {0: 0, 1: 0, 2: 0}
    frames_with_tools = 0
    
    # Sample 10% for speed if it's too large, or go full if needed
    for m_path in tqdm(mask_paths, desc="📊 Scanning Masks"):
        mask = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)
        if mask is None: continue
        
        # Apply the SAME Universal Remapper logic we use in dataset.py
        new_mask = np.zeros_like(mask)
        anatomy_ids = [1, 2, 3, 4, 6, 8, 10, 11, 12, 13, 21, 22, 31, 32, 128]
        tool_ids = [5, 7, 9, 50, 90, 149, 255]
        
        for aid in anatomy_ids: new_mask[mask == aid] = 1
        for tid in tool_ids: new_mask[mask == tid] = 2
        
        # Stats
        total_pixels += mask.size
        class_counts[0] += np.sum(new_mask == 0)
        class_counts[1] += np.sum(new_mask == 1)
        class_counts[2] += np.sum(new_mask == 2)
        
        if 2 in new_mask:
            frames_with_tools += 1
            
    print("\n" + "="*40)
    print("📋 SURGICAL DATASET STATISTICAL AUDIT")
    print("="*40)
    print(f"Total Frames Analyzed: {len(mask_paths)}")
    print(f"Frames with Instruments: {frames_with_tools} ({frames_with_tools/len(mask_paths)*100:.1f}%)")
    print("-" * 20)
    print("Pixel Distribution:")
    for cls, count in class_counts.items():
        name = ["Background", "Anatomy", "Instruments"][cls]
        print(f"  - {name}: {count/total_pixels*100:.2f}%")
    print("="*40)

if __name__ == "__main__":
    audit()
