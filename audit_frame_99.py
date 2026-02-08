import cv2
import numpy as np
import os

def audit_frame_99():
    mask_path = r"C:\Users\akash\Downloads\cholecseg8k\video01\video01_00080\frame_99_endo_mask.png"
    mask = cv2.imread(mask_path, 0)
    unique_vals = np.unique(mask)
    print(f"Unique values: {unique_vals}")
    
    for val in unique_vals:
        bin_mask = (mask == val).astype(np.uint8) * 255
        cv2.imwrite(f"frame_99_index_{val}.png", bin_mask)
        print(f"Saved: frame_99_index_{val}.png (Area: {np.sum(mask==val)})")

if __name__ == "__main__":
    audit_frame_99()
