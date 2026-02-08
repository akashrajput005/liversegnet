from PIL import Image
import numpy as np
import sys

mask_path = sys.argv[1]
mask = Image.open(mask_path)
mask_np = np.array(mask)
unique_pixels = np.unique(mask_np.reshape(-1, mask_np.shape[-1]), axis=0)
print(f"Unique pixel values in {mask_path}:")
for p in unique_pixels:
    print(p)
