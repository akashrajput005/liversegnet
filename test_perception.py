import cv2
import numpy as np
import sys
import os
sys.path.insert(0, 'src')
from infer import InferenceEngine

print("-" * 30)
print("INFERENCE TEST START")
print("-" * 30)

engine = InferenceEngine(
    model_path="models/deeplabv3plus_resnet50.pth",
    architecture='deeplabv3plus',
    encoder='resnet50',
    img_size=(256, 256),
    num_classes=3
)

image_path = "critical_frames/frame_00_original.png"
image = cv2.imread(image_path)
if image is None:
    print(f"Error: Could not load {image_path}")
    sys.exit(1)

mask, overlay, occlusion, distance, liver_found, inst_found = engine.predict_image(image)

print(f"Liver found: {liver_found}")
print(f"Instrument found: {inst_found}")
print(f"Occlusion: {occlusion:.2f}%")
print(f"Distance: {distance:.2f}px")

os.makedirs("test_output", exist_ok=True)
cv2.imwrite("test_output/final_test_overlay.png", overlay)
print("Saved result to test_output/final_test_overlay.png")
print("-" * 30)
print("INFERENCE TEST COMPLETE")
print("-" * 30)
