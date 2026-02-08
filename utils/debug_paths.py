import os
import glob

paths = {
    'choleseg8k': r"C:\Users\akash\Downloads\cholecseg8k",
    'cholecinstanceseg': r"C:\Users\akash\Downloads\cholecinstanceseg\cholecinstanceseg",
}

print(f"Checking CholeSeg8k at {paths['choleseg8k']}")
video_dirs = glob.glob(os.path.join(paths['choleseg8k'], "video*"))
print(f"Found {len(video_dirs)} video dirs: {video_dirs[:2]}")

for vdir in video_dirs[:1]:
    frame_dirs = glob.glob(os.path.join(vdir, "video*"))
    print(f" - Found {len(frame_dirs)} frame dirs in {vdir}: {frame_dirs[:2]}")
    for fdir in frame_dirs[:1]:
        images = glob.glob(os.path.join(fdir, "*_endo.png"))
        print(f"   - Found {len(images)} images in {fdir}")

print(f"\nChecking CholecInstanceSeg at {paths['cholecinstanceseg']}")
for split in ['train', 'val']:
    ann_split_dir = os.path.join(paths['cholecinstanceseg'], split)
    print(f"Checking split {split} at {ann_split_dir}")
    if os.path.exists(ann_split_dir):
        vids = os.listdir(ann_split_dir)
        print(f" - Found {len(vids)} video dirs: {vids[:2]}")
    else:
        print(f" - Split dir NOT FOUND")
