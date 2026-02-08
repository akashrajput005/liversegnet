import torch
import os
from models.deeplab_liver import LiverSegModelA
from models.unet_tools import UNetResNet34
from datasets.choleseg8k import CholeSeg8kDataset
from datasets.cholec_instance import CholecInstanceSegDataset
from utils.transforms import Compose, Resize, ToTensor, Normalize
import traceback

def simple_dry_run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Using device: {device} ---")
    
    paths = {
        'choleseg8k': r"C:\Users\akash\Downloads\cholecseg8k",
        'cholecinstanceseg': r"C:\Users\akash\Downloads\cholecinstanceseg\cholecinstanceseg",
        'reference': r"C:\Users\akash\Downloads\cholecinstance_seg_reference_image_set"
    }

    transform = Compose([
        Resize((256, 256)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("\n--- Verifying Model A (DeepLabV3+) ---")
    try:
        model_a = LiverSegModelA(num_classes=2, pretrained=False).to(device)
        ds_a = CholeSeg8kDataset(paths['choleseg8k'], stage=1, transform=transform)
        print(f"CholeSeg8k Dataset Size: {len(ds_a)}")
        if len(ds_a) > 0:
            img, mask = ds_a[0]
            img = img.unsqueeze(0).to(device)
            with torch.no_grad():
                out = model_a(img)
                if isinstance(out, dict): out = out['out']
                print(f"Model A Forward Pass Success: Output shape {out.shape}")
        else:
            print("[WARNING] Model A Dataset Empty")
    except Exception as e:
        print(f"[ERROR] Model A Failed: {e}")
        traceback.print_exc()

    print("\n--- Verifying Model B (UNet) ---")
    try:
        model_b = UNetResNet34(num_classes=3, pretrained=False).to(device)
        ds_b = CholecInstanceSegDataset(paths['cholecinstanceseg'], paths['reference'], stage=1, transform=transform)
        print(f"CholecInstanceSeg Dataset Size: {len(ds_b)}")
        if len(ds_b) > 0:
            img, mask = ds_b[0]
            img = img.unsqueeze(0).to(device)
            with torch.no_grad():
                out = model_b(img)
                print(f"Model B Forward Pass Success: Output shape {out.shape}")
        else:
            print("[WARNING] Model B Dataset Empty")
    except Exception as e:
        print(f"[ERROR] Model B Failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    simple_dry_run()
