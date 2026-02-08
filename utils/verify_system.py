import torch
import logging
import os
from models.deeplab_liver import LiverSegModelA
from models.unet_tools import UNetResNet34
from datasets.choleseg8k import CholeSeg8kDataset
from datasets.cholec_instance import CholecInstanceSegDataset
from utils.transforms import Compose, Resize, ToTensor

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def verify_system():
    paths = {
        'choleseg8k': r"C:\Users\akash\Downloads\cholecseg8k",
        'cholecinstanceseg': r"C:\Users\akash\Downloads\cholecinstanceseg\cholecinstanceseg",
        'reference': r"C:\Users\akash\Downloads\cholecinstance_seg_reference_image_set"
    }

    logging.info("--- 1. Validating Dataset Loaders ---")
    try:
        ds_a = CholeSeg8kDataset(paths['choleseg8k'], stage=1)
        logging.info(f"CholeSeg8k: Found {len(ds_a)} samples.")
        
        ds_b = CholecInstanceSegDataset(paths['cholecinstanceseg'], paths['reference'], stage=1)
        logging.info(f"CholecInstanceSeg: Found {len(ds_b)} samples.")
    except Exception as e:
        logging.error(f"Dataset Loader Error: {e}")

    logging.info("\n--- 2. Validating Model Instantiation ---")
    try:
        model_a = LiverSegModelA(num_classes=2)
        logging.info("Model A (DeepLabV3+) instantiated successfully.")
        
        model_b = UNetResNet34(num_classes=2)
        logging.info("Model B (U-Net) instantiated successfully.")
    except Exception as e:
        logging.error(f"Model Instantiation Error: {e}")

    logging.info("\n--- 3. Verifying Forward Pass (Dummy Data) ---")
    try:
        dummy_input = torch.randn(1, 3, 256, 256)
        
        out_a = model_a(dummy_input)
        if isinstance(out_a, dict): out_a = out_a['out']
        logging.info(f"Model A Forward Pass: Success. Output shape: {out_a.shape}")
        
        out_b = model_b(dummy_input)
        logging.info(f"Model B Forward Pass: Success. Output shape: {out_b.shape}")
    except Exception as e:
        logging.error(f"Forward Pass Error: {e}")

    logging.info("\n--- Verification Complete ---")

if __name__ == "__main__":
    verify_system()
