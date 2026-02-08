import sys
import os
import traceback
sys.path.append(os.getcwd())
try:
    from training.train_pipeline import start_training
    from datasets.choleseg8k import CholeSeg8kDataset
    from datasets.cholec_instance import CholecInstanceSegDataset
except ImportError as e:
    print(f"[CRITICAL] Import Error: {str(e)}")
    sys.exit(1)

if __name__ == "__main__":
    paths = {
        'choleseg8k': r"C:\Users\akash\Downloads\cholecseg8k",
        'cholecinstanceseg': r"C:\Users\akash\Downloads\cholecinstanceseg\cholecinstanceseg",
        'reference': r"C:\Users\akash\Downloads\cholecinstance_seg_reference_image_set"
    }
    
    # Dataset diagnostics
    print("--- Dataset Diagnostics ---")
    d1 = CholeSeg8kDataset(paths['choleseg8k'])
    print(f"CholeSeg8k Dataset Size: {len(d1)}")
    
    d2 = CholecInstanceSegDataset(paths['cholecinstanceseg'], paths['reference'])
    print(f"CholecInstanceSeg Dataset Size: {len(d2)}")
    
    if len(d1) == 0 or len(d2) == 0:
        print("[WARNING] One or more datasets are empty. Training may fail.")

    print("\n--- Clinical Hardware Dry Run: Model A (Anatomy) ---")
    try:
        start_training(model_type='A', stage=1, root_dirs=paths, dry_run=True)
        print("[SUCCESS] Model A Dry Run Passed.")
    except Exception as e:
        traceback.print_exc()
        print(f"[ERROR] Model A Dry Run Failed: {str(e)}")

    print("\n--- Clinical Hardware Dry Run: Model B (Tools) ---")
    try:
        start_training(model_type='B', stage=1, root_dirs=paths, dry_run=True)
        print("[SUCCESS] Model B Dry Run Passed.")
    except Exception as e:
        traceback.print_exc()
        print(f"[ERROR] Model B Dry Run Failed: {str(e)}")
