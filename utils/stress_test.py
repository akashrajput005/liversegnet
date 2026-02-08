import torch
import time
import logging
import psutil
from models.deeplab_liver import LiverSegModelA
from models.unet_tools import UNetResNet34
from inference.engine import ParallelInferenceEngine

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def stress_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using Device: {device}")
    
    # 1. Initialize Engines
    logging.info("Initializing models for stress test...")
    model_a = LiverSegModelA(num_classes=2)
    model_b = UNetResNet34(num_classes=2)
    engine = ParallelInferenceEngine(model_a, model_b, device=device)
    
    # 2. Mock Stress Load
    dummy_img_path = "dummy_test.png"
    from PIL import Image
    import numpy as np
    Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)).save(dummy_img_path)
    
    iters = 20
    logging.info(f"Starting Stress Test: {iters} concurrent inference iterations...")
    
    start_time = time.time()
    for i in range(iters):
        # Tracking memory before each run
        gpu_mem = torch.cuda.memory_allocated(device) / (1024**2) if device.type == 'cuda' else 0
        cpu_load = psutil.cpu_percent()
        
        results = engine.run_inference(dummy_img_path)
        
        if i % 5 == 0:
            logging.info(f"Iteration {i}: GPU Mem: {gpu_mem:.2f} MiB | CPU: {cpu_load}% | Risk: {results['risk_status']}")
            
    end_time = time.time()
    avg_time = (end_time - start_time) / iters
    logging.info(f"Average Inference Time (Parallel Streams): {avg_time:.4f}s")
    logging.info(f"Throughput: {1/avg_time:.2f} fps")
    
    # Clean up
    import os
    if os.path.exists(dummy_img_path): os.remove(dummy_img_path)
    logging.info("Stress Test Complete.")

if __name__ == "__main__":
    import traceback
    try:
        stress_test()
    except Exception as e:
        traceback.print_exc()
