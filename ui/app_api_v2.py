from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
import cv2
import numpy as np
import os
import sys
import uuid
from typing import Dict, List, Optional
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from infer import InferenceEngine

app = FastAPI(title="LiverSegNet API", version="2.0")

# Load config
CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'configs', 'config.yaml')
import yaml
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

# Initialize multiple engines for ensemble
def initialize_engines():
    engines = {}
    
    # U-Net Baseline
    unet_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'unet_resnet34.pth')
    if os.path.exists(unet_path):
        engines['unet'] = InferenceEngine(
            model_path=unet_path,
            architecture='unet',
            encoder='resnet34',
            img_size=config['input_size']
        )
        print("✅ U-Net engine loaded")
    
    # DeepLabV3+ Advanced
    deeplab_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'deeplabv3plus_resnet50.pth')
    if os.path.exists(deeplab_path):
        engines['deeplabv3plus'] = InferenceEngine(
            model_path=deeplab_path,
            architecture='deeplabv3plus',
            encoder='resnet50',
            img_size=config['input_size']
        )
        print("✅ DeepLabV3+ engine loaded")
    
    # Stage 1 Anatomy
    stage1_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'deeplabv3plus_resnet50_stage1.pth')
    if os.path.exists(stage1_path):
        engines['stage1'] = InferenceEngine(
            model_path=stage1_path,
            architecture='deeplabv3plus',
            encoder='resnet50',
            img_size=config['input_size'],
            num_classes=2
        )
        print("✅ Stage 1 engine loaded")
    
    return engines

engines = initialize_engines()
active_engine = engines.get('deeplabv3plus', list(engines.values())[0] if engines else None)

UPLOAD_DIR = "uploads"
RESULTS_DIR = "results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def calculate_advanced_metrics(mask: np.ndarray, ground_truth: Optional[np.ndarray] = None) -> Dict:
    """Calculate comprehensive metrics for segmentation results"""
    h, w = mask.shape
    liver_mask = (mask == 1).astype(np.uint8)
    inst_mask = (mask == 2).astype(np.uint8)
    
    # Basic metrics
    liver_pixels = np.sum(liver_mask)
    inst_pixels = np.sum(inst_mask)
    total_pixels = h * w
    
    # Area percentages
    liver_percent = (liver_pixels / total_pixels) * 100
    inst_percent = (inst_pixels / total_pixels) * 100
    
    # Connected components analysis
    num_liver_regions = cv2.connectedComponents(liver_mask)[0] - 1
    num_inst_regions = cv2.connectedComponents(inst_mask)[0] - 1
    
    # Contour analysis
    liver_contours = cv2.findContours(liver_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    inst_contours = cv2.findContours(inst_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    
    # Calculate perimeter and area for each region
    liver_perimeter = sum(cv2.arcLength(cnt, True) for cnt in liver_contours)
    inst_perimeter = sum(cv2.arcLength(cnt, True) for cnt in inst_contours)
    
    # Compactness (4π*Area/Perimeter²)
    liver_compactness = []
    inst_compactness = []
    
    for cnt in liver_contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter > 0:
            compactness = 4 * np.pi * area / (perimeter ** 2)
            liver_compactness.append(compactness)
    
    for cnt in inst_contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter > 0:
            compactness = 4 * np.pi * area / (perimeter ** 2)
            inst_compactness.append(compactness)
    
    metrics = {
        'liver_area_percent': float(liver_percent),
        'instrument_area_percent': float(inst_percent),
        'liver_regions': int(num_liver_regions),
        'instrument_regions': int(num_inst_regions),
        'liver_perimeter': float(liver_perimeter),
        'instrument_perimeter': float(inst_perimeter),
        'liver_compactness_mean': float(np.mean(liver_compactness)) if liver_compactness else 0.0,
        'instrument_compactness_mean': float(np.mean(inst_compactness)) if inst_compactness else 0.0,
        'liver_detected': bool(liver_pixels > 100),
        'instrument_detected': bool(inst_pixels > 50)
    }
    
    return metrics

def ensemble_prediction(image: np.ndarray, engines_dict: Dict) -> Dict:
    """Perform ensemble prediction using multiple models"""
    results = {}
    
    for name, engine in engines_dict.items():
        try:
            start_time = time.time()
            mask, overlay, occlusion, distance, liver_found, inst_found = engine.predict_image(image)
            inference_time = time.time() - start_time
            
            results[name] = {
                'mask': mask,
                'overlay': overlay,
                'occlusion': occlusion,
                'distance': distance,
                'liver_found': liver_found,
                'inst_found': inst_found,
                'inference_time': inference_time,
                'metrics': calculate_advanced_metrics(mask)
            }
        except Exception as e:
            print(f"Error in {name} engine: {e}")
            continue
    
    return results

def _normalize_mask_for_ensemble(model_name: str, mask: np.ndarray) -> np.ndarray:
    """Normalize per-model mask to global class ids: 0=bg, 1=liver, 2=instrument.

    Stage1 model is 2-class (0=bg, 1=liver). Convert to global ids.
    """
    if model_name == 'stage1':
        out = np.zeros_like(mask, dtype=np.uint8)
        out[mask == 1] = 1
        return out
    return mask.astype(np.uint8)

@app.get("/")
async def root():
    return {
        "message": "LiverSegNet API v2.0",
        "models_loaded": list(engines.keys()),
        "active_model": active_engine.__class__.__name__ if active_engine else None
    }

@app.get("/models")
async def get_available_models():
    return {
        "available_models": list(engines.keys()),
        "model_info": {
            "unet": {
                "name": "U-Net ResNet34",
                "type": "baseline",
                "strengths": ["fast", "memory_efficient", "good_baseline"]
            },
            "deeplabv3plus": {
                "name": "DeepLabV3+ ResNet50", 
                "type": "advanced",
                "strengths": ["precise_boundaries", "multi_scale", "high_accuracy"]
            },
            "stage1": {
                "name": "Stage 1 Anatomy",
                "type": "specialized",
                "strengths": ["liver_specialized", "2_class", "anatomy_focused"]
            }
        }
    }

@app.post("/segment_image")
async def segment_image(file: UploadFile = File(...), model: Optional[str] = None, ensemble: bool = False):
    file_id = str(uuid.uuid4())
    ext = file.filename.split('.')[-1]
    input_path = os.path.join(UPLOAD_DIR, f"{file_id}.{ext}")
    
    with open(input_path, "wb") as buffer:
        buffer.write(await file.read())
        
    image = cv2.imread(input_path)
    
    if ensemble and len(engines) > 1:
        # Ensemble prediction
        results = ensemble_prediction(image, engines)

        if not results:
            return JSONResponse(status_code=500, content={"message": "No models produced a valid result."})
        
        # Combine results (weighted average based on model confidence)
        combined_mask = np.zeros_like(image[:, :, 0], dtype=np.float32)
        total_weight = 0
        
        for name, result in results.items():
            weight = 1.0  # Can be adjusted based on validation performance
            if name == 'deeplabv3plus':
                weight = 1.5  # Higher weight for advanced model
            elif name == 'stage1':
                weight = 0.8  # Lower weight for specialized model
                
            norm_mask = _normalize_mask_for_ensemble(name, result['mask'])
            combined_mask += norm_mask.astype(np.float32) * float(weight)
            total_weight += weight
        
        combined_mask = (combined_mask / max(total_weight, 1e-6)).round().astype(np.uint8)
        
        # Generate overlay for combined result
        from src.analytics import get_overlay, calculate_occlusion, calculate_min_distance
        overlay = get_overlay(image, combined_mask)
        occlusion = calculate_occlusion((combined_mask == 1).astype(np.uint8), (combined_mask == 2).astype(np.uint8))
        distance = calculate_min_distance((combined_mask == 1).astype(np.uint8), (combined_mask == 2).astype(np.uint8))
        
        # Calculate final metrics
        final_metrics = calculate_advanced_metrics(combined_mask)
        
        # Save overlay
        output_path = os.path.join(RESULTS_DIR, f"{file_id}_ensemble_overlay.png")
        cv2.imwrite(output_path, overlay)
        
        # Prepare response with ensemble results
        response = {
            "mode": "ensemble",
            "models_used": list(results.keys()),
            "occlusion_percent": float(occlusion) if np.isfinite(occlusion) else 0.0,
            "distance_px": float(distance) if np.isfinite(distance) else -1.0,
            "overlay_url": f"/results/{file_id}_ensemble_overlay.png",
            "metrics": final_metrics,
            "individual_results": {}
        }
        
        # Add individual model results
        for name, result in results.items():
            response["individual_results"][name] = {
                "inference_time": result["inference_time"],
                "metrics": result["metrics"],
                "liver_detected": result["liver_found"],
                "instrument_detected": result["inst_found"]
            }
        
        return response
    
    else:
        # Single model prediction
        selected_engine = engines.get(model, active_engine) if model else active_engine
        
        if not selected_engine:
            return JSONResponse(
                status_code=400, 
                content={"message": f"Model '{model}' not available. Available: {list(engines.keys())}"}
            )
        
        start_time = time.time()
        mask, overlay, occlusion, distance, liver_found, inst_found = selected_engine.predict_image(image)
        inference_time = time.time() - start_time
        
        # Calculate advanced metrics
        metrics = calculate_advanced_metrics(mask)
        
        output_path = os.path.join(RESULTS_DIR, f"{file_id}_overlay.png")
        cv2.imwrite(output_path, overlay)
        
        # Save mask separately for analysis
        mask_path = os.path.join(RESULTS_DIR, f"{file_id}_mask.png")
        cv2.imwrite(mask_path, mask * 85)  # Scale for visibility
        
        return {
            "mode": "single_model",
            "model_used": model or "default",
            "inference_time": inference_time,
            "occlusion_percent": float(occlusion) if np.isfinite(occlusion) else 0.0,
            "distance_px": float(distance) if np.isfinite(distance) else -1.0,
            "liver_detected": liver_found,
            "instrument_detected": inst_found,
            "overlay_url": f"/results/{file_id}_overlay.png",
            "mask_url": f"/results/{file_id}_mask.png",
            "metrics": metrics
        }

@app.post("/segment_video")
async def segment_video(file: UploadFile = File(...), model: Optional[str] = None):
    file_id = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_DIR, f"{file_id}.mp4")
    
    with open(input_path, "wb") as buffer:
        buffer.write(await file.read())
        
    selected_engine = engines.get(model, active_engine) if model else active_engine
    
    if not selected_engine:
        return JSONResponse(
            status_code=400, 
            content={"message": f"Model '{model}' not available. Available: {list(engines.keys())}"}
        )
    
    output_path = os.path.join(RESULTS_DIR, f"{file_id}_segmented.mp4")
    start_time = time.time()
    success = selected_engine.predict_video(input_path, output_path)
    processing_time = time.time() - start_time
    
    if success:
        video_url = f"/results/{file_id}_segmented.avi"
        return {
            "video_url": video_url,
            "model_used": model or "default",
            "processing_time": processing_time
        }
    else:
        return JSONResponse(status_code=400, content={"message": "Video processing failed"})

@app.post("/reload_engine")
async def reload_engine():
    global engines, active_engine
    engines = initialize_engines()
    active_engine = engines.get('deeplabv3plus', list(engines.values())[0] if engines else None)
    return {
        "message": "Engines reloaded successfully",
        "models_loaded": list(engines.keys()),
        "active_model": active_engine.__class__.__name__ if active_engine else None
    }

@app.get("/results/{filename}")
async def get_result(filename: str):
    return FileResponse(os.path.join(RESULTS_DIR, filename))

@app.get("/health")
async def health_check():
    gpu_available = False
    if engines:
        try:
            first_engine = list(engines.values())[0]
            gpu_available = 'cuda' in str(first_engine.device)
        except:
            pass
    
    return {
        "status": "healthy",
        "engines_loaded": len(engines),
        "models": list(engines.keys()),
        "gpu_available": gpu_available
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
