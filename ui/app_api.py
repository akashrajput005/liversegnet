from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
import cv2
import numpy as np
import os
import sys
import uuid

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from infer import InferenceEngine

app = FastAPI()

# Load config
CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'configs', 'config.yaml')
import yaml
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

# Initialize engine with Advanced Model
MODEL_NAME = f"{config['active_model']}_{config['active_encoder']}.pth"
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', MODEL_NAME)

engine = InferenceEngine(
    model_path=MODEL_PATH,
    architecture=config['active_model'],
    encoder=config['active_encoder'],
    img_size=config['input_size']
)

UPLOAD_DIR = "uploads"
RESULTS_DIR = "results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

@app.post("/segment_image")
async def segment_image(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    ext = file.filename.split('.')[-1]
    input_path = os.path.join(UPLOAD_DIR, f"{file_id}.{ext}")
    
    with open(input_path, "wb") as buffer:
        buffer.write(await file.read())
        
    image = cv2.imread(input_path)
    mask, overlay, occlusion, distance, liver_found, inst_found = engine.predict_image(image)
    
    output_path = os.path.join(RESULTS_DIR, f"{file_id}_overlay.png")
    cv2.imwrite(output_path, overlay)
    
    # Sanitize for JSON (inf -> -1)
    safe_occlusion = float(occlusion) if np.isfinite(occlusion) else 0.0
    safe_distance = float(distance) if np.isfinite(distance) else -1.0
    
    return {
        "occlusion_percent": safe_occlusion,
        "distance_px": safe_distance,
        "liver_detected": liver_found,
        "tool_detected": inst_found,
        "overlay_url": f"/results/{file_id}_overlay.png"
    }

@app.post("/segment_video")
async def segment_video(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_DIR, f"{file_id}.mp4")
    
    with open(input_path, "wb") as buffer:
        buffer.write(await file.read())
        
    output_path = os.path.join(RESULTS_DIR, f"{file_id}_segmented.mp4")
    success = engine.predict_video(input_path, output_path)
    
    if success:
        # Engine converts .mp4 to .avi for browser compatibility
        video_url = f"/results/{file_id}_segmented.avi"
        return {"video_url": video_url}
    else:
        return JSONResponse(status_code=400, content={"message": "Video processing failed"})

@app.post("/reload_engine")
async def reload_engine():
    global engine
    # Reload config
    with open(CONFIG_PATH, 'r') as f:
        new_config = yaml.safe_load(f)
    
    MODEL_NAME = f"{new_config['active_model']}_{new_config['active_encoder']}.pth"
    MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', MODEL_NAME)
    
    engine = InferenceEngine(
        model_path=MODEL_PATH,
        architecture=new_config['active_model'],
        encoder=new_config['active_encoder'],
        img_size=new_config['input_size']
    )
    return {"message": f"Engine reloaded with {new_config['active_model']}"}

@app.get("/results/{filename}")
async def get_result(filename: str):
    return FileResponse(os.path.join(RESULTS_DIR, filename))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
