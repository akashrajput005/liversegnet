#!/usr/bin/env python3
"""
Test UI functionality with extracted critical frames
"""

import requests
import json
import os
from pathlib import Path

def test_frame_on_ui(frame_path):
    """Test a specific frame on the UI"""
    try:
        with open(frame_path, 'rb') as f:
            files = {'file': (os.path.basename(frame_path), f, 'image/png')}
            response = requests.post("http://localhost:8000/segment_image", files=files)
        
        if response.status_code == 200:
            result = response.json()
            print(f"PASS {os.path.basename(frame_path)}:")
            print(f"   Model: {result.get('model_used', 'N/A')}")
            print(f"   Liver Detected: {result.get('liver_detected', False)}")
            print(f"   Instruments Detected: {result.get('instrument_detected', False)}")
            print(f"   Inference Time: {result.get('inference_time', 0):.3f}s")
            return True
        else:
            print(f"FAIL {os.path.basename(frame_path)}: API Error {response.status_code}")
            return False
    except Exception as e:
        print(f"ERROR {os.path.basename(frame_path)}: {e}")
        return False

def main():
    print("Testing UI with Critical Frames")
    print("=" * 40)
    
    frames_dir = Path("critical_frames")
    if not frames_dir.exists():
        print("ERROR: Critical frames directory not found. Run extract_critical_frames.py first.")
        return
    
    # Test original frames
    original_frames = sorted(frames_dir.glob("frame_*_original.png"))
    
    if not original_frames:
        print("ERROR: No frames found to test.")
        return
    
    print(f"Testing {len(original_frames)} critical frames...")
    
    success_count = 0
    for frame_path in original_frames[:5]:  # Test first 5 frames
        if test_frame_on_ui(frame_path):
            success_count += 1
    
    print(f"\nResults: {success_count}/{len(original_frames[:5])} frames processed successfully")
    
    if success_count == len(original_frames[:5]):
        print("SUCCESS: UI is working perfectly with critical frames!")
    else:
        print("WARNING: Some issues detected - check API server status")

if __name__ == "__main__":
    main()
