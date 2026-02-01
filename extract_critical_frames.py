#!/usr/bin/env python3
"""
Extract critical frames with liver and multiple tools for UI testing
"""

import os
import cv2
import numpy as np
import random
from pathlib import Path
import shutil

try:
    from src.cholec_dataset import CholecSeg8kDataset, get_transforms
except ImportError:
    from cholec_dataset import CholecSeg8kDataset, get_transforms

import yaml

def load_config():
    """Load configuration"""
    with open('configs/config.yaml', 'r') as f:
        return yaml.safe_load(f)

def analyze_frame_complexity(mask):
    """Analyze frame complexity based on liver and tool presence"""
    unique, counts = np.unique(mask, return_counts=True)
    
    liver_pixels = counts[unique == 1].sum() if 1 in unique else 0
    tool_pixels = counts[unique == 2].sum() if 2 in unique else 0
    total_pixels = mask.size
    
    # Calculate complexity scores
    liver_ratio = liver_pixels / total_pixels
    tool_ratio = tool_pixels / total_pixels
    
    # Check for multiple tools (disconnected regions)
    tool_mask = (mask == 2).astype(np.uint8)
    num_contours, _ = cv2.findContours(tool_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    multiple_tools = len(num_contours) >= 2
    
    # Edge complexity (liver-tool boundaries)
    liver_mask = (mask == 1).astype(np.uint8)
    edges = cv2.Canny(liver_mask, 100, 200)
    edge_complexity = np.sum(edges > 0) / total_pixels
    
    complexity_score = {
        'liver_ratio': liver_ratio,
        'tool_ratio': tool_ratio,
        'multiple_tools': multiple_tools,
        'num_tools': len(num_contours),
        'edge_complexity': edge_complexity,
        'overall_score': liver_ratio * 0.3 + tool_ratio * 0.3 + multiple_tools * 0.2 + edge_complexity * 0.2
    }
    
    return complexity_score

def extract_critical_frames():
    """Extract and save raw surgical frames for AI testing (unlabeled)"""
    print("🔍 Extracting Raw Surgical Frames for AI Testing")
    print("=" * 60)
    
    # Load config and dataset
    config = load_config()
    dataset_root = config.get('cholecseg8k_path') or config.get('unified_dataset_path')
    
    if not dataset_root or not os.path.exists(dataset_root):
        print(f"❌ Dataset path not found: {dataset_root}")
        return
    
    # Create dataset
    dataset = CholecSeg8kDataset(
        root_dir=dataset_root,
        transform=get_transforms(is_train=False),
        target_size=tuple(config.get('input_size', [256, 256])),
        max_samples=None,
    )
    
    print(f"📊 Dataset loaded: {len(dataset)} frames")
    
    # Analyze all frames for good surgical scenes
    print("🔍 Analyzing frames for clear liver and tools...")
    frame_scores = []
    
    for i in range(min(len(dataset), 1000)):  # Analyze first 1000 frames
        try:
            image, mask = dataset[i]
            mask_np = mask.numpy()
            
            complexity = analyze_frame_complexity(mask_np)
            complexity['index'] = i
            complexity['dataset_path'] = dataset.image_paths[i] if hasattr(dataset, 'image_paths') else f"frame_{i}"
            
            frame_scores.append(complexity)
            
            if i % 100 == 0:
                print(f"   Analyzed {i}/{min(len(dataset), 1000)} frames...")
                
        except Exception as e:
            continue
    
    # Sort by complexity score
    frame_scores.sort(key=lambda x: x['overall_score'], reverse=True)
    
    print(f"✅ Analyzed {len(frame_scores)} frames")
    
    # Select best raw surgical frames (unlabeled)
    raw_frames = []
    
    # Category 1: Clear liver + multiple tools visible
    category1 = [f for f in frame_scores if f['liver_ratio'] > 0.1 and f['multiple_tools'] and f['num_tools'] >= 2][:5]
    
    # Category 2: High tool density for instrument detection
    category2 = [f for f in frame_scores if f['tool_ratio'] > 0.15 and f['num_tools'] >= 3][:5]
    
    # Category 3: Good liver visibility for organ detection
    category3 = [f for f in frame_scores if f['liver_ratio'] > 0.2 and f['edge_complexity'] > 0.01][:5]
    
    # Category 4: Balanced surgical scenes
    category4 = [f for f in frame_scores if 0.1 < f['liver_ratio'] < 0.4 and 0.1 < f['tool_ratio'] < 0.4][:5]
    
    raw_frames = category1 + category2 + category3 + category4
    
    # Remove duplicates
    seen_indices = set()
    unique_frames = []
    for frame in raw_frames:
        if frame['index'] not in seen_indices:
            unique_frames.append(frame)
            seen_indices.add(frame['index'])
    
    raw_frames = unique_frames[:15]  # Keep top 15 unique raw frames
    
    print(f"🎯 Selected {len(raw_frames)} raw surgical frames for AI testing")
    
    # Create output directory
    output_dir = Path("critical_frames")
    output_dir.mkdir(exist_ok=True)
    
    # Extract and save RAW frames (no pre-segmentation)
    print("💾 Extracting raw surgical frames...")
    
    for i, frame_info in enumerate(raw_frames):
        try:
            idx = frame_info['index']
            image, mask = dataset[idx]
            
            # Convert tensors to numpy
            if hasattr(image, 'numpy'):
                image_np = image.numpy().transpose(1, 2, 0)
            else:
                image_np = image
            
            if hasattr(mask, 'numpy'):
                mask_np = mask.numpy()
            else:
                mask_np = mask
            
            # Denormalize image if needed
            if image_np.max() <= 1.0:
                image_np = (image_np * 255).astype(np.uint8)
            
            # Ensure image is uint8
            image_np = image_np.astype(np.uint8)
            
            # Save ONLY the raw original image (no segmentation)
            cv2.imwrite(str(output_dir / f"frame_{i:02d}_raw.png"), image_np)
            
            # Save metadata for reference (but not the segmentation)
            metadata = {
                'frame_index': idx,
                'liver_ratio': frame_info['liver_ratio'],
                'tool_ratio': frame_info['tool_ratio'],
                'num_tools': frame_info['num_tools'],
                'multiple_tools': frame_info['multiple_tools'],
                'edge_complexity': frame_info['edge_complexity'],
                'overall_score': frame_info['overall_score'],
                'note': 'Raw surgical frame - AI segmentation needed'
            }
            
            with open(output_dir / f"frame_{i:02d}_info.txt", 'w') as f:
                for key, value in metadata.items():
                    f.write(f"{key}: {value}\n")
            
            print(f"   ✅ Raw Frame {i:2d}: Liver={frame_info['liver_ratio']:.3f}, Tools={frame_info['tool_ratio']:.3f}, Count={frame_info['num_tools']}")
            
        except Exception as e:
            print(f"   ❌ Error processing frame {i}: {e}")
            continue
    
    print(f"\n🎉 Raw surgical frames extraction complete!")
    print(f"📁 Output directory: {output_dir.absolute()}")
    print(f"📊 Raw frames saved: {len([f for f in output_dir.glob('*.png')])} (unsegmented surgical scenes)")
    print(f"� Purpose: Test AI segmentation on raw surgical footage")
    
    # Create summary
    print(f"\n📋 Raw Frame Categories Summary:")
    print(f"   Clear Liver + Multiple Tools: {len(category1)}")
    print(f"   High Tool Density: {len(category2)}")
    print(f"   Good Liver Visibility: {len(category3)}")
    print(f"   Balanced Surgical Scenes: {len(category4)}")
    
    return output_dir

def create_ui_test_script():
    """Create a simple script to test UI with extracted frames"""
    script_content = '''#!/usr/bin/env python3
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
    
    print(f"\\nResults: {success_count}/{len(original_frames[:5])} frames processed successfully")
    
    if success_count == len(original_frames[:5]):
        print("SUCCESS: UI is working perfectly with critical frames!")
    else:
        print("WARNING: Some issues detected - check API server status")

if __name__ == "__main__":
    main()
'''
    
    with open("test_ui_with_frames.py", "w", encoding='utf-8') as f:
        f.write(script_content)
    
    print("Created test_ui_with_frames.py for UI testing")

if __name__ == "__main__":
    # Extract critical frames
    output_dir = extract_critical_frames()
    
    # Create UI test script
    create_ui_test_script()
    
    print(f"\nReady to test UI!")
    print(f"1. Open http://localhost:8501 in your browser")
    print(f"2. Upload frames from: {output_dir}")
    print(f"3. Or run: python test_ui_with_frames.py")
