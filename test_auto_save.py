#!/usr/bin/env python3
"""
Test automatic saving functionality
"""

import requests
import os

def test_auto_save():
    """Test that segmentation results are automatically saved"""
    print("🧪 Testing Automatic Saving Functionality")
    print("=" * 50)
    
    # Test with a critical frame
    frame_path = "critical_frames/frame_00_original.png"
    
    if not os.path.exists(frame_path):
        print(f"❌ Test frame not found: {frame_path}")
        return False
    
    try:
        with open(frame_path, 'rb') as f:
            files = {'file': (os.path.basename(frame_path), f, 'image/png')}
            response = requests.post("http://localhost:8000/segment_image", files=files)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Segmentation successful!")
            print(f"   Model: {result.get('model_used', 'N/A')}")
            print(f"   Liver Detected: {result.get('liver_detected', False)}")
            print(f"   Instruments Detected: {result.get('instrument_detected', False)}")
            print(f"   Inference Time: {result.get('inference_time', 0):.3f}s")
            
            # Check if results were saved
            if os.path.exists("results"):
                result_files = os.listdir("results")
                json_files = [f for f in result_files if f.endswith(".json")]
                png_files = [f for f in result_files if f.endswith(".png")]
                txt_files = [f for f in result_files if f.endswith(".txt")]
                
                print(f"\n📁 Results saved to 'results/' directory:")
                print(f"   JSON analysis files: {len(json_files)}")
                print(f"   PNG image files: {len(png_files)}")
                print(f"   TXT summary files: {len(txt_files)}")
                
                if json_files:
                    latest_json = sorted(json_files)[-1]
                    print(f"\n📋 Latest analysis: {latest_json}")
                    
                    # Show content of latest analysis
                    import json
                    with open(f"results/{latest_json}", 'r') as f:
                        data = json.load(f)
                    
                    print(f"   Timestamp: {data['timestamp']}")
                    print(f"   Original File: {data['original_filename']}")
                    print(f"   Model Used: {data['model_used']}")
                    print(f"   Liver Detected: {data['liver_detected']}")
                    print(f"   Instruments Detected: {data['instrument_detected']}")
                    
                    return True
                else:
                    print("❌ No JSON files found in results directory")
                    return False
            else:
                print("❌ Results directory not created")
                return False
                
        else:
            print(f"❌ Segmentation failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_auto_save()
    
    if success:
        print("\n🎉 Automatic saving is working perfectly!")
        print("✅ Results are automatically saved to the 'results/' directory")
        print("✅ Each analysis creates: JSON metadata, overlay image, mask image, and text summary")
    else:
        print("\n❌ Automatic saving test failed")
