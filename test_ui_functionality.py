#!/usr/bin/env python3
"""
Test script to verify all UI functionality
"""

import requests
import json
import time
import os

def test_api_health():
    """Test API health endpoint"""
    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            data = response.json()
            print("✅ API Health Check:")
            print(f"   Status: {data['status']}")
            print(f"   Engines Loaded: {data['engines_loaded']}")
            print(f"   Models Available: {data['models']}")
            print(f"   GPU Available: {data['gpu_available']}")
            return True
        else:
            print(f"❌ API Health failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API Health error: {e}")
        return False

def test_model_engines():
    """Test if all model engines are loaded"""
    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            data = response.json()
            models = data['models']
            
            expected_models = ['unet', 'deeplabv3plus', 'stage1']
            missing_models = [m for m in expected_models if m not in models]
            
            if not missing_models:
                print("✅ All Model Engines Loaded:")
                for model in models:
                    print(f"   - {model}")
                return True
            else:
                print(f"❌ Missing Models: {missing_models}")
                return False
        return False
    except Exception as e:
        print(f"❌ Model engines test error: {e}")
        return False

def test_streamlit_uis():
    """Test if Streamlit UIs are running"""
    ui_ports = [8501, 8502]
    ui_names = ["app_v2.py", "app.py"]
    
    all_running = True
    for port, name in zip(ui_ports, ui_names):
        try:
            response = requests.get(f"http://localhost:{port}", timeout=2)
            if response.status_code == 200:
                print(f"✅ Streamlit UI ({name}) running on port {port}")
            else:
                print(f"❌ Streamlit UI ({name}) not responding on port {port}")
                all_running = False
        except Exception as e:
            print(f"❌ Streamlit UI ({name}) error on port {port}: {e}")
            all_running = False
    
    return all_running

def test_model_files():
    """Test if all model checkpoint files exist"""
    model_files = [
        'models/deeplabv3plus_resnet50.pth',
        'models/deeplabv3plus_resnet50_stage1.pth',
        'models/unet_resnet34.pth',
        'models/unet_resnet34_fast.pth'
    ]
    
    all_exist = True
    print("✅ Model Files Check:")
    for model_file in model_files:
        if os.path.exists(model_file):
            size_mb = os.path.getsize(model_file) / (1024*1024)
            print(f"   ✅ {model_file} ({size_mb:.1f}MB)")
        else:
            print(f"   ❌ {model_file} (MISSING)")
            all_exist = False
    
    return all_exist

def test_inference_capability():
    """Test basic inference capability (without actual image)"""
    try:
        # Test reload engine endpoint
        response = requests.post("http://localhost:8000/reload_engine")
        if response.status_code == 200:
            data = response.json()
            print("✅ Engine Reload Test:")
            print(f"   Message: {data['message']}")
            print(f"   Models: {data['models_loaded']}")
            return True
        else:
            print(f"❌ Engine reload failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Inference capability test error: {e}")
        return False

def main():
    print("🔍 LIVERSEGNET UI FUNCTIONALITY TEST")
    print("=" * 50)
    
    tests = [
        ("API Health", test_api_health),
        ("Model Engines", test_model_engines),
        ("Streamlit UIs", test_streamlit_uis),
        ("Model Files", test_model_files),
        ("Inference Capability", test_inference_capability)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Testing {test_name}...")
        result = test_func()
        results.append((test_name, result))
        time.sleep(0.5)  # Small delay between tests
    
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL SYSTEMS OPERATIONAL!")
        print("\n🚀 Ready to use:")
        print("   • API Server: http://localhost:8000")
        print("   • Streamlit UI v2: http://localhost:8501")
        print("   • Streamlit UI v1: http://localhost:8502")
        print("   • API Docs: http://localhost:8000/docs")
    else:
        print("⚠️  Some issues detected - check failed tests above")
    
    return passed == total

if __name__ == "__main__":
    main()
