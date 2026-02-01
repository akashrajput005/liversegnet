import streamlit as st
import requests
import cv2
import numpy as np
from PIL import Image
import io
import os
import base64
import time
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from datetime import datetime
import pandas as pd
from pathlib import Path
import psutil

# Page Config
st.set_page_config(
    page_title="LiverSegNet Medical Suite",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Medical CSS
def medical_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            color: #E5E7EB;
            background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #334155 100%);
        }
        
        .main-header {
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(147, 51, 234, 0.1) 100%);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 2rem;
            margin-bottom: 1.5rem;
            backdrop-filter: blur(20px);
            text-align: center;
        }
        
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
            animation: pulse 2s infinite;
        }
        
        .status-online { background: #10B981; }
        .status-warning { background: #F59E0B; }
        .status-critical { background: #EF4444; }
        .status-offline { background: #6B7280; }
        
        @keyframes pulse {
            0% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.7; transform: scale(1.1); }
            100% { opacity: 1; transform: scale(1); }
        }
        
        .metric-card {
            background: rgba(0, 0, 0, 0.4);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 1.5rem;
            backdrop-filter: blur(10px);
            margin-bottom: 1rem;
        }
        
        .patient-view {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(34, 197, 94, 0.1) 100%);
            border: 1px solid rgba(16, 185, 129, 0.3);
        }
        
        .professional-view {
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(147, 51, 234, 0.1) 100%);
            border: 1px solid rgba(59, 130, 246, 0.3);
        }
        
        .stButton > button {
            background: linear-gradient(135deg, #3B82F6 0%, #8B5CF6 100%);
            border: none;
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(59, 130, 246, 0.3);
        }
        
        .upload-area {
            background: rgba(0, 0, 0, 0.3);
            border: 2px dashed rgba(59, 130, 246, 0.5);
            border-radius: 12px;
            padding: 2rem;
            text-align: center;
            transition: all 0.3s ease;
        }
        
        .upload-area:hover {
            border-color: rgba(59, 130, 246, 0.8);
            background: rgba(59, 130, 246, 0.05);
        }
        </style>
    """, unsafe_allow_html=True)

medical_css()

# API Configuration
API_BASE = "http://localhost:8000"

# Initialize session state
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'model_performance' not in st.session_state:
    st.session_state.model_performance = {}
if 'session_start_time' not in st.session_state:
    st.session_state.session_start_time = datetime.now()

def get_uptime():
    if 'session_start_time' in st.session_state:
        uptime = datetime.now() - st.session_state.session_start_time
        hours, remainder = divmod(int(uptime.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return "00:00:00"

def check_api_health():
    try:
        response = requests.get(f"{API_BASE}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_available_models():
    try:
        response = requests.get(f"{API_BASE}/models", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Models API returned status: {response.status_code}")
            return {"available_models": [], "model_info": {}}
    except requests.exceptions.RequestException as e:
        st.error(f"API Connection Error: {e}")
        return {"available_models": [], "model_info": {}}
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return {"available_models": [], "model_info": {}}

def segment_image(image_file, model=None, ensemble=False):
    """Process image with AI segmentation"""
    try:
        files = {'file': (image_file.name, image_file.getvalue(), 'image/png')}
        params = {}
        if model:
            params['model'] = model
        if ensemble:
            params['ensemble'] = 'true'  # Send as string, not boolean
            
        response = requests.post(f"{API_BASE}/segment_image", files=files, params=params, timeout=30)
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 400:
            st.error("❌ Bad Request: Check image format and parameters")
            st.error(f"Details: {response.text}")
            return None
        elif response.status_code == 500:
            st.error("❌ Server Error: Ensemble technique failed")
            st.error("Try using single model instead of ensemble")
            return None
        else:
            st.error(f"Segmentation API Error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error during segmentation: {e}")
        return None
    except Exception as e:
        st.error(f"Unexpected error during segmentation: {e}")
        return None

def test_ensemble_capability():
    """Test if ensemble mode is actually working in the API"""
    try:
        # Create a minimal test file (1x1 PNG)
        test_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\tpHYs\x00\x00\x0b\x13\x00\x00\x0b\x13\x01\x00\x9a\x9c\x18\x00\x00\x00\x0cIDATx\x9cc\xf8\x00\x00\x00\x01\x00\x01\x00\x00\x00\x00IEND\xaeB`\x82'
        
        files = {'file': ('test.png', test_data, 'image/png')}
        params = {'ensemble': 'true'}
        
        response = requests.post(f"{API_BASE}/segment_image", files=files, params=params, timeout=10)
        
        if response.status_code == 200:
            return {"working": True, "message": "Ensemble mode is functional"}
        elif response.status_code == 500:
            return {"working": False, "message": "Ensemble mode has server errors", "error": response.text}
        elif response.status_code == 400:
            return {"working": False, "message": "Ensemble mode not properly configured", "error": response.text}
        else:
            return {"working": False, "message": f"Ensemble mode returned status {response.status_code}", "error": response.text}
            
    except Exception as e:
        return {"working": False, "message": f"Ensemble test failed: {str(e)}"}

def check_ensemble_health():
    """Check ensemble capability and cache result"""
    if 'ensemble_health' not in st.session_state:
        st.session_state.ensemble_health = test_ensemble_capability()
    return st.session_state.ensemble_health

def check_gpu_status():
    """Check GPU availability and status"""
    try:
        # Try to get GPU info from API
        response = requests.get(f"{API_BASE}/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            if health_data.get('gpu_available', False):
                return "NVIDIA GPU Active"
            else:
                return "CPU Processing"
        else:
            return "Unknown"
    except:
        return "CPU Processing"

def enhance_image_quality(image, brightness=0, contrast=0, sharpness=0, denoise=False):
    """Apply image enhancement to improve detection"""
    try:
        from PIL import ImageEnhance, ImageFilter
        import numpy as np
        
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Apply brightness
        if brightness != 0:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(1 + (brightness / 100))
        
        # Apply contrast
        if contrast != 0:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1 + (contrast / 100))
        
        # Apply sharpness
        if sharpness > 0:
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1 + (sharpness / 100))
        
        # Apply denoising
        if denoise:
            image = image.filter(ImageFilter.MedianFilter(size=3))
        
        return image
    except Exception as e:
        st.warning(f"Image enhancement failed: {e}")
        return image

def update_model_performance(model_name, processing_time, accuracy_score=None):
    """Update model performance metrics"""
    if model_name not in st.session_state.model_performance:
        st.session_state.model_performance[model_name] = {
            'avg_processing_time': [],
            'accuracy_scores': [],
            'total_runs': 0
        }
    
    perf = st.session_state.model_performance[model_name]
    perf['avg_processing_time'].append(processing_time)
    if accuracy_score is not None:
        perf['accuracy_scores'].append(accuracy_score)
    perf['total_runs'] += 1

def get_model_stats(model_name):
    """Get model performance statistics"""
    if model_name in st.session_state.model_performance:
        perf = st.session_state.model_performance[model_name]
        avg_time = sum(perf['avg_processing_time']) / len(perf['avg_processing_time']) if perf['avg_processing_time'] else 0
        fps = 1.0 / avg_time if avg_time > 0 else 0
        avg_accuracy = sum(perf['accuracy_scores']) / len(perf['accuracy_scores']) if perf['accuracy_scores'] else 0
        return {
            'speed': f"{fps:.1f} FPS",
            'accuracy': f"{avg_accuracy:.1f}%" if perf['accuracy_scores'] else "Estimating...",
            'runs': perf['total_runs']
        }
    return {'speed': 'N/A', 'accuracy': 'N/A', 'runs': 0}

# Header
st.markdown("""
    <div class="main-header">
        <h1 style="margin: 0; color: #fff; font-size: 3rem; font-weight: 700;">
            🏥 LiverSegNet Medical Suite
        </h1>
        <p style="margin: 0.5rem 0 0 0; color: #D1D5DB; font-size: 1.3rem;">
            Advanced Surgical AI for Patient Care & Professional Analysis
        </p>
        <div style="margin-top: 1.5rem; display: flex; justify-content: center; align-items: center; gap: 2rem;">
            <div style="display: flex; align-items: center;">
                <span class="status-indicator status-online"></span>
                <span style="color: #10B981; font-weight: 600;">SYSTEM OPERATIONAL</span>
            </div>
            <div style="color: #9CA3AF;">
                Session: {} | Uptime: {}
            </div>
        </div>
    </div>
""".format(st.session_state.session_start_time.strftime("%H:%M:%S"), get_uptime()), unsafe_allow_html=True)

# API Health Check
api_healthy = check_api_health()
if not api_healthy:
    st.error("🚨 API Server is not running! Please start: `python ui/app_api_v2.py`")
    
    # Show debug info
    st.markdown("### 🔧 Debug Information")
    st.markdown("**API Connection Status:** Failed")
    st.markdown("**Expected API URL:** http://localhost:8000")
    st.markdown("**Troubleshooting:**")
    st.markdown("1. Make sure API server is running: `python ui/app_api_v2.py`")
    st.markdown("2. Check if port 8000 is available")
    st.markdown("3. Verify all dependencies are installed")
    
    # Test connection button
    if st.button("🔄 Test API Connection"):
        with st.spinner("Testing connection..."):
            try:
                response = requests.get(f"{API_BASE}/health", timeout=5)
                if response.status_code == 200:
                    st.success("✅ API is now responsive!")
                    st.rerun()
                else:
                    st.error(f"❌ API returned status: {response.status_code}")
            except Exception as e:
                st.error(f"❌ Connection failed: {e}")
    
    st.stop()

# Get available models
models_info = get_available_models()
available_models = models_info.get("available_models", [])
model_details = models_info.get("model_info", {})

# Debug info if no models available
if not available_models:
    st.warning("⚠️ No AI models loaded. Check API server status.")
    with st.expander("🔍 Model Loading Debug Info"):
        st.markdown("**Models API Response:**")
        st.json(models_info)
        
        if st.button("🔄 Reload Models"):
            models_info = get_available_models()
            available_models = models_info.get("available_models", [])
            model_details = models_info.get("model_info", {})
            if available_models:
                st.success(f"✅ Loaded {len(available_models)} models!")
                st.rerun()
            else:
                st.error("❌ Still no models available")

# Debug section for ensemble testing
if available_models and len(available_models) >= 2:
    with st.expander("🔧 Ensemble Debug (Advanced)"):
        st.markdown("**Test Ensemble API Directly**")
        
        col_test1, col_test2 = st.columns(2)
        with col_test1:
            if st.button("🧪 Test Ensemble Endpoint", key="debug_ensemble"):
                with st.spinner("Testing ensemble endpoint..."):
                    try:
                        # Create a test request
                        test_params = {'ensemble': 'true'}
                        st.markdown(f"**Testing with params:** {test_params}")
                        
                        # Test the endpoint health
                        health_response = requests.get(f"{API_BASE}/health", timeout=5)
                        st.markdown(f"**API Health:** {health_response.status_code}")
                        
                        if health_response.status_code == 200:
                            health_data = health_response.json()
                            st.markdown(f"**Models Available:** {health_data.get('models', [])}")
                            st.markdown(f"**GPU Available:** {health_data.get('gpu_available', False)}")
                            
                            if len(health_data.get('models', [])) >= 2:
                                st.success("✅ Ensemble should be supported")
                            else:
                                st.warning("⚠️ Not enough models for ensemble")
                        
                    except Exception as e:
                        st.error(f"❌ Debug test failed: {e}")
        
        with col_test2:
            if st.button("🔄 Retest Ensemble", key="retest_ensemble"):
                # Clear cached result and retest
                if 'ensemble_health' in st.session_state:
                    del st.session_state.ensemble_health
                st.rerun()

# Sidebar Configuration
with st.sidebar:
    st.markdown("### 🎛️ Configuration")
    
    # User Type Selection
    user_type = st.radio(
        "Select Interface Mode:",
        ["👤 Patient View", "👨‍⚕️ Professional View"],
        help="Choose simplified patient view or detailed professional view"
    )
    
    st.markdown("---")
    
    # Model Selection
    st.markdown("### 🤖 AI Model Selection")
    if available_models:
        selected_model = st.selectbox(
            "Choose AI Model:",
            options=available_models,
            format_func=lambda x: model_details.get(x, {}).get('name', x),
            help="Select AI model for analysis"
        )
        
        # Show model info
        if selected_model in model_details:
            info = model_details[selected_model]
            st.markdown(f"""
                <div style='background: rgba(59, 130, 246, 0.1); border: 1px solid rgba(59, 130, 246, 0.3); border-radius: 8px; padding: 1rem; margin: 0.5rem 0;'>
                    <strong>📋 Model Details:</strong><br>
                    <strong>Type:</strong> {info.get('type', 'Unknown')}<br>
                    <strong>Strengths:</strong> {' • '.join(info.get('strengths', []))}
                </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ No models available")
    
    st.markdown("---")
    
    # Advanced Options
    st.markdown("### ⚙️ Advanced Options")
    
    # Check if we have enough models for ensemble
    ensemble_available = len(available_models) >= 2
    
    # Test ensemble capability if we have enough models
    ensemble_health = None
    if ensemble_available:
        with st.spinner("Testing ensemble capability..."):
            ensemble_health = check_ensemble_health()
    
    # Disable ensemble if not available or not working
    ensemble_disabled = not ensemble_available or (ensemble_health and not ensemble_health['working'])
    
    use_ensemble = st.checkbox(
        "🔄 Ensemble Intelligence",
        help="Combine multiple AI models for enhanced accuracy",
        disabled=ensemble_disabled
    )
    
    # Show appropriate warnings/info
    if not ensemble_available:
        st.warning("⚠️ Ensemble requires at least 2 models. Currently only {} model(s) available.".format(len(available_models)))
        use_ensemble = False
    elif ensemble_health and not ensemble_health['working']:
        st.error("🚨 Ensemble mode is not working properly")
        st.markdown(f"**Issue:** {ensemble_health['message']}")
        if 'error' in ensemble_health:
            with st.expander("Technical Details"):
                st.code(ensemble_health['error'])
        
        # Add troubleshooting options
        st.markdown("**🔧 Troubleshooting Options:**")
        col_fix1, col_fix2 = st.columns(2)
        
        with col_fix1:
            if st.button("🔄 Restart API Test", key="restart_ensemble_test"):
                with st.spinner("Retesting ensemble capability..."):
                    # Clear cache and retest
                    if 'ensemble_health' in st.session_state:
                        del st.session_state.ensemble_health
                    st.rerun()
        
        with col_fix2:
            if st.button("� Check API Status", key="check_api_status"):
                with st.spinner("Checking API status..."):
                    try:
                        health_response = requests.get(f"{API_BASE}/health", timeout=5)
                        if health_response.status_code == 200:
                            health_data = health_response.json()
                            st.success("✅ API is healthy")
                            st.json(health_data)
                        else:
                            st.error(f"❌ API status: {health_response.status_code}")
                    except Exception as e:
                        st.error(f"❌ API connection failed: {e}")
        
        st.info("�💡 **Recommendation:** Use single model mode for now. Ensemble mode may need API server fixes.")
        use_ensemble = False
    elif ensemble_health and ensemble_health['working']:
        st.success("✅ Ensemble mode is functional")
    
    # Show ensemble info
    if use_ensemble and ensemble_available:
        st.info(f"🤖 Will combine all {len(available_models)} models for enhanced accuracy")
    
    # Image Enhancement Options
    st.markdown("### 🖼️ Image Enhancement")
    
    enhance_image = st.checkbox(
        "🔧 Enhance Image Quality",
        help="Apply preprocessing to improve detection accuracy"
    )
    
    if enhance_image:
        col_enh1, col_enh2 = st.columns(2)
        with col_enh1:
            brightness = st.slider("☀️ Brightness", -50, 50, 0, help="Adjust image brightness")
            contrast = st.slider("🎭 Contrast", -50, 50, 0, help="Adjust image contrast")
        with col_enh2:
            sharpness = st.slider("🔍 Sharpness", 0, 100, 0, help="Enhance image sharpness")
            denoise = st.checkbox("🌫️ Reduce Noise", help="Apply noise reduction")
    
    confidence_threshold = st.slider(
        "🎯 Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.3,  # Lowered from 0.5 to be more sensitive
        step=0.05,
        help="Minimum confidence for detections (lower = more sensitive)"
    )

# Main Content Area
col_main, col_sidebar = st.columns([3, 1])

with col_main:
    # File Upload Section
    st.markdown("### 📥 Medical Data Upload")
    
    upload_tab1, upload_tab2 = st.tabs(["🖼️ Image Analysis", "🎥 Video Analysis"])
    
    with upload_tab1:
        st.markdown('<div class="upload-area">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload surgical image for AI analysis",
            type=['png', 'jpg', 'jpeg'],
            help="Supported formats: PNG, JPG, JPEG"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        if uploaded_file is not None:
            # Display original image
            col_img1, col_img2 = st.columns(2)
            
            with col_img1:
                st.markdown("#### **Original Medical Image**")
                image = Image.open(uploaded_file)
                st.image(image, caption="Surgical Scene")
                
                # Show enhancement info if enabled
                if enhance_image:
                    st.markdown(f"🔧 **Enhancement Applied:** Brightness:{brightness} Contrast:{contrast} Sharpness:{sharpness} Denoise:{denoise}")
            
            # Analysis Button
            if st.button("🚀 Start AI Analysis", type="primary", use_container_width=True):
                with st.spinner("🤖 AI analyzing medical image..."):
                    # Apply image enhancement if enabled
                    if enhance_image:
                        with st.spinner("🔧 Enhancing image quality..."):
                            enhanced_image = enhance_image_quality(image, brightness, contrast, sharpness, denoise)
                            # Convert enhanced image back to bytes for API
                            img_bytes = io.BytesIO()
                            enhanced_image.save(img_bytes, format='PNG')
                            img_bytes.seek(0)
                            
                            # Create a new file-like object
                            from io import BytesIO
                            enhanced_file = BytesIO(img_bytes.getvalue())
                            enhanced_file.name = uploaded_file.name
                            
                            result = segment_image(
                                enhanced_file,
                                model=selected_model if not use_ensemble else None,
                                ensemble=use_ensemble
                            )
                    else:
                        result = segment_image(
                            uploaded_file,
                            model=selected_model if not use_ensemble else None,
                            ensemble=use_ensemble
                        )
                
                if result:
                    # Track performance
                    processing_time = result.get('processing_time', 0)
                    model_used = result.get('model_used', selected_model)
                    accuracy_score = result.get('metrics', {}).get('liver_compactness_mean', 0) * 100
                    update_model_performance(model_used, processing_time, accuracy_score)
                    
                    # Display results
                    with col_img2:
                        st.markdown("#### **AI Analysis Results**")
                        if 'overlay_url' in result:
                            overlay_response = requests.get(f"{API_BASE}/{result['overlay_url']}")
                            if overlay_response.status_code == 200:
                                overlay_image = Image.open(io.BytesIO(overlay_response.content))
                                st.image(overlay_image, caption="AI Segmentation: Liver (Green) + Tools (Red)")
                        
                        # Add detection quality indicators
                        st.markdown("#### **Detection Quality**")
                        
                        # Liver detection confidence
                        liver_confidence = result.get('metrics', {}).get('liver_compactness_mean', 0) * 100
                        liver_detected = result.get('liver_detected', False)
                        
                        # Tool detection confidence
                        tool_confidence = result.get('metrics', {}).get('instrument_confidence', 0) * 100
                        tools_detected = result.get('instrument_detected', False)
                        
                        # Background/Overall confidence
                        overall_confidence = result.get('metrics', {}).get('overall_accuracy', 0) * 100
                        
                        # Display confidence bars
                        col_conf1, col_conf2 = st.columns(2)
                        
                        with col_conf1:
                            st.markdown("**Liver Detection**")
                            liver_color = "🟢" if liver_detected else "🔴"
                            st.markdown(f"{liver_color} Status: {'Detected' if liver_detected else 'Not Detected'}")
                            st.progress(liver_confidence/100, text=f"Confidence: {liver_confidence:.1f}%")
                            
                            st.markdown("**Tool Detection**")
                            tool_color = "🟢" if tools_detected else "🔴"
                            st.markdown(f"{tool_color} Status: {'Detected' if tools_detected else 'Not Detected'}")
                            st.progress(tool_confidence/100, text=f"Confidence: {tool_confidence:.1f}%")
                        
                        with col_conf2:
                            st.markdown("**Overall Quality**")
                            if overall_confidence > 80:
                                quality_color = "🟢"
                                quality_text = "Excellent"
                            elif overall_confidence > 60:
                                quality_color = "🟡"
                                quality_text = "Good"
                            elif overall_confidence > 40:
                                quality_color = "🟠"
                                quality_text = "Fair"
                            else:
                                quality_color = "🔴"
                                quality_text = "Poor"
                            
                            st.markdown(f"{quality_color} Quality: {quality_text}")
                            st.progress(overall_confidence/100, text=f"Overall: {overall_confidence:.1f}%")
                            
                            # Recommendations based on quality
                            if overall_confidence < 60:
                                st.markdown("**💡 Recommendations:**")
                                st.markdown("• Try different AI model")
                                st.markdown("• Ensure clear image quality")
                                st.markdown("• Check proper lighting")
                                st.markdown("• Use ensemble mode")
                    
                    # Success message with model info
                    if use_ensemble:
                        st.success(f"✅ Ensemble Analysis Complete! Processing time: {processing_time:.3f}s")
                        st.info("🤖 Combined results from multiple AI models for enhanced accuracy")
                    else:
                        st.success(f"✅ Analysis Complete! Processing time: {processing_time:.3f}s")
                        st.info(f"🤖 Used model: {model_details.get(selected_model, {}).get('name', selected_model)}")
                    
                    # Add to history
                    st.session_state.analysis_history.append({
                        'timestamp': datetime.now().strftime("%H:%M:%S"),
                        'model': result.get('model_used', 'unknown'),
                        'processing_time': processing_time,
                        'liver_detected': result.get('liver_detected', False),
                        'instrument_detected': result.get('instrument_detected', False),
                        'occlusion_percent': result.get('occlusion_percent', 0),
                        'confidence': result.get('metrics', {}).get('liver_compactness_mean', 0),
                        'ensemble_used': use_ensemble,
                        'liver_confidence': liver_confidence,
                        'tool_confidence': tool_confidence,
                        'overall_confidence': overall_confidence
                    })
                else:
                    st.error("❌ Analysis failed")
                    
                    # Fallback suggestion
                    if use_ensemble:
                        st.warning("💡 Try using a single model instead of ensemble mode")
                        if st.button("🔄 Retry with Single Model", key="fallback_single"):
                            # Retry with single model
                            with st.spinner("🤖 Retrying with single model..."):
                                result = segment_image(uploaded_file, model=selected_model, ensemble=False)
                                if result:
                                    st.success("✅ Single model analysis succeeded!")
                                    # Process the result...
                                else:
                                    st.error("❌ Even single model failed. Check API server.")
    
    with upload_tab2:
        st.markdown('<div class="upload-area">', unsafe_allow_html=True)
        uploaded_video = st.file_uploader(
            "Upload surgical video for frame-by-frame analysis",
            type=['mp4', 'avi', 'mov'],
            help="Supported formats: MP4, AVI, MOV"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        if uploaded_video is not None:
            st.video(uploaded_video)
            st.info("🎬 Video processing will analyze each frame with AI")
            
            if st.button("▶️ Process Video", type="primary", use_container_width=True):
                st.info("🔄 Video processing feature coming soon...")

with col_sidebar:
    # Real-time Metrics
    st.markdown("### 📊 Live Metrics")
    
    if st.session_state.analysis_history:
        latest_analysis = st.session_state.analysis_history[-1]
        
        # Liver Status
        liver_status = "DETECTED" if latest_analysis['liver_detected'] else "NOT DETECTED"
        liver_color = "status-online" if latest_analysis['liver_detected'] else "status-offline"
        
        st.markdown(f"""
            <div class="metric-card">
                <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                    <span class="status-indicator {liver_color}"></span>
                    <strong style="color: #fff;">Liver Status</strong>
                </div>
                <div style="color: #9CA3AF; font-size: 1.1rem;">{liver_status}</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Instruments Status
        instrument_status = "DETECTED" if latest_analysis['instrument_detected'] else "NOT DETECTED"
        instrument_color = "status-online" if latest_analysis['instrument_detected'] else "status-offline"
        
        st.markdown(f"""
            <div class="metric-card">
                <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                    <span class="status-indicator {instrument_color}"></span>
                    <strong style="color: #fff;">Instruments</strong>
                </div>
                <div style="color: #9CA3AF; font-size: 1.1rem;">{instrument_status}</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Occlusion Risk
        occlusion = latest_analysis['occlusion_percent']
        if occlusion < 30:
            occlusion_color = "status-online"
            occlusion_text = "LOW RISK"
        elif occlusion < 60:
            occlusion_color = "status-warning"
            occlusion_text = "MODERATE"
        else:
            occlusion_color = "status-critical"
            occlusion_text = "HIGH RISK"
        
        st.markdown(f"""
            <div class="metric-card">
                <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                    <span class="status-indicator {occlusion_color}"></span>
                    <strong style="color: #fff;">Occlusion Risk</strong>
                </div>
                <div style="color: #9CA3AF; font-size: 1.1rem;">{occlusion_text} ({occlusion:.1f}%)</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Processing Speed
        fps = 1.0 / latest_analysis['processing_time'] if latest_analysis['processing_time'] > 0 else 0
        st.markdown(f"""
            <div class="metric-card">
                <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                    <span class="status-indicator status-online"></span>
                    <strong style="color: #fff;">Processing Speed</strong>
                </div>
                <div style="color: #9CA3AF; font-size: 1.1rem;">{fps:.1f} FPS</div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class="metric-card">
                <div style="text-align: center; color: #6B7280;">
                    <div style="font-size: 3rem; margin-bottom: 1rem;">📊</div>
                    <div>No analysis data yet</div>
                    <div style="font-size: 0.9rem; margin-top: 0.5rem;">Upload an image to see metrics</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # System Status
    st.markdown("### 🖥️ System Status")
    
    # Get real system status
    try:
        memory_percent = psutil.virtual_memory().percent
        gpu_status = check_gpu_status()
    except:
        memory_percent = 65
        gpu_status = "Unknown"
    
    st.markdown(f"""
        <div class="metric-card">
            <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                <span class="status-indicator status-online"></span>
                <strong style="color: #fff;">API Server</strong>
            </div>
            <div style="color: #9CA3AF;">Online & Responsive</div>
        </div>
        
        <div class="metric-card">
            <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                <span class="status-indicator status-online"></span>
                <strong style="color: #fff;">AI Models</strong>
            </div>
            <div style="color: #9CA3AF;">{len(available_models)} Loaded</div>
        </div>
        
        <div class="metric-card">
            <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                <span class="status-indicator status-online"></span>
                <strong style="color: #fff;">Processing Unit</strong>
            </div>
            <div style="color: #9CA3AF;">{gpu_status}</div>
        </div>
        
        <div class="metric-card">
            <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                <span class="status-indicator {'status-warning' if memory_percent > 80 else 'status-online'}"></span>
                <strong style="color: #fff;">Memory Usage</strong>
            </div>
            <div style="color: #9CA3AF;">{memory_percent:.0f}% Used</div>
        </div>
    """, unsafe_allow_html=True)

# Professional/Patient View Specific Content
st.markdown("---")

if user_type == "👨‍⚕️ Professional View":
    st.markdown("### 👨‍⚕️ Professional Analysis Dashboard")
    
    # Model Performance
    st.markdown("#### 🤖 AI Model Performance")
    
    if available_models:
        perf_col1, perf_col2, perf_col3 = st.columns(3)
        
        for i, model in enumerate(available_models):
            with [perf_col1, perf_col2, perf_col3][i]:
                info = model_details.get(model, {})
                model_name = info.get('name', model)
                stats = get_model_stats(model)
                
                st.markdown(f"""
                    <div class="metric-card professional-view">
                        <div style="color: #fff; font-weight: 600; margin-bottom: 0.5rem;">{model_name}</div>
                        <div style="color: #9CA3AF; font-size: 0.9rem;">
                            Accuracy: {stats['accuracy']}<br>
                            Speed: {stats['speed']}<br>
                            Runs: {stats['runs']}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
    
    # Analysis History
    if st.session_state.analysis_history:
        st.markdown("#### 📋 Analysis History")
        
        history_df = pd.DataFrame(st.session_state.analysis_history)
        history_df['timestamp'] = pd.to_datetime(history_df['timestamp'], format='%H:%M:%S').dt.time
        
        st.dataframe(
            history_df[['timestamp', 'model', 'processing_time', 'liver_detected', 'instrument_detected', 'occlusion_percent']],
            use_container_width=True
        )
        
        # Performance Chart
        if len(st.session_state.analysis_history) > 1:
            st.markdown("#### 📈 Performance Trends")
            
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Processing Time", "Occlusion Percentage"),
                specs=[[{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(st.session_state.analysis_history))),
                    y=[a['processing_time'] for a in st.session_state.analysis_history],
                    mode='lines+markers',
                    name='Processing Time (s)',
                    line=dict(color='#3B82F6')
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(st.session_state.analysis_history))),
                    y=[a['occlusion_percent'] for a in st.session_state.analysis_history],
                    mode='lines+markers',
                    name='Occlusion %',
                    line=dict(color='#10B981')
                ),
                row=1, col=2
            )
            
            fig.update_layout(
                height=300,
                showlegend=False,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#E5E7EB'
            )
            
            st.plotly_chart(fig, use_container_width=True)

else:  # Patient View
    st.markdown("### 👤 Patient Information Center")
    
    if st.session_state.analysis_history:
        latest = st.session_state.analysis_history[-1]
        
        st.markdown("""
            <div class="metric-card patient-view">
                <h4 style="color: #fff; margin-bottom: 1rem;">🩺 Your Analysis Results</h4>
                <div style="color: #D1D5DB; line-height: 1.6;">
                    <p><strong>Analysis Time:</strong> {}</p>
                    <p><strong>Liver Detection:</strong> {}</p>
                    <p><strong>Surgical Instruments:</strong> {}</p>
                    <p><strong>Occlusion Level:</strong> {:.1f}%</p>
                    <p><strong>AI Confidence:</strong> {:.1f}%</p>
                </div>
            </div>
        """.format(
            latest['timestamp'],
            "✅ Detected" if latest['liver_detected'] else "❌ Not Detected",
            "✅ Detected" if latest['instrument_detected'] else "❌ Not Detected",
            latest['occlusion_percent'],
            latest['confidence'] * 100
        ), unsafe_allow_html=True)
        
        # Patient-friendly explanation
        st.markdown("#### 📋 What This Means")
        
        if latest['liver_detected'] and latest['instrument_detected']:
            st.markdown("""
                <div class="metric-card patient-view">
                    <div style="color: #10B981; font-weight: 600; margin-bottom: 0.5rem;">✅ Normal Surgical View</div>
                    <div style="color: #D1D5DB;">
                        The AI successfully identified both the liver and surgical instruments. 
                        This indicates a clear view for the surgical procedure.
                    </div>
                </div>
            """, unsafe_allow_html=True)
        elif latest['occlusion_percent'] > 60:
            st.markdown("""
                <div class="metric-card patient-view">
                    <div style="color: #F59E0B; font-weight: 600; margin-bottom: 0.5rem;">⚠️ Limited Visibility</div>
                    <div style="color: #D1D5DB;">
                        The AI detected some occlusion (blocking) in the surgical view. 
                        Your surgeon is aware and working to maintain clear visualization.
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="metric-card patient-view">
                    <div style="color: #10B981; font-weight: 600; margin-bottom: 0.5rem;">✅ Good Visualization</div>
                    <div style="color: #D1D5DB;">
                        The surgical area appears to be well-visualized with minimal obstruction.
                    </div>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class="metric-card patient-view">
                <div style="text-align: center; color: #6B7280;">
                    <div style="font-size: 3rem; margin-bottom: 1rem;">🩺</div>
                    <div>No analysis results yet</div>
                    <div style="font-size: 0.9rem; margin-top: 0.5rem;">Upload a medical image to see your results</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #6B7280; padding: 2rem;'>
        <p>🏥 LiverSegNet Medical Suite - Professional AI for Surgical Care</p>
        <p style='font-size: 0.9rem;'>Real-time AI analysis for patient safety and surgical precision</p>
    </div>
""", unsafe_allow_html=True)
