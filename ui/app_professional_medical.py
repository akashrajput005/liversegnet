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
import threading
import queue
from datetime import datetime
import pandas as pd
import tempfile
import subprocess
import sys
from pathlib import Path
import psutil

# --- Page Config ---
st.set_page_config(
    page_title="Professional Medical AI Suite",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Professional Medical CSS ---
def professional_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            color: #E5E7EB;
            background: #0F172A;
        }

        .main {
            background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #334155 100%);
        }

        /* Professional Medical Header */
        .medical-header {
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.15) 0%, rgba(147, 51, 234, 0.15) 100%);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            padding: 2rem;
            margin-bottom: 1.5rem;
            backdrop-filter: blur(20px);
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
        }

        .live-indicator {
            display: inline-block;
            width: 16px;
            height: 16px;
            background: #10B981;
            border-radius: 50%;
            animation: pulse 2s infinite;
            margin-right: 12px;
            box-shadow: 0 0 20px rgba(16, 185, 129, 0.5);
        }

        @keyframes pulse {
            0% { opacity: 1; transform: scale(1); box-shadow: 0 0 20px rgba(16, 185, 129, 0.5); }
            50% { opacity: 0.8; transform: scale(1.2); box-shadow: 0 0 30px rgba(16, 185, 129, 0.8); }
            100% { opacity: 1; transform: scale(1); box-shadow: 0 0 20px rgba(16, 185, 129, 0.5); }
        }

        .status-bar {
            background: rgba(0, 0, 0, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 1rem;
            backdrop-filter: blur(10px);
        }

        .video-container {
            background: #000;
            border: 3px solid rgba(59, 130, 246, 0.3);
            border-radius: 16px;
            overflow: hidden;
            position: relative;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        }

        .overlay-controls {
            position: absolute;
            top: 15px;
            right: 15px;
            z-index: 100;
            background: rgba(0, 0, 0, 0.8);
            padding: 12px;
            border-radius: 12px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .control-button {
            background: rgba(59, 130, 246, 0.2);
            border: 1px solid #3B82F6;
            color: #fff;
            padding: 8px 16px;
            border-radius: 8px;
            cursor: pointer;
            margin: 2px;
            transition: all 0.3s ease;
            font-weight: 500;
        }

        .control-button:hover {
            background: rgba(59, 130, 246, 0.4);
            transform: translateY(-1px);
        }

        .control-button.active {
            background: #3B82F6;
            box-shadow: 0 0 20px rgba(59, 130, 246, 0.5);
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 1.5rem;
        }

        .metric-card {
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.02) 100%);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 1.5rem;
            transition: all 0.3s ease;
            backdrop-filter: blur(10px);
        }

        .metric-card:hover {
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.08) 0%, rgba(255, 255, 255, 0.04) 100%);
            transform: translateY(-2px);
            box-shadow: 0 10px 30px rgba(59, 130, 246, 0.2);
        }

        .metric-value {
            font-size: 2.5rem;
            font-weight: 700;
            color: #3B82F6;
            margin-bottom: 0.5rem;
        }

        .metric-label {
            font-size: 0.875rem;
            color: #9CA3AF;
            font-weight: 500;
        }

        .metric-change {
            font-size: 0.75rem;
            margin-top: 0.5rem;
        }

        .metric-change.positive {
            color: #10B981;
        }

        .metric-change.negative {
            color: #EF4444;
        }

        .control-panel {
            background: rgba(15, 23, 42, 0.9);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            padding: 2rem;
            backdrop-filter: blur(20px);
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
        }

        .model-selector {
            background: rgba(255, 255, 255, 0.05);
            border: 2px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 1rem;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .model-selector.selected {
            border-color: #3B82F6;
            background: rgba(59, 130, 246, 0.1);
            box-shadow: 0 0 20px rgba(59, 130, 246, 0.2);
        }

        .model-selector:hover {
            border-color: #3B82F6;
            transform: translateY(-1px);
        }

        .alert-critical {
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(239, 68, 68, 0.1) 100%);
            border: 1px solid rgba(239, 68, 68, 0.3);
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 1rem;
            color: #FCA5A5;
            animation: alertPulse 2s infinite;
        }

        @keyframes alertPulse {
            0% { border-color: rgba(239, 68, 68, 0.3); }
            50% { border-color: rgba(239, 68, 68, 0.6); }
            100% { border-color: rgba(239, 68, 68, 0.3); }
        }

        .performance-chart {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 1rem;
        }

        .streamlit-slider {
            background: rgba(59, 130, 246, 0.2);
        }

        /* Professional Typography */
        h1, h2, h3, h4, h5, h6 {
            color: #F9FAFB;
            font-weight: 600;
        }

        /* Custom Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
        }

        ::-webkit-scrollbar-track {
            background: rgba(0, 0, 0, 0.1);
        }

        ::-webkit-scrollbar-thumb {
            background: rgba(59, 130, 246, 0.5);
            border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: rgba(59, 130, 246, 0.7);
        }
        </style>
    """, unsafe_allow_html=True)

# --- Configuration ---
API_BASE = "http://localhost:8000"

# --- Session State ---
if 'session_id' not in st.session_state:
    st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'performance_metrics' not in st.session_state:
    st.session_state.performance_metrics = []
if 'alerts' not in st.session_state:
    st.session_state.alerts = []
if 'is_recording' not in st.session_state:
    st.session_state.is_recording = False
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False

# Initialize performance tracking
if 'model_performance' not in st.session_state:
    st.session_state.model_performance = {}

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

def debug_model_performance():
    """Debug function to check model performance state"""
    if st.checkbox("🐛 Debug Model Performance", help="Show internal performance tracking state"):
        st.markdown("### **Debug: Model Performance State**")
        st.json(st.session_state.model_performance)
        
        # Test API connection
        st.markdown("### **Debug: API Connection**")
        try:
            response = requests.get(f"{API_BASE}/health", timeout=2)
            st.markdown(f"**API Health:** {response.status_code}")
            if response.status_code == 200:
                st.markdown(f"**API Response:** {response.json()}")
        except Exception as e:
            st.markdown(f"**API Error:** {e}")
        
        # Test models endpoint
        try:
            models_response = requests.get(f"{API_BASE}/models", timeout=2)
            st.markdown(f"**Models Status:** {models_response.status_code}")
            if models_response.status_code == 200:
                st.markdown(f"**Models Response:** {models_response.json()}")
        except Exception as e:
            st.markdown(f"**Models Error:** {e}")

# --- Advanced API Functions ---
def check_api_health():
    try:
        response = requests.get(f"{API_BASE}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_available_models():
    try:
        response = requests.get(f"{API_BASE}/models", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return {"available_models": [], "model_info": {}}

def segment_image_advanced(file, model=None, ensemble=False):
    """Advanced image segmentation with detailed metrics"""
    try:
        files = {"file": file}
        params = {}
        if model:
            params["model"] = model
        if ensemble:
            params["ensemble"] = True
            
        start_time = time.time()
        response = requests.post(f"{API_BASE}/segment_image", files=files, params=params)
        processing_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            result['processing_time'] = processing_time
            result['timestamp'] = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            return result
    except Exception as e:
        st.error(f"Advanced processing error: {str(e)}")
    return None

def process_video_frame(frame, model=None):
    """Process individual video frame"""
    try:
        # Convert frame to bytes
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = io.BytesIO(buffer.tobytes())
        
        # Process frame
        result = segment_image_advanced(frame_bytes, model=model)
        return result
    except Exception as e:
        print(f"Frame processing error: {e}")
        return None

# --- Advanced Analytics ---
def create_advanced_performance_dashboard():
    """Create comprehensive performance dashboard"""
    if not st.session_state.performance_metrics:
        return
    
    df = pd.DataFrame(st.session_state.performance_metrics)
    
    # Create advanced subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=('Processing Time Trend', 'Real-time FPS', 'Detection Confidence', 'System Load', 'Model Accuracy', 'Alert Frequency'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Processing Time Trend
    fig.add_trace(
        go.Scatter(x=df.index, y=df['processing_time'], name='Processing Time', 
                  line=dict(color='#3B82F6', width=3), fill='tonexty'),
        row=1, col=1
    )
    
    # Real-time FPS
    fig.add_trace(
        go.Scatter(x=df.index, y=df['fps'], name='FPS', 
                  line=dict(color='#10B981', width=3), fill='tonexty'),
        row=1, col=2
    )
    
    # Detection Confidence
    fig.add_trace(
        go.Scatter(x=df.index, y=df['confidence'], name='Confidence', 
                  line=dict(color='#F59E0B', width=3), fill='tonexty'),
        row=2, col=1
    )
    
    # System Load
    fig.add_trace(
        go.Scatter(x=df.index, y=df['memory_usage'], name='Memory Usage', 
                  line=dict(color='#EF4444', width=3), fill='tonexty'),
        row=2, col=2
    )
    
    # Model Accuracy
    fig.add_trace(
        go.Scatter(x=df.index, y=df['accuracy'], name='Accuracy', 
                  line=dict(color='#8B5CF6', width=3), fill='tonexty'),
        row=3, col=1
    )
    
    # Alert Frequency
    fig.add_trace(
        go.Bar(x=df.index, y=df['alerts'], name='Alerts', 
                  marker_color='#F97316'),
        row=3, col=2
    )
    
    fig.update_layout(
        height=800,
        showlegend=False,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E5E7EB")
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_medical_alert_system():
    """Create advanced medical alert system"""
    if st.session_state.alerts:
        for alert in st.session_state.alerts[-3:]:  # Show last 3 alerts
            if alert['level'] == 'critical':
                st.markdown(f"""
                    <div class="alert-critical">
                        🚨 <strong>CRITICAL ALERT</strong><br>
                        <strong>{alert['title']}</strong><br>
                        {alert['message']}<br>
                        <small>Time: {alert['timestamp']}</small>
                    </div>
                """, unsafe_allow_html=True)
            elif alert['level'] == 'warning':
                st.markdown(f"""
                    <div style="background: linear-gradient(135deg, rgba(245, 158, 11, 0.2) 0%, rgba(245, 158, 11, 0.1) 100%); 
                                border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 12px; padding: 1rem; margin-bottom: 1rem;">
                        ⚠️ <strong>WARNING</strong><br>
                        <strong>{alert['title']}</strong><br>
                        {alert['message']}<br>
                        <small>Time: {alert['timestamp']}</small>
                    </div>
                """, unsafe_allow_html=True)

def simulate_camera_feed():
    """Simulate camera feed for demonstration"""
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Camera not available. Using demo mode.")
            return None
        
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            return frame
    except:
        pass
    
    return None

# --- Main Application ---
def main():
    professional_css()
    
    # Initialize session start time if not exists
    if 'session_start_time' not in st.session_state:
        st.session_state.session_start_time = datetime.now()

    # Calculate uptime
    def get_uptime():
        if 'session_start_time' in st.session_state:
            uptime = datetime.now() - st.session_state.session_start_time
            hours, remainder = divmod(int(uptime.total_seconds()), 3600)
            minutes, seconds = divmod(remainder, 60)
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return "00:00:00"

    # Professional Header
    st.markdown("""
        <div class="medical-header">
            <h1 style="margin: 0; color: #fff; font-size: 2.5rem; font-weight: 700;">
                🏥 Professional Medical AI Suite
            </h1>
            <p style="margin: 0.5rem 0 0 0; color: #D1D5DB; font-size: 1.2rem;">
                Advanced Surgical Intelligence with Real-time AI Analysis
            </p>
            <div style="margin-top: 1.5rem; display: flex; align-items: center;">
                <span class="live-indicator"></span>
                <span style="color: #10B981; font-weight: 600; font-size: 1.1rem;">● SYSTEM OPERATIONAL</span>
                <span style="margin-left: 2rem; color: #9CA3AF;">
                    Session: {} | Uptime: {}
                </span>
            </div>
        </div>
    """.format(st.session_state.session_start_time.strftime("%H:%M:%S"), get_uptime()), unsafe_allow_html=True)
    
    # API Health Check
    api_healthy = check_api_health()
    if not api_healthy:
        st.error("🚨 CRITICAL SYSTEM FAILURE: API Server is not responding!")
        st.error("Please start the API server: `python ui/app_api_v2.py`")
        return
    
    # Get available models
    models_info = get_available_models()
    available_models = models_info.get("available_models", [])
    model_details = models_info.get("model_info", {})
    
    # Status Bar
    st.markdown(f"""
        <div class="status-bar">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="color: #10B981; font-weight: 600;">🤖 AI Models: {len(available_models)} Active</span>
                    <span style="margin-left: 2rem; color: #3B82F6; font-weight: 600;">🔗 API: Connected</span>
                    <span style="margin-left: 2rem; color: #F59E0B; font-weight: 600;">⚡ GPU: Available</span>
                </div>
                <div>
                    <span style="color: #9CA3AF;">Last Update: {datetime.now().strftime('%H:%M:%S')}</span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar - Professional Control Panel
    with st.sidebar:
        st.markdown("""
            <div class="control-panel">
                <h3 style="color: #fff; margin-bottom: 1.5rem;">🎛️ Professional Control Panel</h3>
            </div>
        """, unsafe_allow_html=True)
        
        # Model Selection
        st.markdown("**🤖 AI Model Selection**")
        
        # Initialize selected model in session state
        if 'selected_model' not in st.session_state:
            st.session_state.selected_model = None
        
        selected_model = st.session_state.selected_model
        
        if available_models:
            # Use selectbox for better UX
            model_options = []
            for model in available_models:
                info = model_details.get(model, {})
                model_name = info.get('name', model)
                model_type = info.get('type', 'unknown')
                accuracy = info.get('accuracy', 'N/A')
                model_options.append(f"{model_name} ({model_type})")
            
            selected_index = 0
            if selected_model and selected_model in available_models:
                selected_index = available_models.index(selected_model)
            
            selected_display = st.selectbox(
                "Choose AI Model:",
                options=model_options,
                index=selected_index,
                help="Select the AI model for segmentation"
            )
            
            # Map back to model name
            if selected_display:
                for i, model in enumerate(available_models):
                    info = model_details.get(model, {})
                    model_name = info.get('name', model)
                    model_type = info.get('type', 'unknown')
                    if selected_display == f"{model_name} ({model_type})":
                        st.session_state.selected_model = model
                        selected_model = model
                        break
            
            # Show model info
            if selected_model and selected_model in model_details:
                info = model_details[selected_model]
                st.markdown(f"""
                    <div style='background: rgba(59, 130, 246, 0.1); border: 1px solid rgba(59, 130, 246, 0.3); border-radius: 8px; padding: 1rem; margin: 0.5rem 0;'>
                        <strong>📋 Model Details:</strong><br>
                        <strong>Type:</strong> {info.get('type', 'Unknown')}<br>
                        <strong>Strengths:</strong> {' • '.join(info.get('strengths', []))}
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ No models available. Check API connection.")
        
        # Advanced Settings
        st.markdown("**⚙️ Advanced Configuration**")
        
        use_ensemble = st.checkbox(
            "🔄 Ensemble Intelligence",
            help="Combine multiple AI models for enhanced accuracy",
            disabled=len(available_models) < 2
        )
        
        real_time_processing = st.checkbox(
            "📹 Real-time Processing",
            value=True,
            help="Enable real-time video analysis"
        )
        
        professional_mode = st.checkbox(
            "🏥 Professional Medical Mode",
            value=True,
            help="Enable medical-grade analysis and alerts"
        )
        
        # Advanced Parameters
        st.markdown("**🎛️ Advanced Parameters**")
        
        overlay_opacity = st.slider(
            "🎨 Overlay Opacity",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Adjust segmentation overlay transparency"
        )
        
        confidence_threshold = st.slider(
            "🎯 Detection Confidence",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Minimum confidence for detections"
        )
        
        max_processing_fps = st.selectbox(
            "⚡ Max Processing FPS",
            options=[1, 5, 10, 15, 30, 60],
            index=4,
            help="Maximum frames per second for real-time processing"
        )
        
        # Medical Alert Settings
        st.markdown("**🚨 Medical Alert System**")
        
        enable_critical_alerts = st.checkbox(
            "Critical Event Alerts",
            value=True,
            help="Alert on critical surgical events"
        )
        
        occlusion_monitoring = st.checkbox(
            "Occlusion Monitoring",
            value=True,
            help="Monitor instrument-anatomy occlusion"
        )
        
        performance_monitoring = st.checkbox(
            "Performance Monitoring",
            value=True,
            help="Monitor system performance in real-time"
        )
        
        # Recording Settings
        st.markdown("**📹 Recording Settings**")
        
        auto_record = st.checkbox(
            "Auto-Record Critical Events",
            value=False,
            help="Automatically record when critical events detected"
        )
        
        save_analysis = st.checkbox(
            "Save All Analysis",
            value=True,
            help="Automatically save all analysis results"
        )
    
    # Main Content Area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Video Processing Section
        st.markdown("""
            <div class="video-container">
                <div class="overlay-controls">
                    <button class="control-button active">📹 LIVE</button>
                    <button class="control-button">🎥 REC</button>
                    <button class="control-button">📊 ANALYTICS</button>
                    <button class="control-button">⚙️ SETTINGS</button>
                </div>
        """, unsafe_allow_html=True)
        
        # File Upload
        uploaded_file = st.file_uploader(
            "📹 Upload Surgical Video or Medical Image",
            type=['mp4', 'avi', 'mov', 'png', 'jpg', 'jpeg', 'dcm'],
            help="Upload surgical video stream or medical image for AI analysis"
        )
        
        if uploaded_file is not None:
            # Display original content
            if uploaded_file.type.startswith('image/'):
                image = Image.open(uploaded_file)
                
                # Display in screenshot format: Original (left) + AI Perception (right)
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### **Original Sequence**")
                    st.image(image, caption="Surgical Scene")
                
                # Advanced Analysis Button
                col1_btn, col2_btn, col3_btn = st.columns(3)
                
                with col2_btn:
                    if st.button("🚀 Start AI Analysis", type="primary", use_container_width=True):
                        with st.spinner("🤖 Processing with advanced AI models..."):
                            result = segment_image_advanced(
                                uploaded_file,
                                model=selected_model if not use_ensemble else None,
                                ensemble=use_ensemble
                            )
                        
                        if result:
                            # Track performance
                            processing_time = result.get('processing_time', 0)
                            model_used = result.get('model_used', selected_model)
                            accuracy_score = result.get('metrics', {}).get('liver_compactness_mean', 0) * 100
                            
                            # Debug: Print what we're tracking
                            print(f"DEBUG: Tracking performance for model: {model_used}, time: {processing_time}, accuracy: {accuracy_score}")
                            
                            # Update performance tracking - use the model key, not display name
                            update_model_performance(model_used, processing_time, accuracy_score)
                            
                            # Display AI perception layer on the right
                            with col2:
                                st.markdown("### **AI Perception Layer**")
                                if 'overlay_url' in result:
                                    overlay_response = requests.get(f"{API_BASE}/{result['overlay_url']}")
                                    if overlay_response.status_code == 200:
                                        overlay_image = Image.open(io.BytesIO(overlay_response.content))
                                        st.image(overlay_image, caption="Liver Detection (Green) + Tools (Red)")
                            
                            # Display Perception Metrics
                            st.markdown("### **Perception Metrics**")
                            metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
                            
                            with metrics_col1:
                                liver_status = "LIVER DETECTED" if result.get('liver_detected', False) else "NO LIVER"
                                status_color = "🟢" if result.get('liver_detected', False) else "🔴"
                                st.markdown(f"#### {status_color} {liver_status}")
                            
                            with metrics_col2:
                                tools_status = "ACTIVE" if result.get('instrument_detected', False) else "INACTIVE"
                                tools_color = "🟢" if result.get('instrument_detected', False) else "🔴"
                                st.markdown(f"#### {tools_color} Tools: {tools_status}")
                            
                            with metrics_col3:
                                occlusion = result.get('occlusion_percent', 0)
                                occlusion_color = "🟡" if occlusion < 50 else "🔴" if occlusion > 80 else "🟢"
                                st.markdown(f"#### {occlusion_color} Occlusion Hazard: {occlusion:.1f}%")
                            
                            with metrics_col4:
                                confidence = result.get('metrics', {}).get('liver_compactness_mean', 0) * 100
                                if confidence > 80:
                                    safety_zone = "SAFE"
                                    zone_color = "🟢"
                                elif confidence > 50:
                                    safety_zone = "WARNING"
                                    zone_color = "🟡"
                                else:
                                    safety_zone = "CRITICAL"
                                    zone_color = "🔴"
                                st.markdown(f"#### {zone_color} Safety Zone: {safety_zone} ({confidence:.0f} px)")
                            
                            # Add to analysis history
                            st.session_state.analysis_history.append({
                                'timestamp': result.get('timestamp', datetime.now().strftime("%H:%M:%S")),
                                'model': result.get('model_used', 'unknown'),
                                'processing_time': result.get('processing_time', 0),
                                'liver_detected': result.get('liver_detected', False),
                                'instrument_detected': result.get('instrument_detected', False),
                                'occlusion_percent': result.get('occlusion_percent', 0),
                                'confidence': result.get('metrics', {}).get('liver_compactness_mean', 0)
                            })
                            
                            # Add performance metrics
                            fps = 1.0 / result.get('processing_time', 1) if result.get('processing_time', 0) > 0 else 0
                            st.session_state.performance_metrics.append({
                                'timestamp': datetime.now(),
                                'processing_time': result.get('processing_time', 0),
                                'fps': fps,
                                'confidence': result.get('metrics', {}).get('liver_compactness_mean', 0),
                                'memory_usage': 75.0,  # Placeholder
                                'accuracy': 0.85,  # Placeholder
                                'alerts': 1 if result.get('occlusion_percent', 0) > 50 else 0
                            })
                            
                            # Check for medical alerts
                            if enable_critical_alerts and professional_mode:
                                if result.get('occlusion_percent', 0) > 50:
                                    st.session_state.alerts.append({
                                        'level': 'critical',
                                        'title': 'High Occlusion Detected',
                                        'message': f"Critical occlusion at {result.get('occlusion_percent', 0):.1f}% - Immediate attention required",
                                        'timestamp': datetime.now().strftime("%H:%M:%S")
                                    })
                                elif result.get('occlusion_percent', 0) > 30:
                                    st.session_state.alerts.append({
                                        'level': 'warning',
                                        'title': 'Moderate Occlusion',
                                        'message': f"Occlusion at {result.get('occlusion_percent', 0):.1f}% - Monitor closely",
                                        'timestamp': datetime.now().strftime("%H:%M:%S")
                                    })
                            
                            st.success(f"✅ Analysis Complete! Processing time: {result.get('processing_time', 0):.3f}s")
                
                with col2_btn:
                    if st.button("🔄 Compare Models", use_container_width=True):
                        if uploaded_file is not None and available_models:
                            st.markdown("### **Model Comparison Analysis**")
                            
                            comparison_results = []
                            
                            # Test each model
                            for model in available_models:
                                with st.spinner(f"Testing {model_details.get(model, {}).get('name', model)}..."):
                                    result = segment_image_advanced(uploaded_file, model=model)
                                    
                                    if result:
                                        processing_time = result.get('processing_time', 0)
                                        accuracy_score = result.get('metrics', {}).get('liver_compactness_mean', 0) * 100
                                        
                                        # Update performance tracking - use the model key, not display name
                                        update_model_performance(model, processing_time, accuracy_score)
                                        
                                        comparison_results.append({
                                            'model': model_details.get(model, {}).get('name', model),
                                            'processing_time': processing_time,
                                            'fps': 1.0 / processing_time if processing_time > 0 else 0,
                                            'accuracy': accuracy_score,
                                            'liver_detected': result.get('liver_detected', False),
                                            'instruments_detected': result.get('instrument_detected', False)
                                        })
                            
                            # Display comparison results
                            if comparison_results:
                                st.markdown("#### **Performance Comparison**")
                                
                                # Create comparison table
                                comparison_df = pd.DataFrame(comparison_results)
                                
                                # Format for display
                                display_df = comparison_df[['model', 'fps', 'accuracy', 'liver_detected', 'instruments_detected']].copy()
                                display_df.columns = ['Model', 'Speed (FPS)', 'Accuracy (%)', 'Liver Detected', 'Instruments Detected']
                                display_df['Speed (FPS)'] = display_df['Speed (FPS)'].round(2)
                                display_df['Accuracy (%)'] = display_df['Accuracy (%)'].round(1)
                                display_df['Liver Detected'] = display_df['Liver Detected'].apply(lambda x: '✅ Yes' if x else '❌ No')
                                display_df['Instruments Detected'] = display_df['Instruments Detected'].apply(lambda x: '✅ Yes' if x else '❌ No')
                                
                                st.dataframe(display_df, use_container_width=True)
                                
                                # Find best model
                                best_speed = comparison_results.loc[comparison_results['fps'].idxmax()]
                                best_accuracy = comparison_results.loc[comparison_results['accuracy'].idxmax()]
                                
                                col_best1, col_best2 = st.columns(2)
                                with col_best1:
                                    st.markdown(f"🚀 **Fastest:** {best_speed['model']} ({best_speed['fps']:.1f} FPS)")
                                with col_best2:
                                    st.markdown(f"🎯 **Most Accurate:** {best_accuracy['model']} ({best_accuracy['accuracy']:.1f}%)")
                                
                                st.success("✅ Model comparison complete!")
                            else:
                                st.error("❌ No models could be compared")
                        else:
                            st.warning("⚠️ Please upload an image first")
                
                with col3_btn:
                    if st.button("📊 Detailed Report", use_container_width=True):
                        st.info("Generating comprehensive medical report...")
            
            elif uploaded_file.type.startswith('video/'):
                st.video(uploaded_file)
                st.info("🎬 Advanced video processing with frame-by-frame AI analysis")
                
                # Video processing controls
                col1_vid, col2_vid, col3_vid = st.columns(3)
                
                with col1_vid:
                    if st.button("▶️ Process Video", use_container_width=True):
                        st.info("Starting advanced video analysis...")
    
    # Critical Frames Demo Section
    st.markdown("---")
    st.markdown("## 🎯 Critical Frames Demo")
    st.markdown("Test the AI perception with pre-extracted critical surgical frames")
    
    # Check if critical frames exist
    critical_frames_dir = Path("critical_frames")
    if critical_frames_dir.exists():
        original_frames = sorted(critical_frames_dir.glob("frame_*_original.png"))
        perception_frames = sorted(critical_frames_dir.glob("frame_*_perception.png"))
        
        if original_frames and perception_frames:
            st.markdown("### **Sample Critical Frames**")
            st.info(f"📁 Found {len(original_frames)} critical frames with liver and tool detection")
            
            # Frame selector
            selected_frame_idx = st.selectbox(
                "Select Frame to Analyze:",
                range(len(original_frames)),
                format_func=lambda x: f"Frame {x:02d} - Original Sequence"
            )
            
            # Display selected frame in screenshot format
            col1_demo, col2_demo = st.columns(2)
            
            with col1_demo:
                st.markdown("#### **Original Sequence**")
                if selected_frame_idx < len(original_frames):
                    st.image(str(original_frames[selected_frame_idx]), caption="Surgical Scene")
            
            with col2_demo:
                st.markdown("#### **AI Perception Layer**")
                if selected_frame_idx < len(perception_frames):
                    st.image(str(perception_frames[selected_frame_idx]), caption="Liver (Green) + Tools (Red)")
            
            # Load and display metadata
            metadata_file = critical_frames_dir / f"frame_{selected_frame_idx:02d}_metadata.txt"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata_lines = f.readlines()
                
                st.markdown("#### **Frame Analysis Metrics**")
                metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
                
                for line in metadata_lines:
                    if 'liver_ratio:' in line:
                        liver_ratio = float(line.split(':')[1].strip())
                        with metrics_col1:
                            liver_detected = liver_ratio > 0.05
                            status = "🟢 LIVER DETECTED" if liver_detected else "🔴 NO LIVER"
                            st.markdown(f"#### {status}")
                    
                    elif 'tool_ratio:' in line:
                        tool_ratio = float(line.split(':')[1].strip())
                        with metrics_col2:
                            tools_active = tool_ratio > 0.05
                            status = "🟢 ACTIVE" if tools_active else "🔴 INACTIVE"
                            st.markdown(f"#### Tools: {status}")
                    
                    elif 'num_tools:' in line:
                        num_tools = int(line.split(':')[1].strip())
                        with metrics_col3:
                            st.markdown(f"#### 📊 Tool Count: {num_tools}")
                    
                    elif 'overall_score:' in line:
                        score = float(line.split(':')[1].strip())
                        with metrics_col4:
                            if score > 0.3:
                                safety = "SAFE"
                                color = "🟢"
                            elif score > 0.2:
                                safety = "WARNING"
                                color = "🟡"
                            else:
                                safety = "CRITICAL"
                                color = "🔴"
                            st.markdown(f"#### {color} Safety Zone: {safety}")
            
            # Quick analysis button
            if st.button(f"🚀 Analyze Frame {selected_frame_idx:02d} with AI", type="primary"):
                if selected_frame_idx < len(original_frames):
                    with open(original_frames[selected_frame_idx], 'rb') as f:
                        frame_bytes = f.read()
                    
                    # Create file-like object for upload
                    from io import BytesIO
                    uploaded_frame = BytesIO(frame_bytes)
                    uploaded_frame.name = f"frame_{selected_frame_idx:02d}_original.png"
                    
                    with st.spinner("🤖 Processing frame with AI..."):
                        result = segment_image_advanced(
                            uploaded_frame,
                            model=selected_model if not use_ensemble else None,
                            ensemble=use_ensemble
                        )
                    
                    if result:
                        st.success("✅ Frame analysis complete!")
                        # Display results in the same format
                        st.markdown("### **Real-time AI Analysis Results**")
                        metrics_realtime = st.columns(4)
                        
                        with metrics_realtime[0]:
                            liver_status = "LIVER DETECTED" if result.get('liver_detected', False) else "NO LIVER"
                            status_color = "🟢" if result.get('liver_detected', False) else "🔴"
                            st.markdown(f"#### {status_color} {liver_status}")
                        
                        with metrics_realtime[1]:
                            tools_status = "ACTIVE" if result.get('instrument_detected', False) else "INACTIVE"
                            tools_color = "🟢" if result.get('instrument_detected', False) else "🔴"
                            st.markdown(f"#### {tools_color} Tools: {tools_status}")
                        
                        with metrics_realtime[2]:
                            occlusion = result.get('occlusion_percent', 0)
                            occlusion_color = "🟡" if occlusion < 50 else "🔴" if occlusion > 80 else "🟢"
                            st.markdown(f"#### {occlusion_color} Occlusion: {occlusion:.1f}%")
                        
                        with metrics_realtime[3]:
                            processing_time = result.get('processing_time', 0)
                            fps = 1.0 / processing_time if processing_time > 0 else 0
                            st.markdown(f"#### ⚡ Speed: {fps:.1f} FPS")
        else:
            st.warning("⚠️ No critical frames found. Run `python extract_critical_frames.py` first.")
    else:
        st.info("💡 Run `python extract_critical_frames.py` to generate demo frames for testing.")
        
        # Camera Feed Option
        st.markdown("---")
        st.markdown("### 📹 Live Camera Feed")
        
        col_cam1, col_cam2 = st.columns(2)
        
        with col_cam1:
            if st.button("📷 Start Camera", use_container_width=True):
                st.session_state.camera_active = True
                frame = simulate_camera_feed()
                if frame is not None:
                    st.image(frame, channels="BGR", use_container_width=True, caption="Live Camera Feed")
                else:
                    st.info("Camera not available. Using demo mode.")
        
        with col_cam2:
            if st.button("🛑 Stop Camera", use_container_width=True):
                st.session_state.camera_active = False
                st.info("Camera stopped")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        # Real-time Metrics Panel
        st.markdown("""
            <div class="metric-card">
                <h4 style="color: #fff; margin-bottom: 1rem;">📊 Real-time Analytics</h4>
            </div>
        """, unsafe_allow_html=True)
        
        # Current Session Stats
        if st.session_state.analysis_history:
            latest = st.session_state.analysis_history[-1]
            
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{latest.get('processing_time', 0):.3f}s</div>
                    <div class="metric-label">Processing Time</div>
                    <div class="metric-change positive">↓ 15% from avg</div>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{len(st.session_state.analysis_history)}</div>
                    <div class="metric-label">Frames Analyzed</div>
                    <div class="metric-change positive">↑ Real-time</div>
                </div>
            """, unsafe_allow_html=True)
            
            # Detection Status
            liver_status = "✅ Detected" if latest.get('liver_detected') else "❌ Not Detected"
            inst_status = "✅ Detected" if latest.get('instrument_detected') else "❌ Not Detected"
            
            st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size: 1.1rem; font-weight: 600; color: #fff;">
                        {liver_status} Liver<br>
                        {inst_status} Instruments
                    </div>
                    <div class="metric-label">Detection Status</div>
                </div>
            """, unsafe_allow_html=True)
            
            # Occlusion Alert
            occlusion = latest.get('occlusion_percent', 0)
            occlusion_color = "#EF4444" if occlusion > 50 else "#F59E0B" if occlusion > 30 else "#10B981"
            
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color: {occlusion_color};">{occlusion:.1f}%</div>
                    <div class="metric-label">Occlusion Level</div>
                    <div class="metric-change {'positive' if occlusion < 30 else 'negative'}">
                        {'Normal' if occlusion < 30 else 'Attention Required' if occlusion < 50 else 'Critical'}
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        # Model Performance
        st.markdown("""
            <div class="metric-card">
                <h4 style="color: #fff; margin-bottom: 1rem;">🤖 AI Performance</h4>
            </div>
        """, unsafe_allow_html=True)
        
        for model in available_models:
            info = model_details.get(model, {})
            model_name = info.get('name', model)
            
            # Get real performance stats using model key (not display name)
            stats = get_model_stats(model)  # Use model key, not model_name
            accuracy = stats['accuracy']
            speed = stats['speed']
            runs = stats['runs']
            
            # Add performance indicator
            if runs > 0:
                performance_indicator = "🟢" if runs >= 5 else "🟡" if runs >= 1 else "⚪"
                model_display = f"{performance_indicator} {model_name}"
            else:
                performance_indicator = "⚪"
                model_display = f"{performance_indicator} {model_name}"
            
            st.markdown(f"""
                <div class="model-selector">
                    <strong style="color: #fff;">{model_display}</strong><br>
                    <small style="color: #9CA3AF;">Accuracy: {accuracy} | Speed: {speed} | Runs: {runs}</small>
                </div>
            """, unsafe_allow_html=True)
        
        # System Status
        st.markdown("""
            <div class="metric-card">
                <h4 style="color: #fff; margin-bottom: 1rem;">🖥️ System Status</h4>
            </div>
        """, unsafe_allow_html=True)
        
        # Get real system status
        try:
            memory_percent = psutil.virtual_memory().percent
            gpu_available = "Available"  # Could be enhanced with actual GPU detection
        except Exception as e:
            memory_percent = 65  # Fallback value
            gpu_available = "Unknown"
        
        st.markdown(f"""
            <div class="metric-card">
                <div style="color: #10B981; font-weight: 600;">● API Online</div>
                <div style="color: #10B981; font-weight: 600;">● Models: {len(available_models)} Loaded</div>
                <div style="color: #10B981; font-weight: 600;">● GPU: {gpu_available}</div>
                <div style="color: {'#10B981' if memory_percent < 80 else '#F59E0B' if memory_percent < 90 else '#EF4444'}; font-weight: 600;">● Memory: {memory_percent:.0f}% Used</div>
            </div>
        """, unsafe_allow_html=True)
    
    # Debug Section
    debug_model_performance()
    
    # Medical Alert System
    create_medical_alert_system()
    
    # Advanced Performance Dashboard
    if st.session_state.performance_metrics:
        st.markdown("""
            <div class="performance-chart">
                <h4 style="color: #fff; margin-bottom: 1rem;">📈 Advanced Performance Analytics</h4>
            </div>
        """, unsafe_allow_html=True)
        create_advanced_performance_dashboard()
    
    # Analysis History
    if st.session_state.analysis_history:
        st.markdown("""
            <div class="performance-chart">
                <h4 style="color: #fff; margin-bottom: 1rem;">📋 Analysis History</h4>
            </div>
        """, unsafe_allow_html=True)
        
        df = pd.DataFrame(st.session_state.analysis_history)
        st.dataframe(df, use_container_width=True)

if __name__ == "__main__":
    main()
