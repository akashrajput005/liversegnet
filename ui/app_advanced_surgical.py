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
from pathlib import Path

# --- Page Config ---
st.set_page_config(
    page_title="Advanced Surgical AI Suite",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Advanced CSS ---
def local_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            color: #E0E0E0;
        }

        .main {
            background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0F172A 100%);
        }

        /* Advanced Medical Interface */
        .medical-header {
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(147, 51, 234, 0.1) 100%);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            backdrop-filter: blur(10px);
        }

        .live-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            background: #22C55E;
            border-radius: 50%;
            animation: pulse 2s infinite;
            margin-right: 8px;
        }

        @keyframes pulse {
            0% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.7; transform: scale(1.1); }
            100% { opacity: 1; transform: scale(1); }
        }

        .metric-card {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 0.5rem;
            transition: all 0.3s ease;
        }

        .metric-card:hover {
            background: rgba(255, 255, 255, 0.08);
            transform: translateY(-2px);
            box-shadow: 0 8px 32px rgba(59, 130, 246, 0.2);
        }

        .control-panel {
            background: rgba(15, 23, 42, 0.8);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 1.5rem;
            backdrop-filter: blur(15px);
        }

        .video-container {
            background: #000;
            border: 2px solid rgba(59, 130, 246, 0.3);
            border-radius: 12px;
            overflow: hidden;
            position: relative;
        }

        .overlay-controls {
            position: absolute;
            top: 10px;
            right: 10px;
            z-index: 10;
            background: rgba(0, 0, 0, 0.7);
            padding: 10px;
            border-radius: 8px;
        }

        .status-online {
            color: #22C55E;
            font-weight: 600;
        }

        .status-processing {
            color: #F59E0B;
            font-weight: 600;
        }

        .alert-critical {
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid rgba(239, 68, 68, 0.3);
            border-radius: 8px;
            padding: 1rem;
            color: #EF4444;
        }

        .performance-chart {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 1rem;
        }

        /* Model Selection Cards */
        .model-card {
            background: rgba(255, 255, 255, 0.05);
            border: 2px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 0.5rem;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .model-card.selected {
            border-color: #3B82F6;
            background: rgba(59, 130, 246, 0.1);
        }

        .model-card:hover {
            border-color: #3B82F6;
            transform: translateY(-1px);
        }

        /* Real-time Metrics */
        .realtime-metric {
            font-size: 2rem;
            font-weight: 700;
            color: #3B82F6;
        }

        .metric-label {
            font-size: 0.875rem;
            color: #9CA3AF;
            margin-top: 0.25rem;
        }

        /* Alert System */
        .alert-banner {
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(245, 158, 11, 0.1) 100%);
            border: 1px solid rgba(239, 68, 68, 0.3);
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)

# --- Configuration ---
API_BASE = "http://localhost:8000"

# --- Session State Initialization ---
if 'session_id' not in st.session_state:
    st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'performance_metrics' not in st.session_state:
    st.session_state.performance_metrics = []
if 'alerts' not in st.session_state:
    st.session_state.alerts = []

# --- API Functions ---
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

def segment_image_realtime(file, model=None, ensemble=False):
    """Segment image with real-time processing"""
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
            return result
    except Exception as e:
        st.error(f"Real-time processing error: {str(e)}")
    return None

# --- Advanced Analytics ---
def create_performance_dashboard():
    """Create real-time performance dashboard"""
    if not st.session_state.performance_metrics:
        return
    
    df = pd.DataFrame(st.session_state.performance_metrics)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Processing Time', 'FPS', 'Detection Confidence', 'Memory Usage'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Processing Time
    fig.add_trace(
        go.Scatter(x=df.index, y=df['processing_time'], name='Processing Time', line=dict(color='#3B82F6')),
        row=1, col=1
    )
    
    # FPS
    fig.add_trace(
        go.Scatter(x=df.index, y=df['fps'], name='FPS', line=dict(color='#22C55E')),
        row=1, col=2
    )
    
    # Confidence
    fig.add_trace(
        go.Scatter(x=df.index, y=df['confidence'], name='Confidence', line=dict(color='#F59E0B')),
        row=2, col=1
    )
    
    # Memory (placeholder)
    fig.add_trace(
        go.Scatter(x=df.index, y=df['memory_usage'], name='Memory Usage', line=dict(color='#EF4444')),
        row=2, col=2
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_alert_system():
    """Create advanced alert system"""
    if st.session_state.alerts:
        for alert in st.session_state.alerts[-5:]:  # Show last 5 alerts
            if alert['level'] == 'critical':
                st.markdown(f"""
                    <div class="alert-banner">
                        🚨 <strong>{alert['title']}</strong><br>
                        {alert['message']}<br>
                        <small>{alert['timestamp']}</small>
                    </div>
                """, unsafe_allow_html=True)
            elif alert['level'] == 'warning':
                st.markdown(f"""
                    <div class="alert-banner" style="border-color: rgba(245, 158, 11, 0.3); background: rgba(245, 158, 11, 0.1);">
                        ⚠️ <strong>{alert['title']}</strong><br>
                        {alert['message']}<br>
                        <small>{alert['timestamp']}</small>
                    </div>
                """, unsafe_allow_html=True)

# --- Main Application ---
def main():
    local_css()
    
    # Advanced Header
    st.markdown("""
        <div class="medical-header">
            <h1 style="margin: 0; color: #fff; font-size: 2.5rem;">
                🏥 Advanced Surgical AI Suite
            </h1>
            <p style="margin: 0.5rem 0 0 0; color: #9CA3AF; font-size: 1.1rem;">
                Real-time Surgical Perception with Multi-Model Intelligence
            </p>
            <div style="margin-top: 1rem;">
                <span class="live-indicator"></span>
                <span class="status-online">● SYSTEM ONLINE</span>
                <span style="margin-left: 2rem; color: #9CA3AF;">
                    Session: {} | FPS: <span id="fps">0</span>
                </span>
            </div>
        </div>
    """.format(st.session_state.session_id), unsafe_allow_html=True)
    
    # API Health Check
    api_healthy = check_api_health()
    if not api_healthy:
        st.error("🚨 CRITICAL: API Server is not running! Please start the API server first.")
        st.info("Run: `python ui/app_api_v2.py`")
        return
    
    # Get available models
    models_info = get_available_models()
    available_models = models_info.get("available_models", [])
    model_details = models_info.get("model_info", {})
    
    # Sidebar - Advanced Control Panel
    with st.sidebar:
        st.markdown("""
            <div class="control-panel">
                <h3>🎛️ Advanced Control Panel</h3>
            </div>
        """, unsafe_allow_html=True)
        
        # Model Selection with Cards
        st.markdown("**🤖 Model Selection**")
        
        selected_model = None
        for model in available_models:
            info = model_details.get(model, {})
            model_name = info.get('name', model)
            model_type = info.get('type', 'Unknown')
            
            # Create model card
            card_class = "model-card"
            if 'selected_model' in st.session_state and st.session_state.selected_model == model:
                card_class += " selected"
            
            st.markdown(f"""
                <div class="{card_class}" onclick="selectModel('{model}')">
                    <strong>{model_name}</strong><br>
                    <small>Type: {model_type} | Status: ✅ Active</small>
                </div>
            """, unsafe_allow_html=True)
            
            if st.button(f"Select {model_name}", key=f"select_{model}"):
                st.session_state.selected_model = model
                selected_model = model
        
        # Advanced Settings
        st.markdown("**⚙️ Advanced Settings**")
        
        use_ensemble = st.checkbox(
            "🔄 Enable Ensemble Mode",
            help="Combine multiple models for enhanced accuracy",
            disabled=len(available_models) < 2
        )
        
        real_time_mode = st.checkbox(
            "📹 Real-time Processing",
            value=True,
            help="Enable real-time video processing"
        )
        
        overlay_opacity = st.slider(
            "🎨 Overlay Opacity",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1
        )
        
        confidence_threshold = st.slider(
            "🎯 Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05
        )
        
        # Performance Settings
        st.markdown("**⚡ Performance Settings**")
        
        max_fps = st.selectbox(
            "Max FPS",
            options=[1, 5, 10, 15, 30],
            index=3,
            help="Maximum frames per second for real-time processing"
        )
        
        gpu_acceleration = st.checkbox(
            "🚀 GPU Acceleration",
            value=True,
            help="Use GPU for faster processing"
        )
        
        # Alert Settings
        st.markdown("**🚨 Alert System**")
        
        enable_alerts = st.checkbox(
            "Enable Critical Alerts",
            value=True,
            help="Alert on critical surgical events"
        )
        
        occlusion_alert = st.checkbox(
            "Occlusion Alerts",
            value=True,
            help="Alert when instruments occlude anatomy"
        )
        
        performance_alert = st.checkbox(
            "Performance Alerts",
            value=True,
            help="Alert on performance degradation"
        )
    
    # Main Content Area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Video Processing Section
        st.markdown("""
            <div class="video-container">
                <div class="overlay-controls">
                    <button style="background: rgba(0,0,0,0.7); color: white; border: 1px solid #3B82F6; padding: 5px 10px; border-radius: 5px; cursor: pointer;">
                        📹 LIVE
                    </button>
                </div>
        """, unsafe_allow_html=True)
        
        # File Upload for Video/Image
        uploaded_file = st.file_uploader(
            "📹 Upload Surgical Video or Image",
            type=['mp4', 'avi', 'mov', 'png', 'jpg', 'jpeg'],
            help="Upload surgical video stream or image for analysis"
        )
        
        if uploaded_file is not None:
            # Display original content
            if uploaded_file.type.startswith('image/'):
                image = Image.open(uploaded_file)
                st.image(image, use_column_width=True, caption="Original Image")
                
                # Process button
                if st.button("🚀 Start Advanced Analysis", type="primary"):
                    with st.spinner("Processing with advanced AI models..."):
                        result = segment_image_realtime(
                            uploaded_file,
                            model=selected_model if not use_ensemble else None,
                            ensemble=use_ensemble
                        )
                    
                    if result:
                        # Display results
                        if 'overlay_url' in result:
                            overlay_response = requests.get(f"{API_BASE}/{result['overlay_url']}")
                            if overlay_response.status_code == 200:
                                overlay_image = Image.open(io.BytesIO(overlay_response.content))
                                st.image(overlay_image, use_column_width=True, caption="AI Segmentation Overlay")
                        
                        # Add to history
                        st.session_state.analysis_history.append({
                            'timestamp': datetime.now().strftime("%H:%M:%S"),
                            'model': result.get('model_used', 'unknown'),
                            'processing_time': result.get('processing_time', 0),
                            'liver_detected': result.get('liver_detected', False),
                            'instrument_detected': result.get('instrument_detected', False)
                        })
                        
                        # Add performance metrics
                        st.session_state.performance_metrics.append({
                            'timestamp': datetime.now(),
                            'processing_time': result.get('processing_time', 0),
                            'fps': 1.0 / result.get('processing_time', 1),
                            'confidence': 0.85,  # Placeholder
                            'memory_usage': 75.0  # Placeholder
                        })
                        
                        # Check for alerts
                        if enable_alerts:
                            if result.get('occlusion_percent', 0) > 50:
                                st.session_state.alerts.append({
                                    'level': 'critical',
                                    'title': 'High Occlusion Detected',
                                    'message': f"Instrument occlusion at {result.get('occlusion_percent', 0):.1f}%",
                                    'timestamp': datetime.now().strftime("%H:%M:%S")
                                })
            
            elif uploaded_file.type.startswith('video/'):
                st.video(uploaded_file)
                st.info("🎬 Advanced video processing will be implemented with frame-by-frame analysis")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        # Real-time Metrics Panel
        st.markdown("""
            <div class="metric-card">
                <h4>📊 Real-time Metrics</h4>
            </div>
        """, unsafe_allow_html=True)
        
        # Current Session Stats
        if st.session_state.analysis_history:
            latest = st.session_state.analysis_history[-1]
            
            st.markdown(f"""
                <div class="metric-card">
                    <div class="realtime-metric">{latest.get('processing_time', 0):.3f}s</div>
                    <div class="metric-label">Processing Time</div>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
                <div class="metric-card">
                    <div class="realtime-metric">{len(st.session_state.analysis_history)}</div>
                    <div class="metric-label">Frames Processed</div>
                </div>
            """, unsafe_allow_html=True)
            
            liver_status = "✅" if latest.get('liver_detected') else "❌"
            inst_status = "✅" if latest.get('instrument_detected') else "❌"
            
            st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size: 1.2rem; font-weight: 600;">
                        {liver_status} Liver<br>
                        {inst_status} Instruments
                    </div>
                    <div class="metric-label">Detection Status</div>
                </div>
            """, unsafe_allow_html=True)
        
        # Model Performance
        st.markdown("""
            <div class="metric-card">
                <h4>🤖 Model Performance</h4>
            </div>
        """, unsafe_allow_html=True)
        
        for model in available_models:
            info = model_details.get(model, {})
            accuracy = info.get('accuracy', 'N/A')
            speed = info.get('speed', 'N/A')
            
            st.markdown(f"""
                <div class="metric-card">
                    <strong>{info.get('name', model)}</strong><br>
                    <small>Accuracy: {accuracy} | Speed: {speed}</small>
                </div>
            """, unsafe_allow_html=True)
        
        # System Status
        st.markdown("""
            <div class="metric-card">
                <h4>🖥️ System Status</h4>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
            <div class="metric-card">
                <span class="status-online">● API Online</span><br>
                <span class="status-online">● Models Loaded: {len(available_models)}</span><br>
                <span class="status-online">● GPU: {'Active' if gpu_acceleration else 'CPU'}</span>
            </div>
        """, unsafe_allow_html=True)
    
    # Alert System
    create_alert_system()
    
    # Performance Dashboard
    if st.session_state.performance_metrics:
        st.markdown("""
            <div class="performance-chart">
                <h4>📈 Performance Analytics</h4>
            </div>
        """, unsafe_allow_html=True)
        create_performance_dashboard()
    
    # Analysis History
    if st.session_state.analysis_history:
        st.markdown("""
            <div class="performance-chart">
                <h4>📋 Analysis History</h4>
            </div>
        """, unsafe_allow_html=True)
        
        df = pd.DataFrame(st.session_state.analysis_history)
        st.dataframe(df, use_container_width=True)

if __name__ == "__main__":
    main()
