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
from plotly.subplots import make_subplots
import json
from datetime import datetime
import pandas as pd
from pathlib import Path

# Page Config
st.set_page_config(
    page_title="LiverSegNet - Advanced Surgical Perception Hub",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Professional Medical CSS
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
            padding-top: 1rem;
        }
        
        .surgical-header {
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(147, 51, 234, 0.1) 100%);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            backdrop-filter: blur(20px);
        }
        
        .live-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            background: #10B981;
            border-radius: 50%;
            animation: pulse 2s infinite;
            margin-right: 8px;
        }
        
        @keyframes pulse {
            0% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.7; transform: scale(1.1); }
            100% { opacity: 1; transform: scale(1); }
        }
        
        .video-container {
            background: #000;
            border: 2px solid rgba(59, 130, 246, 0.3);
            border-radius: 12px;
            overflow: hidden;
            position: relative;
        }
        
        .metrics-card {
            background: rgba(0, 0, 0, 0.4);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 1rem;
            backdrop-filter: blur(10px);
        }
        
        .status-active {
            color: #10B981;
            font-weight: 600;
        }
        
        .status-critical {
            color: #EF4444;
            font-weight: 600;
        }
        
        .status-warning {
            color: #F59E0B;
            font-weight: 600;
        }
        
        .stButton > button {
            background: linear-gradient(135deg, #3B82F6 0%, #8B5CF6 100%);
            border: none;
            border-radius: 8px;
            font-weight: 600;
        }
        
        .stSelectbox > div > div {
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .stFileUploader > div {
            background: rgba(0, 0, 0, 0.3);
            border: 2px dashed rgba(59, 130, 246, 0.5);
            border-radius: 12px;
        }
        </style>
    """, unsafe_allow_html=True)

professional_css()

# API Configuration
API_BASE = "http://localhost:8000"

def segment_image_advanced(image_file, model=None, ensemble=False):
    """Process image with AI segmentation"""
    try:
        files = {'file': (image_file.name, image_file, 'image/png')}
        data = {}
        if model:
            data['model'] = model
        if ensemble:
            data['ensemble'] = 'true'
            
        response = requests.post(f"{API_BASE}/segment_image", files=files, data=data)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Connection Error: {e}")
        return None

# Header
st.markdown("""
<div class="surgical-header">
    <h1 style="margin: 0; color: #fff; font-size: 2.5rem; font-weight: 700;">
        🏥 LiverSegNet
    </h1>
    <p style="margin: 0.5rem 0 0 0; color: #9CA3AF; font-size: 1.1rem;">
        Advanced Surgical Perception Hub
    </p>
</div>
""", unsafe_allow_html=True)

# Main Content
col1, col2 = st.columns([2, 1])

with col1:
    # Data Ingestion Section
    st.markdown("### 📥 Data Ingestion")
    
    uploaded_file = st.file_uploader(
        "Upload surgical image or video",
        type=['png', 'jpg', 'jpeg', 'mp4', 'avi'],
        help="Select surgical content for AI analysis"
    )
    
    if uploaded_file is not None:
        if uploaded_file.type.startswith('image/'):
            # Display in screenshot format
            img_col1, img_col2 = st.columns(2)
            
            with img_col1:
                st.markdown("#### **Original Sequence**")
                image = Image.open(uploaded_file)
                st.image(image, use_column_width=True, caption="Surgical Scene")
            
            # Analyze button
            analyze_col1, analyze_col2, analyze_col3 = st.columns([1, 2, 1])
            with analyze_col2:
                if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
                    with st.spinner("🤖 Processing with AI..."):
                        result = segment_image_advanced(uploaded_file)
                    
                    if result:
                        with img_col2:
                            st.markdown("#### **AI perception layer**")
                            if 'overlay_url' in result:
                                overlay_response = requests.get(f"{API_BASE}/{result['overlay_url']}")
                                if overlay_response.status_code == 200:
                                    overlay_image = Image.open(io.BytesIO(overlay_response.content))
                                    st.image(overlay_image, use_column_width=True, caption="Liver detection + tools")
                        
                        # Update metrics with REAL AI results
                        st.success("✅ Analysis Complete!")
                        
                        # Update the metrics container with real data
                        with metrics_container:
                            # Clear previous metrics
                            metrics_container.empty()
                            
                            with metrics_container:
                                # Liver Status
                                liver_col1, liver_col2 = st.columns([1, 2])
                                with liver_col1:
                                    if result.get('liver_detected', False):
                                        st.markdown('<div class="live-indicator"></div>', unsafe_allow_html=True)
                                    else:
                                        st.markdown('<div class="live-indicator" style="background: #EF4444;"></div>', unsafe_allow_html=True)
                                with liver_col2:
                                    liver_status = "LIVER DETECTED" if result.get('liver_detected', False) else "NO LIVER"
                                    st.markdown(f"**{liver_status}**")
                                
                                st.markdown("---")
                                
                                # Tools Status
                                tools_col1, tools_col2 = st.columns([1, 2])
                                with tools_col1:
                                    if result.get('instrument_detected', False):
                                        st.markdown('<div class="live-indicator"></div>', unsafe_allow_html=True)
                                    else:
                                        st.markdown('<div class="live-indicator" style="background: #EF4444;"></div>', unsafe_allow_html=True)
                                with tools_col2:
                                    tools_status = "ACTIVE" if result.get('instrument_detected', False) else "INACTIVE"
                                    st.markdown(f"**Tools: {tools_status}**")
                                
                                st.markdown("---")
                                
                                # Occlusion Hazard
                                st.markdown("**Occlusion Hazard**")
                                occlusion = result.get('occlusion_percent', 0)
                                occlusion_level = st.progress(occlusion/100, text=f"{occlusion:.1f}%")
                                
                                st.markdown("---")
                                
                                # Safety Zone
                                st.markdown("**Safety Zone**")
                                confidence = result.get('metrics', {}).get('liver_compactness_mean', 0) * 100
                                if confidence > 80:
                                    safety_zone = "SAFE"
                                    color_class = "status-active"
                                elif confidence > 50:
                                    safety_zone = "WARNING"
                                    color_class = "status-warning"
                                else:
                                    safety_zone = "CRITICAL"
                                    color_class = "status-critical"
                                st.markdown(f'<p class="{color_class}">{safety_zone} ({confidence:.0f} px)</p>', unsafe_allow_html=True)
                                
                                # Additional metrics
                                st.markdown("---")
                                st.markdown("**Processing Info**")
                                processing_time = result.get('processing_time', 0)
                                fps = 1.0 / processing_time if processing_time > 0 else 0
                                st.markdown(f"⚡ **Speed:** {fps:.1f} FPS")
                                st.markdown(f"🤖 **Model:** {result.get('model_used', 'N/A')}")
                    else:
                        st.error("❌ Analysis failed. Please check API connection.")

with col2:
    # Perception Metrics
    st.markdown("### 📊 Perception Metrics")
    
    metrics_container = st.container()
    
    with metrics_container:
        # Initial state - waiting for analysis
        liver_col1, liver_col2 = st.columns([1, 2])
        with liver_col1:
            st.markdown('<div class="live-indicator" style="background: #6B7280;"></div>', unsafe_allow_html=True)
        with liver_col2:
            st.markdown('**<span style="color: #6B7280;">WAITING FOR ANALYSIS</span>**', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Tools Status
        tools_col1, tools_col2 = st.columns([1, 2])
        with tools_col1:
            st.markdown('<div class="live-indicator" style="background: #6B7280;"></div>', unsafe_allow_html=True)
        with tools_col2:
            st.markdown('**<span style="color: #6B7280;">Tools: INACTIVE</span>**', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Occlusion Hazard
        st.markdown("**Occlusion Hazard**")
        occlusion_level = st.progress(0.0, text="0.0%")
        
        st.markdown("---")
        
        # Safety Zone
        st.markdown("**Safety Zone**")
        st.markdown('<p style="color: #6B7280;">NOT ANALYZED</p>', unsafe_allow_html=True)

# Critical Frames Demo (Bottom Section)
st.markdown("---")
st.markdown("### 🎯 Raw Surgical Frames - Test AI Segmentation")

critical_frames_dir = Path("critical_frames")
if critical_frames_dir.exists():
    raw_frames = sorted(critical_frames_dir.glob("frame_*_raw.png"))
    
    if raw_frames:
        # Frame selector
        selected_frame = st.selectbox(
            "Select raw surgical frame:",
            range(len(raw_frames)),
            format_func=lambda x: f"Frame {x:02d} - Raw Surgical Scene"
        )
        
        # Display frame in screenshot format
        frame_col1, frame_col2 = st.columns(2)
        
        with frame_col1:
            st.markdown("#### **Original Sequence**")
            st.image(str(raw_frames[selected_frame]), use_column_width=True, caption="Raw Surgical Scene (Unsegmented)")
        
        with frame_col2:
            st.markdown("#### **AI perception layer**")
            # Initially show placeholder
            ai_placeholder = st.empty()
            ai_placeholder.info("🤖 Click 'Analyze with AI' to generate segmentation")
        
        # Analyze button
        analyze_col1, analyze_col2, analyze_col3 = st.columns([1, 2, 1])
        with analyze_col2:
            if st.button(f"🚀 Analyze Frame {selected_frame:02d} with AI", type="primary", use_container_width=True):
                with st.spinner("🤖 AI performing real-time segmentation..."):
                    # Load raw frame and send to AI
                    with open(raw_frames[selected_frame], 'rb') as f:
                        frame_bytes = f.read()
                    
                    from io import BytesIO
                    frame_file = BytesIO(frame_bytes)
                    frame_file.name = f"frame_{selected_frame:02d}_raw.png"
                    
                    # Get AI segmentation
                    result = segment_image_advanced(frame_file)
                
                if result:
                    # Display AI perception layer
                    with frame_col2:
                        ai_placeholder.empty()
                        st.markdown("#### **AI perception layer**")
                        if 'overlay_url' in result:
                            overlay_response = requests.get(f"{API_BASE}/{result['overlay_url']}")
                            if overlay_response.status_code == 200:
                                overlay_image = Image.open(io.BytesIO(overlay_response.content))
                                st.image(overlay_image, use_column_width=True, caption="AI Segmentation: Liver (Green) + Tools (Red)")
                    
                    # Update metrics with real AI results
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
                    
                    st.success(f"✅ AI Analysis Complete! Processing time: {result.get('processing_time', 0):.3f}s")
                    
                    # Show frame info
                    info_file = critical_frames_dir / f"frame_{selected_frame:02d}_info.txt"
                    if info_file.exists():
                        with open(info_file, 'r') as f:
                            info_lines = f.readlines()
                        st.markdown("#### **Frame Information**")
                        for line in info_lines:
                            st.text(line.strip())
                else:
                    st.error("❌ AI analysis failed. Please check API connection.")
        
        # Instructions
        with st.expander("📋 How to use", expanded=False):
            st.markdown("""
            1. **Select** a raw surgical frame from the dropdown
            2. **View** the original unsegmented surgical scene on the left
            3. **Click** "Analyze with AI" to perform real-time segmentation
            4. **See** AI perception layer with liver (green) and tools (red) on the right
            5. **Review** real-time metrics and analysis results
            
            These are raw surgical frames that need AI segmentation - just like in your screenshot!
            """)
    else:
        st.warning("⚠️ No raw frames found. Run `python extract_critical_frames.py` first.")
else:
    st.info("💡 Run `python extract_critical_frames.py` to generate raw surgical frames for testing.")
