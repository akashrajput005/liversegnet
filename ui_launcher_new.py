import streamlit as st
import subprocess
import sys
import os
from pathlib import Path

# Page Config
st.set_page_config(
    page_title="LiverSegNet UI Launcher",
    page_icon="🏥",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #334155 100%);
        padding: 2rem;
    }
    .launcher-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    .launcher-card:hover {
        background: rgba(255, 255, 255, 0.08);
        border-color: rgba(59, 130, 246, 0.5);
        transform: translateY(-2px);
    }
    .status-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .status-online { background: #10B981; }
    .status-offline { background: #EF4444; }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <h1 style='color: #fff; font-size: 3rem; margin-bottom: 0.5rem;'>🏥 LiverSegNet</h1>
        <p style='color: #9CA3AF; font-size: 1.2rem;'>Professional Surgical AI Suite - Choose Your Interface</p>
    </div>
""", unsafe_allow_html=True)

# Check API Status
def check_api_status():
    try:
        import requests
        response = requests.get("http://localhost:8000/health", timeout=2)
        return response.status_code == 200
    except:
        return False

api_status = check_api_status()

# API Status Indicator
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if api_status:
        st.markdown("""
            <div style='background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 12px; padding: 1rem; text-align: center;'>
                <span class='status-indicator status-online'></span>
                <span style='color: #10B981; font-weight: 600;'>API Server Online</span>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div style='background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 12px; padding: 1rem; text-align: center;'>
                <span class='status-indicator status-offline'></span>
                <span style='color: #EF4444; font-weight: 600;'>API Server Offline</span>
            </div>
        """, unsafe_allow_html=True)

# UI Options
st.markdown("## 🚀 Choose Interface")

# Clean Surgical UI (Recommended)
st.markdown("""
    <div class='launcher-card'>
        <h3>🎯 Clean Surgical Interface <span style='background: #10B981; color: white; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.8rem;'>RECOMMENDED</span></h3>
        <p style='color: #9CA3AF; margin: 1rem 0;'>Professional medical interface matching your screenshot exactly</p>
        <ul style='color: #9CA3AF;'>
            <li>✅ Original Sequence + AI Perception Layer</li>
            <li>✅ Real-time surgical metrics</li>
            <li>✅ Raw frame testing</li>
            <li>✅ Clean, professional design</li>
        </ul>
    </div>
""", unsafe_allow_html=True)

if st.button("🚀 Launch Clean Surgical UI", type="primary", use_container_width=True):
    st.success("✅ Launching Clean Surgical UI...")
    st.markdown("""
        <div style='background: rgba(59, 130, 246, 0.1); border: 1px solid rgba(59, 130, 246, 0.3); border-radius: 8px; padding: 1rem; margin: 1rem 0;'>
            <h4>📋 Opening in new tab...</h4>
            <p><strong>URL:</strong> http://localhost:8503</p>
            <p><strong>Note:</strong> This will open in a new browser tab</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Launch the UI
    try:
        subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", 
            "ui/app_surgical_clean.py", 
            "--server.port", "8503",
            "--server.headless", "true"
        ])
        st.info("🌐 UI launching at http://localhost:8503")
    except Exception as e:
        st.error(f"❌ Error launching UI: {e}")

st.markdown("---")

# Professional Medical UI
st.markdown("""
    <div class='launcher-card'>
        <h3>🏥 Professional Medical Interface</h3>
        <p style='color: #9CA3AF; margin: 1rem 0;'>Full-featured medical interface with advanced options</p>
        <ul style='color: #9CA3AF;'>
            <li>✅ Multiple model support</li>
            <li>✅ Ensemble capabilities</li>
            <li>✅ Advanced analytics</li>
            <li>✅ Critical frames demo</li>
        </ul>
    </div>
""", unsafe_allow_html=True)

if st.button("🏥 Launch Professional Medical UI", use_container_width=True):
    st.success("✅ Launching Professional Medical UI...")
    st.markdown("""
        <div style='background: rgba(59, 130, 246, 0.1); border: 1px solid rgba(59, 130, 246, 0.3); border-radius: 8px; padding: 1rem; margin: 1rem 0;'>
            <h4>📋 Opening in new tab...</h4>
            <p><strong>URL:</strong> http://localhost:8504</p>
            <p><strong>Note:</strong> This will open in a new browser tab</p>
        </div>
    """, unsafe_allow_html=True)
    
    try:
        subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", 
            "ui/app_professional_medical.py", 
            "--server.port", "8504",
            "--server.headless", "true"
        ])
        st.info("🌐 UI launching at http://localhost:8504")
    except Exception as e:
        st.error(f"❌ Error launching UI: {e}")

st.markdown("---")

# Advanced Surgical UI
st.markdown("""
    <div class='launcher-card'>
        <h3>🔬 Advanced Surgical Interface</h3>
        <p style='color: #9CA3AF; margin: 1rem 0;'>Advanced interface for research and development</p>
        <ul style='color: #9CA3AF;'>
            <li>✅ Real-time processing</li>
            <li>✅ Model comparison</li>
            <li>✅ Research tools</li>
            <li>✅ Detailed metrics</li>
        </ul>
    </div>
""", unsafe_allow_html=True)

if st.button("🔬 Launch Advanced Surgical UI", use_container_width=True):
    st.success("✅ Launching Advanced Surgical UI...")
    st.markdown("""
        <div style='background: rgba(59, 130, 246, 0.1); border: 1px solid rgba(59, 130, 246, 0.3); border-radius: 8px; padding: 1rem; margin: 1rem 0;'>
            <h4>📋 Opening in new tab...</h4>
            <p><strong>URL:</strong> http://localhost:8505</p>
            <p><strong>Note:</strong> This will open in a new browser tab</p>
        </div>
    """, unsafe_allow_html=True)
    
    try:
        subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", 
            "ui/app_advanced_surgical.py", 
            "--server.port", "8505",
            "--server.headless", "true"
        ])
        st.info("🌐 UI launching at http://localhost:8505")
    except Exception as e:
        st.error(f"❌ Error launching UI: {e}")

st.markdown("---")

# Standard UI v2.0
st.markdown("""
    <div class='launcher-card'>
        <h3📊 Standard UI v2.0</h3>
        <p style='color: #9CA3AF; margin: 1rem 0;'>Classic interface with full feature set</p>
        <ul style='color: #9CA3AF;'>
            <li>✅ Multi-tab interface</li>
            <li>✅ Video processing</li>
            <li>✅ Model analytics</li>
            <li>✅ Results management</li>
        </ul>
    </div>
""", unsafe_allow_html=True)

if st.button("📊 Launch Standard UI v2.0", use_container_width=True):
    st.success("✅ Launching Standard UI v2.0...")
    st.markdown("""
        <div style='background: rgba(59, 130, 246, 0.1); border: 1px solid rgba(59, 130, 246, 0.3); border-radius: 8px; padding: 1rem; margin: 1rem 0;'>
            <h4>📋 Opening in new tab...</h4>
            <p><strong>URL:</strong> http://localhost:8506</p>
            <p><strong>Note:</strong> This will open in a new browser tab</p>
        </div>
    """, unsafe_allow_html=True)
    
    try:
        subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", 
            "ui/app_v2.py", 
            "--server.port", "8506",
            "--server.headless", "true"
        ])
        st.info("🌐 UI launching at http://localhost:8506")
    except Exception as e:
        st.error(f"❌ Error launching UI: {e}")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #6B7280; padding: 2rem;'>
        <p>🏥 LiverSegNet Professional Suite</p>
        <p style='font-size: 0.9rem;'>Choose the interface that best fits your needs</p>
    </div>
""", unsafe_allow_html=True)
