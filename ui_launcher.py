#!/usr/bin/env python3
"""
LiverSegNet AI - UI Launcher
Choose between different UI modes
"""

import streamlit as st
import subprocess
import sys
import os
import time
import requests

def check_api_status():
    """Check if API server is running"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=3)
        return response.status_code == 200
    except:
        return False

def start_api_server():
    """Start the API server in background"""
    try:
        subprocess.Popen([
            sys.executable, "ui/app_api_v2.py"
        ], cwd=os.getcwd())
        return True
    except Exception as e:
        st.error(f"Failed to start API server: {e}")
        return False

def main():
    st.set_page_config(
        page_title="LiverSegNet AI - UI Launcher",
        page_icon="🏥",
        layout="centered",
        initial_sidebar_state="collapsed"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .main {
            background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #334155 100%);
        }
        .launcher-container {
            max-width: 800px;
            margin: 0 auto;
            padding: 2rem;
        }
        .ui-card {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 2rem;
            margin-bottom: 1.5rem;
            transition: all 0.3s ease;
            cursor: pointer;
        }
        .ui-card:hover {
            background: rgba(255, 255, 255, 0.08);
            transform: translateY(-2px);
            box-shadow: 0 10px 30px rgba(59, 130, 246, 0.2);
            border-color: #3B82F6;
        }
        .ui-card h3 {
            color: #fff;
            margin-bottom: 0.5rem;
        }
        .ui-card p {
            color: #9CA3AF;
            margin-bottom: 1rem;
        }
        .feature-tag {
            display: inline-block;
            background: rgba(59, 130, 246, 0.2);
            color: #3B82F6;
            padding: 0.25rem 0.75rem;
            border-radius: 12px;
            font-size: 0.875rem;
            margin-right: 0.5rem;
            margin-bottom: 0.5rem;
        }
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .status-online {
            background: #10B981;
            box-shadow: 0 0 10px rgba(16, 185, 129, 0.5);
        }
        .status-offline {
            background: #EF4444;
            box-shadow: 0 0 10px rgba(239, 68, 68, 0.5);
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
        <div class="launcher-container">
            <div style="text-align: center; margin-bottom: 3rem;">
                <h1 style="color: #fff; font-size: 3rem; font-weight: 700; margin-bottom: 1rem;">
                    🏥 LiverSegNet AI
                </h1>
                <p style="color: #9CA3AF; font-size: 1.2rem;">
                    Advanced Surgical Intelligence Suite - UI Launcher
                </p>
            </div>
    """, unsafe_allow_html=True)
    
    # Check API Status
    api_status = check_api_status()
    
    # Status Bar
    st.markdown(f"""
        <div style="background: rgba(0, 0, 0, 0.6); border: 1px solid rgba(255, 255, 255, 0.1); 
                    border-radius: 12px; padding: 1rem; margin-bottom: 2rem;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span class="status-indicator {'status-online' if api_status else 'status-offline'}"></span>
                    <span style="color: {'#10B981' if api_status else '#EF4444'}; font-weight: 600;">
                        API Server: {'Online' if api_status else 'Offline'}
                    </span>
                </div>
                <div>
                    <span style="color: #9CA3AF;">Models Available: 3</span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    if not api_status:
        st.warning("⚠️ API Server is not running. Some features may not work correctly.")
        if st.button("🚀 Start API Server", type="primary"):
            with st.spinner("Starting API server..."):
                if start_api_server():
                    st.success("✅ API Server started successfully!")
                    time.sleep(2)
                    st.rerun()
                else:
                    st.error("❌ Failed to start API Server")
    
    # UI Options
    st.markdown("## 🎛️ Choose UI Mode")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Standard UI
        if st.markdown("""
            <div class="ui-card" onclick="selectUI('standard')">
                <h3>📊 Standard UI</h3>
                <p>Enhanced interface with auto-save and basic analytics</p>
                <div>
                    <span class="feature-tag">Auto-Save</span>
                    <span class="feature-tag">4 Tabs</span>
                    <span class="feature-tag">Results Management</span>
                </div>
                <div style="margin-top: 1rem;">
                    <small style="color: #6B7280;">Recommended for most users</small>
                </div>
            </div>
        """, unsafe_allow_html=True):
            pass
        
        if st.button("🚀 Launch Standard UI", key="standard", use_container_width=True, type="primary"):
            st.info("🚀 Launching Standard UI...")
            # Launch standard UI
            subprocess.Popen([
                sys.executable, "-m", "streamlit", "run", 
                "ui/app_v2.py", "--server.port", "8501"
            ], cwd=os.getcwd())
            st.success("✅ Standard UI launched on http://localhost:8501")
            st.markdown('[Open Standard UI](http://localhost:8501)', unsafe_allow_html=True)
    
    with col2:
        # Professional Medical UI
        if st.markdown("""
            <div class="ui-card" onclick="selectUI('professional')">
                <h3>🏥 Professional Medical UI</h3>
                <p>Advanced medical interface with real-time analytics</p>
                <div>
                    <span class="feature-tag">Real-time</span>
                    <span class="feature-tag">Medical Alerts</span>
                    <span class="feature-tag">Advanced Metrics</span>
                </div>
                <div style="margin-top: 1rem;">
                    <small style="color: #6B7280;">Professional medical interface</small>
                </div>
            </div>
        """, unsafe_allow_html=True):
            pass
        
        if st.button("🏥 Launch Professional UI", key="professional", use_container_width=True, type="primary"):
            st.info("🏥 Launching Professional Medical UI...")
            # Launch professional UI
            subprocess.Popen([
                sys.executable, "-m", "streamlit", "run", 
                "ui/app_professional_medical.py", "--server.port", "8502"
            ], cwd=os.getcwd())
            st.success("✅ Professional UI launched on http://localhost:8502")
            st.markdown('[Open Professional UI](http://localhost:8502)', unsafe_allow_html=True)
    
    # Advanced Options
    st.markdown("---")
    st.markdown("## 🔧 Advanced Options")
    
    col3, col4 = st.columns(2)
    
    with col3:
        # Advanced Surgical UI
        if st.markdown("""
            <div class="ui-card" onclick="selectUI('advanced')">
                <h3>🔬 Advanced Surgical UI</h3>
                <p>Research-grade interface with experimental features</p>
                <div>
                    <span class="feature-tag">Experimental</span>
                    <span class="feature-tag">Research</span>
                </div>
            </div>
        """, unsafe_allow_html=True):
            pass
        
        if st.button("🔬 Launch Advanced UI", key="advanced", use_container_width=True):
            st.info("🔬 Launching Advanced Surgical UI...")
            subprocess.Popen([
                sys.executable, "-m", "streamlit", "run", 
                "ui/app_advanced_surgical.py", "--server.port", "8503"
            ], cwd=os.getcwd())
            st.success("✅ Advanced UI launched on http://localhost:8503")
            st.markdown('[Open Advanced UI](http://localhost:8503)', unsafe_allow_html=True)
    
    with col4:
        # Custom Configuration
        if st.markdown("""
            <div class="ui-card" onclick="selectUI('custom')">
                <h3>⚙️ Custom Configuration</h3>
                <p>Launch with custom parameters and settings</p>
                <div>
                    <span class="feature-tag">Custom</span>
                    <span class="feature-tag">Flexible</span>
                </div>
            </div>
        """, unsafe_allow_html=True):
            pass
        
        with st.expander("⚙️ Custom Launch Options"):
            port = st.number_input("Port", value=8504, min_value=8000, max_value=9000)
            ui_file = st.selectbox("UI File", [
                "ui/app_v2.py",
                "ui/app_professional_medical.py", 
                "ui/app_advanced_surgical.py"
            ])
            
            if st.button("🚀 Launch Custom", use_container_width=True):
                st.info(f"🚀 Launching {ui_file} on port {port}...")
                subprocess.Popen([
                    sys.executable, "-m", "streamlit", "run", 
                    ui_file, "--server.port", str(port)
                ], cwd=os.getcwd())
                st.success(f"✅ Custom UI launched on http://localhost:{port}")
                st.markdown(f'[Open Custom UI](http://localhost:{port})', unsafe_allow_html=True)
    
    # System Information
    st.markdown("---")
    st.markdown("## 📊 System Information")
    
    col5, col6, col7 = st.columns(3)
    
    with col5:
        st.metric("API Status", "🟢 Online" if api_status else "🔴 Offline")
    
    with col6:
        st.metric("Available UIs", "3")
    
    with col7:
        st.metric("Models", "3")
    
    # Quick Links
    st.markdown("---")
    st.markdown("## 🔗 Quick Links")
    
    col8, col9, col10 = st.columns(3)
    
    with col8:
        if st.button("📚 Documentation", use_container_width=True):
            st.info("Documentation would open here")
    
    with col9:
        if st.button("🐛 Report Issue", use_container_width=True):
            st.info("Issue tracker would open here")
    
    with col10:
        if st.button("💾 Backup Results", use_container_width=True):
            st.info("Backup utility would run here")
    
    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
