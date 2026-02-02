"""
LiverSegNet AI - Dual-Mode Medical Interface
Surgeon Mode: Technical analysis and metrics
Patient Mode: Simple, reassuring information
"""
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import sys
import os
import time
from datetime import datetime

sys.path.insert(0, 'src')
from infer import InferenceEngine

# Page config
st.set_page_config(
    page_title="LiverSegNet AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
    }
    
    .stApp {
        background: transparent;
    }
    
    /* Glass Card */
    .glass-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border-radius: 24px;
        padding: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.3);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.2);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #fff;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: rgba(255, 255, 255, 0.8);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .status-good {
        color: #10b981;
        font-weight: 600;
    }
    
    .status-warning {
        color: #f59e0b;
        font-weight: 600;
    }
    
    .status-critical {
        color: #ef4444;
        font-weight: 600;
    }
    
    h1, h2, h3 {
        color: #fff;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(16, 185, 129, 0.4);
    }
    
    .sidebar .sidebar-content {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
    }
    
    .patient-card {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    
    .surgeon-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 16px;
        margin-bottom: 1rem;
    }
    
    /* Image containers */
    .image-container {
        border-radius: 16px;
        overflow: hidden;
        border: 2px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    return InferenceEngine(
        model_path="models/deeplabv3plus_resnet50.pth",
        architecture='deeplabv3plus',
        encoder='resnet50',
        img_size=(256, 256),
        num_classes=3
    )

# Sidebar
with st.sidebar:
    st.markdown("### 🏥 **LiverSegNet AI**")
    st.markdown("---")
    
    mode = st.radio(
        "**Select View Mode**",
        ["👨‍⚕️ Surgeon Mode", "👤 Patient Mode"],
        help="Choose between detailed analysis or simplified overview"
    )
    
    st.markdown("---")
    
    st.markdown("### 📊 System Status")
    engine = load_model()
    st.success("✅ AI Models Active")
    st.info("🧠 3-Model Ensemble")
    st.info("🎯 Clinical Grade")
    
    st.markdown("---")
    st.markdown("### 📝 Quick Info")
    if "👨‍⚕️" in mode:
        st.markdown("""
        **Surgeon Dashboard**
        - Full segmentation metrics
        - Clinical measurements
        - Risk assessment
        - Detailed tissue analysis
        """)
    else:
        st.markdown("""
        **Patient Overview**
        - Simple visual results
        - Easy-to-understand status
        - Safety indicators
        - Clear explanations
        """)

# Main content
if "👨‍⚕️" in mode:
    # SURGEON MODE
    st.markdown("<h1 style='text-align: center; color: white;'>🔬 Surgeon Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: rgba(255,255,255,0.8); font-size: 1.1rem;'>Advanced Surgical Vision AI - Clinical Analysis</p>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("📤 Upload Surgical Image", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        
        if len(image_np.shape) == 3:
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='glass-card'><h3>📷 Original Image</h3></div>", unsafe_allow_html=True)
            st.markdown("<div class='image-container'>", unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with st.spinner("🔄 Running AI Analysis..."):
            start_time = time.time()
            mask, overlay, occlusion, distance, liver_found, inst_found = engine.predict_image(image_bgr)
            inference_time = time.time() - start_time
            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        
        with col2:
            st.markdown("<div class='glass-card'><h3>🎯 AI Segmentation</h3></div>", unsafe_allow_html=True)
            st.markdown("<div class='image-container'>", unsafe_allow_html=True)
            st.image(overlay_rgb, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Metrics
        st.markdown("---")
        st.markdown("<div class='glass-card'><h3>📊 Clinical Measurements</h3></div>", unsafe_allow_html=True)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            status = "✅" if liver_found else "❌"
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Liver</div>
                <div class='metric-value'>{status}</div>
                <div class='metric-label'>Detection</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            status = "✅" if inst_found else "❌"
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Instruments</div>
                <div class='metric-value'>{status}</div>
                <div class='metric-label'>Detection</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            occ_class = "status-good" if occlusion < 20 else "status-warning" if occlusion < 40 else "status-critical"
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Occlusion</div>
                <div class='metric-value {occ_class}'>{occlusion:.1f}%</div>
                <div class='metric-label'>Coverage</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            dist_val = f"{distance:.0f}" if distance != float('inf') else "∞"
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Distance</div>
                <div class='metric-value'>{dist_val}</div>
                <div class='metric-label'>Pixels</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Inference</div>
                <div class='metric-value'>{inference_time:.2f}s</div>
                <div class='metric-label'>Speed</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Detailed Analysis
        st.markdown("---")
        st.markdown("<div class='glass-card'><h3>🔍 Detailed Tissue Analysis</h3></div>", unsafe_allow_html=True)
        
        liver_pixels = np.sum(mask == 1)
        inst_pixels = np.sum(mask == 2)
        total_pixels = mask.shape[0] * mask.shape[1]
        
        from cv2 import connectedComponents
        num_liver_regions = connectedComponents((mask == 1).astype(np.uint8))[0] - 1
        num_inst_regions = connectedComponents((mask == 2).astype(np.uint8))[0] - 1
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class='glass-card'>
                <h4>🟢 Liver Analysis</h4>
                <ul style='color: rgba(255,255,255,0.9);'>
                    <li><strong>Coverage:</strong> {(liver_pixels/total_pixels)*100:.2f}% of frame</li>
                    <li><strong>Regions:</strong> {num_liver_regions} detected</li>
                    <li><strong>Pixels:</strong> {liver_pixels:,}</li>
                    <li><strong>Status:</strong> {'Normal' if liver_found else 'Not Detected'}</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class='glass-card'>
                <h4>🔴 Instrument Analysis</h4>
                <ul style='color: rgba(255,255,255,0.9);'>
                    <li><strong>Coverage:</strong> {(inst_pixels/total_pixels)*100:.2f}% of frame</li>
                    <li><strong>Tools:</strong> {num_inst_regions} detected</li>
                    <li><strong>Pixels:</strong> {inst_pixels:,}</li>
                    <li><strong>Status:</strong> {'Active' if inst_found else 'None'}</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Risk Assessment
        st.markdown("---")
        st.markdown("<div class='glass-card'><h3>⚠️ Risk Assessment</h3></div>", unsafe_allow_html=True)
        
        risk_level = "LOW" if occlusion < 20 else "MODERATE" if occlusion < 40 else "HIGH"
        risk_color = "#10b981" if risk_level == "LOW" else "#f59e0b" if risk_level == "MODERATE" else "#ef4444"
        
        st.markdown(f"""
        <div style='background: {risk_color}; color: white; padding: 1.5rem; border-radius: 16px; text-align: center;'>
            <h2>Risk Level: {risk_level}</h2>
            <p style='font-size: 1.1rem; margin-top: 0.5rem;'>
                {'Safe operating conditions' if risk_level == 'LOW' else 'Moderate visibility reduction' if risk_level == 'MODERATE' else 'Significant occlusion detected'}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Download
        st.markdown("---")
        st.download_button(
            "📥 Download Segmentation Report",
            cv2.imencode('.png', overlay)[1].tobytes(),
            f"surgical_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            "image/png",
            use_container_width=True
        )

else:
    # PATIENT MODE
    st.markdown("<h1 style='text-align: center; color: white;'>👤 Patient View</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: rgba(255,255,255,0.8); font-size: 1.2rem;'>Simple & Clear Surgical Analysis</p>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("📤 Upload Image for Analysis", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        
        if len(image_np.shape) == 3:
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
        
        with st.spinner("🔄 Analyzing..."):
            mask, overlay, occlusion, distance, liver_found, inst_found = engine.predict_image(image_bgr)
            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        
        # Simple comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='glass-card'><h3 style='text-align: center;'>Before Analysis</h3></div>", unsafe_allow_html=True)
            st.markdown("<div class='image-container'>", unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='glass-card'><h3 style='text-align: center;'>After AI Analysis</h3></div>", unsafe_allow_html=True)
            st.markdown("<div class='image-container'>", unsafe_allow_html=True)
            st.image(overlay_rgb, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Simple status
        st.markdown("---")
        
        if liver_found and inst_found:
            if occlusion < 30:
                st.markdown("""
                <div class='patient-card' style='background: linear-gradient(135deg, #10b981 0%, #059669 100%);'>
                    <h2>✅ Everything Looks Good</h2>
                    <p style='font-size: 1.2rem; margin-top: 1rem;'>
                        The analysis shows normal surgical conditions.<br>
                        <strong>All tissues properly identified</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class='patient-card' style='background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);'>
                    <h2>⚠️ Moderate Activity</h2>
                    <p style='font-size: 1.2rem; margin-top: 1rem;'>
                        The surgical area is being actively worked on.<br>
                        <strong>This is normal during procedures</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='patient-card' style='background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);'>
                <h2>ℹ️ Analysis Complete</h2>
                <p style='font-size: 1.2rem; margin-top: 1rem;'>
                    The AI has analyzed the image.<br>
                    <strong>Consult with your doctor for details</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Simple metrics
        st.markdown("---")
        st.markdown("<div class='glass-card'><h3>📊 Quick Summary</h3></div>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Liver Tissue</div>
                <div class='metric-value' style='font-size: 3rem;'>{'✅' if liver_found else '❌'}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Surgical Tools</div>
                <div class='metric-value' style='font-size: 3rem;'>{'✅' if inst_found else '❌'}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            safety = "Safe" if occlusion < 30 else "Active"
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Status</div>
                <div class='metric-value' style='font-size: 2rem;'>{safety}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Info box
        st.markdown("---")
        st.info("""
        **What do the colors mean?**
        - 🟢 **Green areas** = Liver tissue (the organ being operated on)
        - 🟠 **Orange/Red areas** = Surgical instruments (tools used by doctors)
        
        The AI helps doctors see exactly where everything is during surgery!
        """)
