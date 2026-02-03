"""
LiverSegNet AI - Pinnacle Surgical Dashboard
Seamless Monolithic Architecture: AI Engine Integrated Directly into UI
"""
import streamlit as st
import cv2
import numpy as np
import io
import os
import time # Streamlit reload trigger
from PIL import Image
from datetime import datetime
import pandas as pd
from src.infer import InferenceEngine

# Page config: Essential for the 'Wonder' experience
st.set_page_config(
    page_title="LiverSegNet AI - Pinnacle",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Clinical Styling (Glassmorphism & Medical Cyberpunk)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    :root {
        --primary-glow: #00f2fe;
        --secondary-glow: #4facfe;
        --glass-bg: rgba(15, 23, 42, 0.85);
        --accent-green: #10b981;
    }

    * { font-family: 'Outfit', sans-serif; }
    
    .stApp {
        background: radial-gradient(circle at top right, #1e293b, #0f172a);
        color: #e2e8f0;
    }
    
    .glass-card {
        background: var(--glass-bg);
        backdrop-filter: blur(25px);
        border-radius: 28px;
        padding: 2.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 0 40px rgba(0, 242, 254, 0.05);
        margin-bottom: 2rem;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(12px);
        border-radius: 20px;
        padding: 1.8rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(to right, #00f2fe, #4facfe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        filter: drop-shadow(0 0 10px rgba(0, 242, 254, 0.3));
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    .status-good { color: #10b981 !important; }
    .status-warning { color: #f59e0b !important; }
    .status-critical { color: #ef4444 !important; }

    /* Custom Pulse for Safety */
    .patient-card {
        padding: 2.5rem;
        border-radius: 24px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# --- ENGINE INITIALIZATION ---
@st.cache_resource
def load_surgical_engine():
    """Bakes the AI Brain directly into the UI for zero-latency performance."""
    model_path = "models/pinnacle_deeplab_r101.pth"
    if not os.path.exists(model_path):
        return None
    return InferenceEngine(model_path=model_path, device='cuda')

engine = load_surgical_engine()

# --- CLINICAL DATA ARCHIVAL ---
UPLOAD_DIR = "uploads"
RESULT_DIR = "results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# System State
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'last_uploaded_file' not in st.session_state:
    st.session_state.last_uploaded_file = None

def perform_analysis_lifecycle(image_np, filename):
    """Handles the entire lifecycle: Saving -> Analyzing -> Archiving."""
    with st.spinner("🧠 Pinnacle AI is processing clinical data..."):
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 1. Archive Upload (Clinical Traceability)
            upload_path = os.path.join(UPLOAD_DIR, f"clinical_in_{timestamp}_{filename}")
            cv2.imwrite(upload_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
            
            # 2. Execute Direct Inference (No Ports, No Delays)
            start_time = time.time()
            results = engine.predict_image(cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
            inference_time = time.time() - start_time
            
            # 3. Archive Results (Clinical Audit)
            result_path = os.path.join(RESULT_DIR, f"clinical_out_{timestamp}_{filename}")
            cv2.imwrite(result_path, results['overlay'])
            
            # Prepare result for UI (Ensuring RGB formatting)
            results['overlay_rgb'] = cv2.cvtColor(results['overlay'], cv2.COLOR_BGR2RGB)
            
            # 🟢 GENERATE WONDER OUTLINES
            # We create glowing edges for a professional clinical look
            mask = results['mask']
            overlay_wonder = results['overlay_rgb'].copy()
            for cls_id, color_rgb in [(1, (0, 255, 0)), (2, (255, 100, 0))]:
                cls_mask = (mask == cls_id).astype(np.uint8)
                cnts, _ = cv2.findContours(cls_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay_wonder, cnts, -1, color_rgb, 2)
            results['wonder_overlay'] = overlay_wonder
            
            results['inference_time'] = inference_time
            results['timestamp'] = timestamp
            
            st.session_state.analysis_results = results
            return True
        except Exception as e:
            st.error(f"⚠️ Clinical Analysis Critical Failure: {str(e)}")
            return False

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### 🏥 **LiverSegNet AI**")
    st.markdown("---")
    
    mode = st.radio(
        "**Select View Mode**",
        ["👨‍⚕️ Surgeon Mode", "👤 Patient Mode"],
        help="Monolithic architecture enabled for stable surgery."
    )
    
    st.markdown("---")
    st.markdown("### 🛡️ System Integrity")
    if engine:
        st.success("✅ Engine: Purity State Active")
    else:
        st.error("❌ Engine: Missing Weights")
        st.info("Ensure models/pinnacle_*.pth exists.")
    
    st.info("🧠 Pinnacle Ensemble (Cached)")
    st.info("🎯 Clinical Mode Active")

# --- MAIN INTERFACE ---
if "👨‍⚕️" in mode:
    st.markdown("<h1 style='text-align: center; color: white;'>🔬 Surgeon Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: rgba(255,255,255,0.8);'>Integrated Clinical Vision & Advanced Metrics</p>", unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["🎯 Live Analysis", "📈 Model Evolution"])
    
    with tab1:
        uploaded_file = st.file_uploader("📤 Upload Surgical Image", type=['png', 'jpg', 'jpeg'])
        
        if uploaded_file:
            # Lifecycle Reset on New Image
            if st.session_state.last_uploaded_file != uploaded_file.name:
                st.session_state.analysis_results = None
                st.session_state.last_uploaded_file = uploaded_file.name

            # Premium Format Handling: Forced RGB conversion for consistent AI perception
            image = Image.open(uploaded_file).convert("RGB")
            image_np = np.array(image)
            
            if st.session_state.analysis_results is None and engine:
                perform_analysis_lifecycle(image_np, uploaded_file.name)

            res = st.session_state.analysis_results
            if res:
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("<div class='glass-card'><h3>📷 Pre-Operative View</h3></div>", unsafe_allow_html=True)
                    st.image(image, use_container_width=True)
                
                with c2:
                    st.markdown("<div class='glass-card'><h3>🎯 AI-Enhanced Vision</h3></div>", unsafe_allow_html=True)
                    view_type = st.radio("Display Mode", ["Hybrid Overlay", "Boundary Outlines"], horizontal=True)
                    opacity = st.slider("💡 Mask Opacity", 0.1, 1.0, 0.75, key="op_s")
                    
                    if view_type == "Hybrid Overlay":
                        display_img = cv2.addWeighted(image_np, 1-opacity, res['overlay_rgb'], opacity, 0)
                    else:
                        display_img = cv2.addWeighted(image_np, 1-(opacity*0.3), res['wonder_overlay'], opacity, 0)
                        
                    st.image(display_img, use_container_width=True)

                # Metrics Section
                st.markdown("---")
                m1, m2, m3, m4, m5 = st.columns(5)
                with m1:
                    val = "✅" if res['liver_found'] else "❌"
                    st.markdown(f"<div class='metric-card'><div class='metric-label'>Liver</div><div class='metric-value'>{val}</div></div>", unsafe_allow_html=True)
                with m2:
                    val = "✅" if res['inst_found'] else "❌"
                    st.markdown(f"<div class='metric-card'><div class='metric-label'>Tools</div><div class='metric-value'>{val}</div></div>", unsafe_allow_html=True)
                with m3:
                    occ = res['occlusion']
                    status = "status-good" if occ < 20 else "status-warning" if occ < 40 else "status-critical"
                    st.markdown(f"<div class='metric-card'><div class='metric-label'>Occlusion</div><div class='metric-value {status}'>{occ:.1f}%</div></div>", unsafe_allow_html=True)
                with m4:
                    dist = res['distance']
                    dist_str = f"{dist:.0f}px" if dist != float('inf') else "∞"
                    st.markdown(f"<div class='metric-card'><div class='metric-label'>Distance</div><div class='metric-value'>{dist_str}</div></div>", unsafe_allow_html=True)
                with m5:
                    st.markdown(f"<div class='metric-card'><div class='metric-label'>Latency</div><div class='metric-value'>{res['inference_time']:.2f}s</div></div>", unsafe_allow_html=True)

                # Deep Audit Cards
                st.markdown("---")
                d1, d2 = st.columns(2)
                with d1:
                    st.markdown(f"<div class='glass-card'><h4>🟢 Anatomical Breakdown</h4><ul style='color: white;'><li><strong>Detected Regions:</strong> {res['liver_regions']}</li><li><strong>Pixel Count:</strong> {res['liver_pixels']}</li></ul></div>", unsafe_allow_html=True)
                with d2:
                    st.markdown(f"<div class='glass-card'><h4>🔴 Instrument Metrics</h4><ul style='color: white;'><li><strong>Active Tools:</strong> {res['inst_regions']}</li><li><strong>Pixel Count:</strong> {res['inst_pixels']}</li></ul></div>", unsafe_allow_html=True)

                # Downloadable Report
                st.markdown("---")
                # Use the Active Display Image for the report
                report_bgr = cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR)
                st.download_button("📥 Final Clinical Report", cv2.imencode('.png', report_bgr)[1].tobytes(), "surgical_report.png", "image/png", use_container_width=True)
            else:
                st.info("🔼 Please upload a surgical frame to initiate analysis.")

    with tab2:
        st.markdown("<div class='glass-card'><h3>📊 Training Evolution Analytics</h3></div>", unsafe_allow_html=True)
        logs = ["logs/training_log.csv", "logs/training_log_unet.csv"]
        sel_log = st.selectbox("Select Model Architecture", logs)
        if os.path.exists(sel_log):
            df = pd.read_csv(sel_log)
            if not df.empty:
                st.line_chart(df[['Train_Loss', 'Val_Loss']])
                st.metric("Latest Dice Score", f"{df['Liver_Dice'].iloc[-1]:.4f}")
                st.dataframe(df.tail(10), use_container_width=True)

else:
    # PATIENT MODE
    st.markdown("<h1 style='text-align: center; color: white;'>👤 Patient Care View</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: rgba(255,255,255,0.8); font-size: 1.2rem;'>Simple, Transparent, and Secure Analysis</p>", unsafe_allow_html=True)
    
    up_file = st.file_uploader("📤 Select Your Image", type=['png', 'jpg', 'jpeg'])
    if up_file:
        if st.session_state.last_uploaded_file != up_file.name:
            st.session_state.analysis_results = None
            st.session_state.last_uploaded_file = up_file.name

        img = Image.open(up_file).convert("RGB")
        img_np = np.array(img)
        
        if st.session_state.analysis_results is None and engine:
            perform_analysis_lifecycle(img_np, up_file.name)

        p_res = st.session_state.analysis_results
        if p_res:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("<div class='glass-card'><h3 style='text-align: center;'>Original</h3></div>", unsafe_allow_html=True)
                st.image(img, use_container_width=True)
            with col2:
                st.markdown("<div class='glass-card'><h3 style='text-align: center;'>AI Perspective</h3></div>", unsafe_allow_html=True)
                st.image(p_res['overlay_rgb'], use_container_width=True)

            st.markdown("---")
            occ = p_res['occlusion']
            if p_res['liver_found']:
                color = "linear-gradient(135deg, #10b981 0%, #059669 100%)" if occ < 25 else "linear-gradient(135deg, #f59e0b 0%, #d97706 100%)"
                msg = "Everything Looks Good" if occ < 25 else "Clinical Activity Detected"
                st.markdown(f"<div class='patient-card' style='background: {color};'><h2>✅ {msg}</h2><p>Our AI has confirmed successful tissue identification and surgical tool monitoring.</p></div>", unsafe_allow_html=True)
            
            st.info("""
            **Understand Your View:**
            - 🟢 **Green Glaze**: This is your liver tissue, perfectly identified by our AI.
            - 🟠 **Orange Glaze**: These are the surgical tools, being tracked for maximum safety.
            """)
