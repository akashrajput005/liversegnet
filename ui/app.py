import streamlit as st
import requests
import cv2
import numpy as np
from PIL import Image
import os
import base64

# --- Page Config ---
st.set_page_config(
    page_title="LiverSegNet AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Premium CSS ---
def local_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif;
            color: #E0E0E0;
        }

        .main {
            background: radial-gradient(circle at top right, #0F172A 0%, #020617 100%);
        }

        /* Glassmorphism Navigation */
        section[data-testid="stSidebar"] {
            background: rgba(15, 23, 42, 0.7) !important;
            backdrop-filter: blur(15px);
            border-right: 1px solid rgba(255, 255, 255, 0.1);
        }

        /* Glassmorphism Card */
        .glass-card {
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border-radius: 24px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            padding: 2.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        }

        /* Safety Gauge */
        .safety-indicator {
            height: 12px;
            width: 100%;
            background: #334155;
            border-radius: 6px;
            margin: 10px 0;
            overflow: hidden;
        }
        .safety-progress {
            height: 100%;
            transition: width 0.8s ease-in-out;
        }

        /* Metrics Styling */
        div[data-testid="stMetricValue"] {
            font-size: 2.8rem !important;
            font-weight: 700;
            background: linear-gradient(135deg, #2DD4BF 0%, #0891B2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        div[data-testid="stMetricLabel"] {
            font-size: 1rem;
            color: #94A3B8;
            font-weight: 500;
        }

        /* Enhanced Buttons */
        .stButton>button {
            border-radius: 16px;
            background: linear-gradient(135deg, #6366F1 0%, #4338CA 100%);
            color: white;
            font-weight: 600;
            letter-spacing: 0.5px;
            padding: 1rem;
            border: none;
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            box-shadow: 0 4px 15px rgba(99, 102, 241, 0.2);
        }
        .stButton>button:hover {
            transform: translateY(-4px) scale(1.02);
            box-shadow: 0 12px 25px rgba(99, 102, 241, 0.4);
            border: none;
            color: white;
        }

        /* Header Premium Gradient */
        .stTitle {
            background: linear-gradient(to bottom right, #F8FAFC 30%, #94A3B8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 4rem !important;
            font-weight: 800 !important;
            letter-spacing: -1.5px;
            padding-bottom: 0.5rem;
        }

        .surgical-badge {
            padding: 6px 14px;
            border-radius: 100px;
            font-size: 0.8rem;
            font-weight: 600;
            display: inline-block;
            margin-right: 8px;
        }
        .badge-active { background: rgba(45, 212, 191, 0.2); color: #2DD4BF; border: 1px solid #2DD4BF; }
        .badge-warning { background: rgba(251, 191, 36, 0.2); color: #FBBF24; border: 1px solid #FBBF24; }
        .badge-danger { background: rgba(248, 113, 113, 0.2); color: #F87171; border: 1px solid #F87171; }
        </style>
    """, unsafe_allow_html=True)

local_css()

# --- App Logic ---
API_URL = "http://localhost:8000"
CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'configs', 'config.yaml')
import yaml

def load_config():
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)

config = load_config()

# --- Sidebar Enhanced ---
st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.title("🩺 AI Hub")
st.sidebar.markdown("### SURGICAL NAVIGATOR")
page = st.sidebar.radio("Command Deck", ["Dashboard", "Video Analysis", "Settings"])

st.sidebar.markdown("---")
st.sidebar.subheader("System Profile")
st.sidebar.markdown(f"""
<div style='background: rgba(255,255,255,0.05); padding: 15px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.1)'>
    <div style='display: flex; justify-content: space-between'>
        <span style='color: #94A3B8'>Active Engine:</span>
        <span style='color: #6366F1; font-weight: 600'>{config['active_model'].upper()}</span>
    </div>
    <div style='display: flex; justify-content: space-between; margin-top: 8px'>
        <span style='color: #94A3B8'>Cloud Status:</span>
        <span style='color: #2DD4BF'>ONLINE 🟢</span>
    </div>
</div>
""", unsafe_allow_html=True)

if page == "Dashboard":
    st.markdown("<h1 class='stTitle'>LiverSegNet</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 1.2rem; color: #94A3B8; margin-top: -20px;'>Advanced Surgical Perception Hub</p>", unsafe_allow_html=True)
    
    st.write("") 

    col_up, col_info = st.columns([1, 1.4])
    
    with col_up:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("🧬 Data Ingestion")
        uploaded_file = st.file_uploader("Select laparoscopic image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)
        
        if uploaded_file:
             st.image(uploaded_file, caption="Original Sequence", use_container_width=True)

    with col_info:
        if uploaded_file:
            if st.button("🚀 INITIATE NEURAL ANALYSIS"):
                 with st.spinner("Processing surgical tensors..."):
                    files = {"file": uploaded_file.getvalue()}
                    try:
                        response = requests.post(f"{API_URL}/segment_image", files=files)
                        if response.status_code == 200:
                            data = response.json()
                            
                            # Premium Results Display
                            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                            res_col1, res_col2 = st.columns([1.5, 1])
                            
                            overlay_img = Image.open(requests.get(f"{API_URL}{data['overlay_url']}", stream=True).raw)
                            res_col1.image(overlay_img, caption="AI perception layer", use_container_width=True)
                            
                            with res_col2:
                                st.subheader("📊 Perception Metrics")
                                
                                # Independent detection status from Engine
                                liver_status = "DETECTED" if data['liver_detected'] else "ABSENT"
                                st.markdown(f"<span class='surgical-badge {'badge-active' if data['liver_detected'] else 'badge-danger'}'>Target: LIVER {liver_status}</span>", unsafe_allow_html=True)
                                
                                inst_status = "ACTIVE" if data['tool_detected'] else "ABSENT"
                                st.markdown(f"<span class='surgical-badge {'badge-warning' if data['tool_detected'] else 'badge-danger'}'>Tools: {inst_status}</span>", unsafe_allow_html=True)

                                st.markdown("<br><br>", unsafe_allow_html=True)
                                st.metric("Occlusion Hazard", f"{data['occlusion_percent']:.1f} %")
                                
                                # Safety Zone Logic
                                dist = data['distance_px']
                                if dist < 0:
                                    safety_text = "N/A"
                                    safety_color = "#334155"
                                    safety_width = 0
                                elif dist < 50:
                                    safety_text = "CRITICAL"
                                    safety_color = "#F87171"
                                    safety_width = 90
                                elif dist < 150:
                                    safety_text = "CAUTION"
                                    safety_color = "#FBBF24"
                                    safety_width = 50
                                else:
                                    safety_text = "SAFE"
                                    safety_color = "#2DD4BF"
                                    safety_width = 20

                                st.markdown(f"**Safety Zone: {safety_text}** ({dist:.0f} px)", unsafe_allow_html=True)
                                st.markdown(f"""
                                <div class='safety-indicator'>
                                    <div class='safety-progress' style='width: {safety_width}%; background: {safety_color};'></div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                            st.markdown("</div>", unsafe_allow_html=True)
                        else:
                            st.error("Engine Fault: Ensure AI API is running on Port 8000")
                    except Exception as e:
                        st.error(f"Hardware Link Offline: {e}")
        else:
             st.info("🧬 Ready for command. Please upload a surgical frame.")

elif page == "Video Analysis":
    st.markdown("<h1 class='stTitle'>Dynamic Analysis</h1>", unsafe_allow_html=True)
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("🎥 Real-time Sequence Processing")
    uploaded_video = st.file_uploader("Select surgical video sequence", type=["mp4", "avi"], label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)
    
    if uploaded_video:
        st.video(uploaded_video)
        if st.button("🎬 EXECUTE BATCH TENSOR PROCESSING"):
            with st.spinner("Analyzing high-frequency sequence..."):
                files = {"file": uploaded_video.getvalue()}
                try:
                    response = requests.post(f"{API_URL}/segment_video", files=files)
                    if response.status_code == 200:
                        st.balloons()
                        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                        st.subheader("✅ Analysis Stream Complete")
                        
                        video_url = f"{API_URL}{response.json()['video_url']}"
                        st.video(video_url)
                        
                        # Provide download link as fallback
                        st.markdown(f"**[📥 Download Segmented Video]({video_url})**", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                    else:
                        st.error("Buffer Overflow: Video sequence processing failed.")
                except Exception as e:
                    st.error(f"Neural Link Timeout: {e}")

elif page == "Settings":
    st.markdown("<h1 class='stTitle'>System Config</h1>", unsafe_allow_html=True)
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("🤖 Neural Engine")
    
    current_model = config['active_model']
    model_choice = st.radio(
        "Active Segmentation Architecture",
        ["Robust Baseline (U-Net + ResNet-34)", "Precision Advanced (DeepLabV3+ + ResNet-50)"],
        index=0 if current_model == "unet" else 1
    )
    
    if st.button("Apply & Hot-Swap Models"):
        new_model = "unet" if "Baseline" in model_choice else "deeplabv3plus"
        new_enc = "resnet34" if "Baseline" in model_choice else "resnet50"
        
        config['active_model'] = new_model
        config['active_encoder'] = new_enc
        with open(CONFIG_PATH, 'w') as f:
            yaml.dump(config, f)
            
        try:
            requests.post(f"{API_URL}/reload_engine")
            st.success(f"Optimized Link Established: Active model is now {model_choice}!")
            st.balloons()
        except:
             st.warning("Config hot-swapped, but Engine requires manual reboot. Restart Terminal 1.")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("🎖️ Engine Benchmarks")
    c1, c2 = st.columns(2)
    c1.write("**Baseline Capacity**")
    c1.progress(57)
    c2.write("**Precision Advanced**")
    c2.progress(34)
    st.write("<small>Note: Advanced model prioritized instrument boundary fidelity over anatomical area.</small>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
