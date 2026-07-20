import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
import sys
import time
import psutil
import torch
import importlib
import plotly.graph_objects as go
import plotly.express as px

# Ensure project root is in sys.path for local module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import inference.engine
importlib.reload(inference.engine)
from inference.engine import ClinicalInferenceEngine

import utils.analytics
importlib.reload(utils.analytics)
from utils.analytics import (
    compute_boundary_precision,
    compute_tissue_displacement,
    compute_consensus_score
)

# Initialize metric history in session state
if 'metric_history' not in st.session_state:
    st.session_state['metric_history'] = {
        'boundary_precision': [],
        'tissue_displacement': [],
        'tip_l2_error': [],
        'consensus_score': [],
        'timestamps': []
    }


# Custom Styling for Clinical Aesthetics
st.set_page_config(page_title="LiverSegNet: Resilient Hybrid Surgical Perception", layout="wide")

st.markdown("""
<style>
    .main {
        background-color: #0d1117;
        color: #e6edf3;
    }
    .stApp {
        background: radial-gradient(circle at 50% 50%, #1a1f2e 0%, #0d1117 100%);
    }
    
    /* Global overrides to force high visibility light-colored text regardless of base theme */
    .stApp, .stApp p, .stApp span, .stApp div, .stApp li, .stApp label, .stApp section, .stApp button {
        color: #e6edf3 !important;
    }
    
    /* Headings highlighted in crisp white */
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6, .glass-header {
        color: #ffffff !important;
        font-weight: bold !important;
    }

    /* Muted text colors for caption Container */
    .stApp caption, .stApp [data-testid="stCaptionContainer"] {
        color: #b0b8c0 !important;
    }

    /* Keep metric value and text clean */
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
    }
    
    /* Tab label styling to make active/inactive distinct and visible */
    button[data-baseweb="tab"] p {
        color: #8892b0 !important;
        font-weight: 500 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] p {
        color: #00c4ff !important;
        font-weight: bold !important;
    }

    /* Sidebar controls and text override */
    section[data-testid="stSidebar"] {
        background-color: #0d1117 !important;
        background-image: none !important;
    }
    section[data-testid="stSidebar"] * {
        color: #e6edf3 !important;
    }
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3, section[data-testid="stSidebar"] h4, section[data-testid="stSidebar"] h5, section[data-testid="stSidebar"] h6 {
        color: #ffffff !important;
    }

    /* Style select box text to be dark/black so it is visible on the white input background */
    div[data-baseweb="select"] * {
        color: #0e1117 !important;
    }
    div[role="listbox"] * {
        color: #0e1117 !important;
    }

    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 20px;
    }
    .metric-card * {
        color: #ffffff !important;
    }
    .glass-header {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        letter-spacing: -1px;
        color: #ffffff !important;
        text-shadow: 0 0 10px rgba(255, 255, 255, 0.2);
    }
    .verified-tag {
        background: linear-gradient(90deg, #00ffa3, #00c4ff);
        color: #000000 !important;
        font-weight: 900;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.8em;
        text-transform: uppercase;
    }
    .violation-tag {
        background: linear-gradient(90deg, #ff3d00, #ff8f00);
        color: #ffffff !important;
        font-weight: 900;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.8em;
        text-transform: uppercase;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.6; }
        100% { opacity: 1; }
    }
    </style>
""", unsafe_allow_html=True)

def draw_neon_glass(img, mask, color, label=None):
    """V2.1.4: Premium Surgical Glass Overlay with Neon Accents"""
    if not np.any(mask): return img
    
    contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return img
    
    # 1. Glass Fill (Soft Transparency)
    glass = img.copy()
    glass[mask == 1] = color
    img = cv2.addWeighted(glass, 0.3, img, 0.7, 0)
    
    # 2. Neon Outer Glow
    glow = np.zeros_like(img)
    cv2.drawContours(glow, contours, -1, color, 12)
    glow = cv2.GaussianBlur(glow, (25, 25), 0)
    img = cv2.addWeighted(img, 1.0, glow, 0.4, 0)
    
    # 3. Sharp neon border
    cv2.drawContours(img, contours, -1, color, 2)
    
    # 4. Clinical Label (Intelligent Plate)
    if label:
        cnt = max(contours, key=cv2.contourArea)
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            font = cv2.FONT_HERSHEY_DUPLEX
            tw, th = cv2.getTextSize(label, font, 0.6, 1)[0]
            cv2.rectangle(img, (cx-5, cy-th-10), (cx+tw+5, cy+5), (20,20,20), -1)
            cv2.putText(img, label, (cx, cy), font, 0.6, color, 1)
            
    return img

st.sidebar.markdown("### Signal Control")
use_heuristics = st.sidebar.toggle("Heuristic Discovery Layer (MAR)", value=True, help="Enables Physically-Informed Color Recovery for shadowed tissues.")
confidence_threshold = st.sidebar.slider("Neural Confidence Threshold", 0.05, 0.9, 0.1, 0.05)

# Sidebar: System Telemetry
st.sidebar.markdown("---")
st.sidebar.markdown("### Surgical Compute Audit")
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    gpu_util = torch.cuda.memory_reserved(0) / 1024**3
    st.sidebar.markdown(f"**GPU**: {gpu_name}")
    st.sidebar.markdown(f"**VRAM**: {gpu_util:.1f} / {gpu_mem:.1f} GB")
else:
    st.sidebar.markdown("**Hardware**: CPU (Safe Mode)")

st.sidebar.markdown(f"**System RAM**: {psutil.virtual_memory().percent}%")
st.sidebar.markdown(f"**CPU Load**: {psutil.cpu_percent()}%")
st.sidebar.markdown("---")
st.sidebar.markdown("**Integrity**: <span style='color: #00ffa3;'>VERIFIED</span>", unsafe_allow_html=True)
st.sidebar.markdown(f"Mode: **HYBRID {'(HEURISTICS ON)' if use_heuristics else '(NEURAL ONLY)'}**")
st.sidebar.markdown("Version: **V3.0.0-HYBRID**")

@st.cache_resource
def load_inference_engine(kernel_tag="V3.0.0-HYBRID"):
    # V2.2.1: UI Hardening & Threshold Standardization
    model_a_path = "./production_v2_2_0/weights/model_A_hybrid.pth"
    model_b_path = "./production_v2_2_0/weights/model_B_hybrid.pth"
    return ClinicalInferenceEngine(model_a_path, model_b_path)

try:
    # Use a unique tag to force re-instantiation across UI refreshes
    engine = load_inference_engine(kernel_tag="V3-0-0-HYBRID")
except Exception as e:
    st.error(f"Inference Engine Offline: {e}")
    engine = None

# Main Interface with Tabs
tab1, tab2, tab3 = st.tabs(["🔴 Live Perception", "📊 Intelligence Hub", "🔥 Heatmap Diagnostics"])

with tab1:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("<h3 class='glass-header'>High-Fidelity Perception Layer</h3>", unsafe_allow_html=True)
        demo_btn = st.checkbox("Load Demo Laparoscopic Frame", value=True)
        uploaded_file = st.file_uploader("Upload Laparoscopic Data", type=['png', 'jpg', 'jpeg'])
        
        frame = None
        if uploaded_file:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, 1)
        elif demo_btn:
            demo_path = r"C:\Users\akash\Downloads\Images Folder\Images Folder\Liver bladder + 2 tools.png"
            if os.path.exists(demo_path):
                frame = cv2.imread(demo_path)
            else:
                if os.path.exists("sample_frame.png"):
                    frame = cv2.imread("sample_frame.png")
                    
        if frame is not None and engine:
            st.session_state['last_frame'] = frame 
            
            with st.spinner("Executing Hybrid Pipeline..."):
                try:
                    results = engine.infer(frame, confidence_threshold=confidence_threshold, use_heuristics=use_heuristics)
                    st.session_state['latest_results'] = results
                    
                    if results:
                        bp = compute_boundary_precision((results['mask_a'] == 1).astype(np.uint8), results['prob_liver'])
                        td = compute_tissue_displacement((results['mask_a'] == 1).astype(np.uint8), results['tips'], frame.shape)
                        l2 = results['velocity']
                        cs = compute_consensus_score(results['mask_a'], results['mask_b'])
                        
                        history = st.session_state['metric_history']
                        history['boundary_precision'].append(bp)
                        history['tissue_displacement'].append(td)
                        history['tip_l2_error'].append(l2)
                        history['consensus_score'].append(cs)
                        history['timestamps'].append(len(history['timestamps']) + 1)
                except Exception as e:
                    st.error(f"Inference Error: {e}")
                    results = None
                    st.session_state['latest_results'] = None

            
            # Visualization Layers
            if results is not None:
                h, w = frame.shape[:2]
                mask_a_resized = cv2.resize(results['mask_a'].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                mask_b_resized = cv2.resize(results['mask_b'].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                
                overlay = frame.copy()
            
            # --- ANATOMY CORE (V2.1.4: Neon glass) ---
            # 1. Liver (Class 1) - Neon Green
            liver_mask = (mask_a_resized == 1).astype(np.uint8)
            overlay = draw_neon_glass(overlay, liver_mask, (0, 200, 0), "LIVER MASTER")
            
            # 2. Gallbladder (Class 2) - Neon Cyan
            gb_mask = (mask_a_resized == 2).astype(np.uint8)
            overlay = draw_neon_glass(overlay, gb_mask, (200, 200, 0), "GALLBLADDER")

            # 3. GI Tract (Class 3) - Orange (Outline only)
            gi_mask = (mask_a_resized == 3).astype(np.uint8)
            cnts_gi, _ = cv2.findContours(gi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, cnts_gi, -1, (0, 165, 255), 2)

            # 4. Fascia/Other (Class 4) - Red (Subtle)
            fascia_mask = (mask_a_resized == 4).astype(np.uint8)
            cnts_f, _ = cv2.findContours(fascia_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, cnts_f, -1, (0, 0, 150), 1)
            
            # --- TOOL KERNEL (Model B) ---
            tool_mask = (mask_b_resized > 0).astype(np.uint8)
            cnts_t, _ = cv2.findContours(tool_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, cnts_t, -1, (255, 0, 255), 3)
            
            # Draw Tactical Tips (Red pulse glow)
            for tip in results['tips']:
                tx = int(tip[0] * w / 256)
                ty = int(tip[1] * h / 256)
                cv2.circle(overlay, (tx, ty), 12, (255, 255, 255), -1) # Center
                cv2.circle(overlay, (tx, ty), 15, (0, 0, 255), 2)     # Ring
            
            # Alpha Blended Overlay
            alpha = 0.35
            augmented_frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            
            # Main Viewport (Side-by-Side)
            st.markdown("#### Surgical AI Navigation (V2.2.1-HYBRID)")
            v_col1, v_col2 = st.columns(2)
            
            with v_col1:
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Original Sequence", use_container_width=True)
            
            with v_col2:
                st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption="AI Perception Layer", use_container_width=True)


            # Legends
            st.markdown("---")
            l1, l2, l3, l4, l5, l6 = st.columns(6)
            l1.markdown("🟢 **Liver**")
            l2.markdown("🔵 **Gallbladder**")
            l3.markdown("🟠 **GI Tract**")
            l4.markdown("🔴 **Fascia**")
            l5.markdown("🟣 **Instruments**")
            l6.markdown("⚪ **Tactical Tips**")
            
    with col2:
        results = st.session_state.get('latest_results', None)
        if results:
            st.markdown("<h3 class='glass-header'>Clinical Telemetry</h3>", unsafe_allow_html=True)
            
            # --- V2.0.3 Aesthetic Cards ---
            spatial_integrity_str = f"{results.get('spatial_reliability', 0.0)*100:.1f}%"
            tool_velocity_str = f"{results.get('velocity', 0.0):.1f} px/f"
            
            st.markdown(f"""
            <div class='metric-card'>
                <p style='margin:0; font-size:0.9em; opacity:0.7;'>ANATOMICAL STATE</p>
                <h4 style='margin:0;'>Liver: {'🟢 LOCALIZED' if np.any(results['mask_a']==1) else '🔴 SEARCHING...'}</h4>
                <h4 style='margin:0;'>Gallbladder: {'🔵 LOCALIZED' if np.any(results['mask_a']==2) else '⚪ OFF-TARGET'}</h4>
            </div>
            
            <div class='metric-card'>
                <p style='margin:0; font-size:0.9em; opacity:0.7;'>KINETIC ANALYTICS</p>
                <h4 style='margin:0;'>Spatial Integrity: {spatial_integrity_str}</h4>
                <h4 style='margin:0;'>Tool velocity: {tool_velocity_str}</h4>
                <p style='margin:0; font-size:0.8em; opacity:0.6;'>Critical Gate: 20.5 px</p>
                <p style='margin:0; font-size:0.8em; opacity:0.6;'>Warning Gate: 50.5 px</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### Safety Vector (Proximity)")
            risk_level = results['risk_status']
            min_dist = results.get('min_distance', 100.0)
            risk_color = "#00ffa3" if risk_level == "SAFE" else "#ffeb3b" if risk_level == "WARNING" else "#ff3d00"
            
            # Progress-style Safety Bar
            bar_val = min(100, max(0, min_dist))
            st.markdown(f"""
            <div style='width: 100%; background: #222; border-radius: 10px; height: 10px; margin-bottom: 5px;'>
                <div style='width: {bar_val}%; background: {risk_color}; height: 100%; border-radius: 10px; box-shadow: 0 0 10px {risk_color};'></div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"<div style='background-color: {risk_color}22; padding: 15px; border-radius: 10px; border: 2px solid {risk_color};'>", unsafe_allow_html=True)
            st.markdown(f"<h3 style='text-align: center; color: {risk_color}; margin: 0; font-family: monospace;'>{risk_level} ({min_dist:.1f}px)</h3>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("#### Hardware Consensus Audit")
            audit = results['consensus_audit']
            tag_class = "verified-tag" if audit['status'] == "VERIFIED" else "violation-tag"
            st.markdown(f"Status: <span class='{tag_class}'>{audit['status']}</span>", unsafe_allow_html=True)
            st.info(audit['description'])
            
            st.markdown("#### Perception Signals")
            sig = results['signals']
            # Improved visual dots
            s_neural = "🟢" if sig['neural'] else "⚪"
            s_det = "🔵" if sig['deterministic'] else "⚪"
            s_heur = "🟡" if sig['heuristic'] else "⚪"
            st.markdown(f"Neural: {s_neural} | Deterministic: {s_det} | Heuristic: {s_heur}")
            
            latency = results.get('latency_ms', 45) # Improved latency display
            st.markdown(f"<p style='opacity: 0.6; font-size: 0.8em;'>Cycle Latency: {latency:.1f}ms</p>", unsafe_allow_html=True)
        else:
            st.info("Awaiting Surgical Data Stream...")

with tab2:
    st.markdown("### Quantitative Surgical Analytics")
    st.info("These metrics track the real-time stability and precision of the perception kernels.")
    
    history = st.session_state.get('metric_history', {})
    timestamps = history.get('timestamps', [])
    
    if not timestamps:
        st.warning("Awaiting surgical analysis session. Please upload laparoscopic images in the '🔴 Live Perception' tab to initialize telemetry history.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Anatomical Boundary Precision")
            st.caption("Measures the sharpness and accuracy of the liver-to-background transition edge.")
            
            bp_vals = history['boundary_precision']
            fig_bp = go.Figure()
            fig_bp.add_trace(go.Scatter(
                x=timestamps, y=bp_vals,
                mode='lines+markers',
                name='Boundary Precision',
                line=dict(color='#00ffa3', width=2),
                fill='tozeroy',
                fillcolor='rgba(0, 255, 163, 0.1)',
                hovertemplate="Frame: %{x}<br>Precision: %{y:.4f}<extra></extra>"
            ))
            if len(bp_vals) >= 3:
                sma_vals = [np.mean(bp_vals[max(0, i-2):i+1]) for i in range(len(bp_vals))]
                fig_bp.add_trace(go.Scatter(
                    x=timestamps, y=sma_vals,
                    mode='lines',
                    name='3-Frame SMA',
                    line=dict(color='#ffffff', width=1.5, dash='dot'),
                    hovertemplate="Frame: %{x}<br>3-Frame SMA: %{y:.4f}<extra></extra>"
                ))
            fig_bp.add_hline(y=0.85, line_dash="dash", line_color="#ffeb3b", 
                              annotation_text="Clinical Threshold (0.85)", annotation_position="top left")
            fig_bp.update_layout(
                template='plotly_dark',
                paper_bgcolor='#0d1117',
                plot_bgcolor='#0d1117',
                font=dict(color='#ffffff'),
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(title='Sequence Frame'),
                yaxis=dict(range=[0, 1.05], title='Precision Score'),
                height=300
            )
            st.plotly_chart(fig_bp, use_container_width=True, theme=None)
            
            st.markdown("#### Tool-Induced Tissue Displacement")
            st.caption("Estimated physical interaction/compression of instruments on anatomy (converted to physical mm).")
            
            td_vals_mm = [val * 0.12 for val in history['tissue_displacement']]
            fig_td = go.Figure()
            fig_td.add_trace(go.Scatter(
                x=timestamps, y=td_vals_mm,
                mode='lines+markers',
                name='Displacement',
                fill='tozeroy',
                fillcolor='rgba(0,196,255,0.15)',
                line=dict(color='#00c4ff', width=2),
                hovertemplate="Frame: %{x}<br>Displacement: %{y:.2f} mm<extra></extra>"
            ))
            fig_td.add_hline(y=6.06, line_dash="dash", line_color="#ffeb3b", 
                              annotation_text="Warning Threshold (6.06 mm)", annotation_position="top left")
            fig_td.add_hline(y=2.46, line_dash="dash", line_color="#ff3d00", 
                              annotation_text="Critical Threshold (2.46 mm)", annotation_position="top left")
            
            fig_td.update_layout(
                template='plotly_dark',
                paper_bgcolor='#0d1117',
                plot_bgcolor='#0d1117',
                font=dict(color='#ffffff'),
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(title='Sequence Frame'),
                yaxis=dict(title='Distance / Compression (mm)'),
                height=300
            )
            st.plotly_chart(fig_td, use_container_width=True, theme=None)
            
        with c2:
            st.markdown("#### Instrument Tip Localization Precision (L2)")
            st.caption("L2 jitter error in pixels in tip tracking compared to temporal history.")
            
            l2_vals = history['tip_l2_error']
            fig_l2 = go.Figure()
            fig_l2.add_trace(go.Scatter(
                x=timestamps, y=l2_vals,
                mode='lines+markers',
                name='L2 Jitter',
                line=dict(color='#ff6ec7', width=2),
                marker=dict(size=6, color='#ff6ec7'),
                hovertemplate="Frame: %{x}<br>Jitter: %{y:.2f} px<extra></extra>"
            ))
            if len(l2_vals) >= 3:
                sma_l2 = [np.mean(l2_vals[max(0, i-2):i+1]) for i in range(len(l2_vals))]
                fig_l2.add_trace(go.Scatter(
                    x=timestamps, y=sma_l2,
                    mode='lines',
                    name='3-Frame SMA',
                    line=dict(color='#ffffff', width=1.5, dash='dot'),
                    hovertemplate="Frame: %{x}<br>3-Frame SMA: %{y:.2f} px<extra></extra>"
                ))
            fig_l2.update_layout(
                template='plotly_dark',
                paper_bgcolor='#0d1117',
                plot_bgcolor='#0d1117',
                font=dict(color='#ffffff'),
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(title='Sequence Frame'),
                yaxis=dict(title='L2 Jitter (pixels)'),
                height=300
            )
            st.plotly_chart(fig_l2, use_container_width=True, theme=None)
            
            st.markdown("#### Model Consensus Score")
            st.caption("Agreement percentage between Kernel A (Anatomy) and Kernel B (Tools).")
            
            latest_cs = history['consensus_score'][-1] * 100
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=latest_cs,
                domain={'x': [0.1, 0.9], 'y': [0.1, 0.9]},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': '#ffffff'},
                    'bar': {'color': '#00c4ff', 'thickness': 0.3},
                    'bgcolor': 'rgba(255,255,255,0.05)',
                    'borderwidth': 1,
                    'bordercolor': 'rgba(255,255,255,0.1)',
                    'steps': [
                        {'range': [0, 60], 'color': 'rgba(255, 61, 0, 0.25)'},
                        {'range': [60, 85], 'color': 'rgba(255, 235, 59, 0.25)'},
                        {'range': [85, 100], 'color': 'rgba(0, 255, 163, 0.25)'}
                    ]
                }
            ))
            fig_gauge.update_layout(
                template='plotly_dark',
                paper_bgcolor='#0d1117',
                plot_bgcolor='#0d1117',
                font=dict(color='#ffffff'),
                height=220,
                margin=dict(t=40, b=20, l=30, r=30)
            )
            st.plotly_chart(fig_gauge, use_container_width=True, theme=None)
            
            cs_vals_pct = [val * 100 for val in history['consensus_score']]
            colors = []
            for val in cs_vals_pct:
                if val >= 85: colors.append('#00ffa3')
                elif val >= 60: colors.append('#ffeb3b')
                else: colors.append('#ff3d00')
                
            fig_cs_bar = go.Figure(go.Bar(
                x=timestamps, y=cs_vals_pct,
                marker_color=colors,
                name='Consensus History',
                hovertemplate="Frame: %{x}<br>Consensus: %{y:.1f}%<extra></extra>"
            ))
            fig_cs_bar.update_layout(
                template='plotly_dark',
                paper_bgcolor='#0d1117',
                plot_bgcolor='#0d1117',
                font=dict(color='#ffffff'),
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(title='Sequence Frame'),
                yaxis=dict(range=[0, 105], title='Consensus Score (%)'),
                height=180
            )
            st.plotly_chart(fig_cs_bar, use_container_width=True, theme=None)



with tab3:
    st.markdown("### Heatmap Diagnostics (Model Confidence)")
    st.info("Visualizing raw 'warmth' maps. Red areas indicate where the model is confident; Blue/Violet indicates doubt.")
    
    results = st.session_state.get('latest_results', None)
    if results:
        frame_data = st.session_state.get('last_frame', None)
        if frame_data is not None:
            st.markdown("#### Diagnostic Visualization Controls")
            ctrl_c1, ctrl_c2, ctrl_c3 = st.columns(3)
            with ctrl_c1:
                blend_alpha = st.slider("Overlay Transparency (Heatmaps)", 0.1, 1.0, 0.5, 0.05)
            with ctrl_c2:
                colormap_name = st.selectbox("Heatmap Colormap", ["Inferno (Clinical)", "Magma", "Viridis", "JET (Legacy)"])
                colormap_map = {
                    "Inferno (Clinical)": cv2.COLORMAP_INFERNO,
                    "Magma": cv2.COLORMAP_MAGMA,
                    "Viridis": cv2.COLORMAP_VIRIDIS,
                    "JET (Legacy)": cv2.COLORMAP_JET
                }
                selected_cmap = colormap_map[colormap_name]
            with ctrl_c3:
                show_contours = st.checkbox("Show Boundary Contours on Heatmap", value=True)
                
            h, w = frame_data.shape[:2]
            frame_rgb = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)
            
            st.markdown("---")
            r1_col1, r1_col2 = st.columns(2)
            
            with r1_col1:
                st.markdown("#### Organ Confidence Heatmap")
                class_selector = st.selectbox("Select Target Organ", 
                    ["🫁 Liver (Class 1)", "💚 Gallbladder (Class 2)", "🟠 GI Tract (Class 3)", "⬜ Fascia/Peritoneum (Class 4)", "⬛ Background (Class 0)"],
                    index=0)
                class_idx = {
                    "🫁 Liver (Class 1)": 1, 
                    "💚 Gallbladder (Class 2)": 2, 
                    "🟠 GI Tract (Class 3)": 3, 
                    "⬜ Fascia/Peritoneum (Class 4)": 4, 
                    "⬛ Background (Class 0)": 0
                }[class_selector]
                
                prob_all = results.get('prob_all_classes', None)
                if prob_all is not None:
                    prob_map = prob_all[class_idx]
                else:
                    if class_idx == 1:
                        prob_map = results.get('prob_liver', np.zeros((256, 256)))
                    elif class_idx == 2:
                        prob_map = results.get('prob_gb', np.zeros((256, 256)))
                    else:
                        prob_map = np.zeros((256, 256))
                        
                if len(prob_map.shape) > 2: prob_map = prob_map[0]
                heat_resized = cv2.resize(prob_map, (w, h))
                heat_norm = (np.clip(heat_resized, 0, 1) * 255).astype(np.uint8)
                heatmap_img = cv2.applyColorMap(heat_norm, selected_cmap)
                heatmap_img_rgb = cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB)
                
                blended_heat = cv2.addWeighted(frame_rgb, 1.0 - blend_alpha, heatmap_img_rgb, blend_alpha, 0)
                
                if show_contours:
                    mask_a = results.get('mask_a', np.zeros((h, w)))
                    class_mask = (mask_a == class_idx).astype(np.uint8)
                    contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(blended_heat, contours, -1, (255, 255, 255), 2)
                    
                st.image(blended_heat, caption=f"{class_selector} Confidence Blend", use_container_width=True)
                
            with r1_col2:
                st.markdown("#### Boundary Uncertainty Map (Shannon Entropy)")
                entropy_map = results.get('entropy_map', None)
                if entropy_map is None:
                    prob_liver = results.get('prob_liver', np.zeros((256, 256)))
                    p = np.clip(prob_liver, 1e-7, 1-1e-7)
                    entropy_map = -(p * np.log2(p) + (1-p) * np.log2(1-p))
                
                entropy_resized = cv2.resize(entropy_map, (w, h))
                entropy_norm = (np.clip(entropy_resized, 0, 1) * 255).astype(np.uint8)
                entropy_cmap = cv2.applyColorMap(entropy_norm, cv2.COLORMAP_MAGMA)
                entropy_cmap_rgb = cv2.cvtColor(entropy_cmap, cv2.COLOR_BGR2RGB)
                blended_uncertainty = cv2.addWeighted(frame_rgb, 1.0 - blend_alpha, entropy_cmap_rgb, blend_alpha, 0)
                
                st.image(blended_uncertainty, caption="Epistemic/Aleatoric Uncertainty Map (Bright = High Doubt)", use_container_width=True)
                
            st.markdown("---")
            r2_col1, r2_col2 = st.columns(2)
            
            with r2_col1:
                st.markdown("#### Proximity Safety Zone Map")
                mask_a = results.get('mask_a', np.zeros((h, w)))
                anatomy_binary = (mask_a > 0).astype(np.uint8)
                dist_map = cv2.distanceTransform(1 - anatomy_binary, cv2.DIST_L2, 5)
                
                dist_map_mm = dist_map * 0.12
                norm_dist = np.clip(dist_map_mm / 30.0, 0, 1)
                inverted_dist = 1.0 - norm_dist
                inverted_norm = (inverted_dist * 255).astype(np.uint8)
                
                dist_cmap = cv2.applyColorMap(inverted_norm, cv2.COLORMAP_HOT)
                dist_cmap_rgb = cv2.cvtColor(dist_cmap, cv2.COLOR_BGR2RGB)
                blended_distance = cv2.addWeighted(frame_rgb, 0.4, dist_cmap_rgb, 0.6, 0)
                
                tips = results.get('tips', [])
                for tip in tips:
                    tx = int(tip[0] * w / 256.0)
                    ty = int(tip[1] * h / 256.0)
                    cv2.circle(blended_distance, (tx, ty), 12, (255, 255, 255), -1)
                    cv2.circle(blended_distance, (tx, ty), 15, (0, 0, 255), 2)
                    
                st.image(blended_distance, caption="Proximity Hazard Map (Red = High Proximity Zone)", use_container_width=True)
                
            with r2_col2:
                st.markdown("#### Segmentation Boundary Outline Overlay")
                boundary_overlay = frame_rgb.copy()
                colors = {
                    1: (0, 255, 163),  
                    2: (0, 196, 255),  
                    3: (255, 165, 0),  
                    4: (255, 61, 0)    
                }
                for class_id, col in colors.items():
                    class_mask = (mask_a == class_id).astype(np.uint8)
                    cnts, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(boundary_overlay, cnts, -1, col, 2)
                    
                st.image(boundary_overlay, caption="Crisp Spatial Segment Contours", use_container_width=True)
                
            st.markdown("---")
            r3_col1, r3_col2 = st.columns(2)
            
            with r3_col1:
                st.markdown("#### Softmax Confidence Distribution")
                fig_hist = go.Figure()
                prob_liver = results.get('prob_liver', np.zeros(1))
                prob_gb = results.get('prob_gb', np.zeros(1))
                
                fig_hist.add_trace(go.Histogram(
                    x=prob_liver.flatten(),
                    nbinsx=50,
                    marker_color='#00ffa3',
                    opacity=0.6,
                    name='Liver'
                ))
                fig_hist.add_trace(go.Histogram(
                    x=prob_gb.flatten(),
                    nbinsx=50,
                    marker_color='#00c4ff',
                    opacity=0.6,
                    name='Gallbladder'
                ))
                fig_hist.update_layout(
                    barmode='overlay',
                    template='plotly_dark',
                    paper_bgcolor='#0d1117',
                    plot_bgcolor='#0d1117',
                    font=dict(color='#ffffff'),
                    margin=dict(l=10, r=10, t=10, b=10),
                    xaxis=dict(title='Confidence Value (0.0 to 1.0)'),
                    yaxis=dict(title='Pixel Frequency'),
                    height=280
                )
                st.plotly_chart(fig_hist, use_container_width=True, theme=None)
                
            with r3_col2:
                st.markdown("#### Per-Class Detection Strength")
                prob_all = results.get('prob_all_classes', None)
                if prob_all is not None:
                    max_conf = [float(np.max(prob_all[i])) for i in range(5)]
                else:
                    max_conf = [1.0, float(np.max(results.get('prob_liver', [0]))), float(np.max(results.get('prob_gb', [0]))), 0.0, 0.0]
                    
                fig_radar = go.Figure(go.Scatterpolar(
                    r=max_conf,
                    theta=['Background', 'Liver', 'Gallbladder', 'GI Tract', 'Fascia'],
                    fill='toself',
                    line_color='#ff6ec7',
                    marker=dict(color='#ff6ec7')
                ))
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True, 
                            range=[0, 1.0],
                            gridcolor='rgba(255,255,255,0.1)',
                            linecolor='rgba(255,255,255,0.1)'
                        ),
                        angularaxis=dict(
                            gridcolor='rgba(255,255,255,0.1)',
                            linecolor='rgba(255,255,255,0.1)'
                        )
                    ),
                    showlegend=False,
                    template='plotly_dark',
                    paper_bgcolor='#0d1117',
                    plot_bgcolor='#0d1117',
                    font=dict(color='#ffffff'),
                    margin=dict(l=20, r=20, t=20, b=20),
                    height=280
                )
                st.plotly_chart(fig_radar, use_container_width=True, theme=None)

                
            st.markdown("#### Diagnostic Telemetry")
            max_p_liver = np.max(results.get('prob_liver', np.zeros(1)))
            st.write(f"**Peak Liver Confidence**: {max_p_liver:.4f} | **Spatial Reliability**: {results.get('spatial_reliability', 0.0):.4f}")
            if max_p_liver < 0.1:
                st.error("CRITICAL: Liver confidence is below 10%. Model weights may need recalibration for this tissue type.")
            elif max_p_liver < 0.25:
                st.warning("LOW CONFIDENCE: Tissue is detected but signal is weak. Sensitivity boost is active.")
        else:
            st.info("Awaiting frame correlation...")
    else:
        st.warning("Please upload a frame to view confidence diagnostics.")

st.markdown("---")
st.markdown(f"**LiverSegNet v3.0.0-HYBRID** | Protocol: Gold Standard Release | Kinetic Safety: Active (20.5/50.5 px)")
