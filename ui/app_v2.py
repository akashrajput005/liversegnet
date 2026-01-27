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

# --- Page Config ---
st.set_page_config(
    page_title="LiverSegNet AI v2.0",
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

        /* Metric Cards */
        .metric-card {
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(147, 51, 234, 0.1) 100%);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            transition: all 0.3s ease;
        }

        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 24px rgba(59, 130, 246, 0.2);
        }

        /* Model Selection */
        .model-selector {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 1rem;
        }

        /* Status Indicators */
        .status-success {
            background: rgba(34, 197, 94, 0.1);
            border: 1px solid rgba(34, 197, 94, 0.3);
            border-radius: 8px;
            padding: 0.5rem 1rem;
            color: #22C55E;
        }

        .status-warning {
            background: rgba(251, 191, 36, 0.1);
            border: 1px solid rgba(251, 191, 36, 0.3);
            border-radius: 8px;
            padding: 0.5rem 1rem;
            color: #FBBF24;
        }

        .status-danger {
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid rgba(239, 68, 68, 0.3);
            border-radius: 8px;
            padding: 0.5rem 1rem;
            color: #EF4444;
        }

        /* Progress Bar */
        .progress-container {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            height: 8px;
            overflow: hidden;
            margin: 1rem 0;
        }

        .progress-bar {
            height: 100%;
            background: linear-gradient(90deg, #3B82F6 0%, #8B5CF6 100%);
            transition: width 0.3s ease;
        }

        /* Charts Container */
        .chart-container {
            background: rgba(255, 255, 255, 0.02);
            border-radius: 16px;
            padding: 1.5rem;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }

        /* Performance Metrics */
        .performance-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin: 1rem 0;
        }

        .performance-item {
            text-align: center;
            padding: 1rem;
            background: rgba(255, 255, 255, 0.03);
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }

        .performance-value {
            font-size: 2rem;
            font-weight: 600;
            color: #3B82F6;
        }

        .performance-label {
            font-size: 0.875rem;
            color: #9CA3AF;
            margin-top: 0.5rem;
        }

        /* Header Styling */
        .header-gradient {
            background: linear-gradient(135deg, #3B82F6 0%, #8B5CF6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        /* Button Styling */
        .stButton > button {
            background: linear-gradient(135deg, #3B82F6 0%, #8B5CF6 100%);
            border: none;
            border-radius: 12px;
            padding: 0.75rem 2rem;
            font-weight: 600;
            transition: all 0.3s ease;
        }

        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 16px rgba(59, 130, 246, 0.3);
        }

        /* File Uploader */
        .stFileUploader > div {
            background: rgba(255, 255, 255, 0.03);
            border: 2px dashed rgba(255, 255, 255, 0.2);
            border-radius: 16px;
            padding: 2rem;
            text-align: center;
            transition: all 0.3s ease;
        }

        .stFileUploader > div:hover {
            border-color: #3B82F6;
            background: rgba(59, 130, 246, 0.05);
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            background: rgba(255, 255, 255, 0.03);
            border-radius: 12px;
            padding: 0.5rem;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            padding: 0.75rem 1.5rem;
            font-weight: 600;
        }

        /* Expander */
        .streamlit-expanderHeader {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            font-weight: 600;
        }

        .streamlit-expanderContent {
            background: rgba(255, 255, 255, 0.02);
            border-radius: 12px;
        }
        </style>
    """, unsafe_allow_html=True)

# --- API Configuration ---
API_BASE = "http://localhost:8000"

def get_available_models():
    """Get available models from API"""
    try:
        response = requests.get(f"{API_BASE}/models")
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return {"available_models": [], "model_info": {}}

def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_BASE}/health")
        return response.status_code == 200
    except:
        return False

def segment_image(file, model=None, ensemble=False):
    """Segment image using API"""
    try:
        files = {"file": file}
        params = {}
        if model:
            params["model"] = model
        if ensemble:
            params["ensemble"] = True
            
        response = requests.post(f"{API_BASE}/segment_image", files=files, params=params)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Error: {str(e)}")
    return None

def create_metrics_dashboard(metrics):
    """Create comprehensive metrics dashboard"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Liver Area", 
            f"{metrics.get('liver_area_percent', 0):.1f}%",
            delta=f"{metrics.get('liver_regions', 0)} regions"
        )
    
    with col2:
        st.metric(
            "Instrument Area", 
            f"{metrics.get('instrument_area_percent', 0):.1f}%",
            delta=f"{metrics.get('instrument_regions', 0)} regions"
        )
    
    with col3:
        st.metric(
            "Liver Compactness", 
            f"{metrics.get('liver_compactness_mean', 0):.3f}",
            delta="Shape quality"
        )
    
    with col4:
        st.metric(
            "Instrument Compactness", 
            f"{metrics.get('instrument_compactness_mean', 0):.3f}",
            delta="Tool quality"
        )
    
    # Create detailed charts
    if metrics:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Area Distribution", "Region Analysis", "Shape Analysis", "Detection Status"),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "indicator"}]]
        )
        
        # Area distribution
        fig.add_trace(
            go.Bar(
                x=["Liver", "Instrument"],
                y=[metrics.get('liver_area_percent', 0), metrics.get('instrument_area_percent', 0)],
                name="Area %",
                marker_color=['#22C55E', '#F59E0B']
            ),
            row=1, col=1
        )
        
        # Region analysis
        fig.add_trace(
            go.Bar(
                x=["Liver Regions", "Instrument Regions"],
                y=[metrics.get('liver_regions', 0), metrics.get('instrument_regions', 0)],
                name="Count",
                marker_color=['#3B82F6', '#EF4444']
            ),
            row=1, col=2
        )
        
        # Shape analysis
        fig.add_trace(
            go.Scatter(
                x=["Liver", "Instrument"],
                y=[metrics.get('liver_compactness_mean', 0), metrics.get('instrument_compactness_mean', 0)],
                mode='markers+lines',
                name="Compactness",
                marker=dict(size=10, color=['#8B5CF6', '#10B981'])
            ),
            row=2, col=1
        )
        
        # Detection status
        liver_detected = metrics.get('liver_detected', False)
        inst_detected = metrics.get('instrument_detected', False)
        
        fig.add_trace(
            go.Indicator(
                mode="number+gauge+delta",
                value=1 if liver_detected and inst_detected else 0.5 if liver_detected or inst_detected else 0,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Detection Quality"},
                gauge={'axis': {'range': [None, 1]},
                       'bar': {'color': "#3B82F6"},
                       'steps': [{'range': [0, 0.5], 'color': "lightgray"},
                                {'range': [0.5, 1], 'color': "gray"}],
                       'threshold': {'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75, 'value': 0.9}}
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=600,
            showlegend=False,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        
        st.plotly_chart(fig, use_container_width=True)

def create_performance_comparison(individual_results):
    """Create performance comparison chart for ensemble results"""
    if not individual_results:
        return
    
    models = list(individual_results.keys())
    inference_times = [individual_results[m]['inference_time'] for m in models]
    liver_metrics = [individual_results[m]['metrics']['liver_area_percent'] for m in models]
    inst_metrics = [individual_results[m]['metrics']['instrument_area_percent'] for m in models]
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Inference Time (s)", "Liver Detection (%)", "Instrument Detection (%)"),
        specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
    )
    
    fig.add_trace(
        go.Bar(x=models, y=inference_times, name="Time", marker_color='#3B82F6'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=models, y=liver_metrics, name="Liver", marker_color='#22C55E'),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Bar(x=models, y=inst_metrics, name="Instrument", marker_color='#F59E0B'),
        row=1, col=3
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    
    st.plotly_chart(fig, use_container_width=True)

# --- Main Application ---
def main():
    local_css()
    
    # Header
    st.markdown("""
        <div style="text-align: center; padding: 2rem 0;">
            <h1 class="header-gradient">LiverSegNet AI v2.0</h1>
            <p style="color: #9CA3AF; font-size: 1.2rem;">Advanced Surgical Perception with Multi-Model Ensemble</p>
        </div>
    """, unsafe_allow_html=True)
    
    # API Health Check
    api_healthy = check_api_health()
    if not api_healthy:
        st.error("🚨 API is not running! Please start the API server first.")
        st.info("Run: `python ui/app_api_v2.py`")
        return
    
    # Get available models
    models_info = get_available_models()
    available_models = models_info.get("available_models", [])
    model_details = models_info.get("model_info", {})
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
            <div class="glass-card">
                <h3>🎛️ Model Configuration</h3>
            </div>
        """, unsafe_allow_html=True)
        
        # Model Selection
        if available_models:
            selected_model = st.selectbox(
                "Select Model",
                options=available_models,
                format_func=lambda x: model_details.get(x, {}).get('name', x),
                index=0 if 'deeplabv3plus' in available_models else 0
            )
            
            # Show model info
            if selected_model in model_details:
                info = model_details[selected_model]
                st.markdown(f"""
                    <div class="model-selector">
                        <strong>Type:</strong> {info.get('type', 'Unknown')}<br>
                        <strong>Strengths:</strong><br>
                        {' • '.join(info.get('strengths', []))}
                    </div>
                """, unsafe_allow_html=True)
        
        # Ensemble Mode
        use_ensemble = st.checkbox(
            "🔄 Enable Ensemble Mode",
            help="Combine multiple models for better accuracy",
            disabled=len(available_models) < 2
        )
        
        if use_ensemble:
            st.info("Ensemble mode will use all available models with weighted averaging")
        
        # Advanced Settings
        st.markdown("""
            <div class="glass-card">
                <h3>⚙️ Advanced Settings</h3>
            </div>
        """, unsafe_allow_html=True)
        
        show_metrics = st.checkbox("📊 Show Detailed Metrics", value=True)
        show_comparison = st.checkbox("🔍 Show Model Comparison", value=True)
        save_masks = st.checkbox("💾 Save Segmentation Masks", value=False)
    
    # Main Content
    tab1, tab2, tab3 = st.tabs(["📸 Image Segmentation", "🎥 Video Processing", "📈 Model Analytics"])
    
    with tab1:
        st.markdown("""
            <div class="glass-card">
                <h3>📸 Upload Surgical Image</h3>
                <p>Upload a surgical image for real-time segmentation and analysis.</p>
            </div>
        """, unsafe_allow_html=True)
        
        # File Upload
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['png', 'jpg', 'jpeg'],
            help="Supported formats: PNG, JPG, JPEG"
        )
        
        if uploaded_file is not None:
            # Display original image
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Original Image**")
                image = Image.open(uploaded_file)
                st.image(image, use_column_width=True)
            
            # Process button
            if st.button("🚀 Start Segmentation", type="primary"):
                with st.spinner("Processing image..."):
                    result = segment_image(
                        uploaded_file, 
                        model=selected_model if not use_ensemble else None,
                        ensemble=use_ensemble
                    )
                
                if result:
                    with col2:
                        st.markdown("**Segmentation Result**")
                        if 'overlay_url' in result:
                            overlay_response = requests.get(f"{API_BASE}/{result['overlay_url']}")
                            if overlay_response.status_code == 200:
                                overlay_image = Image.open(io.BytesIO(overlay_response.content))
                                st.image(overlay_image, use_column_width=True)
                    
                    # Results Section
                    st.markdown("""
                        <div class="glass-card">
                            <h3>📊 Analysis Results</h3>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Basic Metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Occlusion",
                            f"{result.get('occlusion_percent', 0):.1f}%",
                            delta="Hazard Level"
                        )
                    
                    with col2:
                        distance = result.get('distance_px', -1)
                        st.metric(
                            "Distance",
                            f"{distance:.0f}px" if distance > 0 else "N/A",
                            delta="Safety Margin"
                        )
                    
                    with col3:
                        st.metric(
                            "Liver Detected",
                            "✅ Yes" if result.get('liver_detected') else "❌ No",
                            delta="Anatomy"
                        )
                    
                    with col4:
                        st.metric(
                            "Instruments",
                            "✅ Yes" if result.get('instrument_detected') else "❌ No",
                            delta="Tools"
                        )
                    
                    # Performance Metrics
                    if 'inference_time' in result:
                        st.markdown(f"""
                            <div class="status-success">
                                ⚡ Processing Time: {result['inference_time']:.3f}s | 
                                🤖 Model: {result.get('model_used', 'unknown')}
                            </div>
                        """, unsafe_allow_html=True)
                    
                    # Detailed Metrics
                    if show_metrics and 'metrics' in result:
                        st.markdown("""
                            <div class="chart-container">
                                <h4>📈 Detailed Metrics Dashboard</h4>
                            </div>
                        """, unsafe_allow_html=True)
                        create_metrics_dashboard(result['metrics'])
                    
                    # Model Comparison
                    if show_comparison and use_ensemble and 'individual_results' in result:
                        st.markdown("""
                            <div class="chart-container">
                                <h4>🔍 Model Performance Comparison</h4>
                            </div>
                        """, unsafe_allow_html=True)
                        create_performance_comparison(result['individual_results'])
                    
                    # Download Options
                    if save_masks and 'mask_url' in result:
                        mask_response = requests.get(f"{API_BASE}/{result['mask_url']}")
                        if mask_response.status_code == 200:
                            st.download_button(
                                label="📥 Download Segmentation Mask",
                                data=mask_response.content,
                                file_name="segmentation_mask.png",
                                mime="image/png"
                            )
    
    with tab2:
        st.markdown("""
            <div class="glass-card">
                <h3>🎥 Video Processing</h3>
                <p>Upload surgical video for frame-by-frame segmentation.</p>
            </div>
        """, unsafe_allow_html=True)
        
        video_file = st.file_uploader(
            "Choose a video...",
            type=['mp4', 'avi', 'mov'],
            help="Supported formats: MP4, AVI, MOV"
        )
        
        if video_file is not None:
            if st.button("🎬 Process Video", type="primary"):
                with st.spinner("Processing video... This may take several minutes."):
                    # Video processing logic here
                    st.info("Video processing will be implemented with the enhanced API")
    
    with tab3:
        st.markdown("""
            <div class="glass-card">
                <h3>📈 Model Analytics</h3>
                <p>Comprehensive model performance and statistics.</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Model Status
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🤖 Available Models**")
            for model in available_models:
                info = model_details.get(model, {})
                st.markdown(f"""
                    <div class="model-selector">
                        <strong>{info.get('name', model)}</strong><br>
                        Type: {info.get('type', 'Unknown')}<br>
                        Status: ✅ Active
                    </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("**🔧 System Status**")
            st.markdown(f"""
                <div class="status-success">
                    🟢 API Status: Online<br>
                    🤖 Models Loaded: {len(available_models)}<br>
                    ⚡ Ready for Processing
                </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
