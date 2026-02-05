import os
import json
import subprocess
from pathlib import Path
import shutil

import cv2
import numpy as np
import streamlit as st
import plotly.graph_objects as go


CONFIG_PATH = r"C:\Users\Public\liversegnet\configs\config.yaml"
INFER_SCRIPT = r"C:\Users\Public\liversegnet\src\infer.py"
UPLOADS_DIR = Path("uploads")
MODELS_DIR = Path("models")


def run_infer(image_path, encoder):
    cmd = [r".venv\Scripts\python", INFER_SCRIPT, "--image", image_path, "--config", CONFIG_PATH, "--encoder", encoder]
    subprocess.run(cmd, check=True)


def read_masks(encoder):
    masks_path = MODELS_DIR / f"infer_masks_{encoder}.npz"
    if not masks_path.exists():
        return None, None
    data = np.load(str(masks_path))
    return data["liver_mask"], data["instrument_mask"]


def read_metrics(encoder):
    metrics_path = MODELS_DIR / f"infer_metrics_{encoder}.json"
    if not metrics_path.exists():
        return None
    with open(metrics_path, "r") as f:
        return json.load(f)


def compose_overlay(base_bgr, liver_mask, instrument_mask, opacity):
    overlay = base_bgr.copy()
    if liver_mask is not None:
        overlay[liver_mask == 1] = (0, 255, 0)
    if instrument_mask is not None:
        overlay[instrument_mask == 1] = (0, 165, 255)
    blended = cv2.addWeighted(base_bgr, 1 - opacity, overlay, opacity, 0)
    return blended


def mask_only_view(shape, liver_mask, instrument_mask):
    canvas = np.zeros(shape, dtype=np.uint8)
    if liver_mask is not None:
        canvas[liver_mask == 1] = (0, 255, 0)
    if instrument_mask is not None:
        canvas[instrument_mask == 1] = (0, 165, 255)
    return canvas


def save_upload(uploaded_file):
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    temp_path = UPLOADS_DIR / uploaded_file.name
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(temp_path)


def extract_frames(video_path, max_frames, stride):
    cap = cv2.VideoCapture(video_path)
    frames = []
    idx = 0
    kept = 0
    while cap.isOpened() and kept < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % stride == 0:
            frames.append(frame)
            kept += 1
        idx += 1
    cap.release()
    return frames


def render_metrics_cards(metrics, label):
    if not metrics:
        st.warning(f"No metrics found for {label}")
        return
    cols = st.columns(3)
    cols[0].metric("Liver Dice", f"{metrics.get('liver_dice', 0):.3f}")
    cols[1].metric("Inst Dice", f"{metrics.get('instrument_dice', 0):.3f}")
    cols[2].metric("Inst Precision", f"{metrics.get('instrument_precision', 0):.3f}")
    cols = st.columns(3)
    cols[0].metric("Inst Recall", f"{metrics.get('instrument_recall', 0):.3f}")
    cols[1].metric("Inst FP", f"{metrics.get('instrument_false_positive', 0)}")
    cols[2].metric("Area L / I", f"{metrics.get('pixel_area_liver', 0)} / {metrics.get('pixel_area_instrument', 0)}")


st.set_page_config(page_title="LiverSegNet", layout="wide", page_icon="🩺")
st.markdown(
    """
    <style>
    :root { --accent:#7c4dff; --accent2:#00e5ff; --accent3:#00e676; --card:#0f1522; --card2:#121b2d; }
    .stApp { background: radial-gradient(1200px 600px at 12% 8%, #141b2d 0%, #0b0f14 60%, #06080c 100%); color: #e8eef7; }
    .block-container { padding-top: 2rem; }
    .hero { padding: 1.4rem 1.6rem; border: 1px solid #1e2a3a; border-radius: 16px; background: linear-gradient(135deg, rgba(124,77,255,0.18), rgba(0,229,255,0.08)); box-shadow: 0 0 24px rgba(124,77,255,0.15); }
    .glass { border: 1px solid #1e2a3a; border-radius: 16px; background: rgba(12, 16, 26, 0.7); padding: 1rem; }
    .pill { display:inline-block; padding:0.2rem 0.7rem; border-radius:999px; background:rgba(0,229,255,0.12); border:1px solid rgba(0,229,255,0.4); color:#9ff3ff; font-size:0.8rem; margin-left:0.4rem; }
    .stat { background: var(--card); border:1px solid #1e2a3a; border-radius:12px; padding:0.8rem 1rem; }
    .stat h4 { margin:0 0 0.2rem 0; color:#9fb3c8; font-weight:600; }
    .stat p { margin:0; font-size:1.2rem; }
    .stButton>button { background: linear-gradient(135deg, var(--accent), var(--accent2)); border:0; color:#fff; font-weight:600; padding:0.6rem 1.1rem; border-radius:10px; box-shadow:0 6px 18px rgba(124,77,255,0.35); }
    .stButton>button:hover { filter:brightness(1.05); }
    .stFileUploader { background: var(--card2); border-radius:12px; padding:0.6rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <div style="display:flex;align-items:center;justify-content:space-between;gap:1rem;">
        <div>
          <h1 style="margin-bottom:0;">LiverSegNet</h1>
          <p style="margin-top:4px;color:#9fb3c8;">Advanced Surgical Perception Hub</p>
        </div>
        <div>
          <span class="pill">CUDA Ready</span>
          <span class="pill">Precision First</span>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("### Controls")
mode = st.sidebar.selectbox("Mode", ["Patient", "Surgeon"], index=0)
input_type = st.sidebar.selectbox("Input Type", ["Image", "Video"])
opacity = st.sidebar.slider("Overlay Opacity", 0.0, 1.0, 0.55, 0.05)
toggle_overlay = st.sidebar.checkbox("Overlay On", value=True)
mask_mode = st.sidebar.selectbox("Mask Mode", ["Both", "Liver Only", "Instrument Only"])

tab_infer, tab_best = st.tabs(["Inference", "Best Frames"])

ckpt_r101 = Path("models/best_precision.pth").exists()
ckpt_r50 = Path("models/best_precision_r50.pth").exists()

with tab_infer:
    st.markdown("### Data Ingestion")
    uploaded = st.file_uploader("Upload file", type=["png", "jpg", "jpeg", "mp4", "avi", "mov"])
    if uploaded is None:
        st.stop()

    file_path = save_upload(uploaded)

    if "overlay_resnet101" not in st.session_state:
        st.session_state["overlay_resnet101"] = None
    if "overlay_resnet50" not in st.session_state:
        st.session_state["overlay_resnet50"] = None
    if "metrics_resnet101" not in st.session_state:
        st.session_state["metrics_resnet101"] = None
    if "metrics_resnet50" not in st.session_state:
        st.session_state["metrics_resnet50"] = None
    if "video_overlays_resnet101" not in st.session_state:
        st.session_state["video_overlays_resnet101"] = None
    if "video_overlays_resnet50" not in st.session_state:
        st.session_state["video_overlays_resnet50"] = None
    if "masks_resnet101" not in st.session_state:
        st.session_state["masks_resnet101"] = None
    if "masks_resnet50" not in st.session_state:
        st.session_state["masks_resnet50"] = None
    if "base_image" not in st.session_state:
        st.session_state["base_image"] = None

    run_clicked = st.button("Run Inference")

    if input_type == "Video":
        max_frames = st.sidebar.slider("Max Frames", 1, 20, 5, 1)
        stride = st.sidebar.slider("Frame Stride", 1, 10, 3, 1)
        frames = None
    else:
        frames = None

    if mode == "Patient":
        st.markdown("#### Patient Mode")
        if run_clicked:
            if input_type == "Video":
                temp_dir = UPLOADS_DIR / "video_frames"
                temp_dir.mkdir(parents=True, exist_ok=True)
                overlays = []
                frames = extract_frames(file_path, max_frames, stride)
                if not frames:
                    st.error("No frames extracted from video.")
                    st.stop()
                for i, frame in enumerate(frames):
                    frame_path = temp_dir / f"frame_{i:04d}.png"
                    cv2.imwrite(str(frame_path), frame)
                    run_infer(str(frame_path), "resnet101")
                    overlay_path = MODELS_DIR / "infer_overlay_resnet101.png"
                    saved_overlay = temp_dir / f"overlay_{i:04d}.png"
                    shutil.copyfile(str(overlay_path), str(saved_overlay))
                    overlays.append(cv2.imread(str(saved_overlay)))
                st.session_state["video_overlays_resnet101"] = overlays
            else:
                run_infer(file_path, "resnet101")
                st.session_state["base_image"] = cv2.imread(file_path)
                overlay_path = MODELS_DIR / "infer_overlay_resnet101.png"
                st.session_state["overlay_resnet101"] = cv2.imread(str(overlay_path)) if overlay_path.exists() else None
                st.session_state["masks_resnet101"] = read_masks("resnet101")

        if input_type == "Video":
            overlays = st.session_state.get("video_overlays_resnet101")
            if overlays:
                st.image(overlays, channels="BGR", caption="Clinical Overlay Frames")
            else:
                st.info("Click Run Inference to generate video overlays.")
        else:
            base_img = st.session_state.get("base_image")
            liver, inst = st.session_state.get("masks_resnet101") or (None, None)
            if mask_mode == "Liver Only":
                inst = None
            elif mask_mode == "Instrument Only":
                liver = None
            if base_img is None:
                st.info("Click Run Inference to generate overlay.")
            else:
                if not toggle_overlay:
                    st.image(base_img, channels="BGR", caption="Input Image", use_column_width=True)
                else:
                    if mask_mode in ["Liver Only", "Instrument Only"]:
                        masked = mask_only_view(base_img.shape, liver, inst)
                        st.image(masked, channels="BGR", caption="Mask View", use_column_width=True)
                    else:
                        composed = compose_overlay(base_img, liver, inst, opacity)
                        st.image(composed, channels="BGR", caption="Clinical Overlay", use_column_width=True)
        st.stop()

    encoders = ["resnet101", "resnet50"]
    col1, col2 = st.columns(2)
    with col1:
        enc_a = st.selectbox("Encoder A", encoders, index=0)
    with col2:
        enc_b = st.selectbox("Encoder B", encoders, index=1)

    if enc_a == "resnet50" and not ckpt_r50:
        st.error("ResNet50 checkpoint missing: models/best_precision_r50.pth")
        st.stop()
    if enc_b == "resnet50" and not ckpt_r50:
        st.error("ResNet50 checkpoint missing: models/best_precision_r50.pth")
        st.stop()

    if run_clicked:
        if input_type == "Video":
            temp_dir = UPLOADS_DIR / "video_frames"
            temp_dir.mkdir(parents=True, exist_ok=True)
            overlays_a = []
            overlays_b = []
            frames = extract_frames(file_path, max_frames, stride)
            if not frames:
                st.error("No frames extracted from video.")
                st.stop()
            for i, frame in enumerate(frames):
                frame_path = temp_dir / f"frame_{i:04d}.png"
                cv2.imwrite(str(frame_path), frame)
                run_infer(str(frame_path), enc_a)
                overlay_path_a = MODELS_DIR / f"infer_overlay_{enc_a}.png"
                saved_a = temp_dir / f"overlay_{enc_a}_{i:04d}.png"
                shutil.copyfile(str(overlay_path_a), str(saved_a))
                overlays_a.append(cv2.imread(str(saved_a)))
                run_infer(str(frame_path), enc_b)
                overlay_path_b = MODELS_DIR / f"infer_overlay_{enc_b}.png"
                saved_b = temp_dir / f"overlay_{enc_b}_{i:04d}.png"
                shutil.copyfile(str(overlay_path_b), str(saved_b))
                overlays_b.append(cv2.imread(str(saved_b)))
            st.session_state[f"video_overlays_{enc_a}"] = overlays_a
            st.session_state[f"video_overlays_{enc_b}"] = overlays_b
        else:
            run_infer(file_path, enc_a)
            st.session_state["base_image"] = cv2.imread(file_path)
            overlay_path_a = MODELS_DIR / f"infer_overlay_{enc_a}.png"
            st.session_state[f"overlay_{enc_a}"] = cv2.imread(str(overlay_path_a)) if overlay_path_a.exists() else None
            st.session_state[f"metrics_{enc_a}"] = read_metrics(enc_a)
            st.session_state[f"masks_{enc_a}"] = read_masks(enc_a)
            run_infer(file_path, enc_b)
            overlay_path_b = MODELS_DIR / f"infer_overlay_{enc_b}.png"
            st.session_state[f"overlay_{enc_b}"] = cv2.imread(str(overlay_path_b)) if overlay_path_b.exists() else None
            st.session_state[f"metrics_{enc_b}"] = read_metrics(enc_b)
            st.session_state[f"masks_{enc_b}"] = read_masks(enc_b)

    if input_type == "Video":
        overlays_a = st.session_state.get(f"video_overlays_{enc_a}")
        overlays_b = st.session_state.get(f"video_overlays_{enc_b}")
        if overlays_a and overlays_b:
            col_a, col_b = st.columns(2)
            with col_a:
                st.image(overlays_a[:min(4, len(overlays_a))], channels="BGR", caption=f"{enc_a.upper()} frames")
            with col_b:
                st.image(overlays_b[:min(4, len(overlays_b))], channels="BGR", caption=f"{enc_b.upper()} frames")
        else:
            st.info("Click Run Inference to generate video overlays.")
        st.stop()

    base_img = st.session_state.get("base_image")
    img_a = st.session_state.get(f"overlay_{enc_a}")
    img_b = st.session_state.get(f"overlay_{enc_b}")
    masks_a = st.session_state.get(f"masks_{enc_a}") or (None, None)
    masks_b = st.session_state.get(f"masks_{enc_b}") or (None, None)
    if base_img is None or img_a is None or img_b is None:
        st.info("Click Run Inference to generate overlays.")
        st.stop()

    col_a, col_b = st.columns(2)
    with col_a:
        liver_a, inst_a = masks_a
        if mask_mode == "Liver Only":
            inst_a = None
        elif mask_mode == "Instrument Only":
            liver_a = None
        if not toggle_overlay:
            display_a = base_img
        elif mask_mode in ["Liver Only", "Instrument Only"]:
            display_a = mask_only_view(base_img.shape, liver_a, inst_a)
        else:
            display_a = compose_overlay(base_img, liver_a, inst_a, opacity)
        st.image(display_a, channels="BGR", caption=f"{enc_a.upper()} (active checkpoint)")
    with col_b:
        liver_b, inst_b = masks_b
        if mask_mode == "Liver Only":
            inst_b = None
        elif mask_mode == "Instrument Only":
            liver_b = None
        if not toggle_overlay:
            display_b = base_img
        elif mask_mode in ["Liver Only", "Instrument Only"]:
            display_b = mask_only_view(base_img.shape, liver_b, inst_b)
        else:
            display_b = compose_overlay(base_img, liver_b, inst_b, opacity)
        st.image(display_b, channels="BGR", caption=f"{enc_b.upper()} (active checkpoint)")

    metrics_a = st.session_state.get(f"metrics_{enc_a}") or {}
    metrics_b = st.session_state.get(f"metrics_{enc_b}") or {}

    st.subheader("Perception Metrics")
    left, right = st.columns(2)
    with left:
        render_metrics_cards(metrics_a, enc_a)
    with right:
        render_metrics_cards(metrics_b, enc_b)

    bar_metrics = ["liver_dice", "instrument_dice", "instrument_precision", "instrument_recall"]
    fig = go.Figure()
    fig.add_bar(name=enc_a, x=bar_metrics, y=[metrics_a.get(m, 0) for m in bar_metrics])
    fig.add_bar(name=enc_b, x=bar_metrics, y=[metrics_b.get(m, 0) for m in bar_metrics])
    fig.update_layout(barmode="group", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

with tab_best:
    st.markdown("### Best Frames (Top Scoring)")
    best_dir = Path("logs/best_frames")
    csv_path = best_dir / "best_frames_summary.csv"
    if csv_path.exists():
        st.caption(f"Summary: {csv_path}")
    images = sorted(best_dir.glob("*.png"))
    if not images:
        st.info("No best-frame overlays found. Run the extractor to populate this view.")
    else:
        cols = st.columns(4)
        for idx, img_path in enumerate(images[:24]):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            cols[idx % 4].image(img, channels="BGR", caption=img_path.name)
