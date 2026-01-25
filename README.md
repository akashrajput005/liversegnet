# LiverSegNet AI: Advanced Surgical Perception Hub 🩺🚀

LiverSegNet is a research-grade medical computer vision system designed for real-time anatomical and instrument segmentation in laparoscopic surgery. It leverages state-of-the-art Deep Learning (DeepLabV3+ & U-Net) with a **Triple-Head Clinical Ensemble** to provide surgeons with critical clinical insights, including tool-to-organ proximity and occlusion hazards.

## ✨ Core Features
- **Triple-Head Clinical Ensemble**: Combines specialized Stage 1 anatomical grounding with precision Stage 2 instrument tracking
- **Real-Time Clinical Analytics**:
    - **Occlusion Hazards**: Live calculation of liver area obscured by surgical tools
    - **Safety Proximity**: Euclidean distance metrics with color-coded safety zones (SAFE/CAUTION/CRITICAL)
    - **Independent Detection**: Robust organ and instrument status indicators
- **Premium User Experience**: Glassmorphism-inspired Streamlit dashboard with real-time metrics
- **Production Ready**: Clean API/UI separation with FastAPI backend

## 🛠️ Architecture
- **Backend**: FastAPI with PyTorch (CUDA acceleration)
- **Frontend**: Streamlit with Custom CSS (Glassmorphism)
- **Triple-Head Ensemble**: 
    - **Stage 1 Anchor**: DeepLabV3+ ResNet-50 (2-class, anatomical specialist)
    - **U-Net Anchor**: ResNet-34 (3-class, robust baseline)
    - **Advanced Main**: DeepLabV3+ ResNet-50 (3-class, precision tracker)

## 🚀 Quick Start (Local)

### Option 1: One-Click Startup (Easiest)
Simply double-click `start.bat` in the project root!

### Option 2: Manual Startup
See `STARTUP_GUIDE.md` for detailed commands.

### Requirements
- NVIDIA GPU with CUDA support (RTX 3050+ recommended)
- Python 3.8+
- Dependencies: `pip install -r requirements.txt`

## 📂 Project Structure
- `src/`: Core neural logic and training scripts
- `ui/`: Dashboard and Backend API
- `models/`: Pre-trained weights (`.pth` files excluded from Git - download separately)
- `tools/`: Data preprocessing and evaluation utilities
- `configs/`: Centralized parameters for hot-swapping engines
- `start.bat`: One-click application launcher
- `STARTUP_GUIDE.md`: Comprehensive startup and troubleshooting guide

## 🧪 Test Files
Sample surgical frames and videos are provided for testing:
- `surg_interaction_*.png`: Complex surgical scenes with both organs and instruments
- `test_surgical_clip.avi`: 30-frame video sequence

**Note**: Model weights (`.pth` files) are excluded from Git due to size. Download or train models separately.

---
*Developed for Advanced Surgical Intelligence Applications*
