# 🏥 LiverSegNet: Clinical Surgical Vision AI

**LiverSegNet** is an unmatchable, pinnacle-grade AI system designed for real-time surgical segmentation and clinical analytics. It leverages state-of-the-art architectures (**ResNet101 DeepLabV3+** & **EfficientNet-B4 U-Net**) to identify liver tissue and surgical instruments with absolute precision.

## 🚀 Quick Launch
For detailed launch commands and operational procedures, please refer to:
👉 **[LAUNCH.md](file:///c:/Users/Public/LiverSegNet/LAUNCH.md)**

## 🏗️ System Architecture
The system is decoupled for maximum performance and reliability:
- **Backend (Inference API)**: A FastAPI-based engine powered by **Test Time Augmentation (TTA)** and **Ensemble Probability Fusion**.
- **Frontend (Surgeon Dashboard)**: A premium "Cyber-Surgical" Streamlit UI providing live analytics, pixel counts, and safety monitoring.
- **Advanced Training**: An automated multi-stage pipeline utilizing a unified dataset of **8,000+ frames**.

## 📁 Directory Structure
```
├── app.py                # Premium Surgeon Dashboard (Streamlit)
├── src/
│   ├── api.py            # High-performance Inference API
│   ├── infer.py          # Pinnacle Inference Engine (TTA/Ensemble)
│   ├── model.py          # Unified Architecture & Quad-Fusion Loss
│   ├── dataset.py        # Advanced Surgical Augmentations
│   └── train_pinnacle.py # Automated Multi-Stage Training
├── models/               # Standardized Weights (.pth)
├── logs/                 # Clinical CSV Metrics & Historical Data
└── LAUNCH.md             # Command Reference Guide
```

## 🔬 Core Clinical Features
- **Quad-Fusion Loss**: Specialized optimization for Shape, Overlap, Edges, and Class Imbalance.
- **Anatomical Grounding**: Real-time pixel and region counting for clinical precision.
- **Test Time Augmentation (TTA)**: Multi-view cross-validation for zero-jitter masks.
- **Surgical Robustness**: Built-in simulations for smoke, lens blur, and lighting distortions.
- **Surgeon Control**: Dynamic overlay opacity and interactive risk assessment.

---
*Developed for Advanced Surgical Intelligence Applications*
