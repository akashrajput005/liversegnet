# 🩺 LiverSegNet — Hybrid AI Surgical Navigation System

<div align="center">

### **A Resilient Hybrid Perception Pipeline for Intra-operative Laparoscopic Navigation & Kinetic Safety**

*Advancing Explainable Medical AI through Hybrid Intelligence*

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Streamlit-FF4B4B?style=for-the-badge)](https://liversegnet-nq3ncgte2bappazv7kbu7e.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge\&logo=python\&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C?style=for-the-badge\&logo=pytorch\&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-5C3EE8?style=for-the-badge\&logo=opencv\&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge\&logo=streamlit\&logoColor=white)
![Medical AI](https://img.shields.io/badge/Medical_AI-Surgical_Navigation-008CFF?style=for-the-badge)

---

### **Bridging Deep Learning, Explainable AI and Clinical Safety**

LiverSegNet is a **Hybrid AI Surgical Navigation Framework** engineered for **real-time laparoscopic liver segmentation, anatomical recovery, and kinetic risk assessment**.

Unlike conventional segmentation systems that rely solely on neural networks, LiverSegNet introduces a **multi-layer perception architecture** that combines:

🧠 Neural Intelligence
📐 Deterministic Clinical Geometry
🎯 Heuristic Anatomical Recovery

to improve robustness, interpretability, and surgical reliability.

</div>

---

# ✨ Why LiverSegNet?

Modern operating rooms require more than accurate segmentation.

They require:

* Real-time inference
* Clinical reliability
* Explainable AI
* Robust anatomical recovery
* Instrument-aware safety monitoring

LiverSegNet addresses these requirements through a **Hybrid Perception Pipeline** that integrates deep learning with deterministic validation and clinically-inspired heuristic reasoning.

---

# 🧠 Hybrid Perception Architecture

```text
                    Surgical Video
                           │
                           ▼
                Deep Neural Networks
             (DeepLabV3+ / UNet Models)
                           │
                           ▼
                 Primary Segmentation
                           │
        ┌──────────────────┴──────────────────┐
        ▼                                     ▼
Deterministic Geometry                MAR Recovery Engine
(FOV + Safety Logic)              (BGR Anatomical Recovery)
        │                                     │
        └──────────────────┬──────────────────┘
                           ▼
                Hybrid Signal Fusion
                           ▼
             Risk Assessment Engine
                           ▼
          Surgical Navigation Dashboard
```

---

# 🚀 Core Features

## 🧠 Hybrid AI Framework

Combines multiple perception strategies instead of relying on a single neural network.

* Neural segmentation
* Deterministic validation
* Anatomical recovery
* Signal fusion

---

## 🎨 Multicolor Anatomical Recovery (MAR)

A physically-informed recovery module capable of reconstructing attenuated liver regions under:

* Surgical shadows
* Low illumination
* Partial occlusion
* Reduced neural confidence

---

## ⚠️ Kinetic Safety Layer

Real-time monitoring of surgical instrument motion using geometric reasoning.

Features include:

* Dynamic velocity-aware safety zones
* Collision risk estimation
* Anatomical shielding
* Distance threshold analysis

---

## 📊 Explainable Clinical Intelligence

Every decision is classified into one of three transparent signal types:

| Signal           | Purpose                          |
| ---------------- | -------------------------------- |
| 🧠 Neural        | Deep learning segmentation       |
| 📐 Deterministic | Clinical geometry & safety rules |
| 🎯 Heuristic     | Anatomical recovery & refinement |

---

## 🖥 Interactive Surgical Dashboard

Built using **Streamlit** for real-time visualization.

Features include:

* Live segmentation overlays
* Hardware diagnostics
* Confidence visualization
* Safety monitoring
* Temporal smoothing
* Clinical audit information

---

# 📂 Project Structure

```text
LiverSegNet/
│
├── production_v2_2_0/
│   ├── Production Weights
│   └── Release Artifacts
│
├── inference/
│   └── Hybrid Perception Engine
│
├── risk/
│   └── Kinetic Safety Layer
│
├── ui/
│   └── Streamlit Dashboard
│
├── models/
│   ├── DeepLabV3+
│   └── UNet
│
├── datasets/
│   └── Clinical Proxy Dataset Logic
│
├── training/
│   └── Model Training Pipelines
│
├── utils/
│   └── Visualization & Helper Utilities
│
├── docs/
│   └── Technical Documentation
│
└── requirements.txt
```

---

# ⚡ Quick Start

Install dependencies

```bash
pip install -r requirements.txt
```

Launch the dashboard

```bash
streamlit run ui/dashboard.py
```

Run the diagnostic audit

```bash
python master_diag.py
```

---

# 📚 Clinical Documentation

The repository includes detailed documentation covering the design philosophy and technical implementation.

* Hybrid AI Justification
* Master Technical Report
* Development Timeline
* System Architecture
* Clinical Logic
* Model Design

---

# 🧪 AI Models

Primary segmentation models:

* DeepLabV3+
* UNet

Supporting technologies:

* PyTorch
* OpenCV
* Streamlit
* Python

---

# 🛡 Clinical Safety Gates

| Parameter          | Value               |
| ------------------ | ------------------- |
| Critical Threshold | **20.5 px**         |
| Warning Threshold  | **50.5 px**         |
| Temporal Stability | EMA-based smoothing |

---

# 🎯 Design Principles

* Explainable AI
* Human-Centered Clinical Intelligence
* Real-Time Performance
* Modular Architecture
* Hybrid Decision Making
* Transparent Signal Classification

---

# 🔬 Technical Philosophy

Instead of asking:

> "Can a neural network solve everything?"

LiverSegNet asks:

> **"How can deep learning, deterministic geometry, and explainable heuristics work together to improve clinical safety?"**

This philosophy forms the foundation of the Hybrid Perception Pipeline.

---

# 🚀 Future Roadmap

* Transformer-based segmentation models
* Multi-organ perception
* 3D laparoscopic reconstruction
* Depth-aware surgical navigation
* Multi-camera fusion
* Federated clinical learning
* Explainable AI analytics
* Edge deployment optimization

---

# 📜 Version

**LiverSegNet V3.0.0-HYBRID**

*A formalized Hybrid AI framework for intelligent laparoscopic navigation and kinetic safety.*

---

<div align="center">

### **Hybrid Intelligence • Clinical Explainability • Surgical Safety**

⭐ If you find this project useful, consider giving it a star.

</div>
