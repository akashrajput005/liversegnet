# LiverSegNet Development Timeline & MS Planner Tasks

This document contains the detailed task breakdown for the 2-month development cycle of LiverSegNet V3.0.0-HYBRID, distributed across a 3-member team.

## Project Summary
- **Duration**: 2 Months (60 Days)
- **Team Size**: 3 (Lead Dev, Safety/Inference Dev, UI/Data Dev)
- **Total Man-Days**: 130

---

## Phase 1: Research & Data Foundation (15 Days)

| Task Name | Description | Duration (Days) | Assigned To |
| :--- | :--- | :--- | :--- |
| **Project Setup & Environment Initialization** | Configure CUDA environment, install dependencies, and initialize repository structure. | 3 | Dev 1 |
| **Clinical Proxy Logic Implementation** | Create `choleseg8k.py` to map raw dataset labels to 5 clinical classes. | 6 | Dev 3 |
| **Architectural Design Blueprinting** | Finalize the "Trio-Signal" architecture design (Neural, Deterministic, Heuristic). | 3 | Dev 1 |
| **Baseline Model Evaluation** | Benchmark initial DeepLabV3+ and U-Net architectures on surgical video datasets. | 3 | Dev 2 |

---

## Phase 2: Neural Architecture - Model A (15 Days)

| Task Name | Description | Duration (Days) | Assigned To |
| :--- | :--- | :--- | :--- |
| **Model A Implementation (Anatomy)** | Implement DeepLabV3+ with ResNet50 backbone for organ segmentation. | 5 | Dev 1 |
| **Custom Focal Tversky Loss Development** | Write and validate `losses.py` with alpha/beta/gamma priority tuning for recall. | 3 | Dev 1 |
| **Stage 1: Backbone Training** | Train frozen backbone on surgical textures (5-10 epochs). | 4 | Dev 1 |
| **Stage 2: Clinical Fine-tuning** | Full unfreeze training for clinical-grade organ boundary precision. | 3 | Dev 1 |

---

## Phase 3: Neural Architecture - Model B (15 Days)

| Task Name | Description | Duration (Days) | Assigned To |
| :--- | :--- | :--- | :--- |
| **Model B Implementation (Instruments)** | Implement U-Net with ResNet34 for sharp tool edge and tip tracking. | 5 | Dev 2 |
| **Instrument Shield Logic** | Develop deterministic subtraction logic to ensure 0-pixel anatomy/tool overlap. | 3 | Dev 2 |
| **Stage 1: Instrumental Training** | Train Model B on tool-specific datasets to isolate metallic textures. | 4 | Dev 2 |
| **Stage 2: Metal-Tip Specialization** | Fine-tune for high-precision tip detection and instrument boundaries. | 3 | Dev 2 |

---

## Phase 4: Hybrid Perception Layer (15 Days)

| Task Name | Description | Duration (Days) | Assigned To |
| :--- | :--- | :--- | :--- |
| **Parallel Inference Engine** | Build `inference/engine.py` using parallel GPU streams for Model A/B. | 5 | Dev 2 |
| **Multicolor Anatomical Recovery (MAR)** | Implement HSV-resilient heuristic discovery for shadowed anatomy. | 6 | Dev 3 |
| **Signal Merging & Latency Optimization** | Optimize the combined signal path to achieve <50ms per-frame latency. | 4 | Dev 2 |

---

## Phase 5: Risk Assessment & Geometry (15 Days)

| Task Name | Description | Duration (Days) | Assigned To |
| :--- | :--- | :--- | :--- |
| **Temporal EMA Smoothing** | Implement Exponential Moving Average for jitter-free instrument tip tracking. | 5 | Dev 2 |
| **Kinetic Safety Gates** | Develop velocity-adjusted risk buffers (higher speed = earlier warning). | 6 | Dev 2 |
| **Proximity Telemetry Calculation** | Write geometry logic for real-time pixel-to-millimeter distance calculation. | 4 | Dev 2 |

---

## Phase 6: Surgical Navigation Dashboard (15 Days)

| Task Name | Description | Duration (Days) | Assigned To |
| :--- | :--- | :--- | :--- |
| **Streamlit UI Layout** | Design the high-fidelity "Surgical Cockpit" dashboard with Glassmorphism. | 4 | Dev 3 |
| **Neon-Glow Visualization Engine** | Implement custom silhouetting and glow effects for anatomical overlays. | 6 | Dev 3 |
| **Real-time Telemetry Widgets** | Build dynamic velocity bars and heatmaps for AI confidence/doubt. | 5 | Dev 3 |

---

## Phase 7: Integration & Production Audit (20 Days)

| Task Name | Description | Duration (Days) | Assigned To |
| :--- | :--- | :--- | :--- |
| **System Integration & Debugging** | Merge all streams (Neural, Heuristic, Deterministic) and fix edge cases. | 10 | Dev 1 |
| **Production Audit Protocol** | Implement `master_diag.py` for automated release validation and FPS checks. | 5 | Dev 2 |
| **Hardware Compliance Optimization** | Optimize memory usage to ensure <2.5 GB peak VRAM consumption. | 5 | Dev 3 |

---

## Phase 8: Finalization & Reporting (20 Days)

| Task Name | Description | Duration (Days) | Assigned To |
| :--- | :--- | :--- | :--- |
| **System Specification Documentation** | Write comprehensive V3.0.0-HYBRID release specs (`system_specification.md`). | 7 | Dev 3 |
| **Master Technical Report** | Compile exhaustive report merging audit results and technical deep dives. | 8 | Dev 3 |
| **Validation & Presentation Defense** | Run final clinical validation tests and prepare defense material for panels. | 5 | Dev 1 |
