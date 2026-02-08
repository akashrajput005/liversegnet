# LiverSegNet System Specification — V2.0.7 (PRODUCTION - RECOVERED)

## Overview
LiverSegNet is a clinical-grade dual-stream surgical perception system. Version 2.0.7 incorporates the **Clinical Recovery Protocol**, eliminating anatomical fragmentation and enforcing solid-organ contiguity.

## 1. Neural Architecture
### Model A (Anatomy) — V2.0.7 RECOVERED
- **Architecture**: DeepLabV3+ (ResNet50).
- **Segmentation**: Stage 2 (5-class native: Background, Liver, Gallbladder, GI Tract, Other).
- **Clinical Proxy**: Supervised by deterministic, solid parenchymal masks (Erosion + LCC filtering).
- **Anatomical Contiguity**: Enforces **Exactly 1 component per organ** via backend post-processing.
- **Confidence Layer**: Filtered at **0.3 probability threshold** for zero-flicker localization.

### Model B (Tools) — STABLE
- **Architecture**: U-Net with ResNet34.
- **Purpose**: Specialized for instrument tracking and sharp objects.
- **Kernel Separation**: Employs the **Tool Shield** (Strict Subtraction) to ensure 0-pixel anatomy/instrument overlap.

## 2. Temporal & Kinetic Safety
### EMA Smoothing
- **Logic**: Deterministic Exponential Moving Average (`alpha=0.5`).
- **Precision**: Locks instrument tooltips to physical tips across frames.

### Kinetic Safety Margins
- **Mechanism**: Dynamic Safety Gates scale with instrument velocity.
- **Expansion**: `adj_threshold = base_threshold + (velocity * 0.5)`.

## 3. High-Fidelity Visualization
### Neon-Glow Engine (V2.0.7 Stable)
- **Effect**: Multi-layered silhouette drawing with pulsed surgical indicators.
- **Palette**: Neon Green (Liver), Electric Cyan (GB), Purple (Tools), White/Red (Tips).

### UI: Surgical Cockpit (Glassmorphism)
- **Design**: Premium dashboard with real-time **Kinetic Analytics** and Safety Telemetry.

## 4. Audit & Compliance
- **Anatomical Integrity**: Strictly enforced 1-component liver contiguity.
- **Production Audit**: Verified via `master_diag.py` protocol.
- **Hardware Compliance**: Restricted to **< 2.5 GB VRAM** peak execution.

## 5. Directory Structure
- `models/`: Neural architecture definitions.
- `datasets/`: Clinical Proxy logic (`choleseg8k.py`).
- `inference/`: V2.0.7 stabilized engine with EMA and Kinetic Safety.
- `risk/`: Geometry safety and velocity scaling.
- `ui/`: V2.0.7 Production Dashboard.
- `docs/`: Auditable specifications and validation reports.
