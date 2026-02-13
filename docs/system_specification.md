# LiverSegNet System Specification — V3.0.0 (GOLD RELEASE)

## Overview
LiverSegNet is a clinical-grade dual-stream surgical perception system. Version 3.0.0 incorporates the **Hybrid Perception Pipeline**, combining neural inference with deterministic geometric guards and HSV-resilient heuristic discovery.

## 1. Neural Architecture
- **Neural Thresholds**: Class-specific sensitivity (Liver: 0.08, GB: 0.12).
- **Anatomical Contiguity**: Enforces solid masses via **Clinical Size Filtering** (800px/400px).

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
- `inference/`: V3.0.0 Hybrid Engine with HSV Recovery & Size-Guards.
- `risk/`: Geometry safety with velocity scaling.
- `ui/`: V3.0.0 Dual-Stream Visualization Hub.
- `docs/`: Auditable V3.0.0 specifications and validation reports.
