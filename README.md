# LiverSegNet: Hybrid Surgical Perception System (V2.2.1-HYBRID)

LiverSegNet is a formalized hybrid perception system designed for real-time laparoscopic anatomical segmentation and kinetic safety assessment. It bridges the gap between deep learning and clinical reality by integrating multiple signal streams.

## Key Features
*   **Hybrid Perception Architecture**: Trio-signal processing (Neural, Deterministic, Heuristic).
*   **Multicolor Anatomical Recovery (MAR)**: Physically-informed BGR discovery for shadowed anatomy.
*   **Kinetic Safety Layer**: Real-time instrument tracking with velocity-adjusted risk buffers.
*   **Surgical Navigation Hub**: Streamlit dashboard with detailed hardware compute audits and temporal smoothing.
*   **Clinical Governance**: Fully kill-switchable heuristic layer and transparent signal classification.

## System Structure
```bash
LiverSegNet/
├── production_v2_2_0/    # PRODUCTION WEIGHTS & RELEASE ARTIFACTS
├── inference/            # Core Perception Engine (Hybrid Logic)
├── risk/                 # Kinetic Geometry & Safety Layer
├── ui/                   # Surgical Navigation Dashboard (Streamlit)
├── models/               # Deep Architecture Definitions (DeepLabV3+/UNet)
├── datasets/             # Data loading & Clinical Proxy Logic
├── training/             # Production Training Pipelines
├── utils/                # Surgical Visualization & Logic Utils
├── research/             # ARCHIVE: Debug scripts and validation logs
└── docs/                 # Architectural Documentation & Diagrams
```

## Quick Start
1.  **Run Dashboard**: `streamlit run ui/dashboard.py`
2.  **Execute Audit**: `python master_diag.py`

## Architecture: The Hybrid Edge
LiverSegNet V2.2.1-HYBRID utilizes a **Classified Signal Pipeline**:
1.  **NEURAL**: Primary organ/tool localization using DeepLabV3+ and UNet.
2.  **DETERMINISTIC**: Hard geometric filters, FOV masking, and anatomical shielding.
3.  **HEURISTIC**: BGR color-consistent growth (MAR) for recovering attenuated neural signals.

## Clinical Safety Gates
*   **Critical Threshold**: 20.5 px
*   **Warning Threshold**: 50.5 px
*   **Temporal Stability**: EMA-weighted instrument trajectory smoothing.

*This system is formalized and frozen under v2.2.1-hybrid governance.* All work is in English.🛡️
