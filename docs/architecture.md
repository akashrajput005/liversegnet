# LiverSegNet V2.2.1 Architecture

The system is designed as a **Hybrid Perception Pipeline**, combining neural deep learning, deterministic geometric guards, and heuristic color-based discovery.

![LiverSegNet Hybrid Architecture](architecture_diagram.svg)

```mermaid
graph TD
    subgraph "Input Layer"
        F[Endoscopic Frame]
    end

    subgraph "Inference Engine (ClinicalInferenceEngine)"
        P[Preprocessing: Deterministic]
        
        subgraph "Neural Signal Layer"
            MA[Model A: Anatomy - DeepLabV3+]
            MB[Model B: Instruments - UNet]
        end
        
        subgraph "Deterministic Signal Layer"
            FM[FOV Circle Mask]
            CC[Connected Components Filter]
            IS[Instrument Shielding]
        end
        
        subgraph "Heuristic Signal Layer (Kill-Switchable)"
            MAR[Multicolor Anatomical Recovery]
            HF[Hybrid Fusion - Neural Seeds + BGR Discovery]
        end
        
        subgraph "Safety Layer (GeometrySafetyLayer)"
            KT[Kinetic Telemetry - Velocity/Distance]
            RG[Safety Vector - Risk Gradients]
        end
    end

    subgraph "Output Layer"
        D[Streamlit Dashboard - V2.2.1]
        A[Audit Reports - Comparative Vision]
    end

    F --> P
    P --> MA
    P --> MB
    MA --> FM
    FM --> MAR
    MAR --> HF
    HF --> CC
    MB --> IS
    IS --> CC
    CC --> KT
    KT --> RG
    RG --> D
    RG --> A
```

## Signal Definitions

*   **NEURAL**: Deep learning posterior probabilities (DeepLabV3+ & UNet).
*   **DETERMINISTIC**: Hard geometric constraints (FOV masking, Morphology).
*   **HEURISTIC**: Physically-informed color recovery signatures (BGR MAR).
