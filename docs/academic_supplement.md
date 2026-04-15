# LiverSegNet: Academic Methodology Supplement

This document formalizes the "Trio-Signal" architecture of LiverSegNet with mathematical rigor for conference submission.

## 1. Focal Tversky Loss (FTL)
To prioritize surgical safety (recall) over generic accuracy, we employ the Focal Tversky Loss ($L_{FTL}$). Unlike standard Dice loss, FTL allows asymmetric weighting of False Positives ($FP$) and False Negatives ($FN$).

$$L_{FTL} = \sum_{c} \left( 1 - TI_c \right)^\gamma$$

Where the Tversky Index ($TI_c$) for class $c$ is:
$$TI_c = \frac{TP_c + \epsilon}{TP_c + \alpha FN_c + \beta FP_c + \epsilon}$$

**Parameters in LiverSegNet V2.2.x:**
*   $\alpha = 0.7$: High penalty for False Negatives (Missing anatomy).
*   $\beta = 0.3$: Lower penalty for False Positives (Extra segmentation).
*   $\gamma = 4/3 \approx 1.33$: Focal parameter to focus on hard-to-segment boundaries.

---

## 2. Heuristic Recovery: Multicolor Recovery (MAR)
The MAR module recovers anatomical regions where the neural signal probability $P(x,y) < \tau_{neural}$. It is defined as a Region Growth function $G$ seeded by high-confidence neural detections.

### 2.1 Seed Set ($S$)
$$S = \{ (x,y) \mid P(x,y) > \tau_{seed} \}$$

### 2.2 Color Search Space ($\Omega$)
We define two spectral kernels for recovery:
*   **Anatomy Kernel (BGR)**:
    $$\Omega_{Liver} = \{ p \mid R(p) > 30 \land R(p) > 1.2G(p) \land R(p) > 1.2B(p) \}$$
*   **Shadow Kernel (HSV)**:
    $$\Omega_{GB} = \{ p \mid H(p) \in [20^\circ, 60^\circ] \land S(p) > 40 \land V(p) > 30 \}$$

### 2.3 Growth Function
The final mask $M_{hybrid}$ is the union of seeds and connected components within the color kernels:
$$M_{hybrid} = S \cup \{ p \in \Omega \mid p \text{ is connected to } S \}$$

---

## 3. Deterministic Instrument Shielding
To resolve semantic conflicts (e.g., metal tool classified as tissue), we apply a deterministic subtraction rule.

If $M_{tool}$ is the binary mask from Kernel B (U-Net) and $M_{organ}$ is the mask from Kernel A (DeepLabV3+):
$$M_{final} = M_{organ} \cap \neg M_{tool}$$

This ensures that the "Instrument Shield" always takes precedence in real-time visualization.

---

## 4. Kinetic Safety Vector (KSV)
The risk status $R$ is a function of the minimum distance $d_{min}$ between the set of tool-tip coordinates $T$ and the liver boundary $\partial L$, adjusted by tool velocity $v$:

$$R = \begin{cases} \text{CRITICAL} & \text{if } d_{min} < \delta_{safety}(v) \\ \text{WARNING} & \text{if } \delta_{safety}(v) \leq d_{min} < \Delta_{safe} \\ \text{SAFE} & \text{otherwise} \end{cases}$$

Where $\delta_{safety}(v) = \delta_0 + k \cdot v$ is the **Dynamic Risk Buffer**.
