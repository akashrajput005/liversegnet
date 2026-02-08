import torch
import numpy as np
import cv2
import os
from inference.engine import ClinicalInferenceEngine

def run_audit_mode(engine, frame, mode_name, use_heuristics):
    """Executes a standardized audit for a specific perception mode."""
    print(f"\n--- AUDIT MODE: {mode_name} ---")
    
    # Simulate temporal history
    for _ in range(3):
        results = engine.infer(frame, confidence_threshold=0.1, use_heuristics=use_heuristics)
    
    # 1. Visualization Setup
    mask_a = results['mask_a']
    liver_mask = (mask_a == 1).astype(np.uint8)
    gb_mask = (mask_a == 2).astype(np.uint8)
    
    contours_l, _ = cv2.findContours(liver_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_g, _ = cv2.findContours(gb_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 2. Glass Overlay
    glass = frame.copy()
    glass[mask_a == 1] = [0, 200, 0]
    glass[mask_a == 2] = [200, 200, 0]
    augmented = cv2.addWeighted(glass, 0.4, frame, 0.6, 0)
    
    # 3. Signals & Labels
    font = cv2.FONT_HERSHEY_DUPLEX
    def draw_label(img, contours, text, color):
        if not contours: return
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) < 500: return
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            tw, th = cv2.getTextSize(text, font, 0.6, 1)[0]
            cv2.rectangle(img, (cx-5, cy-th-10), (cx+tw+5, cy+5), (20,20,20), -1)
            cv2.putText(img, text, (cx, cy), font, 0.6, color, 1)

    draw_label(augmented, contours_l, f"LIVER ({mode_name})", (0, 255, 0))
    draw_label(augmented, contours_g, "GALLBLADDER", (255, 255, 0))
    
    # 4. Status Metadata
    cv2.putText(augmented, f"PERCEPTION: {mode_name}", (20, 40), font, 0.8, (255, 255, 255), 2)
    sig = results['signals']
    status_str = f"Neural: {'ON' if sig['neural'] else 'OFF'} | Heuristic: {'ON' if sig['heuristic'] else 'OFF'}"
    cv2.putText(augmented, status_str, (20, 70), font, 0.5, (200, 200, 200), 1)
    
    filename = f"audit_v2_2_1_{mode_name.lower().replace(' ', '_')}.png"
    cv2.imwrite(filename, augmented)
    print(f"Audit Result Saved: {filename}")
    return filename

def final_audit():
    print("--- LIVERSEGNET V2.2.1: HYBRID FORMALIZATION AUDIT ---")
    
    model_a_path = "./production_v2_2_0/weights/model_A_hybrid.pth"
    model_b_path = "./production_v2_2_0/weights/model_B_hybrid.pth"
    sample_frame_path = r"C:\Users\akash\Downloads\cholecseg8k\video01\video01_00080\frame_99_endo.png"
    
    try:
        engine = ClinicalInferenceEngine(model_a_path, model_b_path)
        print("Engine: LOADED (V2.2.1-HYBRID)")
    except Exception as e:
        print(f"Engine Load FAILED: {e}")
        return

    frame = cv2.imread(sample_frame_path)
    if frame is None: 
        print(f"Frame Load FAILED: {sample_frame_path}")
        return
    
    # Comparative Audit
    run_audit_mode(engine, frame, "NEURAL ONLY", use_heuristics=False)
    run_audit_mode(engine, frame, "HYBRID ENHANCED", use_heuristics=True)

    print("\n--- COMPARATIVE AUDIT COMPLETE ---")

if __name__ == "__main__":
    final_audit()
