import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

class GeometrySafetyLayer:
    """
    Deterministic Geometry Layer for Risk Assessment.
    No neural networks here. Purely geometric logic.
    """
    SAFETY_LOGIC_VERSION = "1.0.0-deterministic"

    def __init__(self, warning_threshold=50.5, critical_threshold=20.5):
        self.warning_threshold = warning_threshold  # in pixels (calibration needed for mm)
        self.critical_threshold = critical_threshold

    def extract_tool_tips(self, tool_mask):
        """
        V2.0.5: Shell-Aware Tip Detection (Clinical Grade).
        Identifies the tool 'Base' via distance-to-boundary and reaches for the 'Tip'.
        """
        tips = []
        h, w = tool_mask.shape[:2]
        contours, _ = cv2.findContours(tool_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            cnt_pts = cnt.reshape(-1, 2)
            if len(cnt_pts) < 5: continue
            
            # 1. Base Detection: Point closest to any image shell (fixes vignette issues)
            # dist_to_shell = min(x, w-x, y, h-y)
            dists_to_shell = np.minimum(
                np.minimum(cnt_pts[:, 0], w - 1 - cnt_pts[:, 0]),
                np.minimum(cnt_pts[:, 1], h - 1 - cnt_pts[:, 1])
            )
            base_idx = np.argmin(dists_to_shell)
            base_pt = cnt_pts[base_idx]
            
            # 2. Tip Detection: Point furthest from the Base
            dists_from_base = np.sqrt(np.sum((cnt_pts - base_pt)**2, axis=1))
            tip_idx = np.argmax(dists_from_base)
            tips.append(tuple(cnt_pts[tip_idx]))
            
        return tips

    def calculate_risk(self, anatomy_mask, tool_tips, velocity=0.0):
        """
        Calculates risk based on distance transform of anatomy (liver).
        V2: Includes kinetic margin expansion based on tool velocity.
        """
        # V2: Dynamic Threshold Expansion (Kinetic Safety)
        # Expansion factor: +0.5 pixels per unit of velocity
        VELOCITY_MULTIPLIER = 0.5
        adj_warning = self.warning_threshold + (velocity * VELOCITY_MULTIPLIER)
        adj_critical = self.critical_threshold + (velocity * VELOCITY_MULTIPLIER)

        telemetry = {
            "velocity": float(velocity),
            "adj_warning": float(adj_warning),
            "adj_critical": float(adj_critical),
            "status": "SAFE"
        }

        if not tool_tips:
            return "SAFE", float('inf'), telemetry

        # Distance Transform: Distance of each pixel from the liver boundary
        dist_transform = distance_transform_edt(1 - anatomy_mask)
        
        min_distance = float('inf')
        for tip in tool_tips:
            x, y = int(tip[0]), int(tip[1])
            if 0 <= y < dist_transform.shape[0] and 0 <= x < dist_transform.shape[1]:
                dist = dist_transform[y, x]
                if dist < min_distance:
                    min_distance = dist

        # Risk Classification (using adjusted thresholds)
        if min_distance <= adj_critical:
            status = "CRITICAL"
        elif min_distance <= adj_warning:
            status = "WARNING"
        else:
            status = "SAFE"
            
        telemetry = {
            "min_distance_px": float(min_distance),
            "velocity": float(velocity),
            "adj_warning": float(adj_warning),
            "adj_critical": float(adj_critical),
            "status": status
        }
        
        return status, min_distance, telemetry

    def get_audit_data(self, status, distance, tips, velocity_telemetry):
        return {
            "risk_status": status,
            "min_distance_px": float(distance),
            "num_tips_detected": len(tips),
            "tips": [list(t) for t in tips],
            "kinetic_telemetry": velocity_telemetry
        }
