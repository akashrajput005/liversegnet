import numpy as np

class CalibrationEngine:
    """
    Deterministic Calibration Engine for Surgical Perception.
    Maps image pixel coordinates/distances to physical units (mm).
    """
    def __init__(self, camera_matrix=None, dist_coeffs=None, pixel_pitch=None):
        # Default fallback values for standard LAPAROSCOPIC optics if matrix not provided
        self.camera_matrix = camera_matrix if camera_matrix is not None else np.array([
            [800, 0, 960],
            [0, 800, 540],
            [0, 0, 1]
        ])
        self.dist_coeffs = dist_coeffs if dist_coeffs is not None else np.zeros(5)
        
        # Scaling factor: mm per pixel at a reference depth (e.g., 50mm from lens)
        # This is strictly deterministic for surgical consistency
        self.mm_per_pixel = 0.12  # Calibration constant

    def pixels_to_mm(self, pixel_distance):
        """
        Converts pixel distance to physical mm using deterministic scaling.
        Formula: mm = pixels * mm_per_pixel
        """
        return float(pixel_distance * self.mm_per_pixel)

    def calibrate_risk_thresholds(self, warning_mm=10, critical_mm=5):
        """
        Converts clinical mm thresholds to pixel thresholds for the geometry layer.
        """
        warning_px = warning_mm / self.mm_per_pixel
        critical_px = critical_mm / self.mm_per_pixel
        return int(warning_px), int(critical_px)

    def get_calibration_audit(self):
        """Returns deterministic calibration metadata for audit trails"""
        return {
            "calibration_version": "CALIB-1.0-FIXED",
            "mm_per_pixel_scale": self.mm_per_pixel,
            "optical_assumption": "Standard 10mm Laparoscope"
        }
