import numpy as np
import cv2
from scipy.spatial.distance import cdist

def calculate_occlusion(liver_mask, instrument_mask):
    """
    Calculate percentage of liver area occluded by instruments.
    Uses proximity-based occlusion for better surgical perception.
    """
    liver_area = np.sum(liver_mask)
    if liver_area == 0:
        return 0.0
    
    # Direct overlap
    intersection = np.logical_and(liver_mask, instrument_mask)
    occluded_area = np.sum(intersection)
    
    # Proximity-based: dilate instruments to capture "blocking" effect
    if np.sum(instrument_mask) > 0:
        kernel = np.ones((5,5), np.uint8)
        inst_dilated = cv2.dilate(instrument_mask.astype(np.uint8), kernel, iterations=1)
        proximity_occlusion = np.logical_and(liver_mask, inst_dilated)
        occluded_area = max(occluded_area, np.sum(proximity_occlusion))
    
    return (occluded_area / liver_area) * 100.0

def calculate_min_distance(liver_mask, instrument_mask):
    """
    Calculate minimum Euclidean distance between instrument boundary and liver surface.
    Returns distance in pixels.
    """
    if np.sum(liver_mask) == 0 or np.sum(instrument_mask) == 0:
        return float('inf')
    
    # Find contours
    liver_contours, _ = cv2.findContours(liver_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    inst_contours, _ = cv2.findContours(instrument_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not liver_contours or not inst_contours:
        return float('inf')
    
    # Extract points from all contours
    liver_pts = np.vstack([c.reshape(-1, 2) for c in liver_contours])
    inst_pts = np.vstack([c.reshape(-1, 2) for c in inst_contours])
    
    # Compute min distance between point sets
    distances = cdist(inst_pts, liver_pts)
    return np.min(distances)

def get_overlay(image_bgr, mask):
    """
    Create high-fidelity BGR overlay for CV2 processing.
    image_bgr: (H, W, 3) BGR format
    mask: (H, W) with values 0, 1, 2
    """
    overlay = image_bgr.copy()
    
    # Enhanced colors for better visibility
    LIVER_COLOR_BGR = [50, 255, 100]    # Bright Green
    TOOL_COLOR_BGR = [0, 100, 255]      # Bright Red-Orange
    
    alpha = 0.5  # Increased opacity for better visibility
    
    # 1. Liver Overlay (Class 1) - Draw first so tools appear on top
    mask_liver = (mask == 1)
    if np.any(mask_liver):
        overlay[mask_liver] = (image_bgr[mask_liver] * (1 - alpha) + np.array(LIVER_COLOR_BGR) * alpha).astype(np.uint8)
    
    # 2. Tool Overlay (Class 2) - Draw on top
    mask_tool = (mask == 2)
    if np.any(mask_tool):
        overlay[mask_tool] = (image_bgr[mask_tool] * (1 - alpha) + np.array(TOOL_COLOR_BGR) * alpha).astype(np.uint8)
    
    # 3. Enhanced Outlines with glow effect for better visibility
    for cid, color in [(1, LIVER_COLOR_BGR), (2, TOOL_COLOR_BGR)]:
        c_mask = (mask == cid).astype(np.uint8)
        cnts, _ = cv2.findContours(c_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Draw thicker outer glow
        cv2.drawContours(overlay, cnts, -1, [int(c*0.6) for c in color], 4)
        # Draw sharp inner line
        cv2.drawContours(overlay, cnts, -1, color, 2)
    
    return overlay
