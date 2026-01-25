import numpy as np
import cv2
from scipy.spatial.distance import cdist

def calculate_occlusion(liver_mask, instrument_mask):
    """
    Calculate percentage of liver area occluded by instruments.
    Assumes liver_mask and instrument_mask are binary (0 or 1).
    """
    liver_area = np.sum(liver_mask)
    if liver_area == 0:
        return 0.0
    
    # Occlusion is the intersection of liver and instrument regions.
    # Note: In a 2D projection, the instrument 'overlaps' the liver.
    intersection = np.logical_and(liver_mask, instrument_mask)
    occluded_area = np.sum(intersection)
    
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
    
    # Class mapping using BGR constants (Blue, Green, Red)
    LIVER_COLOR_BGR = [0, 255, 0]    # Pure Green
    TOOL_COLOR_BGR = [0, 0, 255]     # Pure Red
    
    alpha = 0.4
    
    # 1. Tool Overlay (Class 2)
    mask_tool = (mask == 2)
    if np.any(mask_tool):
        overlay[mask_tool] = (image_bgr[mask_tool] * (1 - alpha) + np.array(TOOL_COLOR_BGR) * alpha).astype(np.uint8)
        
    # 2. Liver Overlay (Class 1)
    mask_liver = (mask == 1)
    if np.any(mask_liver):
        overlay[mask_liver] = (image_bgr[mask_liver] * (1 - alpha) + np.array(LIVER_COLOR_BGR) * alpha).astype(np.uint8)
    
    # 3. Sharp Outlines
    for cid, color in [(1, LIVER_COLOR_BGR), (2, TOOL_COLOR_BGR)]:
        c_mask = (mask == cid).astype(np.uint8)
        cnts, _ = cv2.findContours(c_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, cnts, -1, color, 2)
    
    return overlay
