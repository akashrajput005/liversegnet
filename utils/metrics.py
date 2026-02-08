import numpy as np
import torch
import cv2

def calculate_boundary_precision(pred_mask, gt_mask, tolerance=2):
    """
    Measures precision of the segmentation boundary using morphological dilation.
    Useful for critical organs like the liver.
    """
    if np.sum(gt_mask) == 0: return 1.0 if np.sum(pred_mask) == 0 else 0.0
    
    # Extract boundaries
    kernel = np.ones((3,3), np.uint8)
    gt_boundary = cv2.dilate(gt_mask.astype(np.uint8), kernel) - gt_mask.astype(np.uint8)
    
    # Precision: how many predicted boundary pixels are within the 'tolerance' zone of gt boundary
    # For simplicity, we use the overlap of the dilated masks
    overlap = np.logical_and(pred_mask > 0, gt_boundary > 0)
    precision = np.sum(overlap) / (np.sum(pred_mask > 0) + 1e-6)
    return float(precision)

def calculate_tip_accuracy(pred_tips, gt_tips):
    """
    Calculates L2 distance between predicted tool tips and ground truth.
    Critical for tool-tissue interaction safety.
    """
    if not pred_tips or not gt_tips:
        return float('inf')
    
    # Matching tips and calculating L2
    # Simplified version for single tip detection
    dist = np.sqrt(np.sum((np.array(pred_tips[0]) - np.array(gt_tips[0]))**2))
    return float(dist)

def calculate_false_negative_sharps(pred_mask, gt_mask):
    """
    Penalizes missed sharp objects heavily. 
    Safety-critical metric.
    """
    intersection = np.logical_and(pred_mask, gt_mask)
    if np.sum(gt_mask) == 0: return 0.0
    recall = np.sum(intersection) / np.sum(gt_mask)
    return 1.0 - recall
