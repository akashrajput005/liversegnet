import cv2
import numpy as np

def compute_boundary_precision(mask, prob_map):
    """
    Measures the sharpness of the probability transition at the predicted boundary.
    Computes spatial gradients of the probability map and averages them along the contour.
    
    Parameters:
    - mask: 2D binary numpy array of the target structure segmentation (e.g. liver)
    - prob_map: 2D numpy array of probabilities (0.0 to 1.0)
    
    Returns:
    - float: Precision score between 0.0 and 1.0.
    """
    if mask is None or np.sum(mask > 0) == 0 or prob_map is None:
        return 0.0
        
    # Ensure binary mask of type uint8
    binary_mask = (mask > 0).astype(np.uint8)
    
    # Match spatial dimensions between probability map and mask
    if prob_map.shape[:2] != binary_mask.shape[:2]:
        prob_map = cv2.resize(prob_map, (binary_mask.shape[1], binary_mask.shape[0]))
        
    # 1. Extract boundary contour using Canny edge detector
    edges = cv2.Canny((binary_mask * 255).astype(np.uint8), 100, 200)
    if np.sum(edges > 0) == 0:
        return 0.0
        
    # 2. Compute spatial gradients of the probability map
    grad_x = cv2.Sobel(prob_map, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(prob_map, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    # 3. Average the gradient magnitude at edge pixels
    mean_grad = np.mean(grad_mag[edges > 0])
    
    # 4. Normalize to [0.0, 1.0] using sigmoid-like soft scaling
    # Standard Sobel with ksize=3 on 0.0-1.0 map gives magnitudes up to ~4.
    score = float(1.0 - np.exp(-1.8 * mean_grad))
    return min(max(score, 0.0), 1.0)

def compute_tissue_displacement(mask_a, tips, frame_shape):
    """
    Computes the estimated physical tissue displacement or proximity (in pixels).
    Uses the distance transform of the anatomical mask to find distance to closest tool tip.
    
    Parameters:
    - mask_a: 2D binary anatomy mask
    - tips: List of tool tip coordinates (x, y)
    - frame_shape: Tuple representing image dimensions (H, W)
    
    Returns:
    - float: Mean distance in pixels from the detected tips to the anatomy.
    """
    if mask_a is None or not tips or not np.any(mask_a > 0):
        return 0.0
        
    # Standardize masks
    anatomy_binary = (mask_a > 0).astype(np.uint8)
    
    # Distance transform: distance of each background pixel to the nearest foreground pixel
    dist_map = cv2.distanceTransform((1 - anatomy_binary), cv2.DIST_L2, 5)
    
    h_mask, w_mask = mask_a.shape[:2]
    h_frame, w_frame = frame_shape[:2]
    
    distances = []
    for tip in tips:
        # Scale tip coordinates to match mask dimensions
        tx = int(tip[0] * w_mask / 256.0) if w_mask != 256.0 else int(tip[0])
        ty = int(tip[1] * h_mask / 256.0) if h_mask != 256.0 else int(tip[1])
        
        tx = min(max(tx, 0), w_mask - 1)
        ty = min(max(ty, 0), h_mask - 1)
        
        distances.append(dist_map[ty, tx])
        
    if not distances:
        return 0.0
        
    # Scale distance to original frame space
    scale_factor = h_frame / h_mask
    mean_dist_scaled = np.mean(distances) * scale_factor
    return float(mean_dist_scaled)

def compute_consensus_score(mask_a, mask_b):
    """
    Measures spatial separation consensus. 
    1.0 means perfect separation (no overlap between tool and anatomy).
    0.0 means complete overlap.
    
    Parameters:
    - mask_a: 2D binary anatomy mask (resized or raw)
    - mask_b: 2D binary tool mask (resized or raw)
    
    Returns:
    - float: Consensus score [0.0, 1.0]
    """
    if mask_a is None or mask_b is None:
        return 1.0
        
    overlap = np.sum((mask_a > 0) & (mask_b > 0))
    union = np.sum((mask_a > 0) | (mask_b > 0))
    
    if union == 0:
        return 1.0
        
    # Consensus is high when overlap is zero
    consensus = 1.0 - (overlap / (union + 1e-8))
    return float(min(max(consensus, 0.0), 1.0))
