import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset, random_split
from src.cholec_dataset import CholecSeg8kDataset, get_transforms
from src.model import get_model, HybridLoss
from tqdm import tqdm
import torch.cuda.amp as amp
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import time
import albumentations as A
from albumentations.pytorch import ToTensorV2

class DataPreprocessor:
    """Advanced data preprocessing for surgical images"""
    
    def __init__(self):
        self.liver_hsv_ranges = [
            (np.array([35, 40, 40]), np.array([85, 255, 255])),  # Green range
            (np.array([20, 40, 40]), np.array([35, 255, 255])),  # Yellow-green range
        ]
        
        self.instrument_hsv_ranges = [
            (np.array([0, 50, 50]), np.array([10, 255, 255])),    # Red range 1
            (np.array([170, 50, 50]), np.array([180, 255, 255])), # Red range 2
            (np.array([100, 50, 50]), np.array([130, 255, 255])), # Blue range
            (np.array([90, 50, 50]), np.array([110, 255, 255])),  # Cyan range
        ]
    
    def enhance_surgical_image(self, image):
        """Enhance surgical image quality"""
        # Convert to LAB color space for better contrast enhancement
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        # Merge back and convert to RGB
        lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Reduce noise while preserving edges
        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        return denoised
    
    def create_refined_mask(self, color_mask):
        """Create refined segmentation mask with better class separation"""
        h, w = color_mask.shape[:2]
        refined_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(color_mask, cv2.COLOR_BGR2HSV)
        
        # Create liver mask with multiple ranges
        liver_mask = np.zeros((h, w), dtype=np.uint8)
        for lower, upper in self.liver_hsv_ranges:
            range_mask = cv2.inRange(hsv, lower, upper)
            liver_mask = cv2.bitwise_or(liver_mask, range_mask)
        
        # Create instrument mask with multiple ranges
        instrument_mask = np.zeros((h, w), dtype=np.uint8)
        for lower, upper in self.instrument_hsv_ranges:
            range_mask = cv2.inRange(hsv, lower, upper)
            instrument_mask = cv2.bitwise_or(instrument_mask, range_mask)
        
        # Morphological operations for cleanup
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Clean up liver mask
        liver_mask = cv2.morphologyEx(liver_mask, cv2.MORPH_OPEN, kernel_small)
        liver_mask = cv2.morphologyEx(liver_mask, cv2.MORPH_CLOSE, kernel_medium)
        liver_mask = cv2.GaussianBlur(liver_mask, (3, 3), 0)
        
        # Clean up instrument mask
        instrument_mask = cv2.morphologyEx(instrument_mask, cv2.MORPH_OPEN, kernel_small)
        instrument_mask = cv2.morphologyEx(instrument_mask, cv2.MORPH_CLOSE, kernel_medium)
        
        # Remove small noise
        liver_mask = self.remove_small_objects(liver_mask, min_size=50)
        instrument_mask = self.remove_small_objects(instrument_mask, min_size=30)
        
        # Handle overlaps (instruments take priority)
        overlap = cv2.bitwise_and(liver_mask, instrument_mask)
        liver_mask = cv2.bitwise_xor(liver_mask, overlap)
        
        # Assign classes
        refined_mask[liver_mask > 127] = 1  # Liver
        refined_mask[instrument_mask > 127] = 2  # Instruments
        
        return refined_mask
    
    def remove_small_objects(self, mask, min_size=50):
        """Remove small connected objects from mask"""
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        cleaned = np.zeros_like(mask)
        
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_size:
                cleaned[labels == i] = 255
        
        return cleaned
    
    def apply_data_augmentation(self, image, mask):
        """Apply surgical-specific data augmentation"""
        # Random brightness and contrast adjustments
        if np.random.rand() > 0.5:
            brightness = np.random.uniform(0.8, 1.2)
            contrast = np.random.uniform(0.9, 1.1)
            image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness * 10)
        
        # Simulate surgical smoke (haze effect)
        if np.random.rand() > 0.7:
            smoke_layer = np.random.uniform(0, 30, image.shape).astype(np.uint8)
            image = cv2.add(image, smoke_layer)
        
        # Random motion blur (camera shake)
        if np.random.rand() > 0.8:
            kernel_size = np.random.choice([3, 5])
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
            kernel = kernel / kernel_size
            image = cv2.filter2D(image, -1, kernel)
        
        return image, mask

class EnhancedCholecSeg8kDataset(torch.utils.data.Dataset):
    """Enhanced dataset with preprocessing"""
    
    def __init__(self, root_dir, transform=None, target_size=(512, 512), max_samples=1000, preprocess=True):
        self.root_dir = root_dir
        self.transform = transform
        self.target_size = target_size
        self.preprocessor = DataPreprocessor() if preprocess else None
        
        # Collect all frame directories
        self.frame_dirs = []
        for video_dir in os.listdir(root_dir):
            video_path = os.path.join(root_dir, video_dir)
            if os.path.isdir(video_path):
                for frame_dir in os.listdir(video_path):
                    frame_path = os.path.join(video_path, frame_dir)
                    if os.path.isdir(frame_path):
                        img_files = [f for f in os.listdir(frame_path) if f.endswith('_endo.png')]
                        if img_files:
                            self.frame_dirs.append(frame_path)
        
        # Limit samples for faster training
        if max_samples and len(self.frame_dirs) > max_samples:
            self.frame_dirs = np.random.choice(self.frame_dirs, max_samples, replace=False).tolist()
            
        print(f"Loaded {len(self.frame_dirs)} samples with preprocessing")
        
    def __len__(self):
        return len(self.frame_dirs)
    
    def __getitem__(self, idx):
        frame_dir = self.frame_dirs[idx]
        
        # Find the image file
        img_files = [f for f in os.listdir(frame_dir) if f.endswith('_endo.png')]
        img_file = img_files[0]
        img_path = os.path.join(frame_dir, img_file)
        
        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load and process mask
        mask_files = [f for f in os.listdir(frame_dir) if f.endswith('_endo_color_mask.png')]
        if mask_files:
            mask_path = os.path.join(frame_dir, mask_files[0])
            color_mask = cv2.imread(mask_path)
            
            if self.preprocessor:
                # Enhanced image preprocessing
                image = self.preprocessor.enhance_surgical_image(image)
                # Refined mask creation
                mask = self.preprocessor.create_refined_mask(color_mask)
            else:
                # Basic mask conversion
                mask = self.convert_color_mask_to_classes(color_mask)
        else:
            # Fallback to binary mask
            mask_files = [f for f in os.listdir(frame_dir) if f.endswith('_endo_mask.png')]
            mask_path = os.path.join(frame_dir, mask_files[0])
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = np.where(mask > 0, 1, 0)
        
        # Apply surgical-specific augmentation
        if self.preprocessor and np.random.rand() > 0.5:
            image, mask = self.preprocessor.apply_data_augmentation(image, mask)
        
        # Resize
        image = cv2.resize(image, (self.target_size[1], self.target_size[0]))
        mask = cv2.resize(mask, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_NEAREST)
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask'].long()
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).long()
            
        return image, mask
    
    def convert_color_mask_to_classes(self, color_mask):
        """Fallback color mask conversion"""
        h, w = color_mask.shape[:2]
        class_mask = np.zeros((h, w), dtype=np.uint8)
        
        hsv = cv2.cvtColor(color_mask, cv2.COLOR_BGR2HSV)
        
        # Basic liver detection (green)
        liver_mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
        
        # Basic instrument detection (red/blue)
        inst_mask1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
        inst_mask2 = cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
        inst_mask3 = cv2.inRange(hsv, np.array([100, 50, 50]), np.array([130, 255, 255]))
        instrument_mask = inst_mask1 | inst_mask2 | inst_mask3
        
        class_mask[liver_mask > 0] = 1
        class_mask[instrument_mask > 0] = 2
        
        return class_mask

# Use the enhanced dataset in training
def create_enhanced_dataset(config):
    """Create enhanced dataset with preprocessing"""
    datasets = []
    
    if os.path.exists(config['cholecseg8k_path']):
        print(f"Loading Enhanced CholecSeg8k from {config['cholecseg8k_path']}")
        cholec_dataset = EnhancedCholecSeg8kDataset(
            root_dir=config['cholecseg8k_path'],
            transform=get_transforms(is_train=True),
            max_samples=1200,  # Increased for better training
            preprocess=True
        )
        datasets.append(cholec_dataset)
    
    if not datasets:
        raise ValueError("No valid datasets found!")
    
    unified_dataset = ConcatDataset(datasets)
    
    # Create train/val split
    val_size = int(0.2 * len(unified_dataset))
    train_size = len(unified_dataset) - val_size
    train_dataset, val_dataset = random_split(unified_dataset, [train_size, val_size])
    
    print(f"Enhanced dataset - Total: {len(unified_dataset)}, Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    return train_dataset, val_dataset

# Rest of the training code remains the same as in train_advanced.py
# Import all the classes and functions from the previous training script
exec(open('train_advanced.py').read().replace('create_unified_dataset', 'create_enhanced_dataset').replace('CholecSeg8kDataset', 'EnhancedCholecSeg8kDataset'))
