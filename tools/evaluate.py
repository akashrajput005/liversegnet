import os
import torch
import numpy as np
import yaml
from torch.utils.data import DataLoader
try:
    from dataset import LiverInstrumentDataset, get_transforms
    from model import get_model
except ImportError:
    from src.dataset import LiverInstrumentDataset, get_transforms
    from src.model import get_model
from tqdm import tqdm
from sklearn.metrics import jaccard_score

def dice_coefficient(y_true, y_pred, epsilon=1e-6):
    """
    y_true, y_pred: numpy arrays of same shape
    """
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection + epsilon) / (np.sum(y_true) + np.sum(y_pred) + epsilon)

def evaluate_model(model, loader, device, num_classes=3):
    model.eval()
    dice_scores = {c: [] for c in range(1, num_classes)}
    iou_scores = {c: [] for c in range(1, num_classes)}
    
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Evaluating"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            masks = masks.cpu().numpy()
            
            for c in range(1, num_classes):
                y_true = (masks == c).astype(np.float32)
                y_pred = (preds == c).astype(np.float32)
                
                if np.sum(y_true) > 0: # Only evaluate if class exists in ground truth
                    dice = dice_coefficient(y_true, y_pred)
                    iou = jaccard_score(y_true.flatten(), y_pred.flatten(), pos_label=1, zero_division=1.0)
                    dice_scores[c].append(dice)
                    iou_scores[c].append(iou)
                    
    results = {}
    for c in range(1, num_classes):
        results[f'class_{c}_dice'] = np.mean(dice_scores[c]) if dice_scores[c] else 0.0
        results[f'class_{c}_iou'] = np.mean(iou_scores[c]) if iou_scores[c] else 0.0
        
    results['mean_dice'] = np.mean([results[f'class_{c}_dice'] for c in range(1, num_classes)])
    results['mean_iou'] = np.mean([results[f'class_{c}_iou'] for c in range(1, num_classes)])
    
    return results

if __name__ == "__main__":
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    test_dataset = LiverInstrumentDataset(
        unified_root=config['unified_dataset_path'],
        split='val', 
        transform=get_transforms(is_train=False)
    )
    
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])
    
    model = get_model(
        architecture=config['active_model'], 
        encoder=config['active_encoder'], 
        num_classes=3
    ).to(device)
    
    model_path = f"models/{config['active_model']}_{config['active_encoder']}.pth"
    if os.path.exists(model_path):
        print(f"Loading advanced weights from {model_path}...")
        model.load_state_dict(torch.load(model_path))
        results = evaluate_model(model, test_loader, device)
        print("Final Advanced Results:", results)
        
        # Save to JSON for reliability
        import json
        with open('results/metrics_advanced.json', 'w') as f:
            serializable_results = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v for k, v in results.items()}
            json.dump(serializable_results, f, indent=4)
        print("Results saved to results/metrics_advanced.json")
    else:
        print("Final model not found. please train first.")
