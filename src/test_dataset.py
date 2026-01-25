import yaml
import torch
try:
    from dataset import LiverInstrumentDataset, get_transforms
except ImportError:
    from src.dataset import LiverInstrumentDataset, get_transforms
import matplotlib.pyplot as plt
import numpy as np

def test_dataset():
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    dataset = LiverInstrumentDataset(
        cholecseg8k_root=config['cholecseg8k_path'],
        cholecinstance_root=config['cholecinstanceseg_path'],
        ref_image_root=config['reference_images_path'],
        split='train',
        transform=get_transforms(is_train=True)
    )
    
    print(f"Total samples: {len(dataset)}")
    
    if len(dataset) == 0:
        print("Dataset is empty. check paths.")
        return

    # Get a few samples and visualize
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    for i in range(4):
        idx = np.random.randint(0, len(dataset))
        image, mask = dataset[idx]
        
        # Unnormalize image for visualization
        image_np = image.permute(1, 2, 0).numpy()
        image_np = image_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        image_np = np.clip(image_np, 0, 1)
        
        mask_np = mask.numpy()
        
        axes[0, i].imshow(image_np)
        axes[0, i].set_title(f"Sample {idx}")
        axes[0, i].axis('off')
        
        axes[1, i].imshow(mask_np, cmap='jet')
        axes[1, i].set_title(f"Mask {idx}")
        axes[1, i].axis('off')
        
    plt.tight_layout()
    plt.savefig('results/dataset_verification.png')
    print("Verification plot saved to results/dataset_verification.png")
    
    # Check classes
    unique_classes = torch.unique(mask)
    print(f"Unique classes in mask: {unique_classes.tolist()}")

if __name__ == "__main__":
    test_dataset()
