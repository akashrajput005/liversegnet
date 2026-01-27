import torch
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from model import get_model

def create_baseline_models():
    """Create baseline models with ImageNet weights for immediate use"""
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    print("Creating baseline models...")
    
    # 1. Main DeepLabV3+ model (3-class)
    print("Creating DeepLabV3+ ResNet50 (3-class)...")
    main_model = get_model(architecture='deeplabv3plus', encoder='resnet50', num_classes=3)
    torch.save(main_model.state_dict(), 'models/deeplabv3plus_resnet50.pth')
    print("✓ Saved: models/deeplabv3plus_resnet50.pth")
    
    # 2. U-Net anchor (3-class)
    print("Creating U-Net ResNet34 (3-class)...")
    unet_model = get_model(architecture='unet', encoder='resnet34', num_classes=3)
    torch.save(unet_model.state_dict(), 'models/unet_resnet34.pth')
    print("✓ Saved: models/unet_resnet34.pth")
    
    # 3. Stage 1 anatomy model (2-class)
    print("Creating DeepLabV3+ ResNet50 Stage 1 (2-class)...")
    stage1_model = get_model(architecture='deeplabv3plus', encoder='resnet50', num_classes=2)
    torch.save(stage1_model.state_dict(), 'models/deeplabv3plus_resnet50_stage1.pth')
    print("✓ Saved: models/deeplabv3plus_resnet50_stage1.pth")
    
    print("\n✅ All baseline models created successfully!")
    print("These models use ImageNet pre-trained weights and are ready for inference.")

if __name__ == "__main__":
    create_baseline_models()
