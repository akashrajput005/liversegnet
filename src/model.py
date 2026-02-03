import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

def get_model(architecture='unet', encoder='efficientnet-b4', in_channels=3, num_classes=3):
    """Pinnacle Model Factory: Enforces state-of-the-art encoders."""
    if architecture.lower() == 'unet':
        # Upgrade to EfficientNet-B4 for superior surgical detail
        encoder_name = 'efficientnet-b4' if encoder == 'resnet34' or encoder == 'efficientnet-b4' else encoder
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights='imagenet',
            in_channels=in_channels,
            classes=num_classes,
            decoder_attention_type='scse' # Advanced Spatial/Channel Attention
        )
    elif architecture.lower() == 'deeplabv3plus':
        # Upgrade to ResNet101 or EfficientNet-B5 for deep context
        encoder_name = 'resnet101' if encoder == 'resnet50' or encoder == 'resnet101' else encoder
        model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights='imagenet',
            in_channels=in_channels,
            classes=num_classes,
        )
    else:
        raise ValueError(f"Architecture {architecture} not supported.")
    
    return model

class AdvancedSurgicalLoss(nn.Module):
    """The Pinnacle Triple-Fusion Loss: Dice + Focal + Tversky.
    
    Optimizing for:
    - Dice: Global Shape overlap.
    - Focal: Hard-pixel mining (small tools, thin edges).
    - Tversky: High recall for sparse objects (Penalty for missing instruments).
    """
    def __init__(self, num_classes=3):
        super(AdvancedSurgicalLoss, self).__init__()
        mode = 'multiclass' if num_classes > 2 else 'binary'
        
        self.dice = smp.losses.DiceLoss(mode=mode, from_logits=True)
        self.focal = smp.losses.FocalLoss(mode=mode)
        # Tversky with alpha=0.3, beta=0.7 emphasizes Recall (less False Negatives)
        # This ensures the AI is hyper-aware of surgical instruments.
        self.tversky = smp.losses.TverskyLoss(mode=mode, from_logits=True, alpha=0.3, beta=0.7)
        
        # Weighted CE: PIXEL INTENSIFICATION (Based on Statistical Audit)
        # We set Instrument weight to 5.0 to overcome the 2.99% pixel occupancy.
        weights = torch.tensor([1.0, 2.0, 5.0]) 
        self.ce = nn.CrossEntropyLoss(weight=weights)
        
    def forward(self, y_pred, y_true):
        device = y_pred.device
        self.ce.weight = self.ce.weight.to(device)
        
        d_loss = self.dice(y_pred, y_true)
        f_loss = self.focal(y_pred, y_true)
        t_loss = self.tversky(y_pred, y_true)
        ce_loss = self.ce(y_pred, y_true)
        
        # Clinical Alpha-Weights for Pinnacle Stability
        return 0.4 * d_loss + 0.2 * f_loss + 0.2 * t_loss + 0.2 * ce_loss

class HybridLoss(nn.Module):
    """Legacy Hybrid Loss for backward compatibility or baseline testing."""
    def __init__(self, num_classes=3):
        super(HybridLoss, self).__init__()
        mode = 'multiclass' if num_classes > 2 else 'binary'
        self.dice = smp.losses.DiceLoss(mode=mode)
        self.ce = nn.CrossEntropyLoss()
        
    def forward(self, y_pred, y_true):
        return 0.5 * self.dice(y_pred, y_true) + 0.5 * self.ce(y_pred, y_true)

