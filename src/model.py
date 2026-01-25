import segmentation_models_pytorch as smp
import torch.nn as nn

def get_model(architecture='unet', encoder='resnet34', in_channels=3, num_classes=3):
    if architecture.lower() == 'unet':
        model = smp.Unet(
            encoder_name=encoder,
            encoder_weights='imagenet',
            in_channels=in_channels,
            classes=num_classes,
        )
    elif architecture.lower() == 'deeplabv3plus':
        model = smp.DeepLabV3Plus(
            encoder_name=encoder,
            encoder_weights='imagenet',
            in_channels=in_channels,
            classes=num_classes,
        )
    else:
        raise ValueError(f"Architecture {architecture} not supported.")
    
    return model

class HybridLoss(nn.Module):
    def __init__(self, num_classes=3):
        super(HybridLoss, self).__init__()
        self.dice_loss = smp.losses.DiceLoss(mode='multiclass', classes=num_classes)
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, y_pred, y_true):
        # y_pred: (B, C, H, W)
        # y_true: (B, H, W)
        dice = self.dice_loss(y_pred, y_true)
        ce = self.ce_loss(y_pred, y_true)
        return 0.5 * dice + 0.5 * ce
