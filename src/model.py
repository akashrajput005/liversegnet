import segmentation_models_pytorch as smp
import torch
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
    def __init__(self, num_classes=3, ignore_index=255):
        super(HybridLoss, self).__init__()

        if num_classes <= 1:
            raise ValueError('num_classes must be >= 2')

        # Dice should usually focus on foreground classes (exclude background=0)
        if num_classes == 2:
            dice_classes = [1]
        else:
            dice_classes = list(range(1, num_classes))

        self.dice_loss = smp.losses.DiceLoss(mode='multiclass', classes=dice_classes, ignore_index=255)
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=255)
        
    def mixed_supervision_loss_per_sample(self, pred, gt, eps=1e-6):
        """THE CORRECT FIX: Per-sample loss aggregation
        
        Binary-only samples train liver
        Polygon-only samples train instrument  
        Mixed samples train both
        No gradient dilution
        """
        total_loss = 0.0
        count = 0
        
        for b in range(pred.shape[0]):
            # Liver (class 1)
            if gt[b, 1].sum() > 0:
                pred_liver = pred[b, 1]
                gt_liver = gt[b, 1]
                inter = (pred_liver * gt_liver).sum()
                denom = pred_liver.sum() + gt_liver.sum()
                dice_liver = 1 - (2 * inter + eps) / (denom + eps)
                total_loss += dice_liver
                count += 1

            # Instrument (class 2)
            if gt[b, 2].sum() > 0:
                pred_inst = pred[b, 2]
                gt_inst = gt[b, 2]
                inter = (pred_inst * gt_inst).sum()
                denom = pred_inst.sum() + gt_inst.sum()
                dice_inst = 1 - (2 * inter + eps) / (denom + eps)
                total_loss += dice_inst
                count += 1

        if count == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        return total_loss / count
        
    def forward(self, y_pred, y_true):
        # y_pred: (B, C, H, W)
        # y_true: (B, H, W)
        
        # DEBUG: Check mask values
        unique_vals = torch.unique(y_true)
        if torch.any(unique_vals < 0) or torch.any(unique_vals >= 3):
            print(f"🚨 INVALID MASK VALUES: min={unique_vals.min().item()}, max={unique_vals.max().item()}, unique={unique_vals.tolist()}")
            # Clamp invalid values to valid range
            y_true = torch.clamp(y_true, 0, 2)
            # Verify after clamping
            unique_fixed = torch.unique(y_true)
            print(f"✅ Fixed mask values: {unique_fixed.tolist()}")
        
        # Additional validation for extreme cases
        if torch.any(torch.isnan(y_true)) or torch.any(torch.isinf(y_true)):
            print(f"🚨 NaN/Inf in ground truth masks")
            y_true = torch.clamp(y_true, 0, 2)
        
        # Convert to one-hot for per-class loss calculation
        y_true_onehot = torch.nn.functional.one_hot(y_true, num_classes=3).permute(0, 3, 1, 2).float()
        
        # Use CORRECT per-sample loss aggregation
        dice = self.mixed_supervision_loss_per_sample(y_pred, y_true_onehot)
        ce = self.ce_loss(y_pred, y_true)
        
        return 0.5 * dice + 0.5 * ce
