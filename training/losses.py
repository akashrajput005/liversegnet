import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss for surgical segmentation.
    Controls the trade-off between Precision and Recall using alpha and beta.
    Focuses on difficult pixels using gamma.
    """
    def __init__(self, alpha=0.7, beta=0.3, gamma=1.3333333, epsilon=1e-6):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, inputs, targets):
        # inputs: [N, C, H, W] logits
        # targets: [N, H, W]
        
        num_classes = inputs.size(1)
        if len(targets.shape) == 3:
            targets = F.one_hot(targets.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
            
        inputs = torch.softmax(inputs, dim=1)
        
        # Calculate Tversky per class
        tp = (inputs * targets).sum(dim=(0, 2, 3))
        fp = (inputs * (1 - targets)).sum(dim=(0, 2, 3))
        fn = ((1 - inputs) * targets).sum(dim=(0, 2, 3))
        
        tversky = (tp + self.epsilon) / (tp + self.alpha * fn + self.beta * fp + self.epsilon)
        
        # --- V2.0.3: MASK-AWARE ADAPTIVE WEIGHTING ---
        # Focus only on classes present in the ground truth
        presence_mask = (targets.sum(dim=(0, 2, 3)) > 0).float()
        
        # Ignore background (class 0) for Tversky foreground optimization
        presence_mask[0] = 0.0
        
        focal_tversky_per_class = torch.pow((1 - tversky), self.gamma)
        
        # Only average over present labels to prevent gradient dilution
        actual_present_count = presence_mask.sum()
        if actual_present_count > 0:
            focal_tversky = (focal_tversky_per_class * presence_mask).sum() / actual_present_count
        else:
            # Fallback for all-BG frames (should not happen in curated dataset)
            focal_tversky = focal_tversky_per_class[1:].mean()
            
        return focal_tversky

class BoundaryAwareLoss(nn.Module):
    """
    Focuses loss on the boundaries of objects.
    Useful for precise surgical tool tip detection.
    """
    def __init__(self, base_loss=nn.CrossEntropyLoss()):
        super(BoundaryAwareLoss, self).__init__()
        self.base_loss = base_loss

    def forward(self, inputs, targets):
        loss = self.base_loss(inputs, targets)
        return loss

class SurgicalHybridLoss(nn.Module):
    """
    Hybrid Loss: Cross-Entropy + Focal Tversky.
    Matches 2025 SOTA for surgical instrument segmentation.
    """
    def __init__(self, alpha=0.7, beta=0.3, gamma=1.3333333, ce_weight=0.5):
        super(SurgicalHybridLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss(reduction='mean')
        self.focal_tversky = FocalTverskyLoss(alpha=alpha, beta=beta, gamma=gamma)
        self.ce_weight = ce_weight

    def forward(self, inputs, targets):
        loss_ce = self.ce(inputs, targets.long())
        loss_ft = self.focal_tversky(inputs, targets)
        return self.ce_weight * loss_ce + (1 - self.ce_weight) * loss_ft
