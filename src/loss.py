import torch
import torch.nn as nn
import torch.nn.functional as F


def _mask_to_boundary(mask):
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    mask = mask.float()
    dilated = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
    eroded = 1.0 - F.max_pool2d(1.0 - mask, kernel_size=3, stride=1, padding=1)
    boundary = (dilated - eroded).clamp(0, 1)
    return boundary


def dice_loss(logits, target, class_idx):
    probs = torch.softmax(logits, dim=1)
    pred = probs[:, class_idx]
    gt = (target == class_idx).float()
    intersection = (pred * gt).sum()
    union = pred.sum() + gt.sum()
    return 1 - (2 * intersection + 1e-6) / (union + 1e-6)


def focal_loss(logits, target, gamma):
    ce = F.cross_entropy(logits, target, reduction="none")
    pt = torch.exp(-ce)
    return ((1 - pt) ** gamma * ce).mean()


def boundary_loss(logits, target):
    with torch.cuda.amp.autocast(enabled=False):
        logits = logits.float()
        target = target.float()
        probs = torch.softmax(logits, dim=1)
        loss = 0.0
        for cls in [1, 2]:
            gt = (target == cls).float()
            boundary = _mask_to_boundary(gt)
            prob = probs[:, cls].unsqueeze(1)
            loss += F.binary_cross_entropy(prob, boundary)
        return loss / 2.0


class Stage1Loss(nn.Module):
    def __init__(
        self,
        focal_gamma=2.5,
        boundary_weight=1.2,
        dice_weight_bg=0.5,
        dice_weight_liver=1.0,
        dice_weight_inst=1.0,
    ):
        super().__init__()
        self.focal_gamma = focal_gamma
        self.boundary_weight = boundary_weight
        self.dice_weight_bg = dice_weight_bg
        self.dice_weight_liver = dice_weight_liver
        self.dice_weight_inst = dice_weight_inst
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, target, source):
        device = logits.device
        total_loss = torch.tensor(0.0, device=device)

        seg8k_idx = (source == 0).nonzero(as_tuple=True)[0]
        if seg8k_idx.numel() > 0:
            seg_logits = logits[seg8k_idx]
            seg_target = target[seg8k_idx]
            loss = 0.0
            if self.dice_weight_bg > 0:
                loss = loss + self.dice_weight_bg * dice_loss(seg_logits, seg_target, 0)
            loss = loss + self.dice_weight_liver * dice_loss(seg_logits, seg_target, 1)
            loss = loss + self.dice_weight_inst * dice_loss(seg_logits, seg_target, 2)
            loss = loss + focal_loss(seg_logits, seg_target, self.focal_gamma)
            loss = loss + self.boundary_weight * boundary_loss(seg_logits, seg_target)
            total_loss = total_loss + loss

        inst_idx = (source == 1).nonzero(as_tuple=True)[0]
        if inst_idx.numel() > 0:
            inst_logits = logits[inst_idx, 2, :, :]
            inst_target = target[inst_idx].float()
            total_loss = total_loss + self.bce(inst_logits, inst_target)

        return total_loss


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits, target):
        probs = torch.sigmoid(logits)
        target = target.float()
        tp = (probs * target).sum()
        fp = (probs * (1 - target)).sum()
        fn = ((1 - probs) * target).sum()
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1 - tversky


class Stage2Loss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.tversky = TverskyLoss(alpha=alpha, beta=beta)

    def forward(self, logits, target):
        return self.bce(logits, target) + self.tversky(logits, target)
