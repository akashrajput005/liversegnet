import torch


def instrument_precision_recall(logits, target):
    preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)
    pred_inst = preds == 2
    targ_inst = target == 2
    tp = (pred_inst & targ_inst).sum().item()
    fp = (pred_inst & ~targ_inst).sum().item()
    fn = (~pred_inst & targ_inst).sum().item()
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    return precision, recall


def dice_score(logits, target, class_idx):
    preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)
    pred = preds == class_idx
    gt = target == class_idx
    intersection = (pred & gt).sum().item()
    union = pred.sum().item() + gt.sum().item()
    return (2 * intersection + 1e-6) / (union + 1e-6)


def compute_metrics(logits, target):
    inst_prec, inst_rec = instrument_precision_recall(logits, target)
    dice_liver = dice_score(logits, target, 1)
    dice_inst = dice_score(logits, target, 2)
    return {
        "instrument_precision": inst_prec,
        "instrument_recall": inst_rec,
        "dice_liver": dice_liver,
        "dice_instrument": dice_inst,
    }
