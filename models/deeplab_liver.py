import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50

class LiverSegModelA(nn.Module):
    """
    Model A Architecture: DeepLabV3+ with ResNet50 Backbone.
    Specialized for Liver and Anatomy segmentation.
    """
    def __init__(self, num_classes=2, pretrained=True): # Stage 1: Liver, Background
        super(LiverSegModelA, self).__init__()
        # Using torchvision implementation as a robust base
        # DeepLabV3+ (standard DeepLabV3 in torchvision is easily extensible)
        from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
        weights = DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
        self.model = deeplabv3_resnet50(weights=weights)
        
        # Replace the classifier for our specific classes
        # The number of input channels for the classifier is 256 for DeepLabV3 output
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
        
        # DeepLabV3 in torchvision also has an auxiliary classifier
        if self.model.aux_classifier:
            self.model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        return self.model(x)

    def load_stage1_weights(self, path):
        """Helper to load weights from Stage 1 for Stage 2 fine-tuning"""
        checkpoint = torch.load(path)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        
        # Filter out the final classifier layers to handle class count mismatch
        # DeepLabV3 has 'classifier.4' and 'aux_classifier.4'
        filtered_dict = {k: v for k, v in state_dict.items() if 'classifier.4' not in k}
        
        self.load_state_dict(filtered_dict, strict=False)
        print(f"Stage 1 weights (backbone + features) loaded from {path}. Classifier heads re-initialized for Stage 2.")
