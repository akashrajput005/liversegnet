import torch
import torch.nn as nn


class RefinementHead(nn.Module):
    def __init__(self, in_channels=3, mid1=32, mid2=64):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid1),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid1, mid2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid2),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Conv2d(mid2, 1, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


def build_stage1_model(encoder_name="resnet101"):
    import segmentation_models_pytorch as smp

    model = smp.DeepLabV3Plus(
        encoder_name=encoder_name,
        encoder_weights="imagenet",
        classes=3,
        activation=None,
    )
    return model


def build_stage2_model():
    return RefinementHead()
