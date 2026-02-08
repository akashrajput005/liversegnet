import torch
import torch.nn as nn
import torchvision.models as models

class UNetResNet34(nn.Module):
    """
    Model B Architecture: U-Net with ResNet34 Encoder.
    Specialized for Tools, Sharp, and Thin objects.
    """
    def __init__(self, num_classes=3, pretrained=True): # Stage 1: Tools, Background
        super(UNetResNet34, self).__init__()
        
        # Encoder: ResNet34
        from torchvision.models import ResNet34_Weights
        weights = ResNet34_Weights.DEFAULT if pretrained else None
        resnet = models.resnet34(weights=weights)
        self.encoder = nn.ModuleList([
            nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool),
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4 
        ])

        # Decoder Blocks
        self.up4 = self._up_block(512, 256)
        self.up3 = self._up_block(256 + 256, 128)
        self.up2 = self._up_block(128 + 128, 64)
        self.up1 = self._up_block(64 + 64, 64)
        
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def _up_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        x = x.float()
        enc0 = self.encoder[0](x)
        enc1 = self.encoder[1](enc0)
        enc2 = self.encoder[2](enc1)
        enc3 = self.encoder[3](enc2)
        enc4 = self.encoder[4](enc3)

        # Decoder with Skip Connections
        up4 = self.up4(enc4)
        up3 = self.up3(torch.cat([up4, enc3], dim=1))
        up2 = self.up2(torch.cat([up3, enc2], dim=1))
        up1 = self.up1(torch.cat([up2, enc1], dim=1))
        
        # Upsample to original size
        out = torch.nn.functional.interpolate(up1, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        return self.final_conv(out)

    def load_stage1_weights(self, path):
        """Helper to load weights from Stage 1 for Stage 2 fine-tuning"""
        checkpoint = torch.load(path)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        
        # Filter out final convolution layer for class expansion
        filtered_dict = {k: v for k, v in state_dict.items() if 'final_conv' not in k}
        
        self.load_state_dict(filtered_dict, strict=False)
        print(f"Stage 1 weights (encoder + decoder) loaded from {path}. Final Conv re-initialized for Stage 2.")
