import torch
import timm

import torch.nn as nn

class SwinTransformerSegmentation(nn.Module):
    def __init__(self, num_classes=21, pretrained=True):
        super(SwinTransformerSegmentation, self).__init__()
        self.backbone = timm.create_model('swin_base_patch4_window7_224', pretrained=pretrained, features_only=True)
        self.decoder = nn.Conv2d(1024, num_classes, kernel_size=1)  # Adjust the input channels based on the backbone output

    def forward(self, x):
        features = self.backbone(x)[-1]  # Get the last feature map
        output = self.decoder(features)
        output = nn.functional.interpolate(output, size=x.shape[2:], mode='bilinear', align_corners=False)
        return output

if __name__ == "__main__":
    model = SwinTransformerSegmentation(num_classes=21, pretrained=True)
    input_tensor = torch.randn(1, 3, 224, 224)  # Example input
    output = model(input_tensor)
    print(output.shape)  # Should be [1, num_classes, 224, 224]
