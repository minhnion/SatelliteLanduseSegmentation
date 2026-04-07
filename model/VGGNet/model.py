import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import VGG16_Weights

class VGGNetSegmentation(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(VGGNetSegmentation, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Load pretrained VGGNet and extract feature maps at different levels
        vgg = models.vgg16(weights=VGG16_Weights.DEFAULT).features

        # Modify the first layer to accept n_channels
        vgg[0] = nn.Conv2d(n_channels, 64, kernel_size=3, padding=1)

        # Encoder - split VGG into stages for skip connections
        self.enc1 = vgg[:4]    # 224 -> 224 (64 channels)
        self.enc2 = vgg[4:9]   # 224 -> 112 (128 channels)
        self.enc3 = vgg[9:16]  # 112 -> 56 (256 channels)
        self.enc4 = vgg[16:23] # 56 -> 28 (512 channels)
        self.enc5 = vgg[23:30] # 28 -> 14 (512 channels)

        # Decoder - progressively upsample and combine with encoder features
        self.dec5 = self._make_decoder_block(512, 512)  # 14 -> 28
        self.dec4 = self._make_decoder_block(1024, 256) # 28 -> 56
        self.dec3 = self._make_decoder_block(512, 128)  # 56 -> 112
        self.dec2 = self._make_decoder_block(256, 64)   # 112 -> 224

        # Final classification layer
        self.final = nn.Conv2d(128, n_classes, kernel_size=1)

    def _make_decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)      # 224x224, 64
        e2 = self.enc2(e1)     # 112x112, 128
        e3 = self.enc3(e2)     # 56x56, 256
        e4 = self.enc4(e3)     # 28x28, 512
        e5 = self.enc5(e4)     # 14x14, 512

        # Decoder with skip connections
        d5 = self.dec5(e5)     # 28x28, 512
        d5 = torch.cat([d5, e4], dim=1)  # 28x28, 1024

        d4 = self.dec4(d5)     # 56x56, 256
        d4 = torch.cat([d4, e3], dim=1)  # 56x56, 512

        d3 = self.dec3(d4)     # 112x112, 128
        d3 = torch.cat([d3, e2], dim=1)  # 112x112, 256

        d2 = self.dec2(d3)     # 224x224, 64
        d2 = torch.cat([d2, e1], dim=1)  # 224x224, 128

        # Final classification
        out = self.final(d2)   # 224x224, n_classes

        return out

# Example usage
if __name__ == "__main__":
    model = VGGNetSegmentation(n_channels=13, n_classes=5)
    input_tensor = torch.randn(1, 13, 224, 224)
    output = model(input_tensor)
    print(output.shape)
