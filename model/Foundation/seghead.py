
import torch
import torch.nn as nn
import torch.nn.functional as F

class Down(nn.Module):
    """
    Downscaling with maxpool then double conv
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

# Define the DoubleConv block, which consists of two convolutional layers followed by batch normalization and ReLU activation
class DoubleConv(nn.Module):
    """
    (convolution => [BN] => ReLU) * 2
    """
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

# Define the Up block, which upscales the input and concatenates it with the skip connection from the encoder, followed by a DoubleConv block
class Up(nn.Module):
    """
    Upscaling then double conv
    """
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

         # If bilinear interpolation is used, upscale using bilinear interpolation and reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            # Otherwise, use transposed convolution for upscaling
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # breakpoint()
        x1 = self.up(x1)
        # Calculate the difference in size between the input and the skip connection
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        # Pad the input to match the size of the skip connection
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # Concatenate the skip connection with the upscaled input
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# Define the final output convolution layer
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class SegmentationHead(nn.Module):
    """
    Land-use segmentation head using a U-Net-like decoder with ViT-style skip connections.
    Outputs a segmentation map with the specified number of classes.
    """
    def __init__(self, n_classes, bilinear=True):
        super().__init__()
        factor = 2 if bilinear else 1
        # U-Net decoder blocks
        self.up0 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up1 = Up(2048, 1024 // factor, bilinear)
        self.up2 = Up(1024, 512 // factor, bilinear)
        self.up3 = Up(512, 256 // factor, bilinear)
        self.up4 = Up(256, 128, bilinear)
        self.outc = OutConv(128, n_classes)

    def forward(self, feature_maps):
        # Start from the smallest scale and upsample with skip connections
        x = feature_maps[3]  # [batch_size, 1024, 64, 64]
        x = self.up0(x)  # [batch_size, 1024, 128, 128]
        x = self.up1(x, feature_maps[3])  # [batch_size, 512, 128, 128]
        x = self.up2(x, feature_maps[2])  # [batch_size, 256, 256, 256]
        x = self.up3(x, feature_maps[1])  # [batch_size, 128, 128, 128]
        x = self.up4(x, feature_maps[0])                   # [batch_size, 64, 256, 256]]
        x = self.outc(x)                  # [batch_size, n_classes, 256, 256]
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)  # [batch_size, n_classes, 256, 256]
        return x
