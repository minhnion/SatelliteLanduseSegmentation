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
    def __init__(self, n_classes, last_fm_chan, scale_factor, bilinear=True):
        super().__init__()
        factor = 2 if bilinear else 1

        # U-Net decoder blocks with proper channel dimensions
        self.up0 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Correct channel dimensions based on StudentEncoder feature maps
        # feature_map1: [B, 64, H/4, W/4]
        # feature_map2: [B, 128, H/8, W/8]
        # feature_map3: [B, 256, H/16, W/16]
        # feature_map4: [B, 512/2048, H/32, W/32]
        # feature_map5: [B, 32, H/8, W/8]

        # Proper channel handling for decoder
        # self.up1 = Up(input_emb_dim *2 , 256 // factor, bilinear)  # Combine embedding + feature_map4
        # self.up2 = Up(256 + 256, 128 // factor, bilinear)           # Combine up1 + feature_map3
        # self.up3 = Up(128 + 128, 64 // factor, bilinear)            # Combine up2 + feature_map2
        # self.up4 = Up(64 + 64, 64, bilinear)                        # Combine up3 + feature_map1
        # self.outc = OutConv(64, n_classes)

        self.up1 = Up(last_fm_chan*2, last_fm_chan // factor, bilinear)
        self.up2 = Up(last_fm_chan, last_fm_chan // (2 * factor), bilinear)
        self.up3 = Up(last_fm_chan // 2, last_fm_chan // (4 * factor), bilinear)
        self.up4 = Up(last_fm_chan // 4, last_fm_chan // (8 * factor), bilinear)
        self.outc = OutConv(last_fm_chan // (8 * factor), n_classes)

        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=4*scale_factor, mode='bilinear', align_corners=True),
            nn.Conv2d(n_classes, n_classes, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, feature_maps):
        """
        Process feature maps from encoder to produce segmentation map

        Args:
            feature_maps: List of feature maps [feature_map1, feature_map2, feature_map3, feature_map4, feature_map5]
                feature_map1: [B, 64, H/4, W/4]
                feature_map2: [B, 128, H/8, W/8]
                feature_map3: [B, 256, H/16, W/16]
                feature_map4: [B, 512/2048, H/32, W/32]
                feature_map5: [B, 32, H/8, W/8] - The embedding
        """
        # Use embedding (feature_map5) as the starting point
        x = feature_maps[3]  # [batch_size, 32, H/8, W/8]
        # breakpoint()

        # Upsample to match feature_map4 resolution
        x = self.up0(x)  # [batch_size, 32, H/4, W/4]

        # Combine with feature maps in U-Net style with skip connections
        x = self.up1(x, feature_maps[3])  # Combine with feature_map4: [batch_size, 256, H/16, W/16]
        x = self.up2(x, feature_maps[2])  # Combine with feature_map3: [batch_size, 128, H/8, W/8]
        x = self.up3(x, feature_maps[1])  # Combine with feature_map2: [batch_size, 64, H/4, W/4]
        x = self.up4(x, feature_maps[0])  # Combine with feature_map1: [batch_size, 64, H/2, W/2]

        # Final convolution to get class predictions
        x = self.outc(x)  # [batch_size, n_classes, H/2, W/2]

        # Upsample to get original resolution
        x = self.final_up(x)
        return x
