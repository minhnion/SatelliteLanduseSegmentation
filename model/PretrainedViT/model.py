import timm
from torchgeo.models.vit import ViTSmall16_Weights
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.unet_layers import *

def _init_parameter(model):
    for param in model.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform(param)

class EncoderBlock(nn.Module):
    def __init__(self, n_channels):
        super(EncoderBlock, self).__init__()

        # Encoder (Contracting Path)
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2
        self.down4 = Down(512, 1024 // factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x

class DecoderUpConv(nn.Module):
    def __init__(self, n_classes, relu=True, slope=0.1):
        super(DecoderUpConv, self).__init__()

        self.up1 = UpConv(256, 128)
        self.up2 = UpConv(128, 64)
        self.up3 = UpConv(64, 32)
        self.up4 = UpConv(32, 16)
        self.outc = OutConv(16, n_classes)
        self.relu = nn.LeakyReLU(negative_slope=slope) if relu else nn.Identity()


    def forward(self, x):
        x = self.up1(x)
        x = self.relu(x)
        x = self.up2(x)
        x = self.relu(x)
        x = self.up3(x)
        x = self.relu(x)
        x = self.up4(x)
        x = self.relu(x)
        x = self.outc(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, relu=True):
        super(DecoderBlock, self).__init__()

        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.relu = nn.ReLU() if relu else nn.Identity()

    def forward(self, x):
        x = self.upconv(x)
        x = self.relu(x)
        return x

class Decoder(nn.Module):
    def __init__(self, n_classes,):
        super(Decoder, self).__init__()

        self.upconv1 = DecoderBlock(384, 128)
        self.upconv2 = DecoderBlock(128, 64)
        self.upconv3 = DecoderBlock(64, 32)
        self.upconv3 = DecoderBlock(32, n_classes, relu=False)

    def forward(self, x):
        x = self.upconv1(x)
        x = self.upconv2(x)
        x = self.upconv3(x)
        x = self.upconv4(x)
        return x

class SegformerDecoder(nn.Module):
    def __init__(self, in_channels, n_classes, final_resolution=(224, 224)):
        """
        Segformer Decoder.

        Args:
            in_channels (int): Number of input channels from the ViT feature map.
            n_classes (int): Number of output segmentation classes.
            final_resolution (tuple): Desired spatial resolution of the output (H, W).
        """
        super(SegformerDecoder, self).__init__()

        self.final_resolution = final_resolution

        # Segformer Decoder block
        self.mlp1 = nn.Conv2d(in_channels, 256, kernel_size=1)
        self.mlp2 = nn.Conv2d(256, 256, kernel_size=1)
        self.upsample = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)  # Upsample by a factor of 2
        self.classifier = nn.Conv2d(256, n_classes, kernel_size=1)


    def forward(self, x):
        """
        Forward pass of the decoder.

        Args:
            x (Tensor): Input feature map from the ViT encoder.

        Returns:
            Tensor: Segmentation map with shape `[B, n_classes, final_resolution[0], final_resolution[1]]`.
        """
        x = self.mlp1(x)  # Reduce channels to 256
        x = F.relu(x)
        x = self.mlp2(x)  # Another projection
        x = F.relu(x)
        x = self.upsample(x)  # Upsample feature map (intermediate size)
        print(x.shape)
        x = F.interpolate(x, size=self.final_resolution, mode='bilinear', align_corners=False)  # Final upscale
        x = self.classifier(x)  # Generate segmentation logits
        return x

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Upsampling followed by convolution.
        """
        super(UpConv, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.activation = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):
        """
        Upsamples input and applies a convolution with activation.
        """
        x = self.upsample(x)
        x = self.conv(x)
        x = self.activation(x)
        return x

class ViT(nn.Module):
    def __init__(self,encoder, in_channels, n_classes, embedding_dim=384, use_timepoints=False):
        super().__init__()
        self.encoder = encoder
        self.embedding_dim = embedding_dim
        self.output_embed_dim = n_classes
        self.use_timepoints = use_timepoints
        # self.num_timepoints = 3
        num_convs=4
        num_convs_per_upscale=1

        self.channels = [self.embedding_dim // (2 ** i) for i in range(num_convs)]
        self.channels = [self.embedding_dim] + self.channels

        self.in_channels = in_channels


        def _build_upscale_block(channels_in, channels_out):

            conv_kernel_size = 3
            conv_padding = 1
            kernel_size = 2
            stride = 2
            dilation = 1
            padding = 0
            output_padding = 0

            layers = []
            layers.append(nn.ConvTranspose2d(
                channels_in,
                channels_out,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
                output_padding=output_padding,
            ))

            layers += [nn.Sequential(
                      nn.Conv2d(channels_out,
                      channels_out,
                      kernel_size=conv_kernel_size,
                      padding=conv_padding),
                      nn.BatchNorm2d(channels_out),
                      nn.Dropout(),
                      nn.ReLU()) for _ in range(num_convs_per_upscale)]

            return nn.Sequential(*layers)

        self.layers = nn.ModuleList([
            _build_upscale_block(self.channels[i], self.channels[i+1])
            for i in range(len(self.channels) - 1)
        ])

        self.conv = nn.Conv2d(self.channels[-1], self.output_embed_dim, kernel_size = 3, padding=1)

    def forward(self, image, target):
        bs, cxt, h, w = image.shape
        num_timepoints = cxt// self.in_channels
        if self.use_timepoints:
            image = image.reshape(bs, num_timepoints, -1, h, w)
            b, t, c, h, w = image.shape
            image = image.reshape(-1, c, h, w)

        feature = self.encoder.forward_features(image)
        if self.use_timepoints:
            temporal_features = feature.reshape(-1, num_timepoints, feature.shape[1], feature.shape[2]).permute(1,0,2,3)
            feature = torch.amax(temporal_features, dim = 0)
        feature = feature[:,1:,:].view(-1,14,14,self.embedding_dim).permute(0,3,1,2)
        for layer in self.layers:
            feature = layer(feature)
        raw_outputs = self.conv(feature)

        # loss = self.loss_fn(raw_outputs, target)

        outputs = raw_outputs

        return outputs

class PretrainedViT(nn.Module):
    def __init__(self, n_classes, img_size=224, slope=0.3):
        """
        Pretrained Vision Transformer (ViT) with a custom decoder for segmentation.
        """
        super(PretrainedViT, self).__init__()

        # Load the segmentation model weights
        weights = ViTSmall16_Weights.SENTINEL2_ALL_DINO

        # Load ViT encoder from timm
        self.encoder = timm.create_model('vit_small_patch16_224', in_chans=weights.meta['in_chans'])
        self.encoder.load_state_dict(weights.get_state_dict(progress=True), strict=False)

        # # Freeze encoder parameters
        # for param in self.encoder.parameters():
        #     param.requires_grad = True

        self.model = ViT(self.encoder, in_channels=13, n_classes=n_classes)

        # Unfreeze the last 100 layers for fine-tuning
        # params = list(self.encoder.parameters())
        # for i, param in enumerate(reversed(params)):
        #     if i < 100:
        #         param.requires_grad = True

        # Custom decoder layers
        # self.vit_conv = nn.Conv2d(384, 256, kernel_size=1, padding=0)
        # self.up1 = UpConv(256, 128)
        # self.up2 = UpConv(128, 64)
        # self.up3 = UpConv(64, 32)
        # self.up4 = UpConv(32, 16)
        # self.outc = nn.Conv2d(16, n_classes, kernel_size=1)

        # # Activation function
        # self.activation = nn.LeakyReLU(negative_slope=slope)
        # self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass: Extract features, reshape, and decode to segmentation map.
        """
        x = self.model(x, None)
        # Extract features from ViT encoder
        # features = self.encoder.forward_features(x)  # Output: [B, 197, 384]

        # # Separate class token and patch embeddings
        # cls_token, patch_embeddings = features[:, :1, :], features[:, 1:, :]  # [B, 1, 384], [B, 196, 384]

        # # Reshape patch embeddings to 2D spatial feature maps
        # B, N, C = patch_embeddings.shape
        # H = W = int(N**0.5)  # Assuming square grid
        # patch_embeddings = patch_embeddings.permute(0, 2, 1).reshape(B, C, H, W)

        # # Pass through decoder layers
        # x = self.vit_conv(patch_embeddings)
        # x = self.up1(x)
        # x = self.up2(x)
        # x = self.up3(x)
        # x = self.up4(x)
        # x = self.outc(x)
        # x = self.relu(x)

        return x
