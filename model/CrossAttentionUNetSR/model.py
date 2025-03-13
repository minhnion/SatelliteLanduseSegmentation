import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from layers.transformer_layers import pair, Transformer
from layers.unet_layers import *

from model.ESRT.model import ESRT, BasicConv
from model.ESRT.common import default_conv, Upsampler
from model.UNetSR.model import ViT

def init_weights_he(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super(CrossAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, query, key, value):
        attn_output, _ = self.multihead_attn(query, key, value)
        return self.norm(attn_output + query)  # Residual connection

class CrossAttentionUNetSR(nn.Module):
    def __init__(self, n_classes, n_channels=3, bilinear=True):
        super(CrossAttentionUNetSR, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # U-Net Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        # ESRT Transformer (Super-Resolution)
        self.sr = ESRT(n_blocks=1, n_channels=n_channels, upscale=2, encoder_only=True)

        # ViT Transformer
        self.vit = ViT(image_size = 32,patch_size = 8,dim = 2048, depth = 2, heads = 16,mlp_dim = 12,channels = 512)

        # Cross-Attention Layer
        self.cross_attention = CrossAttention(dim=512, num_heads=8)

        # Decoder
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        self.apply(init_weights_he)

    def forward(self, x):
        # U-Net Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)  # [B, 512, H, W]

        # Super-Resolution Transformer Path
        x_sr = self.sr(x)  # ESRT transformer output

        # ViT Path
        x_vit = self.vit(x5)  # ViT transformer output

        # Reshape for Cross-Attention (Convert 4D tensors to 3D sequences)
        b, c, h, w = x5.shape  # U-Net feature map shape
        x_vit_flat = x_vit.view(b, c, -1).permute(0, 2, 1)  # [B, L, C]
        x_sr_flat = x_sr.view(b, c, -1).permute(0, 2, 1)  # [B, L, C]

        # Cross-Attention (Fusing ESRT & ViT Features)
        x_fused = self.cross_attention(x_sr_flat, x_vit_flat, x_vit_flat)

        # Reshape Back to 4D
        L = x_fused.shape[1]  # Length of sequence
        H = W = int(math.sqrt(L))  # Assuming square feature map
        x_fused = x_fused.permute(0, 2, 1).view(b, c, H, W)  # Correct reshaping

        # Decoder
        x = self.up1(x_fused, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
