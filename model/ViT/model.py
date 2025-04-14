import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from layers.transformer_layers import pair, Transformer
from layers.unet_layers import *

def init_weights_he(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

# Vision Transformer class
class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, pool = 'cls', channels = 512, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        # Check if image dimensions are divisible by patch dimensions
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        # Calculate number of patches and patch dimension
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # Patch embedding layer
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # Transformer
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()


    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        return x

class UNet(nn.Module):
    def __init__(self, n_classes, n_channels=3, depth=2, heads=16, dropout=0.2, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder (Contracting Path)
        self.inc = DoubleConv(n_channels, 64) # (B, 64, H, W)
        self.down1 = Down(64, 128) # (B, 128, H/2, W/2)
        self.down2 = Down(128, 256) # (B, 256, H/4, W/4)
        self.down3 = Down(256, 512) # (B, 512, H/8, W/8)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor) # (B, 512, H/16, W/16) if bilinear else (B, 1024, H/16, W/16)

        # Vision Transformer block
        self.vit = ViT(image_size = 32,patch_size = 8,dim = 2048, depth = depth, heads = heads,mlp_dim = 12,channels = 512, dropout=dropout, emb_dropout=dropout)
        self.vit_conv = nn.Conv2d(32,512,kernel_size = 1,padding = 0)
        self.vit_linear = nn.Linear(64,1024)

        # Decoder (Expanding Path)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # Encoder (Contracting Path)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4) # (B, 512, H/16, W/16)
        # print(x5.shape)

        #applying Vision Transformer
        x6 = self.vit(x5)
        x6 = torch.reshape(x6,(-1,32,8,8))
        x7 = self.vit_conv(x6)
        x8 = self.vit_linear(torch.reshape(x7,(-1,512,64)))
        x9 = torch.reshape(x8,(-1,512,32,32)) # (B, 512, 32, 32)

        # Decoder (Expanding Path)
        x = self.up1(x9, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
