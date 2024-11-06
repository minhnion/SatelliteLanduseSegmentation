import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch

from layers.transformer_layers import *
from layers.unet_layers import * 

# Define the PromptedViT class
class PromptedViT(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, prompt_dim, depth, heads, mlp_dim, pool='cls', channels=512, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.prompt_dim = prompt_dim
        self.total_dim = dim  # The expected input dimension for the transformer
        adjusted_dim = dim - prompt_dim  # Adjusted dimension to accommodate prompt token

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        # Check if image dimensions are divisible by patch dimensions
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        # Calculate number of patches and patch dimension
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        # Patch embedding layer with adjusted output dimension
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, adjusted_dim),
        )

        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # Prompt token
        self.prompt_token = nn.Parameter(torch.randn(1, prompt_dim))

        # Transformer
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

    def forward(self, img):
        x = self.to_patch_embedding(img)  # x shape: (batch_size, num_patches, adjusted_dim)
        b, n, _ = x.shape

        # Expand prompt token and concatenate
        prompt_token_expanded = self.prompt_token.expand(b, -1)  # Shape: (batch_size, prompt_dim)
        prompt_token_expanded = prompt_token_expanded.unsqueeze(1).repeat(1, n, 1)  # Shape: (batch_size, num_patches, prompt_dim)

        x = torch.cat((x, prompt_token_expanded), dim=-1)  # Now x shape: (batch_size, num_patches, dim)

        # Add class token and positional embedding
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        # Transformer encoding
        x = self.transformer(x)

        # Pooling
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        return x

class PromptedVitUnet(nn.Module):
    def __init__(self, n_classes, n_channels=3, bilinear=True, prompt_dim=128):
        super(PromptedVitUnet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder (Contracting Path)
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        # Prompted Vision Transformer block
        self.vit = PromptedViT(
            image_size=32,
            patch_size=8,
            dim=2048,  # Total dimension expected by the transformer
            prompt_dim=prompt_dim,
            depth=2,
            heads=16,
            mlp_dim=12,
            pool='cls',
            channels=512,
        )

        # Adjusted convolution layers after ViT
        self.vit_conv = nn.Conv2d(32, 512, kernel_size=1, padding=0)
        self.vit_linear = nn.Linear(64, 1024)

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
        x5 = self.down4(x4)

        # Applying Prompted Vision Transformer
        x6 = self.vit(x5)
        x6 = torch.reshape(x6, (-1, 32, 8, 8))
        x7 = self.vit_conv(x6)
        x8 = self.vit_linear(torch.reshape(x7, (-1, 512, 64)))
        x9 = torch.reshape(x8, (-1, 512, 32, 32))

        # Decoder (Expanding Path)
        x = self.up1(x9, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
