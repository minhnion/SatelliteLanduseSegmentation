import torch
import torch.nn as nn
import torch.nn.functional as F
import satlaspretrain_models

# Utility function for InfoNCE alignment loss
def info_nce_loss(seg_emb, sr_emb, temperature=0.1):
    """
    Compute InfoNCE loss to align embeddings between segmentation and super-resolution tasks.
    
    Args:
        seg_emb (torch.Tensor): Segmentation embedding [batch_size, embed_dim]
        sr_emb (torch.Tensor): Super-resolution embedding [batch_size, embed_dim]
        temperature (float): Temperature parameter for scaling similarity
    
    Returns:
        torch.Tensor: InfoNCE loss value
    """
    seg_emb = F.normalize(seg_emb, dim=1)
    sr_emb = F.normalize(sr_emb, dim=1)
    similarity = torch.mm(seg_emb, sr_emb.t()) / temperature
    labels = torch.arange(similarity.size(0)).to(similarity.device)
    return F.cross_entropy(similarity, labels)

class CVPT(nn.Module):
    """
    Cross-Visual Prompt Tuning module to enhance the encoder's adaptability.
    Adds learnable prompts tuned with cross-attention to the feature map.
    """
    def __init__(self, num_prompts, embed_dim, num_heads, H, W):
        super().__init__()
        # Learnable prompts initialized randomly
        self.prompts = nn.Parameter(torch.randn(num_prompts, embed_dim))
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        # Projection layer to match the feature map spatial dimensions
        self.projection = nn.Linear(num_prompts * embed_dim, embed_dim * H * W)

    def forward(self, feature_map):
        batch_size, C, H, W = feature_map.shape
        # Flatten feature map to tokens for attention
        feature_tokens = feature_map.view(batch_size, C, H*W).permute(2, 0, 1)  # [H*W, batch_size, C]
        # Expand prompts to batch size
        prompts = self.prompts.unsqueeze(1).repeat(1, batch_size, 1)  # [num_prompts, batch_size, embed_dim]
        # Cross-attention: prompts attend to feature tokens
        attended_prompts, _ = self.attention(prompts, feature_tokens, feature_tokens)  # [num_prompts, batch_size, embed_dim]
        attended_prompts = attended_prompts.permute(1, 0, 2).contiguous().view(batch_size, -1)  # [batch_size, num_prompts * embed_dim]
        # Project back to feature map shape and add to original
        adjustment = self.projection(attended_prompts).view(batch_size, C, H, W)
        return feature_map + adjustment

class Encoder(nn.Module):
    """
    Encoder module using a pre-trained Swin Transformer backbone with CVPT.
    Outputs multi-scale feature maps and an enhanced embedding.
    """
    def __init__(self, weights_manager, model_identifier, num_prompts, num_heads, H=16, W=16):
        super().__init__()
        # Load pre-trained Swin Transformer backbone
        self.backbone = weights_manager.get_pretrained_model(model_identifier)
        # CVPT module to enhance the last feature map
        self.cvpt = CVPT(num_prompts, embed_dim=1024, num_heads=num_heads, H=H, W=W)

    def forward(self, x):
        # Get multi-scale feature maps from the backbone
        feature_maps = self.backbone(x)  # List: [B, 128, 128, 128], [B, 256, 64, 64], [B, 512, 32, 32], [B, 1024, 16, 16]
        embedding = feature_maps[-1]  # Last feature map as embedding
        # Enhance embedding with CVPT
        enhanced_embedding = self.cvpt(embedding)
        return feature_maps, enhanced_embedding

class CrossAttention(nn.Module):
    """
    Cross-attention module to produce task-specific embeddings for segmentation and super-resolution.
    Facilitates collaboration between tasks by attending to the shared embedding.
    """
    def __init__(self, embed_dim, num_queries_seg, num_queries_sr, num_heads):
        super().__init__()
        # Learnable queries for each task
        self.queries_seg = nn.Parameter(torch.randn(num_queries_seg, embed_dim))
        self.queries_sr = nn.Parameter(torch.randn(num_queries_sr, embed_dim))
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, embedding):
        batch_size, C, H, W = embedding.shape
        # Flatten embedding to tokens
        embedding_tokens = embedding.view(batch_size, C, H*W).permute(2, 0, 1)  # [H*W, batch_size, C]
        # Segmentation task
        queries_seg = self.queries_seg.unsqueeze(1).repeat(1, batch_size, 1)  # [num_queries_seg, batch_size, embed_dim]
        seg_embedding, _ = self.attention(queries_seg, embedding_tokens, embedding_tokens)
        seg_embedding = seg_embedding.permute(1, 0, 2)  # [batch_size, num_queries_seg, embed_dim]
        # Super-resolution task
        queries_sr = self.queries_sr.unsqueeze(1).repeat(1, batch_size, 1)
        sr_embedding, _ = self.attention(queries_sr, embedding_tokens, embedding_tokens)
        sr_embedding = sr_embedding.permute(1, 0, 2)  # [batch_size, num_queries_sr, embed_dim]
        return seg_embedding, sr_embedding

class UpBlock(nn.Module):
    """
    Upsampling block for the segmentation decoder.
    Combines upsampled features with skip connections and applies convolution.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Conv2d(in_channels + out_channels if in_channels > out_channels else in_channels, 
                            out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x, skip=None):
        x = self.upsample(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x

class SegmentationHead(nn.Module):
    """
    Land-use segmentation head using a U-Net-like decoder with ViT-style skip connections.
    Outputs a segmentation map with the specified number of classes.
    """
    def __init__(self, num_classes, in_channels_list):
        super().__init__()
        # U-Net decoder blocks
        self.up1 = UpBlock(in_channels_list[3], in_channels_list[2])  # 1024 -> 512
        self.up2 = UpBlock(in_channels_list[2], in_channels_list[1])  # 512 -> 256
        self.up3 = UpBlock(in_channels_list[1], in_channels_list[0])  # 256 -> 128
        self.up4 = UpBlock(in_channels_list[0], 64)                   # 128 -> 64
        self.final_conv = nn.Conv2d(64, num_classes, 1)

    def forward(self, feature_maps):
        # Start from the smallest scale and upsample with skip connections
        x = feature_maps[3]  # [batch_size, 1024, 16, 16]
        x = self.up1(x, feature_maps[2])  # [batch_size, 512, 32, 32]
        x = self.up2(x, feature_maps[1])  # [batch_size, 256, 64, 64]
        x = self.up3(x, feature_maps[0])  # [batch_size, 128, 128, 128]
        x = self.up4(x)                   # [batch_size, 64, 256, 256]
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # [batch_size, 64, 512, 512]
        x = self.final_conv(x)            # [batch_size, num_classes, 512, 512]
        return x

class Upsampler(nn.Module):
    """
    Upsampler module for super-resolution, inspired by ESRT.
    Uses pixel shuffle to increase resolution.
    """
    def __init__(self, upscale_factor, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64 * (upscale_factor ** 2), 3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.conv2 = nn.Conv2d(64, 3, 3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pixel_shuffle(x)
        x = self.conv2(x)
        return x

class SuperResolutionHead(nn.Module):
    """
    Super-resolution head that enhances the input resolution using an upsampler.
    Incorporates task-specific embedding to condition the feature map.
    """
    def __init__(self, embed_dim, num_queries, in_channels, upscale_factor):
        super().__init__()
        # Project task-specific embedding to match feature map channels
        self.projector = nn.Linear(num_queries * embed_dim, in_channels)
        self.upsampler = Upsampler(upscale_factor, in_channels)

    def forward(self, feature_map, sr_embedding):
        batch_size, C, H, W = feature_map.shape
        # Flatten and project super-resolution embedding
        sr_embedding_flat = sr_embedding.view(batch_size, -1)
        adjustment = self.projector(sr_embedding_flat).view(batch_size, C, 1, 1)
        # Enhance feature map with task-specific information
        enhanced_feature_map = feature_map + adjustment
        # Upsample to high resolution
        hr_image = self.upsampler(enhanced_feature_map)
        return hr_image

class MultiTaskModel(nn.Module):
    """
    Main multi-task model integrating encoder, cross-attention, and task-specific heads.
    Performs super-resolution and land-use segmentation simultaneously.
    """
    def __init__(self, num_classes=10, upscale_factor=4, num_prompts=10, num_queries_seg=10, 
                 num_queries_sr=10, num_heads=8):
        super().__init__()
        # Initialize weights manager for loading pre-trained model
        weights_manager = satlaspretrain_models.Weights()
        # Encoder with Swin Transformer and CVPT
        self.encoder = Encoder(weights_manager, "Sentinel2_SwinB_SI_RGB", num_prompts, num_heads, H=16, W=16)
        # Cross-attention for task collaboration
        self.cross_attention = CrossAttention(embed_dim=1024, num_queries_seg=num_queries_seg, 
                                            num_queries_sr=num_queries_sr, num_heads=num_heads)
        # Task-specific heads
        self.segmentation_head = SegmentationHead(num_classes, in_channels_list=[128, 256, 512, 1024])
        self.super_resolution_head = SuperResolutionHead(embed_dim=1024, num_queries=num_queries_sr, 
                                                       in_channels=1024, upscale_factor=upscale_factor)

    def forward(self, x, seg_gt=None, hr_gt=None):
        """
        Forward pass of the multi-task model.
        
        Args:
            x (torch.Tensor): Input image [batch_size, 3, H, W]
            seg_gt (torch.Tensor, optional): Segmentation ground truth [batch_size, H, W]
            hr_gt (torch.Tensor, optional): High-resolution ground truth [batch_size, 3, H*upscale_factor, W*upscale_factor]
        
        Returns:
            tuple: (seg_output, hr_output, losses)
                - seg_output: Segmentation map [batch_size, num_classes, H, W]
                - hr_output: High-resolution image [batch_size, 3, H*upscale_factor, W*upscale_factor]
                - losses: Dictionary with 'seg_loss', 'sr_loss', and 'alignment_loss'
        """
        # Encode input image to get feature maps and embedding
        feature_maps, embedding = self.encoder(x)
        # Apply cross-attention to get task-specific embeddings
        seg_embedding, sr_embedding = self.cross_attention(embedding)
        # Generate task outputs
        seg_output = self.segmentation_head(feature_maps)
        hr_output = self.super_resolution_head(feature_maps[-1], sr_embedding)

        # Compute losses if ground truth is provided
        losses = {}
        if seg_gt is not None:
            losses['seg_loss'] = F.cross_entropy(seg_output, seg_gt)
        if hr_gt is not None:
            losses['sr_loss'] = F.l1_loss(hr_output, hr_gt)
        # Alignment loss between task embeddings
        pooled_seg_emb = seg_embedding.mean(dim=1)
        pooled_sr_emb = sr_embedding.mean(dim=1)
        losses['alignment_loss'] = info_nce_loss(pooled_seg_emb, pooled_sr_emb)

        return seg_output, hr_output, losses

# Example usage
if __name__ == "__main__":
    # Initialize model
    model = MultiTaskModel(num_classes=10, upscale_factor=4)
    model.eval()

    # Dummy input (batch_size=1, 3 channels, 512x512)
    input_image = torch.randn(1, 3, 512, 512)
    # Dummy ground truth
    seg_gt = torch.randint(0, 10, (1, 512, 512))
    hr_gt = torch.randn(1, 3, 2048, 2048)  # 4x resolution

    # Forward pass
    seg_output, hr_output, losses = model(input_image, seg_gt, hr_gt)
    print(f"Segmentation output shape: {seg_output.shape}")
    print(f"Super-resolution output shape: {hr_output.shape}")
    print(f"Losses: {losses}")