import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import satlaspretrain_models

class TeacherFoundationBackbone(nn.Module):
    """
    Encoder module using a pre-trained Swin Transformer backbone with CVPT.
    Outputs multi-scale feature maps and an enhanced embedding.
    """
    def __init__(self, weights_manager, model_identifier, n_channels=13):
        super().__init__()
        # Load pre-trained Swin Transformer backbone
        self.backbone = weights_manager.get_pretrained_model(model_identifier)

        # Modify the first layer to accept n channels
        # print(f"Original: {self.backbone.backbone.backbone.features[0][0]}")
        self.backbone.backbone.backbone.features[0][0] = nn.Conv2d(n_channels, 128, kernel_size=4, stride=4)
        # self.backbone.backbone.backbone.features[0][0] = nn.Conv2d(n_channels, 128, kernel_size=1, stride=1)

    def forward(self, x):
        # Get multi-scale feature maps from the backbone
        feature_maps = self.backbone(x)  # List: [B, 128, 128, 128], [B, 256, 64, 64], [B, 512, 32, 32], [B, 1024, 16, 16]
        # Enhance embedding with CVPT
        return feature_maps

class CVPT(nn.Module):
    """
    Cross-Visual Prompt Tuning module using convolution-based projection.
    """
    def __init__(self, num_prompts, embed_dim, num_heads, H, W):
        super().__init__()
        self.num_prompts = num_prompts
        self.embed_dim = embed_dim
        self.H, self.W = H, W

        # Learnable prompts initialized randomly
        self.prompts = nn.Parameter(torch.randn(num_prompts, embed_dim))
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)

        # 1x1 Conv projection for refinement
        self.conv_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)

    def forward(self, feature_map):
        batch_size, C, H, W = feature_map.shape  # Expected: [batch_size, 128, 16, 16]

        # Flatten feature map to tokens
        feature_tokens = feature_map.view(batch_size, C, H * W).permute(2, 0, 1)  # [H*W, batch_size, C]

        # Expand prompts to match batch size
        prompts = self.prompts.unsqueeze(1).repeat(1, batch_size, 1)  # [num_prompts, batch_size, embed_dim]

        # Cross-attention: prompts attend to feature tokens
        attended_prompts, _ = self.attention(prompts, feature_tokens, feature_tokens)  # [num_prompts, batch_size, embed_dim]

        # Aggregate across num_prompts (e.g., mean or sum pooling)
        attended_prompts = attended_prompts.mean(dim=0)  # Now shape is [batch_size, embed_dim]

        # Reshape to match feature map spatially
        attended_prompts = attended_prompts.view(batch_size, C, 1, 1)  # [batch_size, 1024, 1, 1]

        # Apply 1x1 convolution
        adjustment = self.conv_proj(attended_prompts)  # [batch_size, 1024, 1, 1]

        # Broadcast across H, W
        return feature_map + adjustment.expand(-1, -1, H, W)


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

class TeacherEncoder(nn.Module):
    """
    Encoder module using a pre-trained Swin Transformer backbone with CVPT.
    Outputs multi-scale feature maps and an enhanced embedding.
    """
    def __init__(self, backbone_model, num_prompts, num_heads, embed_dim=32, H=16, W=16, num_queries_seg=10, num_queries_sr=10):
        super().__init__()
        # Load pre-trained Swin Transformer backbone
        self.backbone = backbone_model
        # Final embedding dimension
        self.layer_fc = nn.Conv2d(1024, embed_dim, kernel_size=(1, 1), stride=(1, 1))
        # CVPT module to enhance the last feature map
        self.cvpt = CVPT(num_prompts, embed_dim=embed_dim, num_heads=num_heads, H=H, W=W)
        # cross-attention module to enhance the embedding
        self.self_attention = CrossAttention(embed_dim=embed_dim, num_queries_seg=num_queries_seg,
                                            num_queries_sr=num_queries_sr, num_heads=num_heads)

    def forward(self, x):
        # Get multi-scale feature maps from the backbone
        feature_maps = self.backbone(x)  # List: [B, 128, 224, 224], [B, 256, 112, 112], [B, 512, 56, 56], [B, 1024, 28, 28]
        feature_maps = [fm.unsqueeze(0) if fm.dim() == 3 else fm for fm in feature_maps]
        embedding = self.layer_fc(feature_maps[-1])  # Last feature map as embedding
        # Enhance embedding with CVPT
        feature_maps.append(embedding)
        enhanced_embedding = self.cvpt(embedding)
        return feature_maps, enhanced_embedding

if __name__ == '__main__':
    weights_manager = satlaspretrain_models.Weights()
    fm_encoder = TeacherFoundationBackbone(weights_manager, "Sentinel2_SwinB_SI_RGB", n_channels=3)

    teacher_encoder = TeacherEncoder(
        fm_encoder,
        10, 4, 32, 16, 16, 10, 10
    )

    x = torch.randn(8, 3, 128, 128)
    # Test forward pass
    feature_maps, embedding = teacher_encoder(x)
    # Print shapes
    print("Feature maps shapes:")
    for i, fm in enumerate(feature_maps):
        print(f"Feature map {i+1}:", fm.shape)
    print("Embedding shape:", embedding.shape)
