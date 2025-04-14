import torch
import torch.nn as nn
import satlaspretrain_models

from model.Foundation.encoder import *
from model.Foundation.srhead import SuperResolutionHead
from model.Foundation.seghead import SegmentationHead

class FoundationModel(nn.Module):
    def __init__(self, n_channels=3, n_classes=5, num_prompts=10, num_heads=8, embed_dim=1024, feature_map_h=64, feature_map_w=64, upscale_factor=3, num_queries_seg=10, num_queries_sr=10):
        super().__init__()
        weights_manager = satlaspretrain_models.Weights()
        self.fm_encoder =  FoundationBackbone(weights_manager, "Sentinel2_SwinB_SI_RGB", n_channels)
        self.full_encoder = Encoder(backbone_model=self.fm_encoder, num_prompts=num_prompts, num_heads=num_heads, embed_dim=embed_dim, H=feature_map_h, W=feature_map_w)
        self.cross_attention = CrossAttention(embed_dim=1024, num_queries_seg=num_queries_seg,
                                            num_queries_sr=num_queries_sr, num_heads=num_heads)
        self.segmentation_head = SegmentationHead(n_classes)
        self.super_resolution_head = SuperResolutionHead(embed_dim=embed_dim, num_queries=num_queries_sr, in_channels=1024, n_channels=n_channels, upscale_factor=upscale_factor)

    def forward(self, x):
        feature_maps, enhanced_embedding = self.full_encoder(x)
        seg_embedding, sr_embedding = self.cross_attention(enhanced_embedding)
        segmentation_map = self.segmentation_head(feature_maps)
        super_resolution_map = self.super_resolution_head(feature_maps, sr_embedding)
        return super_resolution_map, segmentation_map
