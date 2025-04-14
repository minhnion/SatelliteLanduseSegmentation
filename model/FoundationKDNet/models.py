import torch
import torch.nn as nn
import satlaspretrain_models
import torch.nn.functional as F

from model.FoundationKDNet.teacher_encoder import *
from model.FoundationKDNet.student_encoder import *
from model.FoundationKDNet.srhead import SuperResolutionHead
from model.FoundationKDNet.seghead import SegmentationHead

class FoundationKDModel(nn.Module):
    def __init__(self, n_channels=13, n_classes=5, num_prompts=10, num_heads=4, embed_dim=1024, feature_map_h=64, feature_map_w=64, upscale_factor=3, num_queries_seg=10, num_queries_sr=10, sr_out_chan=None):
        """
        Args:
            n_channels (int): Number of channels in the input image.
            n_classes (int): Number of classes in the output segmentation map.
            num_prompts (int): Number of prompts for the full encoder.
            num_heads (int): Number of attention heads for the full encoder.
            embed_dim (int): Embedding dimension for the full encoder.
            feature_map_h (int): Height of the feature maps.
            feature_map_w (int): Width of the feature maps.
            upscale_factor (int): Upscale factor for the super-resolution head.
            num_queries_seg (int): Number of queries for the segmentation head.
            num_queries_sr (int): Number of queries for the super-resolution head.
        """
        super().__init__()
        weights_manager = satlaspretrain_models.Weights()
        self.fm_encoder = TeacherFoundationBackbone(weights_manager, "Sentinel2_SwinB_SI_RGB", n_channels)
        self.full_encoder = TeacherEncoder(backbone_model=self.fm_encoder, num_prompts=num_prompts, num_heads=num_heads, embed_dim=embed_dim, H=feature_map_h, W=feature_map_w)
        self.cross_attention = CrossAttention(embed_dim=embed_dim, num_queries_seg=num_queries_seg,
                                            num_queries_sr=num_queries_sr, num_heads=num_heads)
        self.segmentation_head = SegmentationHead(n_classes, scale_factor=upscale_factor, last_fm_chan=1024)
        self.super_resolution_head = SuperResolutionHead(embed_dim=embed_dim, last_fm_chan=1024, first_fm_chan=128, num_queries=num_queries_sr, in_channels=embed_dim, n_channels=sr_out_chan if sr_out_chan else n_channels, upscale_factor=upscale_factor)

    def forward(self, x):
        # Get feature maps and enhanced embedding from teacher encoder
        feature_maps, enhanced_embedding = self.full_encoder(x)
        # Get segmentation and super-resolution embeddings from cross attention
        seg_embedding, sr_embedding = self.cross_attention(enhanced_embedding)

        # Generate segmentation and super-resolution maps
        segmentation_map = self.segmentation_head(feature_maps)
        super_resolution_map = self.super_resolution_head(feature_maps, sr_embedding)

        return feature_maps, enhanced_embedding, (super_resolution_map, segmentation_map)

class StudentModel(nn.Module):
    def __init__(self, n_channels=13, n_classes=5, student_model_name='ResNet18', num_heads=4, num_queries_seg=10, num_queries_sr=10, embed_dim=1024, upscale_factor=3):
        super().__init__()
        self.student_encoder = StudentEncoder(model_name=student_model_name, n_channels=13, embed_dim=embed_dim)
        self.cross_attention = CrossAttention(embed_dim=embed_dim, num_queries_seg=num_queries_seg,
                                            num_queries_sr=num_queries_sr, num_heads=num_heads)
        self.segmentation_head = SegmentationHead(n_classes, last_fm_chan=512)
        self.super_resolution_head = SuperResolutionHead(embed_dim=embed_dim, last_fm_chan=512, first_fm_chan=64, num_queries=num_queries_sr, in_channels=512, n_channels=n_channels, upscale_factor=upscale_factor)

    def forward(self, x):
        # Get feature maps and embedding from student encoder
        feature_maps, embedding = self.student_encoder(x)

        # Get the segmentation and super-resolution embeddings
        seg_embedding, sr_embedding = self.cross_attention(embedding)

        # Generate output maps
        segmentation_map = self.segmentation_head(feature_maps)
        super_resolution_map = self.super_resolution_head(feature_maps, sr_embedding)

        return feature_maps, embedding, (super_resolution_map, segmentation_map)
