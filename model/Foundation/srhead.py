import torch
import torch.nn as nn
import math

def default_conv(in_channels, out_channels, kernel_size, bias=True, groups = 1):
    wn = lambda x:torch.nn.utils.weight_norm(x)
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, groups = groups)

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=1, dilation=1, groups=1, relu=True,
                 bn=False, bias=False, up_size=0,fan=False):
        super(BasicConv, self).__init__()
        wn = lambda x:torch.nn.utils.weight_norm(x)
        self.out_channels = out_planes
        self.in_channels = in_planes
        if fan:
            self.conv = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        else:
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None
        self.up_size = up_size
        self.up_sample = nn.Upsample(size=(up_size, up_size), mode='bilinear') if up_size != 0 else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        if self.up_size > 0:
            x = self.up_sample(x)
        return x

class Upsampler(nn.Sequential):
    def __init__(self, scale, n_feats, bn=False, act=False, bias=True):
        conv = default_conv
        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feats))

                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feats))

            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)



class SuperResolutionHead(nn.Module):
    """
    Super-resolution head that enhances the input resolution using an upsampler.
    Incorporates task-specific embedding to condition the feature map.
    """
    def __init__(self, embed_dim, num_queries, in_channels, n_channels, upscale_factor):
        super().__init__()
        conv = default_conv
        # Project task-specific embedding to match feature map channels
        self.conv = nn.Conv2d(in_channels, 64 * (upscale_factor ** 2), 3, padding=1)
        self.projector = nn.Linear(num_queries * embed_dim, in_channels)
        self.tail = nn.Sequential(
            Upsampler(8, n_feats=1024),
            Upsampler(upscale_factor, n_feats=1024),
            conv(in_channels=1024, out_channels=n_channels, kernel_size=3)
        )
        self.up = nn.Sequential(
            Upsampler(upscale_factor, n_feats=128),
            BasicConv(in_planes=128, out_planes=n_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, feature_maps, sr_embedding):
        batch_size, C, H, W = feature_maps[-1].shape
        # Flatten and project super-resolution embedding
        sr_embedding_flat = sr_embedding.reshape(batch_size, -1)
        adjustment = self.projector(sr_embedding_flat).view(batch_size, C, 1, 1)
        # Enhance feature map with task-specific information
        enhanced_feature_map = feature_maps[-1] + adjustment
        res_tail = self.tail(enhanced_feature_map)
        res_up = self.up(feature_maps[0])
        return res_tail + res_up
