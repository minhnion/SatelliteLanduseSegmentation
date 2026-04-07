import torch
import torch.nn as nn
import torch.nn.functional as F

class CBR(nn.Module):
    """Convolution + BatchNorm + ReLU block."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(CBR, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn   = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class SCNet(nn.Module):
    def __init__(self, in_channels_Y, in_channels_alpha, n_classes, scale_factor=2):
        super(SCNet, self).__init__()
        self.scale_factor = scale_factor

        # Encoder with early fusion
        self.cbr1_Y = CBR(in_channels_Y, 64)
        self.cbr1_alpha = CBR(in_channels_alpha, 64)
        self.fuse1 = CBR(128, 64)
        self.pool1 = nn.MaxPool2d(2, return_indices=True)

        self.cbr2 = CBR(64, 128)
        self.pool2 = nn.MaxPool2d(2, return_indices=True)

        self.cbr3 = CBR(128, 256)
        self.pool3 = nn.MaxPool2d(2, return_indices=True)
        self.dropout = nn.Dropout(0.5)

        # Decoder
        self.unpool3 = nn.MaxUnpool2d(2)
        self.dec_cbr3 = CBR(256, 128)
        self.dec_dropout3 = nn.Dropout(0.5)
        self.unpool2 = nn.MaxUnpool2d(2)
        self.dec_cbr2 = CBR(128, 64)
        self.dec_dropout2 = nn.Dropout(0.5)
        self.unpool1 = nn.MaxUnpool2d(2)
        self.dec_cbr1 = CBR(64, 64)

        # SPC and Classifier
        self.spc = nn.Conv2d(64, n_classes * (scale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.classifier = nn.Conv2d(n_classes, n_classes, kernel_size=1)

    def forward(self, Y, alpha):
        # Encoder with early fusion
        Y1 = self.cbr1_Y(Y)
        A1 = self.cbr1_alpha(alpha)
        fused1 = self.fuse1(torch.cat([Y1, A1], dim=1))
        F1p, idx1 = self.pool1(fused1)

        F2 = self.cbr2(F1p)
        F2p, idx2 = self.pool2(F2)

        F3 = self.cbr3(F2p)
        F3p, idx3 = self.pool3(F3)
        encoded = self.dropout(F3p)

        # Decoder
        D3 = self.unpool3(encoded, idx3, output_size=F3.size())
        D3 = self.dec_cbr3(D3)
        D3 = self.dec_dropout3(D3)
        D2 = self.unpool2(D3, idx2, output_size=F2.size())
        D2 = self.dec_cbr2(D2)
        D2 = self.dec_dropout2(D2)
        D1 = self.unpool1(D2, idx1, output_size=fused1.size())
        D1 = self.dec_cbr1(D1)

        # SPC and Classification
        spc_out = self.spc(D1)
        up_out = self.pixel_shuffle(spc_out)
        logits = self.classifier(up_out)
        return logits  # Softmax applied externally if needed
