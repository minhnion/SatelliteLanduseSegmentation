import torch
import torch.nn as nn
import torch.nn.functional as F

def init_weights_he(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3):
        super(BasicBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1)//2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.1)
            
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out) 
        return out
    
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, strided=True):
        super(EncoderBlock, self).__init__()
        out_channels = in_channels*2 if strided else in_channels
        self.layer1 = BasicBlock(in_channels, out_channels, stride=2 if strided else 1)
        self.layer2 = BasicBlock(out_channels, out_channels)
        self.layer3 = BasicBlock(out_channels, out_channels)
        self.layer4 = BasicBlock(out_channels, out_channels)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2 if strided else 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        
    def forward(self, x):
        out = self.layer1(x)
        residual1 = self.downsample(x)
        out = self.layer2(out) + residual1
        residual2 = out
        out = self.layer3(out)
        out = self.layer4(out) + residual2
        return out
    
class DecoderBlock(nn.Module):
    def __init__(self, in_channels):
        super(DecoderBlock, self).__init__()
        self.layer1 = BasicBlock(in_channels, in_channels // 4, kernel_size=1)
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels // 4, in_channels // 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(in_channels // 4)
        )
        self.layer3 = BasicBlock(in_channels // 4, in_channels // 2, kernel_size=1)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        return out
    
class InitialBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InitialBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3)
        self.bn = nn.BatchNorm2d(out_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.maxpool(out)
        return out
    
class FinalBlock(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(FinalBlock, self).__init__()
        
        self.transposeconv1 = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=3, stride=2, output_padding=0)
        self.bn1 = nn.BatchNorm2d(in_channels // 2)
        
        self.conv1 = nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=2)
        self.bn2 = nn.BatchNorm2d(in_channels // 2)
        
        self.conv2 = nn.Conv2d(in_channels // 2, out_channels=n_classes, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(n_classes)
        
#         self.softmax = nn.Softmax(dim=-1)    
        
    def forward(self, x):
        out = self.transposeconv1(x)
        out = self.bn1(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.conv2(out)
        out = self.bn3(out)
#         out = self.softmax(out)
        return out
    
class LinkNet(nn.Module):
    def __init__(self, n_classes, n_channels=3):
        super(LinkNet, self).__init__()
        self.initblock = InitialBlock(5, 64)
        self.encoder1 = EncoderBlock(64, strided=False)
        self.encoder2 = EncoderBlock(64)
        self.encoder3 = EncoderBlock(128)
        self.encoder4 = EncoderBlock(256)
        self.decoder4 = DecoderBlock(512)
        self.decoder3 = DecoderBlock(256)
        self.decoder2 = DecoderBlock(128)
        self.decoder1 = DecoderBlock(64)
        self.finalblock = FinalBlock(32, n_classes)
        
        # Apply He initialization
        self.apply(init_weights_he)
        
    def forward(self, x):
        out = self.initblock(x)
        residual1 = self.encoder1(out)
        residual2 = self.encoder2(residual1)
        residual3 = self.encoder3(residual2)
        out = self.encoder4(residual3)
        out = self.decoder4(out) + residual3
        out = self.decoder3(out) + residual2
        out = self.decoder2(out) + residual1
        out = self.decoder1(out)
        out = self.finalblock(out)
        return out