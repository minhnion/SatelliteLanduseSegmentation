from torchvision import transforms as T, models
import torch.nn as nn
import torch

class StudentEncoder(nn.Module):
    def __init__(self, model_name, embed_dim=32, n_channels=13):
        super(StudentEncoder, self).__init__()
        if model_name == 'ResNet18':
            backbone = models.resnet18(pretrained=True)
            self.embedding_dim = 512
        elif model_name == 'ResNet50':
            backbone = models.resnet50(pretrained=True)
            self.embedding_dim = 2048
        else:
            raise ValueError(f"Model {model_name} not supported")

        self.output_dim = embed_dim

        # Remove the final FC layer and keep feature extractors
        self.conv1 = nn.Conv2d(n_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        # self.conv1 = nn.Conv2d(n_channels, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1  # output: 64 channels
        self.layer2 = backbone.layer2  # output: 128 channels
        self.layer3 = backbone.layer3  # output: 256 channels
        self.layer4 = backbone.layer4  # output: 512 channels (ResNet18) or 2048 (ResNet50),
        self.layer5  = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(self.embedding_dim, self.output_dim, kernel_size=(3, 3), padding=(1, 1)),  # Sửa stride
            nn.BatchNorm2d(self.output_dim),  # Thêm BatchNorm để ổn định
            nn.ReLU(inplace=True)
        )

    def get_feature_maps(self, x):
        """Extract multi-scale feature maps"""
        # Initial convolution block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Get feature maps from each layer
        feature_map1 = self.layer1(x)      # [B, 64, H/4, W/4]
        feature_map2 = self.layer2(feature_map1)  # [B, 128, H/8, W/8]
        feature_map3 = self.layer3(feature_map2)  # [B, 256, H/16, W/16]
        feature_map4 = self.layer4(feature_map3)  # [B, 512, H/32, W/32]
        feature_map5 = self.layer5(feature_map4)  # [B, 32, H/8, W/8]

        return [feature_map1, feature_map2, feature_map3, feature_map4, feature_map5]

    def forward(self, x):
        """Forward pass returning both feature maps and final output"""
        # Get multi-scale feature maps
        feature_maps = self.get_feature_maps(x)
        # Get final embedding (last feature map)
        embedding = feature_maps[-1]
        return feature_maps, embedding

if __name__ == '__main__':
    student_encoder = StudentEncoder('ResNet18', embed_dim=32)
    x = torch.randn(8, 3, 128, 128)
    # Test forward pass
    feature_maps, embedding = student_encoder(x)
    # Print shapes
    print("Feature maps shapes:")
    for i, fm in enumerate(feature_maps):
        print(f"Feature map {i+1}:", fm.shape)
    print("Embedding shape:", embedding.shape)
