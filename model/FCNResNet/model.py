import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights

class FCNResnet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(FCNResnet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Load pretrained FCN ResNet50 model with proper weights specification
        self.model_weights = FCN_ResNet50_Weights.DEFAULT
        self.backbone = fcn_resnet50(weights=self.model_weights)

        # Modify the first convolutional layer to accept n_channels input
        self.backbone.backbone.conv1 = nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Modify the classifier to output n_classes
        self.backbone.classifier[4] = nn.Conv2d(512, n_classes, kernel_size=1)

    def forward(self, x):
        x = self.backbone(x)['out']
        return x

# Example usage
if __name__ == "__main__":
    model = FCNResnet(n_channels=13, n_classes=5)
    input_tensor = torch.randn(1, 13, 224, 224)
    output = model(input_tensor)
    print(output.shape)
