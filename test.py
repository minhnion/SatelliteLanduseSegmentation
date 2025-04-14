import torch
import torch.nn as nn
import torchvision.models as models

class FCN32s(nn.Module):
    def __init__(self, num_classes=21):
        """
        Initialize the FCN-32s model based on VGG16.

        Args:
            num_classes (int): Number of segmentation classes (default: 21 for PASCAL VOC).
        """
        super(FCN32s, self).__init__()

        # Load pre-trained VGG16 model
        vgg16 = models.vgg16(pretrained=True)

        # Extract the feature extractor (convolutional layers)
        self.features = vgg16.features

        # Replace the classifier with convolutional layers
        self.fcn = nn.Sequential(
            # FC6: Replace with 7x7 conv (originally 4096 units)
            nn.Conv2d(512, 4096, kernel_size=7, padding=0),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            # FC7: Replace with 1x1 conv (originally 4096 units)
            nn.Conv2d(4096, 4096, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            # FC8: Replace with 1x1 conv to output num_classes
            nn.Conv2d(4096, num_classes, kernel_size=1, padding=0)
        )

        # Upsampling layer to recover input resolution (stride 32)
        self.upsample = nn.ConvTranspose2d(
            num_classes,
            num_classes,
            kernel_size=64,
            stride=32,
            padding=16,
            bias=False
        )

        # Initialize upsampling weights for bilinear interpolation
        self._initialize_weights()

    def forward(self, x):
        """
        Forward pass of the FCN-32s model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, H, W).

        Returns:
            torch.Tensor: Output segmentation map of shape (batch_size, num_classes, H, W).
        """
        # Feature extraction
        x = self.features(x)  # Shape: (batch_size, 512, H/32, W/32)

        # Fully convolutional layers
        x = self.fcn(x)       # Shape: (batch_size, num_classes, H/32, W/32)

        # Upsample to original resolution
        x = self.upsample(x)  # Shape: (batch_size, num_classes, H, W)

        return x

    def _initialize_weights(self):
        """Initialize the upsampling layer with bilinear interpolation weights."""
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = self._get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0]
                )
                m.weight.data.copy_(initial_weight)

    def _get_upsampling_weight(self, in_channels, out_channels, kernel_size):
        """Generate bilinear interpolation weights for upsampling."""
        factor = (kernel_size + 1) // 2
        if kernel_size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = torch.arange(kernel_size).float()
        filt = (1 - torch.abs(og - center) / factor)
        weight = torch.zeros((in_channels, out_channels, kernel_size, kernel_size))
        for i in range(in_channels):
            for j in range(out_channels):
                if i == j:
                    weight[i, j] = filt.unsqueeze(0) * filt.unsqueeze(1)
        return weight

# Example usage
if __name__ == "__main__":
    # Create model instance
    model = FCN32s(num_classes=21)

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Test with dummy input (batch_size=1, channels=3, height=224, width=224)
    input_tensor = torch.randn(1, 3, 224, 224).to(device)
    output = model(input_tensor)

    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")  # Should be (1, 21, 224, 224)
