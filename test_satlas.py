import satlaspretrain_models
import torch.nn as nn
import torch
from torchsummary import summary

import torch

def get_model_size(model):
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size = (param_size + buffer_size) / (1024 ** 2)  # Convert to MB
    return total_size

# Load pre-trained model
weights_manager = satlaspretrain_models.Weights()
model = weights_manager.get_pretrained_model('Sentinel2_SwinB_SI_RGB')

# Print the model architecture
# print(model)
n_channels = 13
model.backbone.backbone.features[0][0] = nn.Conv2d(n_channels, 128, kernel_size=(1, 1), stride=(1, 1))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
dummy_input = torch.randn(1, n_channels, 512, 512, device=device)

torch.cuda.reset_peak_memory_stats(device)  # Reset memory stats
torch.cuda.synchronize()  # Ensure all operations are finished

with torch.no_grad():
    feature_maps = model(dummy_input)
    for feature_map in feature_maps:
        print(feature_map.shape)

print(f"Model size: {get_model_size(model):.2f} MB")
peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # Convert to MB
print(f"Peak GPU memory usage: {peak_memory:.2f} MB")

# summary(model, (n_channels, 512, 512))
