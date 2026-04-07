import torch

model = torch.load('model/PretrainedViT/B13_vitb16_mae_ep99.pth')

# Check the model architecture
print(model)
