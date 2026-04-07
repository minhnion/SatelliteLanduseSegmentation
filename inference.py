from model.ViT.model import UNet
import torch
from utils.image_utils import class_to_rgb, open_tif_image
import rasterio

def inference(image, model, device):
    # Set the model to evaluation mode
    model.eval()

    # Convert the image to a PyTorch tensor
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0).to(device)  # (N, C, H, W)

    # Perform inference
    with torch.no_grad():
        prediction = model(image_tensor)

    # Convert the prediction to a NumPy array
    prediction = prediction.squeeze(0).cpu().numpy()

    return prediction

if __name__ == "__main__":
    # Load the model
    model = UNet(n_channels=13, n_classes=5)
    model.load_state_dict(torch.load("model/PretrainedViTUNet/best_pretrained_13bands_finetuned_model.pth"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
