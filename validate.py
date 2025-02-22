from sklearn.metrics import precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import torch
import wandb
import os
import shutil
import numpy as np
from utils.image_utils import *
from torch.utils.data import DataLoader
from data.data import LandCoverDataset
from utils.data_utils import ResizeAndToClassTransform
from model.ViT.model import UNet
import warnings
from rasterio.errors import NotGeoreferencedWarning

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

def plot_validations(inputs, outputs, masks, epoch, batch_size, batch_index, input_path, num_samples=5, image_dir=None):
    # Convert tensors to numpy arrays
    inputs = inputs.cpu().numpy()
    outputs = torch.argmax(outputs, dim=1).cpu().numpy()
    if not isinstance(masks, np.ndarray):
        masks = masks.cpu().numpy()

    samples = len(inputs) if num_samples=='all' else num_samples
    for i in range(samples):
        # Create a new figure for each sample
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'Epoch {epoch + 1} - Sample {i + 1}')

        # Plot input image
        axs[0].set_title(f'Input {i+1}')
        image = inputs[i].transpose(1, 2, 0)  # Convert CHW to HWC
        image = image[:, :, :3]
        # image = (image * 255).astype(np.int64)
        divise_factor = 1
        if np.issubdtype(image.dtype, np.integer):
            max_value = np.max(image)
            divise_factor = 2 ** int(np.ceil(np.log2(max_value)))  # Normalize for display
        else:
            divise_factor = np.max(image)
        image = (image / divise_factor).astype(float)
        axs[0].imshow(image)
        axs[0].axis('off')

        # Plot prediction
        axs[1].set_title(f'Prediction {i+1}')
        axs[1].imshow(class_to_rgb(outputs[i]))
        axs[1].axis('off')

        # Plot ground truth
        axs[2].set_title(f'Groundtruth {i+1}')
        axs[2].imshow(class_to_rgb(masks[i]))
        axs[2].axis('off')

        plt.tight_layout()  # Adjust layout to prevent overlap
        input_name = os.path.basename(input_path[0])
        plt.savefig(image_dir+ f"{input_name}.png")
        plt.close()


def validate(model, data_loader, classes, image_dir=None):
    assert model is not None
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)  # Ensure the model is on the correct device

    class_idx_unidentifiable = classes.index('unidentifiable')

    with torch.no_grad():
        for batch_index, (inputs, masks, inputs_path, mask_path) in enumerate(data_loader):
            print(f"{batch_index+1}/{len(data_loader)}")
            inputs = inputs.to(device)  # Move inputs to the same device as the model
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            labels = masks.cpu().numpy()

            valid_mask = labels != class_idx_unidentifiable  # Mask to ignore "unidentifiable" class
            preds = preds[valid_mask]
            labels = labels[valid_mask]

            # Precision and recall for each class
            precision_per_class = precision_score(labels, preds, zero_division=0, average=None, labels=list(range(len(classes))))
            recall_per_class = recall_score(labels, preds, zero_division=0, average=None, labels=list(range(len(classes))))

            for i, class_name in enumerate(classes):
                if class_name == 'unidentifiable':
                    continue
                if precision_per_class[i] < 0.6 or recall_per_class[i] < 0.6:
                    plot_validations(inputs, outputs, masks, input_path=inputs_path, epoch=1, batch_size=1, batch_index=batch_index, num_samples='all', image_dir=image_dir)

if __name__ == "__main__":
    input_folder = '/mnt/henryng/2700km_cut256'
    size = (256, 256)
    validate_transform = ResizeAndToClassTransform(size=size, augment=False)
    dataset = LandCoverDataset(input_folder, validate_transform)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)
    model_path = '/mnt/anhtn/log/weights/dataset:gg_earth_cut64_dataset_model:ViTUnet_epoch:200_bs:16_lr:0.0001_datetime:20241130_015542/best_weight.pth'
    image_dir = '/mnt/anhtn/validate'
    model = UNet(n_classes=5, n_channels=5, depth=12, heads=12, dropout=0.1)
    model.load_state_dict(torch.load(model_path, weights_only=True))

    classes = ['unidentifiable', 'forest', 'rice_field', 'water', 'residential']
    validate(model, data_loader, classes, image_dir)
