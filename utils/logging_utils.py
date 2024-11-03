import torch
import matplotlib.pyplot as plt
import numpy as np

from utils.image_utils import *

def plot_metrics(train_losses, val_losses, val_precisions, val_recalls):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 10))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, 'o-', label='Train Loss')
    plt.plot(epochs, val_losses, 'o-', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs, val_precisions, 'o-', label='Val Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs, val_recalls, 'o-', label='Val Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_predictions(inputs, outputs, masks, epoch, num_samples=5):
    # Convert tensors to numpy arrays
    inputs = inputs.cpu().numpy().astype(np.int64)
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
        image = (image / np.max(image))  # Normalize for display
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
        plt.show()
