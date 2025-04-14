import torch
import matplotlib.pyplot as plt
import numpy as np

from utils.image_utils import *

def plot_metrics(train_losses, val_losses, val_precisions, val_recalls, image_dir=None):
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
    plt.savefig(image_dir + 'train_result.png')
    plt.show()
    plt.close()

def plot_predictions(inputs, outputs, masks, epoch, batch_size, batch_index, CLASSES_TO_RGB, classes, num_samples=5, image_dir=None, sr_images=None, groundtruths=None):
    # Convert tensors to numpy arrays
    inputs = inputs.cpu().numpy()
    outputs = torch.argmax(outputs, dim=1).cpu().numpy()
    masks = masks.cpu().numpy() if not isinstance(masks, np.ndarray) else masks
    sr_images = sr_images.cpu().numpy() if sr_images is not None else None
    groundtruths = groundtruths.cpu().numpy() if groundtruths is not None else None
    num_subplots = 5 if sr_images is not None else 3

    samples = len(inputs) if num_samples == 'all' else num_samples
    for i in range(samples):
        fig, axs = plt.subplots(1, num_subplots, figsize=(15, 5))
        fig.suptitle(f'Epoch {epoch + 1} - Sample {i + 1}')

        # Plot input image
        axs[0].set_title(f'Input {i+1}')
        image = inputs[i].transpose(1, 2, 0)[:, :, :3] if inputs[i].shape[0] == 13 else inputs[i].transpose(1, 2, 0)[:, :, 4:1:-1]
        divise_factor = np.max(image) if not np.issubdtype(image.dtype, np.integer) else 2 ** int(np.ceil(np.log2(np.max(image))))
        image = (image / divise_factor).astype(float)
        axs[0].imshow(image)
        axs[0].axis('off')
        idx = 1

        if sr_images is not None:
            axs[idx].set_title(f'SR Images {i+1}')
            sr_image = sr_images[i].transpose(1, 2, 0)[:, :, :3]
            sr_image = np.nan_to_num(sr_image)
            # Clip values to be between 0 and 1
            sr_image = np.clip(sr_image, 0, 1)
            axs[idx].imshow(sr_image)
            axs[idx].axis('off')
            idx += 1

        if groundtruths is not None:
            axs[idx].set_title(f'Groundtruth {i+1}')
            groundtruth = groundtruths[i].transpose(1, 2, 0)[:, :, :3]
            groundtruth = np.nan_to_num(groundtruth)
            # Clip values to be between 0 and 1
            groundtruth = np.clip(groundtruth, 0, 1)
            axs[idx].imshow(groundtruth)
            axs[idx].axis('off')
            idx += 1

        axs[idx].set_title(f'Prediction {i+1}')
        axs[idx].imshow(class_to_rgb(outputs[i], CLASSES_TO_RGB, classes))
        axs[idx].axis('off')
        idx+=1

        axs[idx].set_title(f'Groundtruth {i+1}')
        axs[idx].imshow(class_to_rgb(masks[i], CLASSES_TO_RGB, classes))
        axs[idx].axis('off')

        plt.tight_layout()
        if image_dir:
            plt.savefig(f"{image_dir}{batch_index * batch_size + i + 1}.png")
        plt.close()
