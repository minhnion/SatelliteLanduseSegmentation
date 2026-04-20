import torch
import numpy as np
from pathlib import Path

from utils.image_utils import *

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def _normalize_channel(channel):
    channel = np.nan_to_num(channel.astype(np.float32))
    low, high = np.percentile(channel, [2, 98])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        high = channel.max() if channel.size else 1.0
        low = channel.min() if channel.size else 0.0
    scale = high - low
    if scale <= 0:
        return np.zeros_like(channel, dtype=np.float32)
    return np.clip((channel - low) / scale, 0.0, 1.0)


def _to_display_rgb(sample):
    image = sample.transpose(1, 2, 0)
    channels = image.shape[2]
    if channels >= 13:
        display = image[:, :, :3]
        display = np.stack([_normalize_channel(display[:, :, i]) for i in range(3)], axis=-1)
        return display
    if channels == 2:
        vv = _normalize_channel(image[:, :, 0])
        vh = _normalize_channel(image[:, :, 1])
        avg = 0.5 * (vv + vh)
        return np.stack([vv, avg, vh], axis=-1)
    if channels == 1:
        gray = _normalize_channel(image[:, :, 0])
        return np.stack([gray, gray, gray], axis=-1)
    display = image[:, :, :min(3, channels)]
    if display.shape[2] < 3:
        display = np.repeat(display, 3, axis=2)[:, :, :3]
    return np.stack([_normalize_channel(display[:, :, i]) for i in range(3)], axis=-1)

def plot_metrics(train_losses, val_losses, val_precisions, val_recalls, image_dir=None):
    if plt is None:
        print("matplotlib is not installed; skipping train metric plot generation.")
        return
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
    if image_dir:
        image_dir = Path(image_dir)
        image_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(image_dir / 'train_result.png')
    plt.close()

def plot_predictions(inputs, outputs, masks, epoch, batch_size, batch_index, CLASSES_TO_RGB, classes, num_samples=5, image_dir=None, sr_images=None, groundtruths=None):
    if plt is None:
        return
    # Convert tensors to numpy arrays
    inputs = inputs.cpu().numpy()
    outputs = torch.argmax(outputs, dim=1).cpu().numpy()
    masks = masks.cpu().numpy() if not isinstance(masks, np.ndarray) else masks
    sr_images = sr_images.cpu().numpy() if sr_images is not None else None
    groundtruths = groundtruths.cpu().numpy() if groundtruths is not None else None
    num_subplots = 5 if sr_images is not None else 3

    samples = len(inputs) if num_samples == 'all' else min(num_samples, len(inputs))
    for i in range(samples):
        fig, axs = plt.subplots(1, num_subplots, figsize=(15, 5))
        fig.suptitle(f'Epoch {epoch + 1} - Sample {i + 1}')

        # Plot input image
        axs[0].set_title(f'Input {i+1}')
        image = _to_display_rgb(inputs[i])
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
            image_dir = Path(image_dir)
            image_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(image_dir / f"{batch_index * batch_size + i + 1}.png")
        plt.close()
