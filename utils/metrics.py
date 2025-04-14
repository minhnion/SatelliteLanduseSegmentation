import numpy as np
from utils.data_utils import to_numpy_array
from skimage.metrics import structural_similarity as ssim

def calculate_iou(preds, labels, num_classes):
    iou_per_class = []

    for class_id in range(num_classes):
        intersection = np.logical_and(preds == class_id, labels == class_id).sum()
        union = np.logical_or(preds == class_id, labels == class_id).sum()

        if union == 0:
            iou_per_class.append(np.nan)  # Avoid division by zero
        else:
            iou_per_class.append(intersection / union)

    return iou_per_class

def compute_psnr(img1, img2) -> float:
    max_pixel = 255 if img1.dtype == np.uint8 else 1.0
    img1, img2 = to_numpy_array(img1), to_numpy_array(img2)
    assert img1.shape == img2.shape, "Input images must have the same shape"

    img1 = np.clip(img1, 0, 1)
    img2 = np.clip(img2, 0, 1)

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')  # No difference between images

    return 20 * np.log10(max_pixel / np.sqrt(mse))

def compute_ssim(img1, img2) -> float:
    img1, img2 = to_numpy_array(img1), to_numpy_array(img2)
    assert img1.shape == img2.shape, "Input images must have the same shape"

    if img1.ndim == 3:  # Multi-channel image (C, H, W) or (H, W, C)
        ssim_values = [ssim(img1[c], img2[c], data_range=1.0) for c in range(img1.shape[0])]
        return np.mean(ssim_values)  # Average SSIM across channels
    else:
        return ssim(img1, img2, data_range=img2.max() - img2.min())  # Grayscale image
