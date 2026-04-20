import numpy as np
from utils.data_utils import to_numpy_array

try:
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    ssim = None


def compute_confusion_matrix(labels, preds, num_classes):
    labels = np.asarray(labels).reshape(-1)
    preds = np.asarray(preds).reshape(-1)
    valid = (
        (labels >= 0) & (labels < num_classes) &
        (preds >= 0) & (preds < num_classes)
    )
    labels = labels[valid].astype(np.int64, copy=False)
    preds = preds[valid].astype(np.int64, copy=False)
    counts = np.bincount(num_classes * labels + preds, minlength=num_classes * num_classes)
    return counts.reshape(num_classes, num_classes)


def compute_precision_recall(labels, preds, num_classes):
    confusion = compute_confusion_matrix(labels, preds, num_classes)
    true_positive = np.diag(confusion).astype(np.float64)
    predicted_positive = confusion.sum(axis=0).astype(np.float64)
    actual_positive = confusion.sum(axis=1).astype(np.float64)

    precision_per_class = np.divide(
        true_positive,
        predicted_positive,
        out=np.zeros(num_classes, dtype=np.float64),
        where=predicted_positive != 0,
    )
    recall_per_class = np.divide(
        true_positive,
        actual_positive,
        out=np.zeros(num_classes, dtype=np.float64),
        where=actual_positive != 0,
    )

    support = actual_positive
    total_support = support.sum()
    if total_support == 0:
        weighted_precision = 0.0
        weighted_recall = 0.0
    else:
        weighted_precision = float(np.sum(precision_per_class * support) / total_support)
        weighted_recall = float(np.sum(recall_per_class * support) / total_support)

    return weighted_precision, weighted_recall, precision_per_class, recall_per_class, confusion


def normalize_confusion_matrix(confusion, normalize='true'):
    confusion = np.asarray(confusion, dtype=np.float64)
    if normalize == 'true':
        row_sums = confusion.sum(axis=1, keepdims=True)
        return np.divide(confusion, row_sums, out=np.zeros_like(confusion), where=row_sums != 0)
    if normalize == 'pred':
        col_sums = confusion.sum(axis=0, keepdims=True)
        return np.divide(confusion, col_sums, out=np.zeros_like(confusion), where=col_sums != 0)
    if normalize == 'all':
        total = confusion.sum()
        return confusion / total if total else np.zeros_like(confusion)
    return confusion

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
    if ssim is None:
        raise ImportError("scikit-image is required for SSIM computation")
    img1, img2 = to_numpy_array(img1), to_numpy_array(img2)
    assert img1.shape == img2.shape, "Input images must have the same shape"

    if img1.ndim == 3:  # Multi-channel image (C, H, W) or (H, W, C)
        ssim_values = [ssim(img1[c], img2[c], data_range=1.0) for c in range(img1.shape[0])]
        return np.mean(ssim_values)  # Average SSIM across channels
    else:
        return ssim(img1, img2, data_range=img2.max() - img2.min())  # Grayscale image
