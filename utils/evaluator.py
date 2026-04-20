import json
from pathlib import Path

import numpy as np
import torch

from utils.logging_utils import plot_predictions
from utils.metrics import calculate_iou, compute_confusion_matrix, compute_precision_recall, compute_psnr, compute_ssim, normalize_confusion_matrix

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    from tqdm import tqdm
except ImportError:
    class _SimpleTqdm:
        def __init__(self, iterable=None, *args, **kwargs):
            self.iterable = iterable
            self.n = 0

        def __iter__(self):
            for item in self.iterable:
                yield item
                self.n += 1

        def set_postfix(self, **kwargs):
            return None

    def tqdm(iterable=None, *args, **kwargs):
        return _SimpleTqdm(iterable, *args, **kwargs)

try:
    import wandb
except ImportError:
    wandb = None


def _write_metrics_json(metrics_path, payload):
    if not metrics_path:
        return
    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)


def _plot_confusion_matrix(confusion, classes, title):
    if plt is None:
        return None
    fig, ax = plt.subplots(figsize=(12, 8))
    image = ax.imshow(confusion, cmap=plt.cm.Blues)
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    for row_index in range(confusion.shape[0]):
        for col_index in range(confusion.shape[1]):
            value = confusion[row_index, col_index]
            ax.text(col_index, row_index, f"{value:.2f}", ha="center", va="center", color="black")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    return fig


def evaluate_on_test_set(model, test_loader, classes, CLASSES_TO_RGB, image_dir=None, wandb_setup=True, num_samples=5, metrics_path=None):
    assert model is not None
    model.eval()
    device = next(model.parameters()).device
    model.to(device)  # Ensure the model is on the correct device

    # class_idx_unidentifiable = classes.index('unidentifiable')

    all_labels = []
    all_preds = []
    all_labels_each_cls = []
    all_preds_each_cls = []
    progress_bar = tqdm(test_loader, desc="Testing", leave=True)

    with torch.inference_mode():
        for batch_index, (inputs, masks, *_) in enumerate(progress_bar):
            inputs = inputs.to(device)  # Move inputs to the same device as the model
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            labels = masks.cpu().numpy()

            # valid_mask = labels != class_idx_unidentifiable  # Mask to ignore "unidentifiable" class
            # preds = preds[valid_mask]
            # labels = labels[valid_mask]
            batch_size = test_loader.batch_size

            plot_predictions(inputs, outputs, masks, classes=classes, epoch=1, batch_size=batch_size, batch_index=batch_index, num_samples=num_samples, image_dir = image_dir, CLASSES_TO_RGB=CLASSES_TO_RGB)

            all_preds.extend(preds.flatten())
            all_labels.extend(labels.flatten())

    # Precision and recall mean
    precision, recall, precision_per_class, recall_per_class, confusion = compute_precision_recall(
        all_labels, all_preds, len(classes)
    )
    iou_per_class = calculate_iou(np.array(all_preds), np.array(all_labels), len(classes))
    mean_iou = float(np.nanmean(iou_per_class))

    print(f'Test precision: {precision}, Test recall: {recall}, Test mIoU: {mean_iou}')

    log_test = {
        'precision': float(precision),
        'recall': float(recall),
        'mean_iou': float(mean_iou),
    }

    for i, class_name in enumerate(classes):
        # if (i==class_idx_unidentifiable):
            # continue
        class_iou = float(iou_per_class[i]) if not np.isnan(iou_per_class[i]) else None
        print(f'Class: {class_name} - Precision: {precision_per_class[i]:.4f}, Recall: {recall_per_class[i]:.4f}, IoU: {class_iou}')
        log_test[f'precision_{class_name}'] = float(precision_per_class[i])
        log_test[f'recall_{class_name}'] = float(recall_per_class[i])
        log_test[f'iou_{class_name}'] = class_iou

    # log test
    if wandb_setup and wandb is not None:
        wandb.log({'Test log': log_test})
    _write_metrics_json(metrics_path, log_test)

    # Confusion Matrix
    cm = normalize_confusion_matrix(confusion, normalize='true')
    fig = _plot_confusion_matrix(cm, classes, "Normalized Confusion Matrix")
    if fig is not None and image_dir:
        image_dir = Path(image_dir)
        image_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(image_dir / "test_confusion_matrix.png")
    if fig is not None and wandb_setup and wandb is not None:
        wandb.log({"Confusion-matrix plot": wandb.Image(fig)})
    if fig is not None:
        plt.close(fig)

def evaluate_sr_on_test_set(model, test_loader, classes, CLASSES_TO_RGB, image_dir=None, wandb_setup=True, num_samples=5):
    assert model is not None
    model.eval()
    device = next(model.parameters()).device
    model.to(device)

    # class_idx_unidentifiable = classes.index('Unidentifiable')

    psnr, ssim = 0.0, 0.0
    batch_size = test_loader.batch_size

    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Testing", leave=True)

        for batch_index, (inputs, masks, lr_images, *_) in enumerate(progress_bar):
            inputs, masks, lr_images = (
                inputs.to(device, non_blocking=True),
                masks.to(device, non_blocking=True),
                lr_images.to(device, non_blocking=True)
            )

            sr_images = model(lr_images)

            # Compute PSNR and SSIM efficiently
            psnr += torch.mean(torch.tensor([
                compute_psnr(sr.cpu().numpy(), inp.cpu().numpy()) for sr, inp in zip(sr_images, inputs)
            ])).item()

            ssim += torch.mean(torch.tensor([
                compute_ssim(sr.cpu().numpy(), inp.cpu().numpy()) for sr, inp in zip(sr_images, inputs)
            ])).item()

            # Ignore unidentifiable class
            # valid_mask = labels != class_idx_unidentifiable
            # preds, labels = preds[valid_mask], labels[valid_mask]

            plot_predictions(lr_images, lr_images, masks, classes=classes, epoch=1, num_samples="all", sr_images=sr_images, groundtruths=inputs, image_dir=image_dir, batch_index=batch_index, batch_size=batch_size, CLASSES_TO_RGB=CLASSES_TO_RGB)

    psnr /= len(test_loader)
    ssim /= len(test_loader)

    print(f'Test PSNR: {psnr:.4f}, Test SSIM: {ssim:.4f}')

    log_test = {
        'psnr': psnr,
        'ssim': ssim
    }

    if wandb_setup and wandb is not None:
        wandb.log({'Test log': log_test})


def evaluate_sr_seg_on_test_set(model, test_loader, classes, CLASSES_TO_RGB, image_dir=None, wandb_setup=True, num_samples=5):
    assert model is not None
    model.eval()
    device = next(model.parameters()).device
    model.to(device)

    # class_idx_unidentifiable = classes.index('Unidentifiable')

    all_labels = []
    all_preds = []
    psnr, ssim = 0.0, 0.0
    precision_per_class, recall_per_class = torch.zeros(len(classes)), torch.zeros(len(classes))
    batch_size = test_loader.batch_size

    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Testing", leave=True)

        for batch_index, (inputs, masks, lr_images, *_) in enumerate(progress_bar):
            inputs, masks, lr_images = (
                inputs.to(device, non_blocking=True),
                masks.to(device, non_blocking=True),
                lr_images.to(device, non_blocking=True)
            )

            model_name = model.__class__.__name__
            if model_name == 'FoundationKDModel':
                # SRSeg model
                # Default case (e.g., for other models)
                _, _, (sr_images, seg_masks) = model(lr_images)
            else:
                sr_images, seg_masks = model(lr_images)

            # Compute PSNR and SSIM efficiently
            psnr += torch.mean(torch.tensor([
                compute_psnr(sr.cpu().numpy(), inp.cpu().numpy()) for sr, inp in zip(sr_images, inputs)
            ])).item()

            ssim += torch.mean(torch.tensor([
                compute_ssim(sr.cpu().numpy(), inp.cpu().numpy()) for sr, inp in zip(sr_images, inputs)
            ])).item()

            # Flatten tensors for classification metrics
            preds = torch.argmax(seg_masks, dim=1).cpu().flatten()
            labels = masks.cpu().flatten()

            # Ignore unidentifiable class
            # valid_mask = labels != class_idx_unidentifiable
            # preds, labels = preds[valid_mask], labels[valid_mask]

            all_preds.append(preds)
            all_labels.append(labels)

            plot_predictions(lr_images, seg_masks, masks, classes=classes, epoch=1, num_samples="all", sr_images=sr_images, groundtruths=inputs, image_dir=image_dir, batch_index=batch_index, batch_size=batch_size, CLASSES_TO_RGB=CLASSES_TO_RGB)

    # Compute averages
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # Compute overall precision and recall
    precision_weighted, recall_weighted, precision_per_class, recall_per_class, confusion = compute_precision_recall(
        all_labels.numpy(), all_preds.numpy(), len(classes)
    )

    psnr /= len(test_loader)
    ssim /= len(test_loader)

    print(f'Test Precision: {precision_weighted:.4f}, Test Recall: {recall_weighted:.4f}')
    print(f'Test PSNR: {psnr:.4f}, Test SSIM: {ssim:.4f}')

    log_test = {
        'precision': precision_weighted,
        'recall': recall_weighted,
        'psnr': psnr,
        'ssim': ssim
    }

    for i, class_name in enumerate(classes):
        # if i != class_idx_unidentifiable:
            print(f'Class: {class_name} - Precision: {precision_per_class[i]:.4f}, Recall: {recall_per_class[i]:.4f}')
            log_test[f'{class_name}_precision'] = precision_per_class[i].item()
            log_test[f'{class_name}_recall'] = recall_per_class[i].item()

    if wandb_setup and wandb is not None:
        wandb.log({'Test log': log_test})

    # Confusion Matrix
    cm = normalize_confusion_matrix(confusion, normalize='true')
    fig = _plot_confusion_matrix(cm, classes, "Normalized Confusion Matrix")
    if fig is not None and wandb_setup and wandb is not None:
        wandb.log({"Confusion-matrix plot": wandb.Image(fig)})
    if fig is not None:
        plt.close(fig)

def evaluate_scnet_on_test_set(model, test_loader, classes, CLASSES_TO_RGB, image_dir=None, wandb_setup=True, num_samples=5):
    assert model is not None
    model.eval()
    device = next(model.parameters()).device
    model.to(device)  # Ensure the model is on the correct device

    # class_idx_unidentifiable = classes.index('unidentifiable')

    all_labels = []
    all_preds = []
    all_labels_each_cls = []
    all_preds_each_cls = []
    progress_bar = tqdm(test_loader, desc="Testing", leave=True)

    with torch.inference_mode():
        for batch_index, (inputs, masks, lr_images, fraction_alphas, *_) in enumerate(progress_bar):
            inputs, lr_images, fraction_alphas = inputs.to(device), lr_images.to(device), fraction_alphas.to(device)
            outputs = model(lr_images, fraction_alphas)
            preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            labels = masks.cpu().numpy()

            # valid_mask = labels != class_idx_unidentifiable  # Mask to ignore "unidentifiable" class
            # preds = preds[valid_mask]
            # labels = labels[valid_mask]
            batch_size = test_loader.batch_size

            plot_predictions(inputs, outputs, masks, classes=classes, epoch=1, batch_size=batch_size, batch_index=batch_index, num_samples=num_samples, image_dir = image_dir, CLASSES_TO_RGB=CLASSES_TO_RGB)

            all_preds.extend(preds.flatten())
            all_labels.extend(labels.flatten())

    # Precision and recall mean
    precision, recall, precision_per_class, recall_per_class, confusion = compute_precision_recall(
        all_labels, all_preds, len(classes)
    )

    print(f'Test precision: {precision}, Test recall: {recall}')

    log_test = {
        'precision': precision,
        'recall': recall
    }

    for i, class_name in enumerate(classes):
        # if (i==class_idx_unidentifiable):
            # continue
        print(f'Class: {class_name} - Precision: {precision_per_class[i]:.4f}, Recall: {recall_per_class[i]:.4f}')
        log_test[f'precision_{class_name}'] = precision_per_class[i]
        log_test[f'recall_{class_name}'] = recall_per_class[i]

    # log test
    if wandb_setup and wandb is not None:
        wandb.log({'Test log': log_test})

    # Confusion Matrix
    cm = normalize_confusion_matrix(confusion, normalize='true')
    fig = _plot_confusion_matrix(cm, classes, "Normalized Confusion Matrix")
    if fig is not None and wandb_setup and wandb is not None:
        wandb.log({"Confusion-matrix plot": wandb.Image(fig)})
    if fig is not None:
        plt.close(fig)
