from sklearn.metrics import precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import torch
import wandb
from tqdm import tqdm

from utils.logging_utils import plot_predictions
from utils.metrics import compute_psnr, compute_ssim

def evaluate_on_test_set(model, test_loader, classes, CLASSES_TO_RGB, image_dir=None, wandb_setup=True, num_samples=5):
    assert model is not None
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)

    print(f'Test precision: {precision}, Test recall: {recall}')

    log_test = {
        'precision': precision,
        'recall': recall
    }

    # Precision and recall for each class
    precision_per_class = precision_score(all_labels, all_preds, zero_division=0, average=None, labels=list(range(len(classes))))
    recall_per_class = recall_score(all_labels, all_preds, zero_division=0, average=None, labels=list(range(len(classes))))

    for i, class_name in enumerate(classes):
        # if (i==class_idx_unidentifiable):
            # continue
        print(f'Class: {class_name} - Precision: {precision_per_class[i]:.4f}, Recall: {recall_per_class[i]:.4f}')
        log_test[f'precision_{class_name}'] = precision_per_class[i]
        log_test[f'recall_{class_name}'] = recall_per_class[i]

    # log test
    if wandb_setup:
        wandb.log({'Test log': log_test})

    # Confusion Matrix
    plt.figure(figsize=(12, 8))

    valid_classes = [cls for cls in classes if cls != 'unidentifiable']
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(valid_classes))), normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=valid_classes)
    disp.plot(cmap=plt.cm.Blues)

    plt.title("Normalized Confusion Matrix")
    wandb.log({"Confusion-matrix plot": wandb.Image(plt)})
    plt.close()

def evaluate_sr_on_test_set(model, test_loader, classes, CLASSES_TO_RGB, image_dir=None, wandb_setup=True, num_samples=5):
    assert model is not None
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

    if wandb_setup:
        wandb.log({'Test log': log_test})


def evaluate_sr_seg_on_test_set(model, test_loader, classes, CLASSES_TO_RGB, image_dir=None, wandb_setup=True, num_samples=5):
    assert model is not None
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    precision_weighted = precision_score(all_labels.numpy(), all_preds.numpy(), average='weighted', zero_division=0)
    recall_weighted = recall_score(all_labels.numpy(), all_preds.numpy(), average='weighted', zero_division=0)

    # Compute per-class precision and recall
    precision_per_class = precision_score(all_labels.numpy(), all_preds.numpy(), average=None, zero_division=0, labels=list(range(len(classes))))
    recall_per_class = recall_score(all_labels.numpy(), all_preds.numpy(), average=None, zero_division=0, labels=list(range(len(classes))))

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

    if wandb_setup:
        wandb.log({'Test log': log_test})

    # Confusion Matrix
    plt.figure(figsize=(12, 8))
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(classes))), normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Normalized Confusion Matrix")
    if wandb_setup:
        wandb.log({"Confusion-matrix plot": wandb.Image(plt)})
    plt.show()
    plt.close()

def evaluate_scnet_on_test_set(model, test_loader, classes, CLASSES_TO_RGB, image_dir=None, wandb_setup=True, num_samples=5):
    assert model is not None
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)

    print(f'Test precision: {precision}, Test recall: {recall}')

    log_test = {
        'precision': precision,
        'recall': recall
    }

    # Precision and recall for each class
    precision_per_class = precision_score(all_labels, all_preds, zero_division=0, average=None, labels=list(range(len(classes))))
    recall_per_class = recall_score(all_labels, all_preds, zero_division=0, average=None, labels=list(range(len(classes))))

    for i, class_name in enumerate(classes):
        # if (i==class_idx_unidentifiable):
            # continue
        print(f'Class: {class_name} - Precision: {precision_per_class[i]:.4f}, Recall: {recall_per_class[i]:.4f}')
        log_test[f'precision_{class_name}'] = precision_per_class[i]
        log_test[f'recall_{class_name}'] = recall_per_class[i]

    # log test
    if wandb_setup:
        wandb.log({'Test log': log_test})

    # Confusion Matrix
    plt.figure(figsize=(12, 8))

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(classes))), normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues)

    plt.title("Normalized Confusion Matrix")
    wandb.log({"Confusion-matrix plot": wandb.Image(plt)})
    plt.close()
