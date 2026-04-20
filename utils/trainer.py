import copy
import csv
import json
import gc
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

try:
    import wandb
except ImportError:
    wandb = None

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

from utils.logging_utils import plot_metrics
from utils.metrics import calculate_iou, compute_precision_recall, compute_psnr, compute_ssim


def _append_metrics_row(csv_path, row):
    if not csv_path:
        return
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _write_metrics_json(json_path, payload):
    if not json_path:
        return
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)


def _wandb_watch(model, enabled):
    if enabled and wandb is not None:
        wandb.watch(model, log='all', log_freq=100)


def _wandb_log(payload, enabled):
    if enabled and wandb is not None:
        wandb.log(payload)


def _save_full_model_snapshots(model, save_path):
    try:
        torch.save(model, save_path.replace(".pth", "_full.pth"))
        model_cpu = copy.deepcopy(model).to('cpu')
        torch.save(model_cpu, save_path.replace(".pth", "_cpu_full.pth"))
        del model_cpu
    except Exception as error:
        print(f"Error while saving full model snapshot: {error}")

def train(model, train_loader, val_loader, optimizer,
          scheduler, criterion, classes, device,
          num_epochs=100, save_path=f'best_model.pth', l1_lambda=0.0001,
          early_stop=True, patience=20, image_dir=None,
          train_metrics_path=None, best_metrics_path=None, wandb_setup=True):
    best_val_loss, best_train_loss, best_model_val_precision, best_model_val_recall = float('inf'), float('inf'), float('inf'), float('inf')
    best_model_val_miou = float('-inf')
    early_stop_counter = 0
    best_model_wts = None

    # class_idx_unidentifiable = classes.index('unidentifiable')

    _wandb_watch(model, wandb_setup)

    train_losses, val_losses, val_precisions, val_recalls = [], [], [], []

    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        print(f"\nEpoch: {epoch+1}")
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)

        for inputs, masks, *_ in progress_bar:
            # torch.cuda.empty_cache()
            inputs, masks = inputs.to(device), masks.to(device)
            # print(f"inputs, masks: {inputs.shape}, {masks.shape}")
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, masks)
            if l1_lambda and l1_lambda > 0:
                l1_reg = sum(param.abs().sum() for param in model.parameters())  # Faster than torch.norm(param, p=1)
                loss = loss + l1_lambda * l1_reg
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=running_loss / (progress_bar.n + 1))  # Show avg loss

        del inputs, masks

        epoch_train_loss = running_loss / len(train_loader)
        train_losses.append(epoch_train_loss)

        torch.cuda.empty_cache()

        # Validation
        model.eval()
        val_running_loss = 0.0
        all_preds, all_labels = [], []  # Store as lists for memory efficiency

        precision_per_class = torch.zeros(len(classes), device=device)
        recall_per_class = torch.zeros(len(classes), device=device)

        progress_bar = tqdm(val_loader, desc="Validation", leave=True)

        with torch.no_grad():
            for inputs, masks, *_ in progress_bar:
                inputs, masks = inputs.to(device), masks.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, masks)
                val_running_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)

                # valid_mask = preds != class_idx_unidentifiable  # Mask to ignore "unidentifiable" class
                # preds, masks = preds[valid_mask], masks[valid_mask]

                all_preds.append(preds.flatten().cpu())
                all_labels.append(masks.flatten().cpu())

                progress_bar.set_postfix(loss=val_running_loss / (progress_bar.n + 1))  # Show avg loss

        epoch_val_loss = val_running_loss / len(val_loader)
        val_losses.append(epoch_val_loss)
        scheduler.step(epoch_val_loss)

        # Convert to tensors at once to save memory
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        precision, recall, precision_per_class_np, recall_per_class_np, _ = compute_precision_recall(
            all_labels.numpy(), all_preds.numpy(), len(classes)
        )
        precision_per_class = torch.tensor(precision_per_class_np, device=device)
        recall_per_class = torch.tensor(recall_per_class_np, device=device)
        iou_per_class = calculate_iou(all_preds.numpy(), all_labels.numpy(), len(classes))
        mean_iou = float(np.nanmean(iou_per_class))

        val_precisions.append(precision)
        val_recalls.append(recall)

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Precision: {precision:.4f}, Val Recall: {recall:.4f}, Val mIoU: {mean_iou:.4f}')

        for i, class_name in enumerate(classes):
            # if i != class_idx_unidentifiable:
                class_iou = float(iou_per_class[i]) if not np.isnan(iou_per_class[i]) else None
                print(f'Class: {class_name} - Precision: {precision_per_class[i]:.4f}, Recall: {recall_per_class[i]:.4f}, IoU: {class_iou}')

        log_data = {
            "epoch": epoch + 1,
            "train_loss": epoch_train_loss,
            "val_loss": epoch_val_loss,
            "val_precision": precision,
            "val_recall": recall,
            "val_miou": mean_iou,
        }

        for i, class_name in enumerate(classes):
            # if i != class_idx_unidentifiable:
                log_data[f"{class_name}_precision"] = precision_per_class[i].item()
                log_data[f"{class_name}_recall"] = recall_per_class[i].item()
                log_data[f"{class_name}_iou"] = None if np.isnan(iou_per_class[i]) else float(iou_per_class[i])

        _append_metrics_row(train_metrics_path, log_data)
        _wandb_log({"Train log": log_data}, wandb_setup)

        # Save the best model
        if epoch_val_loss < best_val_loss:
            best_val_loss, best_train_loss = epoch_val_loss, epoch_train_loss
            best_model_val_precision, best_model_val_recall = precision, recall
            best_model_val_miou = mean_iou
            best_model_wts = copy.deepcopy(model.state_dict())

            torch.save(best_model_wts, save_path)  # GPU-compatible
            cpu_save_path = save_path.replace(".pth", "_cpu.pth")
            torch.save({k: v.cpu() for k, v in best_model_wts.items()}, cpu_save_path)  # CPU-compatible

            _save_full_model_snapshots(model, save_path)

            print(f"Best model saved with validation loss: {best_val_loss:.4f}")
            best_payload = {
                "epoch": epoch + 1,
                "train_loss": best_train_loss,
                "val_loss": best_val_loss,
                "val_precision": best_model_val_precision,
                "val_recall": best_model_val_recall,
                "val_miou": best_model_val_miou,
                "save_path": save_path,
                "cpu_save_path": cpu_save_path,
            }
            for i, class_name in enumerate(classes):
                best_payload[f"{class_name}_precision"] = precision_per_class[i].item()
                best_payload[f"{class_name}_recall"] = recall_per_class[i].item()
                best_payload[f"{class_name}_iou"] = None if np.isnan(iou_per_class[i]) else float(iou_per_class[i])
            _write_metrics_json(best_metrics_path, best_payload)
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop and early_stop_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

        torch.cuda.empty_cache()  # Only call once per epoch
    if best_model_wts:
        model.load_state_dict(best_model_wts)
    else:
        model = None

    print(f"Best model -> Train Loss: {best_train_loss:.4f}, Val Loss: {best_val_loss:.4f}, Precision: {best_model_val_precision:.4f}, Recall: {best_model_val_recall:.4f}, mIoU: {best_model_val_miou:.4f}")
    plot_metrics(train_losses, val_losses, val_precisions, val_recalls, image_dir=image_dir)

    return model

def train_sr_only(model, train_loader, val_loader, optimizer,
          scheduler, criterion_sr, classes, device,
          num_epochs=100, save_path=f'best_model.pth', l1_lambda=0.0001,
          early_stop=True, patience=20, image_dir=None):

    best_val_loss_sr = float('inf')
    best_train_loss_sr = float('inf')
    best_psnr = 0.0
    best_ssim = 0.0

    # class_idx_unidentifiable = classes.index('unidentifiable')

    cpu_save_path = save_path.replace('.pth', '_cpu.pth')
    early_stop_counter = 0
    best_wts = None

    _wandb_watch(model, wandb_setup=True)

    train_losses_sr, val_losses_sr = [], []

    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        gc.collect()

        print(f"\nEpoch: {epoch+1}")
        model.train()

        running_loss_sr = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)
        for inputs, masks, lr_images, *_ in progress_bar:
            inputs, masks, lr_images = (
                inputs.to(device, non_blocking=True),
                masks.to(device, non_blocking=True),
                lr_images.to(device, non_blocking=True)
            )

            optimizer.zero_grad()

            sr_images = model(lr_images)
            loss_sr = criterion_sr(sr_images, inputs)

            loss_sr.backward()
            optimizer.step()

            # Use .detach() to avoid storing gradients, but no need for .item() inside loop
            running_loss_sr += loss_sr.detach()

            total_running_loss = (running_loss_sr) / (progress_bar.n + 1)
            progress_bar.set_postfix(loss=total_running_loss.item())  # Convert once

        current_wts = copy.deepcopy(model.state_dict())

        torch.save(current_wts, save_path.replace('best', 'current'))  # GPU-compatible

        # Convert to scalar only after accumulation
        epoch_train_loss_sr = running_loss_sr.item() / len(train_loader)

        scheduler.step(epoch_train_loss_sr)

        train_losses_sr.append(epoch_train_loss_sr)

        torch.cuda.empty_cache()
        gc.collect()

        # Validation
        model.eval()
        val_loss_sr = 0.0
        psnr, ssim = 0.0, 0.0

        progress_bar = tqdm(val_loader, desc=f"Validation epoch {epoch+1}/{num_epochs}", leave=True)

        with torch.no_grad():
            for inputs, masks, lr_images, *_ in progress_bar:
                inputs, masks, lr_images = (
                    inputs.to(device, non_blocking=True),
                    masks.to(device, non_blocking=True),
                    lr_images.to(device, non_blocking=True)
                )

                sr_images = model(lr_images)
                loss_sr = criterion_sr(sr_images, inputs).detach()

                val_loss_sr += loss_sr

                # Compute PSNR and SSIM efficiently
                psnr += torch.mean(torch.tensor([
                    compute_psnr(sr.cpu().numpy(), inp.cpu().numpy()) for sr, inp in zip(sr_images, inputs)
                ])).item()

                ssim += torch.mean(torch.tensor([
                    compute_ssim(sr.cpu().numpy(), inp.cpu().numpy()) for sr, inp in zip(sr_images, inputs)
                ])).item()

        epoch_val_loss_sr = val_loss_sr.item() / len(val_loader)

        epoch_psnr = psnr / len(val_loader)
        epoch_ssim = ssim / len(val_loader)

        val_losses_sr.append(epoch_val_loss_sr)

        # Print out all metrics
        print("="*50)
        print(f"Epoch {epoch + 1}")
        print(f"Train SR Loss: {epoch_train_loss_sr:.4f}")
        print(f"Validation SR Loss: {epoch_val_loss_sr:.4f}")
        print(f"PSNR: {epoch_psnr:.4f}, SSIM: {epoch_ssim:.4f}")
        print("="*50)

        # Logging and Model Saving
        log_data = {
            "epoch": epoch + 1,
            "train_sr_loss": epoch_train_loss_sr,
            "val_sr_loss": epoch_val_loss_sr,
            "val_psnr": epoch_psnr,
            "val_ssim": epoch_ssim,
        }

        _wandb_log({"Validation Metrics": log_data}, enabled=True)

        # Save Best Model
        if epoch_val_loss_sr < best_val_loss_sr:
            best_train_loss_sr, best_val_loss_sr = epoch_train_loss_sr, epoch_val_loss_sr
            best_psnr, best_ssim = epoch_psnr, epoch_ssim
            best_wts = copy.deepcopy(model.state_dict())

            torch.save(best_wts, save_path)  # GPU-compatible
            torch.save({k: v.to('cpu') for k, v in best_wts.items()}, save_path.replace(".pth", "_cpu.pth"))  # CPU-compatible

            print(f"Best model saved with validation loss: {best_val_loss_sr:.4f}")
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        # Early Stopping
        if early_stop and early_stop_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

        torch.cuda.empty_cache()
        gc.collect()

    torch.cuda.empty_cache()  # Free memory after validation
    if best_wts:
        model.load_state_dict(best_wts)
    else:
        model = None

    print(f"Best model -> Train SR Loss: {best_train_loss_sr:.4f}\n"
            f"Val SR Loss: {best_val_loss_sr}\n"
            f"Val PSNR: {best_psnr:.4f}, Val SSIM: {best_ssim:.4f}")

    return model

def train_sr_seg(model, train_loader, val_loader, optimizer,
          scheduler, criterion_seg, criterion_sr, classes, device,
          num_epochs=100, save_path=f'best_model.pth', l1_lambda=0.0001,
          early_stop=True, patience=20, image_dir=None):

    best_val_loss_sr = float('inf')
    best_val_loss_seg = float('inf')
    best_train_loss_sr = float('inf')
    best_train_loss_seg = float('inf')
    best_val_precision = 0.0
    best_val_recall = 0.0
    best_psnr = 0.0
    best_ssim = 0.0

    # class_idx_unidentifiable = classes.index('unidentifiable')

    cpu_save_path = save_path.replace('.pth', '_cpu.pth')
    early_stop_counter = 0
    best_wts = None

    _wandb_watch(model, enabled=True)

    train_losses_sr, train_losses_seg, val_losses_sr, val_losses_seg, val_precisions, val_recalls = [], [], [], [], [], []

    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        gc.collect()

        print(f"\nEpoch: {epoch+1}")
        model.train()

        running_loss_sr, running_loss_seg = 0.0, 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)
        for inputs, masks, lr_images, *_ in progress_bar:
            inputs, masks, lr_images = (
                inputs.to(device, non_blocking=True),
                masks.to(device, non_blocking=True),
                lr_images.to(device, non_blocking=True)
            )

            optimizer.zero_grad()

            model_name = model.__class__.__name__
            if model_name == 'FoundationKDModel':
                _, _, (sr_images, seg_masks) = model(lr_images)
            else:
                sr_images, seg_masks = model(lr_images)
            loss_sr = criterion_sr(sr_images, inputs)
            loss_seg = criterion_seg(seg_masks, masks)
            total_loss = loss_sr + loss_seg

            total_loss.backward()
            optimizer.step()

            # Use .detach() to avoid storing gradients, but no need for .item() inside loop
            running_loss_sr += loss_sr.detach()
            running_loss_seg += loss_seg.detach()

            total_running_loss = (running_loss_sr + running_loss_seg) / (progress_bar.n + 1)
            progress_bar.set_postfix(loss=total_running_loss.item())  # Convert once

        current_wts = copy.deepcopy(model.state_dict())

        torch.save(current_wts, save_path.replace('best', 'current'))  # GPU-compatible

        # Convert to scalar only after accumulation
        epoch_train_loss_sr = running_loss_sr.item() / len(train_loader)
        epoch_train_loss_seg = running_loss_seg.item() / len(train_loader)

        scheduler.step(epoch_train_loss_seg)

        train_losses_sr.append(epoch_train_loss_sr)
        train_losses_seg.append(epoch_train_loss_seg)

        torch.cuda.empty_cache()
        gc.collect()

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        val_loss_sr, val_loss_seg = 0.0, 0.0
        psnr, ssim = 0.0, 0.0

        progress_bar = tqdm(val_loader, desc=f"Validation epoch {epoch+1}/{num_epochs}", leave=True)

        with torch.no_grad():
            for inputs, masks, lr_images, *_ in progress_bar:
                inputs, masks, lr_images = (
                    inputs.to(device, non_blocking=True),
                    masks.to(device, non_blocking=True),
                    lr_images.to(device, non_blocking=True)
                )


                if model_name == 'FoundationKDModel':
                    _, _, (sr_images, seg_masks) = model(lr_images)
                else:
                    sr_images, seg_masks = model(lr_images)
                loss_sr = criterion_sr(sr_images, inputs).detach()
                loss_seg = criterion_seg(seg_masks, masks).detach()

                val_loss_sr += loss_sr
                val_loss_seg += loss_seg

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

                # # Ignore unidentifiable class
                # valid_mask = labels != class_idx_unidentifiable
                # preds, labels = preds[valid_mask], labels[valid_mask]

                all_preds.append(preds)
                all_labels.append(labels)

        epoch_val_loss_sr = val_loss_sr.item() / len(val_loader)
        epoch_val_loss_seg = val_loss_seg.item() / len(val_loader)

        epoch_psnr = psnr / len(val_loader)
        epoch_ssim = ssim / len(val_loader)

        val_losses_sr.append(epoch_val_loss_sr)
        val_losses_seg.append(epoch_val_loss_seg)

        # Convert list of tensors to a single tensor for precision/recall calculation
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        # Compute overall precision and recall
        precision_average, recall_average, precision_per_class, recall_per_class, _ = compute_precision_recall(
            all_labels.numpy(), all_preds.numpy(), len(classes)
        )

        # Print out all metrics
        print("="*50)
        print(f"Epoch {epoch + 1}")
        print(f"Train SR Loss: {epoch_train_loss_sr:.4f}, Train Seg Loss: {epoch_train_loss_seg:.4f}")
        print(f"Validation SR Loss: {epoch_val_loss_sr:.4f}, Validation Seg Loss: {epoch_val_loss_seg:.4f}")
        print(f"PSNR: {epoch_psnr:.4f}, SSIM: {epoch_ssim:.4f}")
        print(f"Overall Precision: {precision_average:.4f}, Overall Recall: {recall_average:.4f}")
        print("Per-Class Precision:")
        for i, p in enumerate(precision_per_class):
            # if i != class_idx_unidentifiable:
                print(f"  Class {classes[i]}: {p:.4f}")
        print("Per-Class Recall:")
        for i, r in enumerate(recall_per_class):
            # if i != class_idx_unidentifiable:
                print(f"  Class {classes[i]}: {r:.4f}")
        print("="*50)

        # Logging and Model Saving
        log_data = {
            "epoch": epoch + 1,
            "train_sr_loss": epoch_train_loss_sr,
            "train_seg_loss": epoch_train_loss_seg,
            "val_sr_loss": epoch_val_loss_sr,
            "val_seg_loss": epoch_val_loss_seg,
            "val_precision": precision_average,
            "val_recall": recall_average,
            "val_psnr": epoch_psnr,
            "val_ssim": epoch_ssim,
        }

        val_precisions.append(precision_average)
        val_recalls.append(recall_average)

        # Log per-class precision and recall
        for i, (p, r) in enumerate(zip(precision_per_class, recall_per_class)):
            # if i != class_idx_unidentifiable:
                log_data[f"precision_{classes[i]}"] = p
                log_data[f"recall_{classes[i]}"] = r

        _wandb_log({"Validation Metrics": log_data}, enabled=True)

        # Save Best Model
        if epoch_val_loss_seg < best_val_loss_seg:
            best_train_loss_seg, best_val_loss_seg = epoch_train_loss_seg, epoch_val_loss_seg
            best_train_loss_sr, best_val_loss_sr = epoch_train_loss_sr, epoch_val_loss_sr
            best_val_precision, best_val_recall = precision_average, recall_average
            best_psnr, best_ssim = epoch_psnr, epoch_ssim
            best_wts = copy.deepcopy(model.state_dict())

            torch.save(best_wts, save_path)  # GPU-compatible
            torch.save({k: v.to('cpu') for k, v in best_wts.items()}, save_path.replace(".pth", "_cpu.pth"))  # CPU-compatible

            print(f"Best model saved with validation loss: {best_val_loss_seg:.4f}")
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        # Early Stopping
        if early_stop and early_stop_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

        torch.cuda.empty_cache()
        gc.collect()

    torch.cuda.empty_cache()  # Free memory after validation
    if best_wts:
        model.load_state_dict(best_wts)
    else:
        model = None

    print(f"Best model -> Train SR Loss: {best_train_loss_sr:.4f}, Train Seg Loss: {best_train_loss_seg:.4f}\n"
            f"Val SR Loss: {best_val_loss_sr}, Val Seg Loss: {best_val_loss_seg}\n"
            f"Val Precision: {best_val_precision:.4f}, Val Recall: {best_val_recall:.4f}\n"
            f"Val PSNR: {best_psnr:.4f}, Val SSIM: {best_ssim:.4f}")

    plot_metrics(train_losses_seg, val_losses_seg, val_precisions, val_recalls, image_dir=image_dir)

    return model

def train_scnet(model, train_loader, val_loader, optimizer,
          scheduler, criterion, classes, device,
          num_epochs=100, save_path=f'best_model.pth', l1_lambda=0.0001,
          early_stop=True, patience=20, image_dir=None):
    best_val_loss, best_train_loss, best_model_val_precision, best_model_val_recall = float('inf'), float('inf'), float('inf'), float('inf')
    early_stop_counter = 0
    best_model_wts = None

    # class_idx_unidentifiable = classes.index('unidentifiable')

    _wandb_watch(model, enabled=True)

    train_losses, val_losses, val_precisions, val_recalls = [], [], [], []

    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        print(f"\nEpoch: {epoch+1}")
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)

        for inputs, masks, lr_images, fraction_alphas, *_ in progress_bar:
            # torch.cuda.empty_cache()
            inputs, masks, lr_images, fraction_alphas = inputs.to(device), masks.to(device), lr_images.to(device), fraction_alphas.to(device)
            # print(f"inputs, masks: {inputs.shape}, {masks.shape}")
            optimizer.zero_grad()

            outputs = model(lr_images, fraction_alphas)
            loss = criterion(outputs, masks)
            if l1_lambda and l1_lambda > 0:
                l1_reg = sum(param.abs().sum() for param in model.parameters())  # Faster than torch.norm(param, p=1)
                loss = loss + l1_lambda * l1_reg
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=running_loss / (progress_bar.n + 1))  # Show avg loss

        del inputs, masks

        epoch_train_loss = running_loss / len(train_loader)
        scheduler.step(epoch_train_loss)
        train_losses.append(epoch_train_loss)

        torch.cuda.empty_cache()

        # Validation
        model.eval()
        val_running_loss = 0.0
        all_preds, all_labels = [], []  # Store as lists for memory efficiency

        precision_per_class = torch.zeros(len(classes), device=device)
        recall_per_class = torch.zeros(len(classes), device=device)

        progress_bar = tqdm(val_loader, desc="Validation", leave=True)

        with torch.no_grad():
            for inputs, masks, lr_images, fraction_alphas, *_ in progress_bar:
                inputs, masks, lr_images, fraction_alphas = inputs.to(device), masks.to(device), lr_images.to(device), fraction_alphas.to(device)
                outputs = model(lr_images, fraction_alphas)
                loss = criterion(outputs, masks)
                val_running_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)

                # valid_mask = preds != class_idx_unidentifiable  # Mask to ignore "unidentifiable" class
                # preds, masks = preds[valid_mask], masks[valid_mask]

                all_preds.append(preds.flatten().cpu())
                all_labels.append(masks.flatten().cpu())

                progress_bar.set_postfix(loss=val_running_loss / (progress_bar.n + 1))  # Show avg loss

        epoch_val_loss = val_running_loss / len(val_loader)
        val_losses.append(epoch_val_loss)
        scheduler.step(epoch_val_loss)

        # Convert to tensors at once to save memory
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        precision, recall, precision_per_class_np, recall_per_class_np, _ = compute_precision_recall(
            all_labels.numpy(), all_preds.numpy(), len(classes)
        )
        precision_per_class = torch.tensor(precision_per_class_np, device=device)
        recall_per_class = torch.tensor(recall_per_class_np, device=device)

        val_precisions.append(precision)
        val_recalls.append(recall)

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Precision: {precision:.4f}, Val Recall: {recall:.4f}')

        for i, class_name in enumerate(classes):
            # if i != class_idx_unidentifiable:
                print(f'Class: {class_name} - Precision: {precision_per_class[i]:.4f}, Recall: {recall_per_class[i]:.4f}')

        log_data = {
            "epoch": epoch + 1,
            "train_loss": epoch_train_loss,
            "val_loss": epoch_val_loss,
            "val_precision": precision,
            "val_recall": recall,
        }

        for i, class_name in enumerate(classes):
            # if i != class_idx_unidentifiable:
                log_data[f"{class_name}_precision"] = precision_per_class[i].item()
                log_data[f"{class_name}_recall"] = recall_per_class[i].item()

        _wandb_log({"Train log": log_data}, enabled=True)

        # Save the best model
        if epoch_val_loss < best_val_loss:
            best_val_loss, best_train_loss = epoch_val_loss, epoch_train_loss
            best_model_val_precision, best_model_val_recall = precision, recall
            best_model_wts = copy.deepcopy(model.state_dict())

            torch.save(best_model_wts, save_path)  # GPU-compatible
            cpu_save_path = save_path.replace(".pth", "_cpu.pth")
            torch.save({k: v.cpu() for k, v in best_model_wts.items()}, cpu_save_path)  # CPU-compatible

            _save_full_model_snapshots(model, save_path)

            print(f"Best model saved with validation loss: {best_val_loss:.4f}")
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop and early_stop_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

        torch.cuda.empty_cache()  # Only call once per epoch
    if best_model_wts:
        model.load_state_dict(best_model_wts)
    else:
        model = None

    print(f"Best model -> Train Loss: {best_train_loss:.4f}, Val Loss: {best_val_loss:.4f}, Precision: {best_model_val_precision:.4f}, Recall: {best_model_val_recall:.4f}")
    plot_metrics(train_losses, val_losses, val_precisions, val_recalls, image_dir=image_dir)

    return model
