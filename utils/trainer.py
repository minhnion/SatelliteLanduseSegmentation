import copy
import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import precision_score, recall_score
import wandb
import time
import gc
from utils.logging_utils import plot_metrics
from utils.metrics import calculate_iou, compute_psnr, compute_ssim
from tqdm import tqdm
import numpy as np

def train(model, train_loader, val_loader, optimizer,
          scheduler, criterion, classes, device,
          num_epochs=100, save_path=f'best_model.pth', l1_lambda=0.0001,
          early_stop=True, patience=20, image_dir=None):
    best_val_loss, best_train_loss, best_model_val_precision, best_model_val_recall = float('inf'), float('inf'), float('inf'), float('inf')
    early_stop_counter = 0
    best_model_wts = None

    # class_idx_unidentifiable = classes.index('unidentifiable')

    wandb.watch(model, log='all', log_freq=100)

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

        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)

        precision_per_class = torch.tensor(precision_score(
            all_labels, all_preds, average=None, zero_division=0, labels=list(range(len(classes)))
        ), device=device)

        recall_per_class = torch.tensor(recall_score(
            all_labels, all_preds, zero_division=0, average=None, labels=list(range(len(classes)))
        ), device=device)

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

        wandb.log({"Train log": log_data})

        # Save the best model
        if epoch_val_loss < best_val_loss:
            best_val_loss, best_train_loss = epoch_val_loss, epoch_train_loss
            best_model_val_precision, best_model_val_recall = precision, recall
            best_model_wts = copy.deepcopy(model.state_dict())

            torch.save(best_model_wts, save_path)  # GPU-compatible
            cpu_save_path = save_path.replace(".pth", "_cpu.pth")
            torch.save({k: v.cpu() for k, v in best_model_wts.items()}, cpu_save_path)  # CPU-compatible

            try:
                torch.save(model, save_path.replace(".pth", "_full.pth"))  # Save full model
                torch.save(model.to('cpu'), cpu_save_path.replace(".pth", "_full.pth"))  # Save CPU version
            except Exception as e:
                print(f"Error: {e}")

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

    wandb.watch(model, log='all', log_freq=100)

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

        wandb.log({"Validation Metrics": log_data})

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

    wandb.watch(model, log='all', log_freq=100)

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
        precision_average = precision_score(all_labels.numpy(), all_preds.numpy(), average='weighted', zero_division=0)
        recall_average = recall_score(all_labels.numpy(), all_preds.numpy(), average='weighted', zero_division=0)

        # Compute per-class precision and recall
        precision_per_class = precision_score(all_labels.numpy(), all_preds.numpy(), average=None, zero_division=0)
        recall_per_class = recall_score(all_labels.numpy(), all_preds.numpy(), average=None, zero_division=0)

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

        wandb.log({"Validation Metrics": log_data})

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

    wandb.watch(model, log='all', log_freq=100)

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

        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)

        precision_per_class = torch.tensor(precision_score(
            all_labels, all_preds, average=None, zero_division=0, labels=list(range(len(classes)))
        ), device=device)

        recall_per_class = torch.tensor(recall_score(
            all_labels, all_preds, zero_division=0, average=None, labels=list(range(len(classes)))
        ), device=device)

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

        wandb.log({"Train log": log_data})

        # Save the best model
        if epoch_val_loss < best_val_loss:
            best_val_loss, best_train_loss = epoch_val_loss, epoch_train_loss
            best_model_val_precision, best_model_val_recall = precision, recall
            best_model_wts = copy.deepcopy(model.state_dict())

            torch.save(best_model_wts, save_path)  # GPU-compatible
            cpu_save_path = save_path.replace(".pth", "_cpu.pth")
            torch.save({k: v.cpu() for k, v in best_model_wts.items()}, cpu_save_path)  # CPU-compatible

            try:
                torch.save(model, save_path.replace(".pth", "_full.pth"))  # Save full model
                torch.save(model.to('cpu'), cpu_save_path.replace(".pth", "_full.pth"))  # Save CPU version
            except Exception as e:
                print(f"Error: {e}")

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
