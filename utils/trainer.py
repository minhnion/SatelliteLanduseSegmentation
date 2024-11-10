import copy
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score
import wandb 

from utils.logging_utils import plot_metrics, plot_predictions

def train(model, train_loader, val_loader, optimizer, scheduler, criterion, classes, device, num_epochs=100, save_path='best_model.pth', image_dir = None, early_stop=True, patience=20):

    best_val_loss, best_train_loss, best_model_val_precision, best_model_val_recall = float('inf'), float('inf'), float('inf'), float('inf')
    early_stop_counter = 0
    best_model_wts = None

    wandb.watch(model, log='all', log_freq=100)

    train_losses, val_losses, val_precisions, val_recalls = [], [], [] , []

    for epoch in range(num_epochs):
        print(f"Epoch: {epoch+1}")
        running_loss = 0.0
        model.train()

        class_idx_unidentifiable = classes.index('unidentifiable')
        print('Unidentifiable class id:', class_idx_unidentifiable)

        for i, (inputs, masks, image_paths, mask_paths) in enumerate(train_loader):
            inputs, masks = inputs.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() 
            
        epoch_train_loss = running_loss / len(train_loader)
        scheduler.step(epoch_train_loss)
        train_losses.append(epoch_train_loss)

        # Validate the model
        model.eval()
        val_running_loss = 0.0

        all_preds, all_labels, val_inputs, val_masks, val_outputs, all_preds_each_cls, all_labels_each_cls = [], [], [], [], [], [], []
        forest_precision, forest_recall = 0, 0
        rice_field_precision, rice_field_recall = 0, 0
        water_precision, water_recall = 0, 0
        residential_precision, residential_recall = 0, 0

        class_idx_to_pc = {
            "forest": [forest_precision, forest_recall],
            "rice_field": [rice_field_precision, rice_field_recall],
            "water": [water_precision, water_recall],
            "residential": [residential_precision, residential_recall]
        }

        with torch.no_grad():
            for inputs, masks, image_paths, mask_paths in val_loader:
                inputs, masks = inputs.to(device), masks.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, masks)
                val_running_loss += loss.item() 
                
                val_inputs.append(inputs)
                val_outputs.append(outputs)
                val_masks.append(masks)

                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                labels = masks.cpu().numpy()
                valid_mask = labels != class_idx_unidentifiable  # Mask to ignore "unidentifiable" class
                preds = preds[valid_mask]
                labels = labels[valid_mask]

                all_preds.extend(preds.flatten())
                all_labels.extend(labels.flatten())
                all_preds_each_cls.extend(preds)
                all_labels_each_cls.extend(labels)

        epoch_val_loss = val_running_loss / len(val_loader)
        val_losses.append(epoch_val_loss)

        # Compute precision, recall, and IoU for each class
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)

        precision_per_class = precision_score(all_labels_each_cls, all_preds_each_cls, average=None, zero_division=0, labels=list(range(len(classes))))
        recall_per_class = recall_score(all_labels_each_cls, all_preds_each_cls, zero_division=0, average=None, labels=list(range(len(classes))))

        val_precisions.append(precision)
        val_recalls.append(recall)

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Precision: {precision:.4f}, Val Recall: {recall:.4f}')
        
        for i, class_name in enumerate(classes):
            if (i==0):
                continue
            print(f'Class: {class_name} - Precision: {precision_per_class[i]:.4f}, Recall: {recall_per_class[i]:.4f}')
            class_idx_to_pc[class_name] = [precision_per_class[i], recall_per_class[i]]

        # Log metrics to wandb
        log_data = {
            "epoch": epoch + 1,
            "train_loss": epoch_train_loss,
            "val_loss": epoch_val_loss,
            "val_precision": precision,
            "val_recall": recall,
        }

        for class_name, metrics in class_idx_to_pc.items():
            log_data[f"{class_name}_precision"] = metrics[0]
            log_data[f"{class_name}_recall"] = metrics[1]

        # Log to wandb 
        wandb.log( {"Train log": log_data})

        # Save the model if the validation loss is the best we've seen so far
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_train_loss = epoch_train_loss
            best_model_val_precision = precision
            best_model_val_recall = recall
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved with validation loss: {best_val_loss:.4f}")
            early_stop_counter = 0  # Reset early stopping counter
        else:
            early_stop_counter += 1

        if early_stop and early_stop_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # Load the best model weights
    if best_model_wts:
        model.load_state_dict(best_model_wts)
    else:
        model = None
    print(f"Best model has train loss: {best_train_loss}, val loss: {best_val_loss} \nprecision: {best_model_val_precision}, recall: {best_model_val_recall}")

    # Plot the metrics
    plot_metrics(train_losses, val_losses, val_precisions, val_recalls, image_dir=image_dir)
    return model