import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import logging
import os
from datetime import datetime
from model.FoundationKDNet.kd_model import KnowledgeDistillationModel
from model.FoundationKDNet.student_encoder import StudentEncoder
from model.FoundationKDNet.teacher_encoder import TeacherEncoder  # You'll need to implement this
from data.dataloader import load_dataloader
from dataset_config.load_config import load_config

def setup_logging(save_dir):
    """Setup logging configuration"""
    log_dir = os.path.join(save_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def train_kd(teacher_model, student_model, train_loader, val_loader,
             num_epochs, save_dir, device='cuda', alpha=0.5):
    """
    Train the knowledge distillation model

    Args:
        teacher_model: Pre-trained teacher model
        student_model: Student model to be trained
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        save_dir: Directory to save checkpoints
        device: Device to train on
        alpha: Weight for knowledge distillation loss
    """
    # Setup logging
    setup_logging(save_dir)

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Initialize knowledge distillation model
    kd_model = KnowledgeDistillationModel(device=device)

    # Training loop
    best_accuracy = 0.0
    for epoch in range(num_epochs):
        # Training phase
        kd_model.student.train()
        train_metrics = {
                'total_loss': 0,
                'sr_loss_ts': 0,
                'seg_loss_ts': 0,
                'sr_loss_gt': 0,
                'seg_loss_gt': 0,
                'feature_loss': 0,
                'embedding_loss': 0
            }
        num_batches = 0

        for batch_idx, data in enumerate(train_loader):
            metrics = kd_model.train_step(data, alpha)
            print(metrics)

            # Update metrics
            for k, v in metrics.items():
                train_metrics[k] += v
            num_batches += 1

            # Log progress
            if (batch_idx + 1) % 10 == 0:
                logging.info(f'Epoch [{epoch+1}/{num_epochs}] '
                           f'Batch [{batch_idx+1}/{len(train_loader)}] '
                           f'Loss: {metrics["total_loss"]:.4f}')

        # Average training metrics
        for k in train_metrics:
            train_metrics[k] /= num_batches

        # Validation phase
        val_metrics = kd_model.evaluate(val_loader)

        # Log epoch metrics
        logging.info(f'Epoch [{epoch+1}/{num_epochs}] '
                    f'Train Loss: {train_metrics["total_loss"]:.4f} '
                    f'Val Loss: {val_metrics["loss"]:.4f} '
                    f'Val Accuracy: {val_metrics["accuracy"]:.2f}%')

        # Save best model
        if val_metrics['accuracy'] > best_accuracy:
            best_accuracy = val_metrics['accuracy']
            checkpoint_path = os.path.join(save_dir, 'best_model.pth')
            kd_model.save_checkpoint(checkpoint_path, epoch, val_metrics)
            logging.info(f'Saved best model with accuracy: {best_accuracy:.2f}%')

        # Save periodic checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            kd_model.save_checkpoint(checkpoint_path, epoch, val_metrics)
            logging.info(f'Saved checkpoint at epoch {epoch+1}')
