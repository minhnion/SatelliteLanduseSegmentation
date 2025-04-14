import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import logging
import os
from datetime import datetime
from model.FoundationKDNet.kd_model import KnowledgeDistillationModel
from model.FoundationKDNet.student_encoder import StudentEncoder
from model.FoundationKDNet.teacher_encoder import TeacherEncoder  # You'll need to implement this
from model.FoundationKDNet.train_kd import train_kd
from data.dataloader import load_dataloader
from dataset_config.load_config import load_config
import argparse
from tqdm import tqdm

# Hàm huấn luyện với tqdm
def train_kd(model, train_loader, val_loader, num_epochs=10):
    # Thanh tiến trình cho epoch
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        model.student.train()
        # Thanh tiến trình cho step trong train_loader
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for step, data in enumerate(train_bar):
            model.set_input(data)
            metrics = model.optimize_parameters()
            # Cập nhật thông tin trên thanh tiến trình
            if step % 10 == 0:
                model.print_info(epoch, step)
                train_bar.set_postfix({
                    'G_loss': f'{metrics["total_loss"]:.5f}',
                    'SR Loss': f'{metrics["sr_loss_ts"] + metrics["sr_loss_gt"]:.5f}',
                    'Seg Loss': f'{metrics["seg_loss_ts"] + metrics["seg_loss_gt"]:.5f}'
                })
        # Evaluate and save
        metrics = model.evaluate(val_loader)
        model.save_ckpt(epoch, step, metrics)
        logging.info(f'Epoch {epoch} Validation Metrics: {metrics}')

# Cấu hình tham số
if __name__ == "__main__":
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    parser = argparse.ArgumentParser(description="Knowledge Distillation for Segmentation and SR")
    parser.add_argument('--n_channels', type=int, default=13, help='Number of input channels')
    parser.add_argument('--n_classes', type=int, default=5, help='Number of classes')
    parser.add_argument('--embed_dim', type=int, default=1024, help='Embedding dimension')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--lr_g', type=float, default=1e-6, help='Learning rate for generator')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--snapshot_dir', type=str, default='./snapshots', help='Directory to save checkpoints')
    parser.add_argument('--T_ckpt_path', type=str, default=None, help='Teacher checkpoint path')
    parser.add_argument('--alpha', type=float, default=0.5, help='Weight for embedding loss')
    args = parser.parse_args()

    # Khởi tạo mô hình và huấn luyện
    model = KnowledgeDistillationModel(args)

    dataset_module = 'dataset_config.north_vn'
    dataset_config = load_config(dataset_module)
    RGB_TO_CLASSES = dataset_config['classes']
    CLASSES_TO_RGB = {}

    for k, v in RGB_TO_CLASSES.items():
        if v not in CLASSES_TO_RGB:
            CLASSES_TO_RGB[v] = k

    classes = []
    for cls in RGB_TO_CLASSES.values():
        if cls not in classes:
            classes.append(cls)
    n_classes = dataset_config['n_classes']
    n_channels = dataset_config['n_channels']
    base_path = dataset_config['base_path']
    num_tiles = dataset_config['num_tiles']
    weights = dataset_config['weights']

    train_loader, val_loader, test_loader = load_dataloader(batch_size=16, classes=classes, RGB_TO_CLASSES=RGB_TO_CLASSES, root_dir='/mnt/hungvv/minh/dataset/new_13bands_dataset_splitted',
                                                            size=(128, 128), mask_scale=2, num_tiles=4, scale_factor=2)

    # Training parameters
    num_epochs = 1
    save_dir = 'checkpoints/kd_training'
    alpha = 0.5  # Weight for knowledge distillation loss

    # Start training
    train_kd(model, train_loader, val_loader, num_epochs)
