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

if __name__ == '__main__':
    # Example usage
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize models
    teacher_model = TeacherEncoder('ResNet50',
                                    num_prompts=10, num_heads=4, embed_dim=32,
                                    H=64, W=64, num_queries_seg=10, num_queries_sr=10)  # You need to implement this
    student_model = StudentEncoder('ResNet18', embed_dim=32)

    # Define data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Create data loaders (you need to implement your dataset class)

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
    train_kd(teacher_model, student_model, train_loader, val_loader,
             num_epochs, save_dir, device, alpha)
