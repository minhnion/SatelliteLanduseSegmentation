import numpy as np
from time import time
from torch.utils.data import DataLoader
from data.data import LandCoverDataset
from utils.data_utils import ResizeAndToClassTransform

def load_dataloader(batch_size):
    root_dir = "dataset/thanhhoa"
    size = (256, 256)
    transform = ResizeAndToClassTransform(size=size, augment=True)

    train_dataset = LandCoverDataset(root_dir=root_dir + '/train', transforms=transform)
    val_dataset = LandCoverDataset(root_dir=root_dir + '/val', transforms=transform)
    test_dataset = LandCoverDataset(root_dir=root_dir + '/test', transforms=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, val_loader, test_loader
