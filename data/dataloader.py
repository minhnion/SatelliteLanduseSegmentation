import numpy as np
from time import time
from torch.utils.data import DataLoader
from data.data import LandCoverDataset
from utils.data_utils import ResizeAndToClassTransform

def load_dataloader(batch_size, root_dir, size=(256,256), rgb_only=False, mask_scale=None, scale_factor=None):
    # print(size)
    train_transform = ResizeAndToClassTransform(size=size, augment=True)
    val_test_transform = ResizeAndToClassTransform(size=size, augment=False)

    train_dataset = LandCoverDataset(root_dir=root_dir + '/train', transforms=train_transform, rgb_only=rgb_only, mask_scale=mask_scale, scale_factor=scale_factor)
    val_dataset = LandCoverDataset(root_dir=root_dir + '/val', transforms=val_test_transform, rgb_only=rgb_only, mask_scale=mask_scale, scale_factor=scale_factor)
    test_dataset = LandCoverDataset(root_dir=root_dir + '/test', transforms=val_test_transform, rgb_only=rgb_only, mask_scale=mask_scale, scale_factor=scale_factor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)

    return train_loader, val_loader, test_loader

def load_one_dataloader(batch_size, root_dir, size=(256,256)):
    val_test_transform = ResizeAndToClassTransform(size=size, augment=False)

    dataset = LandCoverDataset(root_dir=root_dir, transforms=val_test_transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    return loader
