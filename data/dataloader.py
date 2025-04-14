import numpy as np
from time import time
from torch.utils.data import DataLoader
from data.data import LandCoverDataset, LandCoverHRDataset
from utils.data_utils import ResizeAndToClassTransform, ResizeAndToClassHRTransform
import os
from glob import glob
from sklearn.model_selection import train_test_split

def load_dataloader(batch_size, root_dir, RGB_TO_CLASSES, classes, size=(256,256), mask_scale=None, scale_factor=None, num_tiles=None, random_state=42):
    # print(size)
    train_transforms = ResizeAndToClassTransform(size, RGB_TO_CLASSES, classes, augment=True)
    val_test_transforms = ResizeAndToClassTransform(size, RGB_TO_CLASSES, classes, augment=False)

    # Check if train/val/test folders exist
    train_path = os.path.join(root_dir, 'train')
    val_path = os.path.join(root_dir, 'val')
    test_path = os.path.join(root_dir, 'test')

    if os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path):
        print("Train/Val/Test folders found, using existing splits.")

        train_dataset = LandCoverDataset(root_dir=train_path, transforms=train_transforms,
                                        scale_factor=scale_factor, num_tiles=num_tiles, mask_scale=mask_scale)
        val_dataset = LandCoverDataset(root_dir=val_path, transforms=val_test_transforms,
                                    scale_factor=scale_factor, num_tiles=num_tiles, mask_scale=mask_scale)
        test_dataset = LandCoverDataset(root_dir=test_path, transforms=val_test_transforms,
                                        scale_factor=scale_factor, num_tiles=num_tiles, mask_scale=mask_scale)

    else:
        print("No predefined Train/Val/Test splits found. Splitting dataset...")

        # Find all satellite images (_sat.tif or _sat.jpg) and corresponding masks (_mask.png)
        all_images = sorted(glob(os.path.join(root_dir, '*_sat.tif'))) + \
                    sorted(glob(os.path.join(root_dir, '*_sat.jpg')))
        all_masks = sorted(glob(os.path.join(root_dir, '*_mask.png')))

        # Ensure each image has a corresponding mask
        image_names = {os.path.basename(img).replace('_sat.tif', '').replace('_sat.jpg', '') for img in all_images}
        mask_names = {os.path.basename(mask).replace('_mask.png', '') for mask in all_masks}

        valid_names = image_names & mask_names  # Find names that exist in both sets

        # Filter images and masks based on valid names
        all_images = [img for img in all_images if os.path.basename(img).replace('_sat.tif', '').replace('_sat.jpg', '') in valid_names]
        all_masks = [mask for mask in all_masks if os.path.basename(mask).replace('_mask.png', '') in valid_names]

        assert len(all_images) == len(all_masks), "Mismatch between image and mask counts"

        # Split into Train (70%) and Temp (30% for Val + Test)
        train_images, temp_images, train_masks, temp_masks = train_test_split(
            all_images, all_masks, test_size=0.3, random_state=random_state
        )

        # Split Temp into Val (20%) and Test (10%)
        val_images, test_images, val_masks, test_masks = train_test_split(
            temp_images, temp_masks, test_size=1/3, random_state=random_state
        )

        train_file_paths = train_images + train_masks
        val_file_paths = val_images + val_masks
        test_file_paths = test_images + test_masks

        # Create datasets from split data
        train_dataset = LandCoverDataset(root_dir='',
                                        file_paths=train_file_paths,
                                        transforms=train_transforms,
                                        scale_factor=scale_factor,
                                        num_tiles=num_tiles, mask_scale=mask_scale)
        val_dataset = LandCoverDataset(root_dir='',
                                    file_paths=val_file_paths,
                                    transforms=val_test_transforms,
                                    scale_factor=scale_factor,
                                    num_tiles=num_tiles, mask_scale=mask_scale)
        test_dataset = LandCoverDataset(root_dir='',
                                        file_paths=test_file_paths,
                                        transforms=val_test_transforms,
                                        scale_factor=scale_factor,
                                        num_tiles=num_tiles, mask_scale=mask_scale)

    # Create DataLoaders
    num_workers = 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    return train_loader, val_loader, test_loader

def load_hr_dataloader(batch_size, root_dir, RGB_TO_CLASSES, classes, size=(256,256), scale_factor=None, num_tiles=None, random_state=42):
    # print(size)
    train_transforms = ResizeAndToClassHRTransform(size, RGB_TO_CLASSES, classes, augment=True)
    val_test_transforms = ResizeAndToClassHRTransform(size, RGB_TO_CLASSES, classes, augment=False)

    # Check if train/val/test folders exist
    train_path = os.path.join(root_dir, 'train')
    val_path = os.path.join(root_dir, 'val')
    test_path = os.path.join(root_dir, 'test')

    if os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path):
        print("Train/Val/Test folders found, using existing splits.")

        train_dataset = LandCoverHRDataset(root_dir=train_path, transforms=train_transforms,
                                        scale_factor=scale_factor, num_tiles=1)
        val_dataset = LandCoverHRDataset(root_dir=val_path, transforms=val_test_transforms,
                                    scale_factor=scale_factor, num_tiles=1)
        test_dataset = LandCoverHRDataset(root_dir=test_path, transforms=val_test_transforms,
                                        scale_factor=scale_factor, num_tiles=1)

    else:
        print("No predefined Train/Val/Test splits found. Splitting dataset...")

        # Find all satellite images (_sat.tif or _sat.jpg) and corresponding masks (_mask.png)
        all_images = sorted(glob(os.path.join(root_dir, '*_sat.tif'))) + \
                    sorted(glob(os.path.join(root_dir, '*_sat.jpg')))
        all_masks = sorted(glob(os.path.join(root_dir, '*_mask.png')))

        # Ensure each image has a corresponding mask
        image_names = {os.path.basename(img).replace('_sat.tif', '').replace('_sat.jpg', '') for img in all_images if not img.endswith('lr_sat.tif')}
        mask_names = {os.path.basename(mask).replace('_mask.png', '') for mask in all_masks}

        valid_names = image_names & mask_names  # Find names that exist in both sets

        # Filter images and masks based on valid names
        all_images = [img for img in all_images if os.path.basename(img).replace('_sat.tif', '').replace('_sat.jpg', '') in valid_names]
        all_masks = [mask for mask in all_masks if os.path.basename(mask).replace('_mask.png', '') in valid_names]
        # Find corresponding low-resolution images by replacing 'sat' with 'lr_sat'
        all_lr_images = [img.replace('_sat.', '_lr_sat.') for img in all_images]

        # Verify all low-resolution images exist
        valid_lr_images = [lr_img for lr_img in all_lr_images if os.path.exists(lr_img)]

        assert len(valid_lr_images) == len(all_images), "Mismatch between low-res images and high-res images"
        assert len(all_images) == len(all_masks), "Mismatch between image and mask counts"

        # Split into Train (70%) and Temp (30% for Val + Test)
        # train_images, temp_images, train_masks, temp_masks, train_lr_o = train_test_split(
        #     all_images, all_masks, all_lr_images, test_size=0.3, random_state=random_state
        # )

        # # Split Temp into Val (20%) and Test (10%)
        # val_images, test_images, val_masks, test_masks = train_test_split(
        #     temp_images, temp_masks, all_lr_images, test_size=1/3, random_state=random_state
        # )

        train_images, temp_images, train_masks, temp_masks, train_lr_images, temp_lr_images = train_test_split(
            all_images, all_masks, valid_lr_images, test_size=0.3, random_state=random_state
        )
        # Split Temp into Val (20%) and Test (10%)
        val_images, test_images, val_masks, test_masks, val_lr_images, test_lr_images = train_test_split(
            temp_images, temp_masks, temp_lr_images, test_size=1/3, random_state=random_state
        )

        train_file_paths = train_images + train_masks + train_lr_images
        val_file_paths = val_images + val_masks + val_lr_images
        test_file_paths = test_images + test_masks + test_lr_images

        # Create datasets from split data
        train_dataset = LandCoverHRDataset(root_dir='',
                                        file_paths=train_file_paths,
                                        transforms=train_transforms,
                                        scale_factor=scale_factor,
                                        num_tiles=num_tiles)
        val_dataset = LandCoverHRDataset(root_dir='',
                                    file_paths=val_file_paths,
                                    transforms=val_test_transforms,
                                    scale_factor=scale_factor,
                                    num_tiles=num_tiles)
        test_dataset = LandCoverHRDataset(root_dir='',
                                        file_paths=test_file_paths,
                                        transforms=val_test_transforms,
                                        scale_factor=scale_factor,
                                        num_tiles=num_tiles)

    # Create DataLoaders
    num_workers = 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    return train_loader, val_loader, test_loader

def load_one_dataloader(batch_size, root_dir, size=(256,256)):
    val_test_transform = ResizeAndToClassTransform(size=size, augment=False)

    dataset = LandCoverDataset(root_dir=root_dir, transforms=val_test_transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    return loader
