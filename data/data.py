import os
from PIL import Image
from torch.utils.data import Dataset
from utils.image_utils import open_tif_image, split_image, downscale_image, compute_fraction_image
import numpy as np
import torch
import torch.nn as nn

class LandCoverDataset(Dataset):
    def __init__(self, root_dir, transforms=None, patch_size=16, scale_factor=2, num_tiles=4, file_paths=None, mask_scale=None):
        self.root_dir = root_dir
        self.transform = transforms
        self.image_paths = []
        self.mask_paths = []
        self.patch_size = patch_size  # Patch size for fractional mask
        self.scale_factor = scale_factor  # Downscaling factor
        self.num_tiles = num_tiles  # Number of tiles per row/column
        self.mask_scale = mask_scale  # Scale factor for masks

        if not file_paths:
            file_paths = os.listdir(root_dir)

        for file_name in file_paths:
            if file_name.endswith(('.jpg', '.tif')):  # Support for .jpg, .tif, .tiff
                image_path = os.path.join(root_dir, file_name)
                self.image_paths.append(image_path)

                if file_name.endswith('_sat.jpg'):
                    mask_name = file_name.replace('_sat.jpg', '_mask.png')
                elif file_name.endswith('_sat.tif'):
                    mask_name = file_name.replace('_sat.tif', '_mask.png')

                self.mask_paths.append(os.path.join(root_dir, mask_name))

        assert len(self.image_paths) == len(self.mask_paths)

    def __len__(self):
        if self.num_tiles:
            return len(self.image_paths) * (self.num_tiles ** 2)
        return len(self.image_paths)

    def __getitem__(self, idx):
        if self.num_tiles:
            image_idx = idx // (self.num_tiles ** 2)
            tile_idx = idx % (self.num_tiles ** 2)
        else:
            image_idx = idx
            tile_idx = None
        image_path = self.image_paths[image_idx]
        mask_path = self.mask_paths[image_idx]

        # Load image based on file format
        if image_path.endswith((".tif")):
            image = open_tif_image(image_path)
        else:
            image = np.array(Image.open(image_path).convert('RGB'))  # Convert to RGB for other formats
            image = image / 255.0  # Normalize the image to [0, 1] range

        # Load mask
        mask = np.array(Image.open(mask_path).convert('RGB'))  # Open mask as RGB
        assert image.shape[:2] == mask.shape[:2]

        if tile_idx is not None:
            image = split_image(image, self.num_tiles, tile_idx)
            mask = split_image(mask, self.num_tiles, tile_idx)
            assert image.shape[:2] == mask.shape[:2]

        # Apply transformations if any
        if self.transform:
            image, mask, n_classes = self.transform(image, mask, self.mask_scale)

        h_image, w_image = image.shape[1:]
        h_mask, w_mask = mask.shape
        scale = 1 if not self.mask_scale else self.mask_scale
        # print(h_image, w_image, h_mask, w_mask, scale)

        # print(f"Image shape: {image.shape}, Mask shape: {mask.shape}")
        assert h_mask == h_image * scale and w_mask == w_image * scale

        if self.scale_factor:
            lr_image = downscale_image(image.numpy(), self.scale_factor)
            fractional_alpha = compute_fraction_image(lr_image)
            return image, mask, lr_image, fractional_alpha, image_path, mask_path

        # Return image, mask, and their corresponding file paths
        return image, mask, None, None, image_path, mask_path

class LandCoverHRDataset(Dataset):
    def __init__(self, root_dir, transforms=None, scale_factor=16, num_tiles=4, file_paths=None):
        self.root_dir = root_dir
        self.transform = transforms
        self.image_paths = []
        self.lr_image_paths = []
        self.mask_paths = []
        self.scale_factor = scale_factor  # Downscaling factor
        self.num_tiles = num_tiles  # Number of tiles per row/column

        if not file_paths:
            file_paths = os.listdir(root_dir)

        for file_name in file_paths:
            if file_name.endswith(('lr_sat.jpg', 'lr_sat.tif')):  # Support for .jpg, .tif, .tiff
                lr_image_path = os.path.join(root_dir, file_name)
                self.lr_image_paths.append(lr_image_path)

                image_path = os.path.join(root_dir, file_name.replace('lr_sat', 'sat'))
                self.image_paths.append(image_path)

                if image_path.endswith('_sat.jpg'):
                    mask_path = image_path.replace('_sat.jpg', '_mask.png')
                elif file_name.endswith('_sat.tif'):
                    mask_path = image_path.replace('_sat.tif', '_mask.png')

                self.mask_paths.append(mask_path)

        assert len(self.image_paths) == len(self.mask_paths)
        assert len(self.lr_image_paths) == len(self.image_paths)

    def __len__(self):
        if self.num_tiles:
            return len(self.image_paths) * (self.num_tiles ** 2)
        return len(self.image_paths)

    def __getitem__(self, idx):
        if self.num_tiles:
            image_idx = idx // (self.num_tiles ** 2)
            tile_idx = idx % (self.num_tiles ** 2)
        else:
            image_idx = idx
            tile_idx = None
        image_path = self.image_paths[image_idx]
        mask_path = self.mask_paths[image_idx]
        lr_image_path = self.lr_image_paths[image_idx]

        # Load image based on file format
        if image_path.endswith((".tif")):
            image = open_tif_image(image_path)
            image = image[:, :, [2, 1, 0]]  # BGR → RGB conversion
        else:
            image = np.array(Image.open(image_path).convert('RGB'))  # Convert to RGB for other formats
            image = image / 255.0  # Normalize the image to [0, 1] range

        # Load low-resolution image
        if lr_image_path.endswith((".tif")):
            lr_image = open_tif_image(lr_image_path)
            # lr_image = lr_image[:, :, [2, 1, 0]]  # BGR → RGB conversion
        else:
            lr_image = np.array(Image.open(lr_image_path).convert('RGB'))
            lr_image = lr_image / 255.0

        # Load mask
        mask = np.array(Image.open(mask_path).convert('RGB'))  # Open mask as RGB
        h_lr_image, w_lr_image = lr_image.shape[:2]
        h_image, w_image = image.shape[:2]
        # print(f"Image shape: {image.shape}\nMask shape: {mask.shape}\nLR Image shape: {lr_image.shape}")
        assert h_lr_image == h_image // self.scale_factor and w_lr_image == w_image // self.scale_factor

        assert image.shape[:2] == mask.shape[:2]

        if tile_idx is not None:
            image = split_image(image, self.num_tiles, tile_idx)
            mask = split_image(mask, self.num_tiles, tile_idx)
            lr_image = split_image(lr_image, self.num_tiles, tile_idx)
            assert image.shape[:2] == mask.shape[:2]

        # Apply transformations if any
        if self.transform:
            image, mask, lr_image = self.transform(image, mask, lr_image)

        return image, mask, lr_image, image_path, mask_path, lr_image_path
