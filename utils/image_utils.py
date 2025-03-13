import numpy as np
from torchvision.transforms.functional import hflip, vflip, rotate
from PIL import Image
import numpy as np
import torch
import cv2
import rasterio
import torch.nn.functional as F

def class_to_rgb(class_mask):
    # Create an RGB image with the same dimensions as the class mask
    rgb_mask = np.zeros((class_mask.shape[0], class_mask.shape[1], 3), dtype=np.uint8)

    # Map class indices back to RGB values
#     rgb_mask[class_mask == 0] = [0, 0, 255]      # bamboo
#     rgb_mask[class_mask == 5] = [0, 0, 0]        # unknown
    rgb_mask[class_mask == 0] = [0, 0, 0]      # unknown
    rgb_mask[class_mask == 1] = [0, 255, 0]      # forest
    rgb_mask[class_mask == 2] = [255, 0, 0]      # rice_field
    rgb_mask[class_mask == 3] = [0, 255, 255]    # water
    rgb_mask[class_mask == 4] = [255, 255, 0]    # residential

    return rgb_mask

def rgb_to_class(mask):
    class_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)

    # Convert RGB mask values to class indices (adapt your mapping here)
    class_mask[(mask == [0, 0, 0]).all(axis=2)] = 0  # unidentifiable
    class_mask[(mask == [255, 255, 255]).all(axis=2)] = 0  # unidentifiable
    class_mask[(mask == [0, 0, 255]).all(axis=2)] = 0  # lightly_vegetated
    class_mask[(mask == [0, 255, 0]).all(axis=2)] = 1  # forest
    class_mask[(mask == [255, 0, 0]).all(axis=2)] = 2  # rice_field
    class_mask[(mask == [0, 255, 255]).all(axis=2)] = 3  # water
    class_mask[(mask == [255, 255, 0]).all(axis=2)] = 4  # residential
    return class_mask

def open_tif_image(tiff_file, rgb_only=False):
    try:
        with rasterio.open(tiff_file) as src:
            image = src.read([1, 2, 3]) if rgb_only else src.read()
            image = np.nan_to_num(image)  # Replace NaN values with 0
            image = np.transpose(image, (1, 2, 0))
            image_max = np.max(image)
            image_min = np.min(image)
            image = (image - image_min) / (image_max - image_min)
            image_max = np.max(image)
            image_min = np.min(image)
            if np.isnan(image).any():
                print(f"NaN values found in image {tiff_file}")
                raise ValueError
            # print(f"Image min: {image_min}, Image max: {image_max}")
            # print(f"Image dtype: {dtype}")
            return image

    except Exception as e:
        print(f"Error opening file {tiff_file}: {e}")

def split_image(image: np.ndarray, num_tiles: int, tile_idx: int) -> np.ndarray:
    """
    Splits a NumPy image into `num_tiles` parts along both axes (x and y) and returns the part at index `tile_idx`.

    :param image: NumPy array representing the image (H, W, C) or (H, W) for grayscale images.
    :param num_tiles: Number of parts to split the image into per axis.
    :param tile_idx: Index of the part to return (0-based, row-major order).
    :return: The selected image tile as a NumPy array.
    """
    if num_tiles < 1:
        raise ValueError("num_tiles must be at least 1")

    h_splits = np.array_split(image, num_tiles, axis=0)
    tiles = [np.array_split(h, num_tiles, axis=1) for h in h_splits]

    flat_tiles = [tile for row in tiles for tile in row]

    if tile_idx < 0 or tile_idx >= len(flat_tiles):
        raise ValueError("tile_idx must be in range [0, num_tiles*num_tiles-1]")

    return flat_tiles[tile_idx]

def downscale_image(image: np.ndarray, scale_factor: int) -> np.ndarray:
    """
    Downscale an image using bicubic interpolation while handling CHW format.

    Parameters:
        image (np.ndarray): The input image in (C, H, W) format.
        scale_factor (int): The scaling factor (e.g., 2 for half size).

    Returns:
        np.ndarray: The downscaled image in (C, H', W') format.
    """
    # Convert numpy array to PyTorch tensor
    image_tensor = torch.from_numpy(image).float()  # (C, H, W)

    # Downscale using PyTorch's interpolate
    downscaled_tensor = F.interpolate(
        image_tensor.unsqueeze(0),  # Add batch dimension
        scale_factor=1 / scale_factor,
        mode='bicubic',
        align_corners=False,
        recompute_scale_factor=True
    ).squeeze(0)  # Remove batch dimension

    # Convert back to numpy array
    downscaled_image = downscaled_tensor.numpy()

    return downscaled_image

class ResizeAndToClassTransform:
    def __init__(self, size, augment=False):
        self.size = size
        self.augment = augment

    def rgb_to_class(self, mask):
        class_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)

        # Convert RGB mask values to class indices (adapt your mapping here)
        class_mask[(mask == [0, 0, 255]).all(axis=2)] = 0  # bamboo
        class_mask[(mask == [0, 255, 0]).all(axis=2)] = 1  # forest
        class_mask[(mask == [255, 0, 0]).all(axis=2)] = 2  # rice_field
        class_mask[(mask == [0, 255, 255]).all(axis=2)] = 3  # water
        class_mask[(mask == [255, 255, 0]).all(axis=2)] = 4  # residential
        class_mask[(mask == [0, 0, 0]).all(axis=2)] = 0  # unknown
        return class_mask

    def augment_image_and_mask(self, image, mask):
        # Convert NumPy arrays to PyTorch tensors
        image_tensor = torch.from_numpy(image).permute(2, 0, 1)  # (C, H, W)
        mask_tensor = torch.from_numpy(np.array(mask))  # (H, W)

        # Apply augmentations to both image and mask
        if self.augment:
            # Random horizontal flip
            if torch.rand(1).item() > 0.5:
                image_tensor = hflip(image_tensor)
                mask_tensor = hflip(mask_tensor)

            # Random vertical flip
            if torch.rand(1).item() > 0.5:
                image_tensor = vflip(image_tensor)
                mask_tensor = vflip(mask_tensor)

        # Convert tensors back to NumPy arrays (C, H, W -> H, W, C for image)
        image = image_tensor.permute(1, 2, 0).numpy()
        mask = Image.fromarray(mask_tensor.numpy().astype(np.uint8))  # Convert mask back to PIL Image

        return image, mask

    def __call__(self, image, mask):
        # Ensure NaN values are handled
        image = np.nan_to_num(image)

        # Augment the image
        image, mask = self.augment_image_and_mask(image, mask)

        # Resize the image
        image_channel = []
        for i in range(image.shape[2]):
            resized_channel = cv2.resize(image[:, :, i], self.size)
            image_channel.append(resized_channel)

        image_resized = np.stack(image_channel, axis=-1)

        # Resize the mask and convert it to the appropriate format
        mask = mask.resize(self.size, resample=Image.NEAREST)  # Resize mask
        mask = np.array(mask).astype(np.uint8)  # Convert to NumPy array

        # Convert the RGB mask to class indices
        mask = self.rgb_to_class(mask)

        # Convert image and mask to tensors
        mask = torch.tensor(mask, dtype=torch.long)
        image = torch.from_numpy(image_resized).permute(2, 0, 1).float()  # (C, H, W)

        return image, mask
