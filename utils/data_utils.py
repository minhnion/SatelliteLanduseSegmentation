import torchvision.transforms.functional as functional
from torchvision.transforms.functional import hflip, vflip, rotate
from utils.image_utils import rgb_to_class
from PIL import Image
import numpy as np
import torch
import cv2

class ResizeAndToClassTransform:
    def __init__(self, size, RGB_TO_CLASSES, classes, augment=False):
        self.size = size
        self.RGB_TO_CLASSES = RGB_TO_CLASSES
        self.classes = classes
        self.augment = augment

    def augment_image_and_mask(self, image, mask):
        # Convert NumPy arrays to PyTorch tensors
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()  # (C, H, W)
        mask_tensor = torch.from_numpy(np.array(mask)).float()  # (H, W)

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

    def __call__(self, image, mask, mask_scale=None):
        # Ensure NaN values are handled
        image = np.nan_to_num(image)
        mask_scale = 1 if mask_scale is None else mask_scale
        mask = Image.fromarray(mask).resize(tuple(int(x * mask_scale) for x in self.size), resample=Image.NEAREST)
        # Augment the image
        # image, mask = self.augment_image_and_mask(image, mask)

        # Resize the image
        image_channel = []
        for i in range(image.shape[2]):
            resized_channel = cv2.resize(image[:, :, i], self.size)
            image_channel.append(resized_channel)
        image_resized = np.stack(image_channel, axis=-1)

        # Resize the mask and convert it to the appropriate format
        mask_scale = 1 if mask_scale is None else mask_scale
        mask = mask.resize(tuple(x * mask_scale for x in self.size), resample=Image.NEAREST)
        # print(np.array(mask).shape)
        mask = np.array(mask).astype(np.uint8)  # Convert to NumPy array

        # Convert the RGB mask to class indices
        mask = rgb_to_class(mask, self.RGB_TO_CLASSES, self.classes)

        # Convert image and mask to tensors
        mask = torch.tensor(mask, dtype=torch.long)
        image = torch.from_numpy(image_resized).permute(2, 0, 1).float()  # (C, H, W)
        if np.isnan(image).any():
            print(f'Nan values in image after transformation: {image}')
            raise ValueError('Nan values in image after transformation')
        return image, mask, self.classes

class ResizeAndToClassHRTransform:
    def __init__(self, size, RGB_TO_CLASSES, classes, augment=False):
        self.size = size
        self.RGB_TO_CLASSES = RGB_TO_CLASSES
        self.classes = classes
        self.augment = augment

    def augment_image_and_mask(self, image, mask):
        # Convert NumPy arrays to PyTorch tensors
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()  # (C, H, W)
        mask_tensor = torch.from_numpy(np.array(mask)).float()  # (H, W)

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

    def __call__(self, image, mask, lr_image):
        # Ensure NaN values are handled
        image = np.nan_to_num(image)

        # Augment the image
        # image, mask = self.augment_image_and_mask(image, mask)

        # Resize the image
        image_channel = []
        for i in range(image.shape[2]):
            resized_channel = cv2.resize(image[:, :, i], self.size)
            image_channel.append(resized_channel)
        image_resized = np.stack(image_channel, axis=-1)

        # Resize the low-resolution image
        lr_image_channel = []
        for i in range(lr_image.shape[2]):
            lr_size = (self.size[0] // 16, self.size[1] // 16)
            resized_channel = cv2.resize(lr_image[:, :, i], lr_size)
            lr_image_channel.append(resized_channel)
        lr_image_resized = np.stack(lr_image_channel, axis=-1)

        # Resize the mask and convert it to the appropriate format
        # mask is currently numpy array
        mask = Image.fromarray(mask.astype(np.uint8))
        mask = mask.resize(self.size, resample=Image.NEAREST)
        mask = np.array(mask).astype(np.uint8)  # Convert to NumPy array

        # Convert the RGB mask to class indices
        mask = rgb_to_class(mask, self.RGB_TO_CLASSES, self.classes)

        # Convert image and mask to tensors
        mask = torch.tensor(mask, dtype=torch.long)
        image = torch.from_numpy(image_resized).permute(2, 0, 1).float()  # (C, H, W)
        lr_image = torch.from_numpy(lr_image_resized).permute(2, 0, 1).float()  # (C, H, W)
        if np.isnan(image).any():
            print(f'Nan values in image after transformation: {image}')
            raise ValueError('Nan values in image after transformation')
        return image, mask, lr_image


def to_numpy_array(image) -> np.ndarray:
    """
    Convert input to a NumPy array.

    Parameters:
        image: Input image (NumPy array, list, tuple, PIL Image, or Torch tensor).

    Returns:
        np.ndarray: Converted NumPy array.
    """
    if isinstance(image, np.ndarray):
        return image
    elif isinstance(image, torch.Tensor):
        return image.cpu().numpy()  # Move to CPU and convert to NumPy
    elif isinstance(image, (list, tuple)):
        return np.array(image, dtype=np.uint8)
    try:
        from PIL import Image
        if isinstance(image, Image.Image):
            return np.array(image)
    except ImportError:
        pass
    raise ValueError("Unsupported image format. Provide a NumPy array, list, tuple, Torch tensor, or PIL Image.")
