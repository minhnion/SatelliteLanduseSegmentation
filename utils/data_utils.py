import torchvision.transforms.functional as functional
from torchvision.transforms.functional import hflip, vflip, rotate
from PIL import Image
import numpy as np
import torch
import cv2

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