import numpy as np 
from torchvision.transforms.functional import hflip, vflip, rotate
from PIL import Image
import numpy as np
import torch
import cv2
import rasterio

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

def open_tif_image(tiff_file, channel_num=4):
    try:
        with rasterio.open(tiff_file) as src:
            # Check the number of bands in the file
            num_bands = src.count
            # print(f"{tiff_file} has {num_bands} bands.")

            # Only proceed if the file has at least the required number of bands
            if num_bands < channel_num:
                print(f"Error: {tiff_file} does not have enough bands (requires {channel_num}).")
                return

            # Read the bands dynamically based on the channel_num
            bands = []
            for i in range(1, channel_num + 1):
                bands.append(src.read(i))  # Band indices in rasterio start at 1

        # Stack the read bands into an RGB-like array (or multi-channel array if more than 3)
        stacked_bands = np.dstack(bands) 
        stacked_bands_no_nan = np.nan_to_num(stacked_bands)  # Replace NaN values with 0
        
        return stacked_bands_no_nan
    
    except Exception as e:
        print(f"Error opening file {tiff_file}: {e}")

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
