import numpy as np
from torchvision.transforms.functional import hflip, vflip, rotate
from PIL import Image
import numpy as np
import torch
import cv2
import rasterio
import torch.nn.functional as F

import numpy as np

import numpy as np

# Define the mapping from RGB colors to class indices
# RGB_TO_CLASSES = {
#     (0, 0, 0): 0,         # unidentifiable
#     (255, 255, 255): 0,   # unidentifiable (alternative)
#     (0, 0, 255): 0,       # lightly_vegetated -> treated as unidentifiable
#     (0, 255, 0): 1,       # forest
#     (255, 0, 0): 2,       # rice_field
#     (0, 255, 255): 3,     # water
#     (255, 255, 0): 4,     # residential
# }

# RGB_TO_CLASSES = {
#     (0, 255, 255): 0,    # Urban land
#     (255, 255, 0): 1,    # Agriculture land
#     (255, 0, 255): 2,    # Rangeland
#     (0, 255, 0): 3,      # Forest land
#     (0, 0, 255): 4,      # Water
#     (255, 255, 255): 5,  # Barren land
#     (0, 0, 0): 6         # Unknown
# }

# Generate CLASS_TO_RGB with unique class representatives
# CLASS_TO_RGB = {}
# for rgb, class_idx in RGB_TO_CLASSES.items():
#     if class_idx not in CLASS_TO_RGB:
#         CLASS_TO_RGB[class_idx] = rgb  # Store only the first encountered RGB value

def rgb_to_class(mask, RGB_TO_CLASSES, classes: list):
    """Convert an RGB mask to a class index mask."""
    h, w, _ = mask.shape

    # Apply mapping dynamically
    class_mask = np.zeros((h, w), dtype=np.uint8)

    # Apply mapping dynamically
    for rgb, class_name in RGB_TO_CLASSES.items():
        class_mask[np.all(mask == rgb, axis=-1)] = classes.index(class_name)

    return class_mask

def class_to_rgb(class_mask, CLASSES_TO_RGB, classes: list):
    """Convert a class index mask to an RGB image."""
    h, w = class_mask.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for class_name, rgb in CLASSES_TO_RGB.items():
        rgb_mask[class_mask == classes.index(class_name)] = rgb

    return rgb_mask

def open_tif_image(tiff_file):
    try:
        with rasterio.open(tiff_file) as src:
            image = src.read()
            image = np.nan_to_num(image)  # Replace NaN values with 0
            image = np.transpose(image, (1, 2, 0))
            # image_max = np.max(image)
            # image_min = np.min(image)
            # image = (image - image_min) / (image_max - image_min + 1e-6)  # Normalize to [0, 1]
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

import numpy as np
from scipy.ndimage import sobel

def compute_sobel_magnitude(image):
    """
    Compute the Sobel edge magnitude for a 2D image.

    Args:
        image (np.ndarray): 2D image array of shape (H, W).

    Returns:
        np.ndarray: Sobel magnitude array of shape (H, W).
    """
    dx = sobel(image, axis=0)  # Gradient along rows (vertical edges)
    dy = sobel(image, axis=1)  # Gradient along columns (horizontal edges)
    mag = np.sqrt(dx**2 + dy**2)  # Compute magnitude
    return mag

def compute_fraction_image(image):
    """
    Compute fraction image (soft information) for SCNet from either multispectral or RGB images.

    Args:
        image (np.ndarray): Input image with shape (C, H, W), where C is 13 for multispectral
                            (e.g., Sentinel-2 bands) or 3 for RGB. The image should be in float format.

    Returns:
        np.ndarray: Fraction image with shape (C', H, W), where C' is the number of features
                    (5 for multispectral, 4 for RGB).

    Raises:
        ValueError: If the number of input channels is neither 3 nor 13.
    """
    # Ensure the image is in float format for accurate computations
    if image.dtype not in (np.float32, np.float64):
        image = image.astype(np.float32)

    # Get the number of channels
    C = image.shape[0]

    if C == 13:
        # Multispectral image processing (e.g., Sentinel-2 with 13 bands)
        # Band indices based on standard Sentinel-2 ordering: B1 to B12, B8A
        B3 = image[2]   # Green
        B4 = image[3]   # Red
        B5 = image[4]   # Red-edge 1
        B8 = image[7]   # NIR
        B11 = image[10] # SWIR1
        B12 = image[11] # SWIR2

        # Compute normalized difference indices
        # Small epsilon (1e-6) added to denominators to prevent division by zero
        ndvi = (B8 - B4) / (B8 + B4 + 1e-6)      # Normalized Difference Vegetation Index
        ndwi = (B3 - B8) / (B3 + B8 + 1e-6)      # Normalized Difference Water Index
        ndbi = (B11 - B8) / (B11 + B8 + 1e-6)    # Normalized Difference Built-up Index
        rendvi = (B8 - B5) / (B8 + B5 + 1e-6)    # Red-edge NDVI using B5

        # Compute Sobel edge magnitude on the red band (B4)
        sobel_mag = compute_sobel_magnitude(B4)

        # Normalize features to [0, 1]
        # Normalized difference indices are in [-1, 1], so scale to [0, 1]
        features = [ndvi, ndwi, ndbi, rendvi]
        for i, feat in enumerate(features):
            feat = np.clip(feat, -1, 1)  # Clip to theoretical range
            features[i] = (feat + 1) / 2  # Scale from [-1, 1] to [0, 1]

        # Normalize Sobel magnitude to [0, 1] using min-max scaling
        sobel_min = sobel_mag.min()
        sobel_max = sobel_mag.max()
        sobel_mag = (sobel_mag - sobel_min) / (sobel_max - sobel_min + 1e-6)

        # Stack all features into a single array
        fraction_image = np.stack(features + [sobel_mag], axis=0)  # Shape: (5, H, W)

    elif C == 3:
        # RGB image processing
        R = image[0]  # Red channel
        G = image[1]  # Green channel
        B = image[2]  # Blue channel

        # Compute color-based features
        veg_like = (G - R) / (G + R + 1e-6)      # Vegetation-like feature
        water_like = (B - G) / (B + G + 1e-6)    # Water-like feature

        # Compute Sobel magnitude for each channel
        mag_R = compute_sobel_magnitude(R)
        mag_G = compute_sobel_magnitude(G)
        mag_B = compute_sobel_magnitude(B)
        # Take the maximum magnitude across channels for overall edge strength
        sobel_mag_rgb = np.maximum(np.maximum(mag_R, mag_G), mag_B)

        # Compute intensity (grayscale) and its Sobel magnitude
        intensity = 0.299 * R + 0.587 * G + 0.114 * B  # Standard RGB to grayscale weights
        sobel_intensity = compute_sobel_magnitude(intensity)

        # Normalize features to [0, 1]
        features = [veg_like, water_like]
        for i, feat in enumerate(features):
            feat = np.clip(feat, -1, 1)  # Clip to theoretical range
            features[i] = (feat + 1) / 2  # Scale from [-1, 1] to [0, 1]

        # Normalize Sobel features to [0, 1]
        for feat in [sobel_mag_rgb, sobel_intensity]:
            feat_min = feat.min()
            feat_max = feat.max()
            feat[...] = (feat - feat_min) / (feat_max - feat_min + 1e-6)

        # Stack all features into a single array
        fraction_image = np.stack(features + [sobel_mag_rgb, sobel_intensity], axis=0)  # Shape: (4, H, W)

    else:
        raise ValueError("Input image must have 3 (RGB) or 13 (multispectral) channels.")

    return fraction_image
