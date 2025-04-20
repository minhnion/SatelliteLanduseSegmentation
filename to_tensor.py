import os
import torch
import rasterio
import numpy as np
from pathlib import Path
import argparse

def tif_to_tensor(tif_path, output_path=None):
    """
    Convert a TIF image to a PyTorch tensor and save it.

    Args:
        tif_path (str): Path to the input TIF file
        output_path (str, optional): Path to save the tensor. If None, will use the same name with .pt extension.

    Returns:
        torch.Tensor: The loaded tensor
    """
    # Create output path if not provided
    if output_path is None:
        output_path = str(Path(tif_path).with_suffix('.pt'))

    # Open the TIF file
    with rasterio.open(tif_path) as src:
        # Read the image data
        img = src.read()  # This returns a 3D array [bands, height, width]

        # Convert to float32 if needed and normalize if desired
        img = img.astype(np.float32)

        # Convert to PyTorch tensor
        tensor = torch.from_numpy(img)

    # Save the tensor
    torch.save(tensor, output_path)

    print(f"Converted {tif_path} to tensor and saved at {output_path}")
    return tensor


if __name__ == "__main__":
    tif_path = "dataset/sentinel_hr_lr_dataset/anh_25km_sat.tif"
    output_path = "dataset/sentinel_hr_lr_dataset/anh_25km_sat.pt"

    tif_to_tensor(tif_path)
