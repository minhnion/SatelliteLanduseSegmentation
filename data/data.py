import os
from PIL import Image
from torch.utils.data import Dataset
from utils.image_utils import open_tif_image, split_image
import numpy as np

class LandCoverDataset(Dataset):
    def __init__(self, root_dir, transforms=None, num_tiles=4, rgb_only=False):
        self.root_dir = root_dir
        self.transform = transforms
        self.image_paths, self.mask_paths = [], []
        self.num_tiles = num_tiles
        # self.images, self.masks = [], []

        for file_name in os.listdir(root_dir):
            if file_name.endswith((".jpg", ".tif")):  # Support for .jpg, .tif, .tiff
                image_path = os.path.join(root_dir, file_name)
                # image = open_tif_image(image_path)
                self.image_paths.append(image_path)
                if file_name.endswith("_sat.jpg"):
                    mask_name = file_name.replace("_sat.jpg", "_mask.png")
                elif file_name.endswith(("_sat.tif")):
                    mask_name = file_name.replace("_sat.tif", "_mask.png")
                self.mask_paths.append(os.path.join(root_dir, mask_name))
        assert len(self.image_paths) == len(self.mask_paths)

    def __len__(self):
        return len(self.image_paths) * (self.num_tiles ** 2)

    def __getitem__(self, idx):
        image_idx = idx // (self.num_tiles ** 2)
        tile_idx = idx % (self.num_tiles ** 2)
        image_path = self.image_paths[image_idx]
        mask_path = self.mask_paths[image_idx]

        # Load image based on file format
        if image_path.endswith((".tif")):
            image = open_tif_image(image_path)
        else:
            image = np.array(Image.open(image_path).convert('RGB'))  # Convert to RGB for other formats

        # Load mask
        mask = np.array(Image.open(mask_path).convert('RGB'))  # Open mask as RGB
        assert image.shape[:2] == mask.shape[:2]

        image = split_image(image, self.num_tiles, tile_idx)
        mask = split_image(mask, self.num_tiles, tile_idx)
        assert image.shape[:2] == mask.shape[:2]

        # Apply transformations if any
        if self.transform:
            image, mask = self.transform(image, mask)

        # print(image.shape)
        # print(mask.shape)
        assert image.shape[1:] == mask.shape

        # print(np.count_nonzero(mask == 5))

        # Return image, mask, and their corresponding file paths
        return image, mask, image_path, mask_path
