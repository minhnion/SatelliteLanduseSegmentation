import os
from PIL import Image
from torch.utils.data import Dataset
from utils.image_utils import open_tif_image

class LandCoverDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.transform = transforms
        self.image_paths = []
        self.mask_paths = []

        for file_name in os.listdir(root_dir):
            if file_name.endswith((".jpg", ".tif")):  # Support for .jpg, .tif, .tiff
                self.image_paths.append(os.path.join(root_dir, file_name))
                if file_name.endswith("_sat.jpg"):
                    mask_name = file_name.replace("_sat.jpg", "_mask.png")
                elif file_name.endswith(("_sat.tif")):
                    mask_name = file_name.replace("_sat.tif", "_mask.png")
                self.mask_paths.append(os.path.join(root_dir, mask_name))
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # Load image based on file format
        if image_path.endswith((".tif")):
            image = open_tif_image(image_path)
        else:
            image = Image.open(image_path).convert('RGB')  # Convert to RGB for other formats

        # Load mask
        mask = Image.open(mask_path).convert('RGB')  # Open mask as RGB

        # Apply transformations if any
        if self.transform:
            image, mask = self.transform(image, mask)

        # Return image, mask, and their corresponding file paths
        return image, mask, image_path, mask_path