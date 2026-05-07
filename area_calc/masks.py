import numpy as np
from PIL import Image

from area_calc.config import CLASS_NAMES, RGB_TO_CLASS


def load_mask_rgb(mask_path, target_width, target_height):
    with Image.open(mask_path).convert("RGB") as img:
        if img.size != (target_width, target_height):
            img = img.resize((target_width, target_height), resample=Image.NEAREST)
        return np.array(img, dtype=np.uint8)


def build_class_masks(mask_rgb):
    class_masks = {}
    known = np.zeros(mask_rgb.shape[:2], dtype=bool)

    for class_name in CLASS_NAMES:
        mask = np.zeros(mask_rgb.shape[:2], dtype=bool)
        for rgb, mapped in RGB_TO_CLASS.items():
            if mapped == class_name:
                mask |= np.all(mask_rgb == rgb, axis=-1)
        class_masks[class_name] = mask
        known |= mask

    return class_masks, ~known


def area_for_mask(mask_bool, row_pixel_area_m2):
    counts_per_row = mask_bool.sum(axis=1).astype(np.float64)
    return float(np.dot(counts_per_row, row_pixel_area_m2))
