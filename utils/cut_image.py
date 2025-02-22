import os
import numpy as np
import rasterio
from PIL import Image, ImageFile

# Increase the maximum image size limit
Image.MAX_IMAGE_PIXELS = None

# Suppress the DecompressionBombWarning
ImageFile.LOAD_TRUNCATED_IMAGES = True

def cut_image_tif(image_path, output_folder, tile_size):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with rasterio.open(image_path) as raster:
        image = raster.read()  # Read all channels
        image = image.transpose(1, 2, 0)  # Transpose to (height, width, channels)
        image_width = raster.width
        image_height = raster.height

    for top in range(0, image_height, tile_size):
        for left in range(0, image_width, tile_size):
            if top + tile_size > image_height:
                top = image_height - tile_size
            if left + tile_size > image_width:
                left = image_width - tile_size
            right = left + tile_size
            bottom = top + tile_size
            tile = image[top:bottom, left:right]
            tile_path = os.path.join(output_folder, f"{os.path.basename(image_path).split('.')[0].split('_sat')[0]}_{int(np.ceil(top / tile_size))}_{int(np.ceil(left / tile_size))}_sat.tif")
            with rasterio.open(
                tile_path,
                'w',
                driver='GTiff',
                height=tile.shape[0],
                width=tile.shape[1],
                count=tile.shape[2],
                dtype=tile.dtype
            ) as dst:
                for i in range(tile.shape[2]):
                    dst.write(tile[:, :, i], i + 1)

def cut_image_png(image_path, output_folder, tile_size):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image = Image.open(image_path)
    image_width, image_height = image.size
    image = np.array(image)

    for top in range(0, image_height, tile_size):
        for left in range(0, image_width, tile_size):
            if top + tile_size > image_height:
                top = image_height - tile_size
            if left + tile_size > image_width:
                left = image_width - tile_size
            right = left + tile_size
            bottom = top + tile_size
            tile = image[top:bottom, left:right]
            tile_image = Image.fromarray(tile)
            tile_image.save(os.path.join(output_folder, f"{os.path.basename(image_path).split('.')[0].split('_mask')[0]}_{int(np.ceil(top / tile_size))}_{int(np.ceil(left / tile_size))}_mask.png"))

def process_folder(input_folder, output_folder, tile_size=256):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        image_path = os.path.join(input_folder, filename)
        if filename.lower().endswith('.tif'):
            cut_image_tif(image_path, output_folder, tile_size)
        elif filename.lower().endswith('.png'):
            cut_image_png(image_path, output_folder, tile_size)

if __name__ == "__main__":
    input_folder = '/mnt/henryng/final_dataset_new'
    tile_size = 256
    output_folder = input_folder + f'_cut{tile_size}'
    process_folder(input_folder, output_folder, tile_size=tile_size)
