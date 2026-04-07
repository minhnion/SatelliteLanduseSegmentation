from PIL import Image

# Load the image
image_path = "/mnt/hungvv/minh/dataset/deepglobe-classification/119_mask.png"
image = Image.open(image_path)

# Print the dimensions
print("Image dimensions:", image.size)  # Output will be (width, height)
