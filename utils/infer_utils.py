import numpy as np
import torch
from utils.image_utils import class_to_rgb

def infer_patches(model, device, org_shape, image: np.ndarray, patch_size: int = 128, n_classes=5):
    stride = patch_size // 2  # 50% overlap between patches
    org_size = org_shape[:2]
    image_height, image_width = image.shape[:2]

    # Use float32 for precision
    infer_image = np.zeros((org_size[0], org_size[1], n_classes), dtype=np.float32)
    count_map = np.zeros((org_size[0], org_size[1], n_classes), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for top in range(0, image_height, stride):
            for left in range(0, image_width, stride):
                # Ensure patch fits within bounds
                bottom = min(top + patch_size, image_height)
                right = min(left + patch_size, image_width)

                # Extract and pad patch
                patch = image[top:bottom, left:right]
                pad_h, pad_w = patch_size - (bottom - top), patch_size - (right - left)
                if pad_h > 0 or pad_w > 0:
                    patch = np.pad(patch, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')

                patch = np.expand_dims(patch, axis=0)  # Add batch dim
                patch = torch.from_numpy(patch).permute(0, 3, 1, 2).float().to(device)

                # Model inference
                output = model(patch)
                output = torch.softmax(output, dim=1).cpu().numpy()[0]  # Convert to probabilities

                # Remove padding from predictions
                output = output[:, :bottom - top, :right - left]

                # Accumulate softmax probabilities
                infer_image[top:bottom, left:right] += np.transpose(output, (1, 2, 0))
                count_map[top:bottom, left:right] += 1

    # Normalize and convert to final class labels
    infer_image /= np.maximum(count_map, 1)  # Avoid division by zero
    final_output = np.argmax(infer_image, axis=-1)  # Get class labels
    final_output = class_to_rgb(final_output.astype(np.uint8))

    return final_output
