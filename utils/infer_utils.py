import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional

from utils.image_utils import class_to_rgb


CLASS_TO_RGB = {
    "Unidentifiable": (0, 0, 0),
    "Forest": (0, 255, 0),
    "Rice field": (255, 0, 0),
    "Water": (0, 255, 255),
    "Residential": (255, 255, 0),
}

CLASSES = list(CLASS_TO_RGB.keys())


def _axis_starts(size: int, patch_size: int, stride: int):
    if size <= 0:
        raise ValueError(f"Invalid image axis size: {size}")
    if patch_size >= size:
        return [0]

    starts = list(range(0, size - patch_size + 1, stride))
    last = size - patch_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def _segmentation_output(model_output):
    if isinstance(model_output, tuple):
        return model_output[1]
    return model_output


def _validate_infer_params(image, patch_size, stride, model_input_size):
    if image.ndim != 3:
        raise ValueError(f"Expected image with shape (H, W, C), got {image.shape}")
    if patch_size <= 0:
        raise ValueError(f"patch_size must be positive, got {patch_size}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    if model_input_size <= 0:
        raise ValueError(f"model_input_size must be positive, got {model_input_size}")


def infer_patches(
    model,
    device,
    image: np.ndarray,
    patch_size: int = 512,
    model_input_size: int = 512,
    stride: Optional[int] = None,
    n_classes: Optional[int] = None,
):
    """Run sliding-window Sentinel-1 inference with train-time ViTUnet sizing.

    The Sentinel-1 ViTUnet was trained on 512x512 tensors. Current 3km
    inference tiles are smaller than 512px, so the default runs one full-tile
    pass and resizes that tile to 512x512 before
    the model forward pass, then the softmax probabilities are resized back to
    the raw patch footprint before overlap averaging.
    """
    stride = patch_size // 2 if stride is None else stride
    _validate_infer_params(image, patch_size, stride, model_input_size)

    image_height, image_width = image.shape[:2]
    n_classes = int(n_classes or getattr(model, "n_classes", len(CLASSES)))

    probability_sum = np.zeros((image_height, image_width, n_classes), dtype=np.float32)
    count_map = np.zeros((image_height, image_width, 1), dtype=np.float32)

    row_starts = _axis_starts(image_height, patch_size, stride)
    col_starts = _axis_starts(image_width, patch_size, stride)

    model.eval()
    model_dtype = next(model.parameters()).dtype
    use_autocast = device.type == "cuda" and model_dtype == torch.float16

    with torch.inference_mode():
        for top in row_starts:
            bottom = min(top + patch_size, image_height)
            for left in col_starts:
                right = min(left + patch_size, image_width)
                patch = image[top:bottom, left:right]

                patch_tensor = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0)
                patch_tensor = patch_tensor.to(device=device, dtype=model_dtype)
                if patch_tensor.shape[-2:] != (model_input_size, model_input_size):
                    patch_tensor = F.interpolate(
                        patch_tensor,
                        size=(model_input_size, model_input_size),
                        mode="bilinear",
                        align_corners=False,
                    )

                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_autocast):
                    logits = _segmentation_output(model(patch_tensor))

                if logits.shape[1] != n_classes:
                    raise ValueError(f"Model returned {logits.shape[1]} classes, expected {n_classes}")

                probabilities = torch.softmax(logits.float(), dim=1)
                probabilities = F.interpolate(
                    probabilities,
                    size=(bottom - top, right - left),
                    mode="bilinear",
                    align_corners=False,
                )
                probabilities_np = probabilities.squeeze(0).permute(1, 2, 0).cpu().numpy()

                probability_sum[top:bottom, left:right] += probabilities_np
                count_map[top:bottom, left:right] += 1.0

    if np.any(count_map == 0):
        raise RuntimeError("Internal error: some pixels were not covered by inference patches")

    probabilities = probability_sum / count_map
    class_mask = np.argmax(probabilities, axis=-1).astype(np.uint8)
    return class_to_rgb(class_mask, CLASS_TO_RGB, CLASSES)
