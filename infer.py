import os
import warnings

import torch
from dotenv import load_dotenv
from PIL import Image
from rasterio.errors import NotGeoreferencedWarning

from model.ViT.model import UNet
from utils.image_utils import adapt_sentinel1_to_13_channels, open_tif_image
from utils.infer_utils import infer_patches
from utils.parser import parse_infer_args
from utils.seed import set_seed

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
warnings.filterwarnings("ignore", category=FutureWarning)  # Add this line to ignore FutureWarning

classes = ['unidentifiable', 'forest', 'rice_field', 'water', 'residential']
n_classes = len(classes)


def validate_patch_size(model_name, patch_size):
    if model_name == "UNet" and patch_size % 128 != 0:
        raise ValueError(
            f"patch_size={patch_size} is invalid for this ViTUnet checkpoint. "
            "Use a multiple of 128, for example 128 or 256."
        )

def inference_image(device, input_path, patch_size, output_path, model):

    image = open_tif_image(input_path)
    expected_channels = getattr(model, "n_channels", None)
    if expected_channels == 13 and image.shape[2] == 2:
        print(f"Adapting {os.path.basename(input_path)} from 2-band Sentinel-1 to 13 pseudo-bands")
        image = adapt_sentinel1_to_13_channels(image)
    elif expected_channels is not None and image.shape[2] != expected_channels:
        raise ValueError(
            f"Input image {input_path} has {image.shape[2]} bands, but model expects {expected_channels} bands. "
            "This checkpoint cannot be used with the current TIFF files."
        )

    infered_image = infer_patches(model, device, image.shape, image, patch_size)

    infered_image_pil = Image.fromarray(infered_image)
    infered_image_pil.save(output_path)

if __name__ == "__main__":
    try:
        import time  # Import time module

        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        # Load environment variables from .env file
        load_dotenv()

        # Initialize
        set_seed(20)
        args = parse_infer_args()
        if args.cuda and torch.cuda.is_available():
            device = torch.device(f"cuda:{args.gpu_id}")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        else:
            if args.cuda:
                print("CUDA requested but not available in the current environment, falling back to CPU")
            device = torch.device("cpu")
        print(f"Using device: {device}")

        input_path = args.input
        output_folder = args.output
        pretrained_path = args.pretrained
        patch_size = args.patch_size
        model_name = args.model

        assert os.path.exists(input_path), f"Input path {input_path} does not exist"
        assert os.path.exists(pretrained_path), f"Pretrained path {pretrained_path} does not exist"
        validate_patch_size(model_name, patch_size)

        os.makedirs(output_folder, exist_ok=True)

        # Load the model with weight path
        # model = UNet(n_classes=n_classes, n_channels=13).to(device)
        # checkpoint = torch.load(pretrained_path, map_location=device)
        # model.load_state_dict(checkpoint)
        # model = model.eval()

        # Load the model
        if model_name == "UNet":
            model = UNet(n_classes=n_classes, n_channels=13, depth=12, heads=4).to(device)
        elif model_name == "FoundationModel":
            from model.Foundation.model import FoundationModel
            model = FoundationModel(n_classes=n_classes, n_channels=13, upscale_factor=2).to(device)
        else:
            raise ValueError(f"Model {model_name} is not supported")

        # Load the state dictionary into the model
        checkpoint = torch.load(pretrained_path, map_location=device)
        if isinstance(checkpoint, dict):
            # If it's a state dictionary
            model.load_state_dict(checkpoint)
        else:
            # If it's a full model
            model = checkpoint

        if device.type == "cuda":
            model = model.half()
            print("Using FP16 inference on CUDA to reduce memory")

        model = model.eval()

        start_time = time.time()  # Start timing

        if os.path.isfile(input_path):
            output_path = os.path.join(output_folder, f"{os.path.basename(input_path).split('.')[0]}_infered.png")
            inference_image(device, input_path, patch_size, output_path, model)
        else:
            for file_name in os.listdir(input_path):
                if file_name.endswith(".tif"):
                    file_path = os.path.join(input_path, file_name)
                    output_path = os.path.join(output_folder, f"{os.path.basename(file_path).split('.')[0]}_infered.png")
                    inference_image(device, file_path, patch_size, output_path, model)

        end_time = time.time()  # End timing

        inference_time = end_time - start_time
        print(f"Inference time: {inference_time:.2f} seconds")

    except Exception as e:
        raise
