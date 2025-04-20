import torch
from utils.seed import set_seed
from utils.parser import parse_infer_args
from utils.image_utils import open_tif_image
from utils.infer_utils import infer_patches

from model.ViT.model import UNet
from model.Foundation.model import FoundationModel

from datetime import datetime

import os
from dotenv import load_dotenv
import warnings
from rasterio.errors import NotGeoreferencedWarning
from PIL import Image  # Correct import

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
warnings.filterwarnings("ignore", category=FutureWarning)  # Add this line to ignore FutureWarning

classes = ['unidentifiable', 'forest', 'rice_field', 'water', 'residential']
n_classes = len(classes)

def inference_image(device, input_path, patch_size, output_path, model):

    image = open_tif_image(input_path)

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
        device = torch.device("cuda:" + str(args.gpu_id)) if args.cuda else torch.device("cpu")
        print(f"Using device: {device}")

        input_path = args.input
        output_folder = args.output
        pretrained_path = args.pretrained
        patch_size = args.patch_size
        model_name = args.model

        assert os.path.exists(input_path), f"Input path {input_path} does not exist"
        assert os.path.exists(pretrained_path), f"Pretrained path {pretrained_path} does not exist"

        os.makedirs(output_folder, exist_ok=True)

        # Load the model with weight path
        # model = UNet(n_classes=n_classes, n_channels=13).to(device)
        # checkpoint = torch.load(pretrained_path, map_location=device)
        # model.load_state_dict(checkpoint)
        # model = model.eval()

        # Load the model
        if model_name == "UNet":
            model = UNet(n_classes=n_classes, n_channels=13).to(device)
        elif model_name == "FoundationModel":
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
