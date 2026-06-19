import os
import re
import time
import warnings
from pathlib import Path
from typing import Optional

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
warnings.filterwarnings("ignore", category=FutureWarning)


SUPPORTED_TIF_SUFFIXES = {".tif", ".tiff"}
VITUNET_TRAIN_INPUT_SIZE = 512


def checkpoint_to_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if isinstance(checkpoint, dict):
        return checkpoint
    if hasattr(checkpoint, "state_dict"):
        return checkpoint.state_dict()
    raise ValueError("Unsupported checkpoint format")


def inspect_unet_checkpoint(checkpoint):
    state_dict = checkpoint_to_state_dict(checkpoint)
    if "inc.double_conv.0.weight" not in state_dict or "outc.conv.weight" not in state_dict:
        raise ValueError("Checkpoint does not look like model.ViT.model.UNet")

    n_channels = int(state_dict["inc.double_conv.0.weight"].shape[1])
    n_classes = int(state_dict["outc.conv.weight"].shape[0])
    depth_matches = {
        int(match.group(1))
        for key in state_dict
        for match in [re.match(r"vit\.transformer\.layers\.(\d+)\.", key)]
        if match
    }
    depth = (max(depth_matches) + 1) if depth_matches else 8

    to_qkv_key = "vit.transformer.layers.0.0.fn.to_qkv.weight"
    heads = 8
    if to_qkv_key in state_dict:
        heads = int(state_dict[to_qkv_key].shape[0] // (3 * 64))

    return {
        "n_channels": n_channels,
        "n_classes": n_classes,
        "depth": depth,
        "heads": heads,
        "input_size": VITUNET_TRAIN_INPUT_SIZE,
    }


def discover_tif_paths(input_path: Path, recursive: bool = True):
    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_TIF_SUFFIXES:
            raise ValueError(f"Input file is not a GeoTIFF: {input_path}")
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    iterator = input_path.rglob("*") if recursive else input_path.glob("*")
    tif_paths = sorted(
        path for path in iterator
        if path.is_file() and path.suffix.lower() in SUPPORTED_TIF_SUFFIXES
    )
    if not tif_paths:
        search_mode = "recursively" if recursive else "in the top-level folder"
        raise FileNotFoundError(f"No GeoTIFF files found {search_mode}: {input_path}")
    return tif_paths


def output_path_for(tif_path: Path, input_root: Path, output_dir: Path, preserve_dirs: bool):
    output_name = f"{tif_path.stem}_infered.png"
    if preserve_dirs and input_root.is_dir():
        relative_parent = tif_path.relative_to(input_root).parent
        return output_dir / relative_parent / output_name
    return output_dir / output_name


def build_output_plan(tif_paths, input_root: Path, output_dir: Path, preserve_dirs: bool):
    plan = []
    seen_outputs = {}
    for tif_path in tif_paths:
        output_path = output_path_for(tif_path, input_root, output_dir, preserve_dirs)
        if output_path in seen_outputs:
            raise ValueError(
                "Duplicate output path produced by input files: "
                f"{seen_outputs[output_path]} and {tif_path} -> {output_path}. "
                "Use --preserve_dirs or rename duplicate source stems."
            )
        seen_outputs[output_path] = tif_path
        plan.append((tif_path, output_path))
    return plan


def validate_vitunet_runtime_config(checkpoint_config, model_input_size):
    if model_input_size != checkpoint_config["input_size"]:
        raise ValueError(
            f"This ViTUnet checkpoint must be inferred at model_input_size={checkpoint_config['input_size']}. "
            f"Got {model_input_size}."
        )


def validate_image_channels(image, input_path, model, allow_sentinel1_to_13_adapter):
    expected_channels = getattr(model, "n_channels", None)
    actual_channels = image.shape[2]

    if expected_channels == 13 and actual_channels == 2:
        if not allow_sentinel1_to_13_adapter:
            raise ValueError(
                f"{input_path} has 2 Sentinel-1 bands, but the checkpoint expects 13 bands. "
                "Use inference_model/model_sentinel1_best.pth for Sentinel-1 inference. "
                "The legacy pseudo-13-band adapter is disabled by default."
            )
        print(f"Adapting {input_path.name} from 2-band Sentinel-1 to 13 pseudo-bands")
        return adapt_sentinel1_to_13_channels(image)

    if expected_channels is not None and actual_channels != expected_channels:
        raise ValueError(
            f"Input image {input_path} has {actual_channels} bands, "
            f"but model expects {expected_channels} bands."
        )
    return image


def inference_image(
    device,
    input_path: Path,
    output_path: Path,
    model,
    patch_size: int,
    model_input_size: int,
    stride: Optional[int],
    patch_batch_size: int,
    allow_sentinel1_to_13_adapter: bool,
):
    image = open_tif_image(str(input_path))
    if image is None:
        raise ValueError(f"Could not read input image: {input_path}")
    image = validate_image_channels(image, input_path, model, allow_sentinel1_to_13_adapter)

    inferred_image = infer_patches(
        model,
        device,
        image,
        patch_size=patch_size,
        model_input_size=model_input_size,
        stride=stride,
        n_classes=getattr(model, "n_classes", None),
        patch_batch_size=patch_batch_size,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(inferred_image).save(output_path)


def load_model(model_name, checkpoint_path: Path, device, model_input_size: int):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if model_name == "UNet":
        checkpoint_config = inspect_unet_checkpoint(checkpoint)
        validate_vitunet_runtime_config(checkpoint_config, model_input_size)
        model = UNet(
            n_classes=checkpoint_config["n_classes"],
            n_channels=checkpoint_config["n_channels"],
            depth=checkpoint_config["depth"],
            heads=checkpoint_config["heads"],
        ).to(device)
        model.load_state_dict(checkpoint_to_state_dict(checkpoint), strict=True)
        print(
            "Checkpoint config -> "
            f"n_channels: {checkpoint_config['n_channels']}, "
            f"n_classes: {checkpoint_config['n_classes']}, "
            f"depth: {checkpoint_config['depth']}, "
            f"heads: {checkpoint_config['heads']}, "
            f"model_input_size: {checkpoint_config['input_size']}"
        )
        return model.eval()

    if model_name == "FoundationModel":
        from model.Foundation.model import FoundationModel
        model = FoundationModel(n_classes=5, n_channels=13, upscale_factor=2).to(device)
        model.load_state_dict(checkpoint_to_state_dict(checkpoint), strict=True)
        return model.eval()

    raise ValueError(f"Model {model_name} is not supported")


def validate_args(args):
    if args.patch_size <= 0:
        raise ValueError(f"patch_size must be positive, got {args.patch_size}")
    if args.stride is not None and args.stride <= 0:
        raise ValueError(f"stride must be positive, got {args.stride}")
    if args.limit is not None and args.limit <= 0:
        raise ValueError(f"limit must be positive, got {args.limit}")
    if args.patch_batch_size <= 0:
        raise ValueError(f"patch_batch_size must be positive, got {args.patch_batch_size}")


def main():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    load_dotenv()
    set_seed(20)

    args = parse_infer_args()
    validate_args(args)

    if args.cuda and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_id}")
        # Best numerical fidelity defaults to full FP32. TF32 is available as an
        # explicit speed option because it trades mantissa precision for throughput.
        torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
        torch.backends.cudnn.allow_tf32 = bool(args.tf32)
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high" if args.tf32 else "highest")
        print(f"CUDA TF32 enabled: {bool(args.tf32)}")
    else:
        if args.cuda:
            print("CUDA requested but not available in the current environment, falling back to CPU")
        device = torch.device("cpu")
    print(f"Using device: {device}")

    input_path = Path(args.input)
    output_dir = Path(args.output)
    checkpoint_path = Path(args.pretrained)
    if not checkpoint_path.exists() and not args.dry_run:
        raise FileNotFoundError(f"Pretrained checkpoint does not exist: {checkpoint_path}")

    tif_paths = discover_tif_paths(input_path, recursive=not args.no_recursive)
    if args.limit is not None:
        tif_paths = tif_paths[:args.limit]
    output_plan = build_output_plan(tif_paths, input_path, output_dir, args.preserve_dirs)

    print(f"Discovered GeoTIFF files: {len(output_plan)}")
    for tif_path, output_path in output_plan[:5]:
        print(f"  {tif_path} -> {output_path}")
    if len(output_plan) > 5:
        print(f"  ... {len(output_plan) - 5} more")

    if args.dry_run:
        print("Dry run only; no model inference executed.")
        return

    model = load_model(args.model, checkpoint_path, device, args.model_input_size)
    if args.fp16:
        if device.type != "cuda":
            raise ValueError("--fp16 can only be used with CUDA")
        model = model.half()
        print("Using FP16 inference on CUDA")

    effective_stride = args.stride if args.stride is not None else args.patch_size // 2
    print(
        "Inference config -> "
        f"raw_patch_size: {args.patch_size}, "
        f"stride: {effective_stride}, "
        f"model_input_size: {args.model_input_size}, "
        f"patch_batch_size: {args.patch_batch_size}"
    )

    start_time = time.time()
    for index, (tif_path, output_path) in enumerate(output_plan, start=1):
        print(f"[{index}/{len(output_plan)}] {tif_path}")
        inference_image(
            device,
            tif_path,
            output_path,
            model,
            patch_size=args.patch_size,
            model_input_size=args.model_input_size,
            stride=args.stride,
            patch_batch_size=args.patch_batch_size,
            allow_sentinel1_to_13_adapter=args.allow_sentinel1_to_13_adapter,
        )

    elapsed = time.time() - start_time
    print(f"Inference time: {elapsed:.2f} seconds")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
