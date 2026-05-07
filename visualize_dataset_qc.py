import argparse
import csv
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image


RGB_TO_CLASS = {
    (0, 0, 0): "Unidentifiable",
    (0, 0, 255): "Unidentifiable",
    (255, 255, 255): "Unidentifiable",
    (0, 255, 0): "Forest",
    (255, 0, 0): "Rice field",
    (0, 255, 255): "Water",
    (255, 255, 0): "Residential",
}

CLASS_TO_RGB = {
    "Unidentifiable": (0, 0, 255),
    "Forest": (0, 255, 0),
    "Rice field": (255, 0, 0),
    "Water": (0, 255, 255),
    "Residential": (255, 255, 0),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export PNG quicklooks from dataset/ TIFF + label masks for visual QC."
    )
    parser.add_argument("--dataset_dir", type=str, default="dataset")
    parser.add_argument("--output", type=str, default="qc_visualization/dataset")
    parser.add_argument(
        "--mode",
        choices=["all", "top-rice", "list"],
        default="top-rice",
        help="Which images to export.",
    )
    parser.add_argument("--limit", type=int, default=50, help="Max number of images to export.")
    parser.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="Specific stems or filenames to export, e.g. Area_2024_N_o_0 or Area_2024_N_o_0_sat.tif.",
    )
    parser.add_argument("--alpha", type=float, default=0.45, help="Mask overlay alpha.")
    parser.add_argument(
        "--percentile_low",
        type=float,
        default=2.0,
        help="Lower percentile for SAR visualization stretch.",
    )
    parser.add_argument(
        "--percentile_high",
        type=float,
        default=98.0,
        help="Upper percentile for SAR visualization stretch.",
    )
    parser.add_argument(
        "--max_side",
        type=int,
        default=1200,
        help="Resize output PNGs so the longest side is at most this value. Use 0 to keep native size.",
    )
    return parser.parse_args()


def normalize_stem(name):
    name = Path(name).name
    if name.endswith("_sat.tif"):
        return name[: -len("_sat.tif")]
    if name.endswith("_mask.png"):
        return name[: -len("_mask.png")]
    return Path(name).stem


def stretch01(channel, low=2.0, high=98.0):
    arr = np.asarray(channel, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return np.zeros(arr.shape, dtype=np.float32)
    lo, hi = np.percentile(valid, [low, high])
    if hi <= lo:
        lo, hi = float(valid.min()), float(valid.max())
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def sar_to_rgb(image_chw, low, high):
    if image_chw.shape[0] < 2:
        gray = stretch01(image_chw[0], low, high)
        rgb = np.stack([gray, gray, gray], axis=-1)
    else:
        vv = image_chw[0]
        vh = image_chw[1]
        vv_norm = stretch01(vv, low, high)
        vh_norm = stretch01(vh, low, high)
        ratio_norm = stretch01(vv - vh, low, high)
        rgb = np.stack([vv_norm, vh_norm, ratio_norm], axis=-1)
    return (rgb * 255.0).round().astype(np.uint8)


def load_tif_rgb(path, low, high):
    with rasterio.open(path) as src:
        image = src.read()
        meta = {
            "width": src.width,
            "height": src.height,
            "band_count": src.count,
            "crs": str(src.crs),
            "bounds": (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top),
            "descriptions": "|".join(str(desc) for desc in src.descriptions),
        }
    return sar_to_rgb(image, low, high), meta


def load_mask(path, target_size):
    mask = Image.open(path).convert("RGB")
    if mask.size != target_size:
        mask = mask.resize(target_size, resample=Image.NEAREST)
    return np.array(mask, dtype=np.uint8)


def build_known_mask(mask_rgb):
    known = np.zeros(mask_rgb.shape[:2], dtype=bool)
    for rgb in RGB_TO_CLASS:
        known |= np.all(mask_rgb == rgb, axis=-1)
    return known


def overlay_mask(base_rgb, mask_rgb, alpha):
    known = build_known_mask(mask_rgb)
    out = base_rgb.astype(np.float32)
    out[known] = (1.0 - alpha) * out[known] + alpha * mask_rgb[known].astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def class_ratios(mask_rgb):
    total = mask_rgb.shape[0] * mask_rgb.shape[1]
    ratios = {}
    for class_name in CLASS_TO_RGB:
        class_mask = np.zeros(mask_rgb.shape[:2], dtype=bool)
        for rgb, mapped_class in RGB_TO_CLASS.items():
            if mapped_class == class_name:
                class_mask |= np.all(mask_rgb == rgb, axis=-1)
        ratios[class_name] = float(class_mask.sum() / total) if total else 0.0
    return ratios


def resize_if_needed(image, max_side, nearest=False):
    if not max_side or max_side <= 0:
        return image
    pil = Image.fromarray(image)
    w, h = pil.size
    scale = min(1.0, float(max_side) / max(w, h))
    if scale >= 1.0:
        return image
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    resample = Image.NEAREST if nearest else Image.BILINEAR
    return np.array(pil.resize(new_size, resample=resample), dtype=np.uint8)


def save_png(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def select_images(dataset_dir, mode, limit, files, low, high):
    image_paths = sorted(dataset_dir.glob("*_sat.tif"))
    if mode == "list":
        if not files:
            raise ValueError("--files is required when --mode list")
        wanted = {normalize_stem(name) for name in files}
        return [path for path in image_paths if normalize_stem(path.name) in wanted]

    if mode == "all":
        return image_paths[:limit] if limit else image_paths

    scored = []
    for image_path in image_paths:
        mask_path = image_path.with_name(image_path.name.replace("_sat.tif", "_mask.png"))
        if not mask_path.exists():
            continue
        with rasterio.open(image_path) as src:
            target_size = (src.width, src.height)
        mask_rgb = load_mask(mask_path, target_size)
        ratios = class_ratios(mask_rgb)
        scored.append((ratios["Rice field"], image_path))
    scored.sort(reverse=True, key=lambda item: item[0])
    selected = [path for _, path in scored]
    return selected[:limit] if limit else selected


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output)

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")

    selected_paths = select_images(
        dataset_dir,
        args.mode,
        args.limit,
        args.files,
        args.percentile_low,
        args.percentile_high,
    )
    if not selected_paths:
        raise FileNotFoundError("No images selected for visualization")

    rows = []
    for image_path in selected_paths:
        stem = normalize_stem(image_path.name)
        mask_path = image_path.with_name(f"{stem}_mask.png")
        if not mask_path.exists():
            print(f"Skipping {image_path.name}: missing {mask_path.name}")
            continue

        base_rgb, meta = load_tif_rgb(image_path, args.percentile_low, args.percentile_high)
        mask_rgb = load_mask(mask_path, (meta["width"], meta["height"]))
        overlay_rgb = overlay_mask(base_rgb, mask_rgb, args.alpha)
        ratios = class_ratios(mask_rgb)

        base_out = resize_if_needed(base_rgb, args.max_side, nearest=False)
        mask_out = resize_if_needed(mask_rgb, args.max_side, nearest=True)
        overlay_out = resize_if_needed(overlay_rgb, args.max_side, nearest=False)

        save_png(output_dir / "sar_rgb" / f"{stem}_sar_rgb.png", base_out)
        save_png(output_dir / "mask" / f"{stem}_mask.png", mask_out)
        save_png(output_dir / "overlay" / f"{stem}_overlay.png", overlay_out)

        rows.append(
            {
                "stem": stem,
                "image_name": image_path.name,
                "mask_name": mask_path.name,
                "sar_png": str((output_dir / "sar_rgb" / f"{stem}_sar_rgb.png").resolve()),
                "mask_png": str((output_dir / "mask" / f"{stem}_mask.png").resolve()),
                "overlay_png": str((output_dir / "overlay" / f"{stem}_overlay.png").resolve()),
                "width": meta["width"],
                "height": meta["height"],
                "band_count": meta["band_count"],
                "band_descriptions": meta["descriptions"],
                "crs": meta["crs"],
                "left": meta["bounds"][0],
                "bottom": meta["bounds"][1],
                "right": meta["bounds"][2],
                "top": meta["bounds"][3],
                "rice_ratio": ratios["Rice field"],
                "forest_ratio": ratios["Forest"],
                "water_ratio": ratios["Water"],
                "residential_ratio": ratios["Residential"],
                "unidentifiable_ratio": ratios["Unidentifiable"],
            }
        )

    index_path = output_dir / "index.csv"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Exported {len(rows)} image set(s) to {output_dir}")
    print(f"Index CSV: {index_path}")
    print(f"SAR quicklooks: {output_dir / 'sar_rgb'}")
    print(f"Masks: {output_dir / 'mask'}")
    print(f"Overlays: {output_dir / 'overlay'}")


if __name__ == "__main__":
    main()
