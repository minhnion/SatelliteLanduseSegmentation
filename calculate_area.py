import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


CLASS_TO_RGB = {
    "Unidentifiable": (0, 0, 0),
    "Forest": (0, 255, 0),
    "Rice field": (255, 0, 0),
    "Water": (0, 255, 255),
    "Residential": (255, 255, 0),
}

PNG_TILE_PATTERN = re.compile(r"^S1_(\d+)_(\d+)_(\d+)_")


def slugify(name):
    return name.lower().replace(" ", "_")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate landuse area from inferred PNG masks, including per-province totals by indexCode."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="inference_png/Resolution3x3",
        help="Folder containing inferred PNG masks",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="area_output/Resolution3x3",
        help="Folder to save CSV outputs",
    )
    parser.add_argument(
        "--province_label",
        type=str,
        default="mapbox/gadm_resolution_3_province_mapbox_label.json",
        help="JSON file that maps each tile to province indexCode / overlap ratios",
    )
    parser.add_argument(
        "--tile_width_km",
        type=float,
        default=3.0,
        help="Physical tile width in kilometers",
    )
    parser.add_argument(
        "--tile_height_km",
        type=float,
        default=3.0,
        help="Physical tile height in kilometers",
    )
    parser.add_argument(
        "--normalize_intersection_ratios",
        action="store_true",
        help="Normalize provinceOverlapList ratios to sum to 1.0 for each intersection tile",
    )
    return parser.parse_args()


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def extract_tile_key(file_name):
    match = PNG_TILE_PATTERN.match(file_name)
    if not match:
        raise ValueError(f"Unexpected PNG name format, cannot extract tile key: {file_name}")
    cell_code, row, col = match.groups()
    return int(cell_code), int(row), int(col)


def file_preference_key(path):
    return ("(" in path.stem, len(path.name), path.name)


def select_unique_png_files(png_files):
    grouped = defaultdict(list)
    for path in png_files:
        grouped[extract_tile_key(path.name)].append(path)

    selected_files = []
    duplicate_rows = []
    for tile_key in sorted(grouped):
        candidates = sorted(grouped[tile_key], key=file_preference_key)
        kept = candidates[0]
        selected_files.append(kept)
        if len(candidates) > 1:
            duplicate_rows.append(
                {
                    "cell_code": tile_key[0],
                    "grid_row": tile_key[1],
                    "grid_col": tile_key[2],
                    "kept_file": kept.name,
                    "skipped_files": ";".join(path.name for path in candidates[1:]),
                    "duplicate_count": len(candidates) - 1,
                }
            )

    return selected_files, duplicate_rows


def count_class_pixels(image_np):
    counts = {}
    for class_name, rgb in CLASS_TO_RGB.items():
        counts[class_name] = int(np.sum(np.all(image_np == rgb, axis=-1)))
    return counts


def analyze_image(image_path, tile_area_km2):
    tile_key = extract_tile_key(image_path.name)
    with Image.open(image_path).convert("RGB") as img:
        image_np = np.array(img)

    height, width = image_np.shape[:2]
    total_pixels = height * width
    pixel_area_km2 = tile_area_km2 / total_pixels
    pixel_area_m2 = pixel_area_km2 * 1_000_000.0

    class_counts = count_class_pixels(image_np)
    class_area_km2 = {}
    row = {
        "file_name": image_path.name,
        "cell_code": tile_key[0],
        "grid_row": tile_key[1],
        "grid_col": tile_key[2],
        "width_px": width,
        "height_px": height,
        "total_pixels": total_pixels,
        "tile_area_km2": tile_area_km2,
        "pixel_area_m2": pixel_area_m2,
    }

    for class_name, pixel_count in class_counts.items():
        key = slugify(class_name)
        proportion = pixel_count / total_pixels if total_pixels else 0.0
        area_km2 = proportion * tile_area_km2
        area_ha = area_km2 * 100.0
        class_area_km2[class_name] = area_km2
        row[f"{key}_pixels"] = pixel_count
        row[f"{key}_ratio"] = proportion
        row[f"{key}_area_km2"] = area_km2
        row[f"{key}_area_ha"] = area_ha

    return {
        "tile_key": tile_key,
        "row": row,
        "class_counts": class_counts,
        "class_area_km2": class_area_km2,
        "total_pixels": total_pixels,
    }


def load_province_label_map(label_path):
    with label_path.open("r", encoding="utf-8") as f:
        raw_entries = json.load(f)

    label_map = {}
    for entry in raw_entries:
        tile_key = (int(entry["cellCode"]), int(entry["row"]), int(entry["col"]))
        if tile_key in label_map:
            raise ValueError(f"Duplicate tile mapping found in province label file for key {tile_key}")

        is_intersection = bool(entry.get("isIntersection", False))
        if is_intersection:
            overlap_list = entry.get("provinceOverlapList") or []
            if not overlap_list:
                raise ValueError(f"Missing provinceOverlapList for intersection tile {tile_key}")
            allocations = [
                {
                    "province_code": int(item["indexCode"]),
                    "ratio": float(item["ratio"]),
                }
                for item in overlap_list
            ]
        else:
            index_code = entry["indexCode"]
            if isinstance(index_code, list):
                if len(index_code) != 1:
                    raise ValueError(f"Expected one province code for non-intersection tile {tile_key}, got {index_code}")
                index_code = index_code[0]
            allocations = [{"province_code": int(index_code), "ratio": 1.0}]

        ratio_sum = sum(item["ratio"] for item in allocations)
        label_map[tile_key] = {
            "is_intersection": is_intersection,
            "allocations": allocations,
            "raw_ratio_sum": ratio_sum,
        }

    return label_map


def build_summary_rows(per_image_rows, tile_area_km2):
    total_tile_area_km2 = tile_area_km2 * len(per_image_rows)
    total_pixels_all = sum(row["total_pixels"] for row in per_image_rows)
    summary_rows = []
    for class_name in CLASS_TO_RGB:
        key = slugify(class_name)
        total_class_pixels = sum(row[f"{key}_pixels"] for row in per_image_rows)
        total_class_area_km2 = sum(row[f"{key}_area_km2"] for row in per_image_rows)
        total_class_area_ha = sum(row[f"{key}_area_ha"] for row in per_image_rows)
        pixel_ratio_all = total_class_pixels / total_pixels_all if total_pixels_all else 0.0
        area_ratio_all = total_class_area_km2 / total_tile_area_km2 if total_tile_area_km2 else 0.0
        summary_rows.append(
            {
                "class_name": class_name,
                "rgb": CLASS_TO_RGB[class_name],
                "image_count": len(per_image_rows),
                "total_pixels": total_class_pixels,
                "pixel_ratio": pixel_ratio_all,
                "total_area_km2": total_class_area_km2,
                "total_area_ha": total_class_area_ha,
                "area_ratio": area_ratio_all,
            }
        )
    return summary_rows


def add_province_outputs(
    analyses,
    province_label_map,
    output_dir,
    tile_area_km2,
    normalize_intersection_ratios,
):
    per_image_province_rows = []
    province_summary = {}
    missing_label_keys = []

    for analysis in analyses:
        tile_key = analysis["tile_key"]
        mapping = province_label_map.get(tile_key)
        if mapping is None:
            missing_label_keys.append(tile_key)
            continue

        raw_ratio_sum = mapping["raw_ratio_sum"]
        for allocation in mapping["allocations"]:
            raw_ratio = allocation["ratio"]
            if normalize_intersection_ratios and raw_ratio_sum > 0.0:
                effective_ratio = raw_ratio / raw_ratio_sum
            else:
                effective_ratio = raw_ratio

            province_code = allocation["province_code"]
            summary_row = province_summary.setdefault(
                province_code,
                {
                    "province_code": province_code,
                    "tile_hit_count": 0,
                    "assigned_tile_area_km2": 0.0,
                    "assigned_tile_area_ha": 0.0,
                },
            )
            summary_row["tile_hit_count"] += 1
            summary_row["assigned_tile_area_km2"] += tile_area_km2 * effective_ratio
            summary_row["assigned_tile_area_ha"] += tile_area_km2 * effective_ratio * 100.0

            row = {
                "file_name": analysis["row"]["file_name"],
                "cell_code": tile_key[0],
                "grid_row": tile_key[1],
                "grid_col": tile_key[2],
                "province_code": province_code,
                "is_intersection": mapping["is_intersection"],
                "raw_ratio": raw_ratio,
                "effective_ratio": effective_ratio,
                "raw_ratio_sum": raw_ratio_sum,
                "tile_assigned_area_km2": tile_area_km2 * effective_ratio,
                "tile_assigned_area_ha": tile_area_km2 * effective_ratio * 100.0,
                "tile_unassigned_ratio_raw": max(0.0, 1.0 - raw_ratio_sum),
                "tile_excess_ratio_raw": max(0.0, raw_ratio_sum - 1.0),
            }

            for class_name in CLASS_TO_RGB:
                key = slugify(class_name)
                weighted_pixels = analysis["class_counts"][class_name] * effective_ratio
                weighted_area_km2 = analysis["class_area_km2"][class_name] * effective_ratio
                row[f"{key}_pixels_equiv"] = weighted_pixels
                row[f"{key}_area_km2"] = weighted_area_km2
                row[f"{key}_area_ha"] = weighted_area_km2 * 100.0

                summary_row.setdefault(f"{key}_pixels_equiv", 0.0)
                summary_row.setdefault(f"{key}_area_km2", 0.0)
                summary_row.setdefault(f"{key}_area_ha", 0.0)
                summary_row[f"{key}_pixels_equiv"] += weighted_pixels
                summary_row[f"{key}_area_km2"] += weighted_area_km2
                summary_row[f"{key}_area_ha"] += weighted_area_km2 * 100.0

            per_image_province_rows.append(row)

    if missing_label_keys:
        raise ValueError(
            "Missing province labels for some tiles, for example: "
            + ", ".join(str(key) for key in missing_label_keys[:5])
        )

    for province_code, row in province_summary.items():
        total_area_km2 = 0.0
        for class_name in CLASS_TO_RGB:
            total_area_km2 += row[f"{slugify(class_name)}_area_km2"]
        row["total_area_km2"] = total_area_km2
        row["total_area_ha"] = total_area_km2 * 100.0

    per_image_fields = [
        "file_name",
        "cell_code",
        "grid_row",
        "grid_col",
        "province_code",
        "is_intersection",
        "raw_ratio",
        "effective_ratio",
        "raw_ratio_sum",
        "tile_assigned_area_km2",
        "tile_assigned_area_ha",
        "tile_unassigned_ratio_raw",
        "tile_excess_ratio_raw",
    ]
    for class_name in CLASS_TO_RGB:
        key = slugify(class_name)
        per_image_fields.extend(
            [
                f"{key}_pixels_equiv",
                f"{key}_area_km2",
                f"{key}_area_ha",
            ]
        )

    summary_fields = [
        "province_code",
        "tile_hit_count",
        "assigned_tile_area_km2",
        "assigned_tile_area_ha",
        "total_area_km2",
        "total_area_ha",
    ]
    for class_name in CLASS_TO_RGB:
        key = slugify(class_name)
        summary_fields.extend(
            [
                f"{key}_pixels_equiv",
                f"{key}_area_km2",
                f"{key}_area_ha",
            ]
        )

    per_image_csv = output_dir / "per_image_province_area.csv"
    summary_csv = output_dir / "summary_province_area.csv"
    write_csv(per_image_csv, per_image_fields, per_image_province_rows)
    write_csv(
        summary_csv,
        summary_fields,
        [province_summary[code] for code in sorted(province_summary)],
    )

    return per_image_csv, summary_csv


def main():
    args = parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    province_label_path = Path(args.province_label)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_dir}")
    if not province_label_path.exists():
        raise FileNotFoundError(f"Province label file does not exist: {province_label_path}")

    png_files = sorted(input_dir.glob("*.png"))
    if not png_files:
        raise FileNotFoundError(f"No PNG files found in: {input_dir}")

    unique_png_files, duplicate_rows = select_unique_png_files(png_files)
    tile_area_km2 = args.tile_width_km * args.tile_height_km
    analyses = [analyze_image(image_path, tile_area_km2) for image_path in unique_png_files]
    per_image_rows = [item["row"] for item in analyses]

    per_image_fields = [
        "file_name",
        "cell_code",
        "grid_row",
        "grid_col",
        "width_px",
        "height_px",
        "total_pixels",
        "tile_area_km2",
        "pixel_area_m2",
    ]
    for class_name in CLASS_TO_RGB:
        key = slugify(class_name)
        per_image_fields.extend(
            [
                f"{key}_pixels",
                f"{key}_ratio",
                f"{key}_area_km2",
                f"{key}_area_ha",
            ]
        )

    per_image_csv = output_dir / "per_image_area.csv"
    write_csv(per_image_csv, per_image_fields, per_image_rows)

    summary_csv = output_dir / "summary_area.csv"
    write_csv(
        summary_csv,
        [
            "class_name",
            "rgb",
            "image_count",
            "total_pixels",
            "pixel_ratio",
            "total_area_km2",
            "total_area_ha",
            "area_ratio",
        ],
        build_summary_rows(per_image_rows, tile_area_km2),
    )

    duplicate_csv = output_dir / "duplicate_tiles.csv"
    write_csv(
        duplicate_csv,
        ["cell_code", "grid_row", "grid_col", "kept_file", "skipped_files", "duplicate_count"],
        duplicate_rows,
    )

    province_label_map = load_province_label_map(province_label_path)
    province_per_image_csv, province_summary_csv = add_province_outputs(
        analyses=analyses,
        province_label_map=province_label_map,
        output_dir=output_dir,
        tile_area_km2=tile_area_km2,
        normalize_intersection_ratios=args.normalize_intersection_ratios,
    )

    print(f"Processed {len(unique_png_files)} unique PNG tiles from {input_dir}")
    print(f"Skipped {len(duplicate_rows)} duplicated tile keys")
    print(f"Per-image area CSV: {per_image_csv}")
    print(f"Summary area CSV: {summary_csv}")
    print(f"Duplicate tile CSV: {duplicate_csv}")
    print(f"Per-image province CSV: {province_per_image_csv}")
    print(f"Summary province CSV: {province_summary_csv}")
    print(f"Province label JSON: {province_label_path}")
    print(
        f"Assumed tile size: {args.tile_width_km} km x {args.tile_height_km} km = {tile_area_km2:.2f} km2"
    )
    if args.normalize_intersection_ratios:
        print("Intersection ratios were normalized to sum to 1.0 per tile")
    else:
        print("Intersection ratios were used as provided in the JSON file")


if __name__ == "__main__":
    main()
