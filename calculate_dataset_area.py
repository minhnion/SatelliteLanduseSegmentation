import argparse
import csv
import json
from collections import defaultdict
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

CLASS_NAMES = []
for class_name in RGB_TO_CLASS.values():
    if class_name not in CLASS_NAMES:
        CLASS_NAMES.append(class_name)


def slugify(name):
    return name.lower().replace(" ", "_")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Calculate ground-truth area from dataset/ using TIFF georeference and "
            "Mapbox 3km province labels as the current province standard."
        )
    )
    parser.add_argument("--dataset_dir", type=str, default="dataset")
    parser.add_argument(
        "--grid_path",
        type=str,
        default="mapbox/mapbox_grid_3km_filtered_7provinces.json",
        help="GeoJSON grid used as the 3km standard",
    )
    parser.add_argument(
        "--province_label",
        type=str,
        default="mapbox/gadm_resolution_3_province_mapbox_label.json",
        help="JSON file mapping each 3km grid cell to province_code / overlap ratios",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="area_output/dataset_ground_truth_mapbox",
        help="Folder to save CSV outputs",
    )
    parser.add_argument(
        "--normalize_intersection_ratios",
        action="store_true",
        help="Normalize province overlap ratios to sum to 1.0 for each grid cell",
    )
    return parser.parse_args()


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def bounds_intersect(bounds_a, bounds_b):
    left_a, bottom_a, right_a, top_a = bounds_a
    left_b, bottom_b, right_b, top_b = bounds_b
    return not (
        right_a <= left_b
        or right_b <= left_a
        or top_a <= bottom_b
        or top_b <= bottom_a
    )


def feature_bounds(feature):
    ring = feature["geometry"]["coordinates"][0]
    xs = [point[0] for point in ring]
    ys = [point[1] for point in ring]
    return min(xs), min(ys), max(xs), max(ys)


def load_grid_cells(grid_path):
    with grid_path.open("r", encoding="utf-8") as f:
        geojson = json.load(f)

    grid_cells = []
    for feature in geojson["features"]:
        props = feature["properties"]
        grid_cells.append(
            {
                "key": (int(props["cell_code"]), int(props["row"]), int(props["col"])),
                "bounds": feature_bounds(feature),
            }
        )
    return grid_cells


def load_province_label_map(label_path):
    with label_path.open("r", encoding="utf-8") as f:
        raw_entries = json.load(f)

    label_map = {}
    for entry in raw_entries:
        key = (int(entry["cellCode"]), int(entry["row"]), int(entry["col"]))
        if key in label_map:
            raise ValueError(f"Duplicate tile mapping found in province label file for key {key}")

        is_intersection = bool(entry.get("isIntersection", False))
        if is_intersection:
            overlap_list = entry.get("provinceOverlapList") or []
            if not overlap_list:
                raise ValueError(f"Missing provinceOverlapList for intersection tile {key}")
            allocations = [
                {"province_code": int(item["indexCode"]), "ratio": float(item["ratio"])}
                for item in overlap_list
            ]
        else:
            index_code = entry["indexCode"]
            if isinstance(index_code, list):
                if len(index_code) != 1:
                    raise ValueError(f"Expected one province code for non-intersection tile {key}, got {index_code}")
                index_code = index_code[0]
            allocations = [{"province_code": int(index_code), "ratio": 1.0}]

        label_map[key] = {
            "is_intersection": is_intersection,
            "allocations": allocations,
            "raw_ratio_sum": sum(item["ratio"] for item in allocations),
        }

    return label_map


def meters_per_degree_lat(lat_deg):
    lat = np.radians(lat_deg)
    return (
        111132.92
        - 559.82 * np.cos(2.0 * lat)
        + 1.175 * np.cos(4.0 * lat)
        - 0.0023 * np.cos(6.0 * lat)
    )


def meters_per_degree_lon(lat_deg):
    lat = np.radians(lat_deg)
    return (
        111412.84 * np.cos(lat)
        - 93.5 * np.cos(3.0 * lat)
        + 0.118 * np.cos(5.0 * lat)
    )


def build_pixel_geometry(transform, width, height):
    if transform.b != 0.0 or transform.d != 0.0:
        raise ValueError("This script expects north-up, axis-aligned GeoTIFFs")

    col_indices = np.arange(width, dtype=np.float64)
    row_indices = np.arange(height, dtype=np.float64)
    x_centers = transform.c + transform.a * (col_indices + 0.5)
    y_centers = transform.f + transform.e * (row_indices + 0.5)

    pixel_width_deg = abs(transform.a)
    pixel_height_deg = abs(transform.e)
    row_pixel_area_m2 = (
        pixel_width_deg
        * meters_per_degree_lon(y_centers)
        * pixel_height_deg
        * meters_per_degree_lat(y_centers)
    )

    return x_centers, y_centers, row_pixel_area_m2


def load_mask(mask_path, target_width, target_height):
    with Image.open(mask_path).convert("RGB") as img:
        if img.size != (target_width, target_height):
            img = img.resize((target_width, target_height), resample=Image.NEAREST)
        return np.array(img, dtype=np.uint8)


def build_class_masks(mask_rgb):
    class_masks = {}
    covered = np.zeros(mask_rgb.shape[:2], dtype=bool)
    for class_name in CLASS_NAMES:
        mask = np.zeros(mask_rgb.shape[:2], dtype=bool)
        for rgb, mapped_class in RGB_TO_CLASS.items():
            if mapped_class == class_name:
                mask |= np.all(mask_rgb == rgb, axis=-1)
        class_masks[class_name] = mask
        covered |= mask
    unknown_mask = ~covered
    return class_masks, unknown_mask


def area_for_boolean_mask(mask_bool, row_pixel_area_m2):
    counts_per_row = mask_bool.sum(axis=1).astype(np.float64)
    return float(np.dot(counts_per_row, row_pixel_area_m2))


def window_indices(x_centers, y_centers, bounds):
    left, bottom, right, top = bounds
    col_mask = (x_centers >= left) & (x_centers < right)
    row_mask = (y_centers >= bottom) & (y_centers < top)
    if not row_mask.any() or not col_mask.any():
        return None

    row_indices = np.flatnonzero(row_mask)
    col_indices = np.flatnonzero(col_mask)
    return (
        int(row_indices[0]),
        int(row_indices[-1]) + 1,
        int(col_indices[0]),
        int(col_indices[-1]) + 1,
    )


def image_total_stats(class_masks, unknown_mask, row_pixel_area_m2, total_image_area_m2):
    stats = {}
    known_area_m2 = 0.0
    known_pixels = 0
    for class_name in CLASS_NAMES:
        pixels = int(class_masks[class_name].sum())
        area_m2 = area_for_boolean_mask(class_masks[class_name], row_pixel_area_m2)
        stats[class_name] = {"pixels": pixels, "area_m2": area_m2}
        known_pixels += pixels
        known_area_m2 += area_m2

    unknown_pixels = int(unknown_mask.sum())
    unknown_area_m2 = area_for_boolean_mask(unknown_mask, row_pixel_area_m2) if unknown_pixels else 0.0
    stats["unknown"] = {"pixels": unknown_pixels, "area_m2": unknown_area_m2}
    stats["known_total_area_m2"] = known_area_m2
    stats["known_total_pixels"] = known_pixels
    stats["total_image_area_m2"] = total_image_area_m2
    return stats


def build_empty_class_row():
    row = {}
    for class_name in CLASS_NAMES:
        key = slugify(class_name)
        row[f"{key}_pixels"] = 0.0
        row[f"{key}_area_km2"] = 0.0
        row[f"{key}_area_ha"] = 0.0
    return row


def build_empty_class_row_equiv():
    row = {}
    for class_name in CLASS_NAMES:
        key = slugify(class_name)
        row[f"{key}_pixels_equiv"] = 0.0
        row[f"{key}_area_km2"] = 0.0
        row[f"{key}_area_ha"] = 0.0
    return row


def add_class_values(target_row, prefix_mode, class_name, pixels_value, area_m2_value):
    key = slugify(class_name)
    if prefix_mode == "base":
        target_row[f"{key}_pixels"] += pixels_value
    else:
        target_row[f"{key}_pixels_equiv"] += pixels_value
    target_row[f"{key}_area_km2"] += area_m2_value / 1_000_000.0
    target_row[f"{key}_area_ha"] += area_m2_value / 10_000.0


def aggregate_summary_rows(per_image_rows):
    summary_rows = []
    total_image_area_km2 = sum(float(row["image_area_km2"]) for row in per_image_rows)
    total_known_area_km2 = sum(float(row["known_area_km2"]) for row in per_image_rows)
    total_known_pixels = sum(int(row["known_pixels"]) for row in per_image_rows)

    for class_name in CLASS_NAMES:
        key = slugify(class_name)
        total_pixels = sum(float(row[f"{key}_pixels"]) for row in per_image_rows)
        total_area_km2 = sum(float(row[f"{key}_area_km2"]) for row in per_image_rows)
        summary_rows.append(
            {
                "class_name": class_name,
                "image_count": len(per_image_rows),
                "total_pixels": total_pixels,
                "pixel_ratio_over_known": (total_pixels / total_known_pixels) if total_known_pixels else 0.0,
                "total_area_km2": total_area_km2,
                "total_area_ha": total_area_km2 * 100.0,
                "area_ratio_over_known": (total_area_km2 / total_known_area_km2) if total_known_area_km2 else 0.0,
                "area_ratio_over_image_extent": (total_area_km2 / total_image_area_km2) if total_image_area_km2 else 0.0,
            }
        )

    return summary_rows


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    grid_path = Path(args.grid_path)
    province_label_path = Path(args.province_label)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")
    if not grid_path.exists():
        raise FileNotFoundError(f"Grid GeoJSON not found: {grid_path}")
    if not province_label_path.exists():
        raise FileNotFoundError(f"Province label JSON not found: {province_label_path}")

    grid_cells = load_grid_cells(grid_path)
    province_label_map = load_province_label_map(province_label_path)
    grid_keys = {cell["key"] for cell in grid_cells}
    label_keys = set(province_label_map)
    if grid_keys != label_keys:
        raise ValueError(
            f"Grid / province-label mismatch: grid_only={len(grid_keys - label_keys)}, "
            f"label_only={len(label_keys - grid_keys)}"
        )

    image_paths = sorted(dataset_dir.glob("*_sat.tif"))
    if not image_paths:
        raise FileNotFoundError(f"No *_sat.tif files found in {dataset_dir}")

    per_image_rows = []
    per_image_province_rows = []
    outside_coverage_rows = []
    province_summary = {}
    coverage_summary = {
        "dataset_image_count": 0,
        "images_with_mapbox_overlap": 0,
        "images_without_mapbox_overlap": 0,
        "dataset_total_area_km2": 0.0,
        "dataset_known_area_km2": 0.0,
        "dataset_unknown_area_km2": 0.0,
        "mapbox_overlap_area_km2": 0.0,
        "mapbox_outside_area_km2": 0.0,
        "province_allocated_area_km2": 0.0,
        "inside_grid_unallocated_area_km2": 0.0,
    }

    missing_mask_files = []

    for image_path in image_paths:
        mask_path = image_path.with_name(image_path.name.replace("_sat.tif", "_mask.png"))
        if not mask_path.exists():
            missing_mask_files.append(mask_path.name)
            continue

        with rasterio.open(image_path) as src:
            width = src.width
            height = src.height
            transform = src.transform
            bounds = (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
            crs = str(src.crs)
            band_count = src.count
            band_descriptions = src.descriptions

        x_centers, y_centers, row_pixel_area_m2 = build_pixel_geometry(transform, width, height)
        image_area_m2 = float(row_pixel_area_m2.sum() * width)
        mask_rgb = load_mask(mask_path, width, height)
        class_masks, unknown_mask = build_class_masks(mask_rgb)
        image_stats = image_total_stats(class_masks, unknown_mask, row_pixel_area_m2, image_area_m2)

        per_image_row = {
            "file_name": image_path.name,
            "mask_name": mask_path.name,
            "crs": crs,
            "band_count": band_count,
            "band_descriptions": band_descriptions,
            "width_px": width,
            "height_px": height,
            "left": bounds[0],
            "bottom": bounds[1],
            "right": bounds[2],
            "top": bounds[3],
            "image_area_km2": image_area_m2 / 1_000_000.0,
            "image_area_ha": image_area_m2 / 10_000.0,
            "known_area_km2": image_stats["known_total_area_m2"] / 1_000_000.0,
            "known_area_ha": image_stats["known_total_area_m2"] / 10_000.0,
            "unknown_area_km2": image_stats["unknown"]["area_m2"] / 1_000_000.0,
            "unknown_area_ha": image_stats["unknown"]["area_m2"] / 10_000.0,
            "known_pixels": image_stats["known_total_pixels"],
            "unknown_pixels": image_stats["unknown"]["pixels"],
            "mapbox_overlap_area_km2": 0.0,
            "mapbox_overlap_area_ha": 0.0,
            "outside_mapbox_area_km2": 0.0,
            "outside_mapbox_area_ha": 0.0,
            "province_allocated_area_km2": 0.0,
            "province_allocated_area_ha": 0.0,
            "inside_grid_unallocated_area_km2": 0.0,
            "inside_grid_unallocated_area_ha": 0.0,
            "overlapping_grid_cell_count": 0,
        }
        per_image_row.update(build_empty_class_row())

        for class_name in CLASS_NAMES:
            add_class_values(
                per_image_row,
                "base",
                class_name,
                float(image_stats[class_name]["pixels"]),
                image_stats[class_name]["area_m2"],
            )

        per_image_province_acc = {}
        mapbox_overlap_area_m2 = 0.0
        province_allocated_area_m2 = 0.0
        overlap_cell_count = 0

        for grid_cell in grid_cells:
            if not bounds_intersect(bounds, grid_cell["bounds"]):
                continue

            indices = window_indices(x_centers, y_centers, grid_cell["bounds"])
            if indices is None:
                continue

            row_start, row_end, col_start, col_end = indices
            row_slice = slice(row_start, row_end)
            col_slice = slice(col_start, col_end)
            overlap_cell_count += 1
            overlap_area_m2 = float(row_pixel_area_m2[row_slice].sum() * (col_end - col_start))
            mapbox_overlap_area_m2 += overlap_area_m2

            class_window_stats = {}
            for class_name in CLASS_NAMES:
                window_mask = class_masks[class_name][row_slice, col_slice]
                window_pixels = float(window_mask.sum())
                window_area_m2 = area_for_boolean_mask(window_mask, row_pixel_area_m2[row_slice])
                class_window_stats[class_name] = {
                    "pixels": window_pixels,
                    "area_m2": window_area_m2,
                }

            province_mapping = province_label_map[grid_cell["key"]]
            raw_ratio_sum = province_mapping["raw_ratio_sum"]
            for allocation in province_mapping["allocations"]:
                raw_ratio = allocation["ratio"]
                if args.normalize_intersection_ratios and raw_ratio_sum > 0.0:
                    effective_ratio = raw_ratio / raw_ratio_sum
                else:
                    effective_ratio = raw_ratio

                province_code = allocation["province_code"]
                province_row = per_image_province_acc.setdefault(
                    province_code,
                    {
                        "file_name": image_path.name,
                        "mask_name": mask_path.name,
                        "province_code": province_code,
                        "crs": crs,
                        "left": bounds[0],
                        "bottom": bounds[1],
                        "right": bounds[2],
                        "top": bounds[3],
                        "grid_cells_hit": 0,
                        "allocated_area_km2": 0.0,
                        "allocated_area_ha": 0.0,
                    },
                )
                if province_code not in province_summary:
                    province_summary[province_code] = {
                        "province_code": province_code,
                        "image_count": 0,
                        "grid_cell_hits": 0,
                        "allocated_area_km2": 0.0,
                        "allocated_area_ha": 0.0,
                    }
                    province_summary[province_code].update(build_empty_class_row_equiv())

                province_row["grid_cells_hit"] += 1
                province_summary[province_code]["grid_cell_hits"] += 1

                weighted_overlap_m2 = overlap_area_m2 * effective_ratio
                province_row["allocated_area_km2"] += weighted_overlap_m2 / 1_000_000.0
                province_row["allocated_area_ha"] += weighted_overlap_m2 / 10_000.0
                province_summary[province_code]["allocated_area_km2"] += weighted_overlap_m2 / 1_000_000.0
                province_summary[province_code]["allocated_area_ha"] += weighted_overlap_m2 / 10_000.0
                province_allocated_area_m2 += weighted_overlap_m2

                for class_name in CLASS_NAMES:
                    weighted_pixels = class_window_stats[class_name]["pixels"] * effective_ratio
                    weighted_area_m2 = class_window_stats[class_name]["area_m2"] * effective_ratio
                    province_row.setdefault(f"{slugify(class_name)}_pixels_equiv", 0.0)
                    province_row.setdefault(f"{slugify(class_name)}_area_km2", 0.0)
                    province_row.setdefault(f"{slugify(class_name)}_area_ha", 0.0)
                    add_class_values(province_row, "equiv", class_name, weighted_pixels, weighted_area_m2)
                    add_class_values(province_summary[province_code], "equiv", class_name, weighted_pixels, weighted_area_m2)

        per_image_row["mapbox_overlap_area_km2"] = mapbox_overlap_area_m2 / 1_000_000.0
        per_image_row["mapbox_overlap_area_ha"] = mapbox_overlap_area_m2 / 10_000.0
        per_image_row["outside_mapbox_area_km2"] = max(0.0, image_area_m2 - mapbox_overlap_area_m2) / 1_000_000.0
        per_image_row["outside_mapbox_area_ha"] = max(0.0, image_area_m2 - mapbox_overlap_area_m2) / 10_000.0
        per_image_row["province_allocated_area_km2"] = province_allocated_area_m2 / 1_000_000.0
        per_image_row["province_allocated_area_ha"] = province_allocated_area_m2 / 10_000.0
        per_image_row["inside_grid_unallocated_area_km2"] = max(0.0, mapbox_overlap_area_m2 - province_allocated_area_m2) / 1_000_000.0
        per_image_row["inside_grid_unallocated_area_ha"] = max(0.0, mapbox_overlap_area_m2 - province_allocated_area_m2) / 10_000.0
        per_image_row["overlapping_grid_cell_count"] = overlap_cell_count
        per_image_rows.append(per_image_row)

        coverage_summary["dataset_image_count"] += 1
        coverage_summary["dataset_total_area_km2"] += image_area_m2 / 1_000_000.0
        coverage_summary["dataset_known_area_km2"] += image_stats["known_total_area_m2"] / 1_000_000.0
        coverage_summary["dataset_unknown_area_km2"] += image_stats["unknown"]["area_m2"] / 1_000_000.0
        coverage_summary["mapbox_overlap_area_km2"] += mapbox_overlap_area_m2 / 1_000_000.0
        coverage_summary["mapbox_outside_area_km2"] += max(0.0, image_area_m2 - mapbox_overlap_area_m2) / 1_000_000.0
        coverage_summary["province_allocated_area_km2"] += province_allocated_area_m2 / 1_000_000.0
        coverage_summary["inside_grid_unallocated_area_km2"] += max(0.0, mapbox_overlap_area_m2 - province_allocated_area_m2) / 1_000_000.0

        if overlap_cell_count == 0:
            coverage_summary["images_without_mapbox_overlap"] += 1
            outside_coverage_rows.append(
                {
                    "file_name": image_path.name,
                    "mask_name": mask_path.name,
                    "left": bounds[0],
                    "bottom": bounds[1],
                    "right": bounds[2],
                    "top": bounds[3],
                    "image_area_km2": image_area_m2 / 1_000_000.0,
                    "image_area_ha": image_area_m2 / 10_000.0,
                }
            )
        else:
            coverage_summary["images_with_mapbox_overlap"] += 1

        for province_code, province_row in per_image_province_acc.items():
            province_summary[province_code]["image_count"] += 1
            per_image_province_rows.append(province_row)

    if missing_mask_files:
        raise FileNotFoundError(f"Missing masks for some TIFFs, for example: {missing_mask_files[:5]}")

    for province_code in sorted(province_summary):
        province_summary[province_code]["total_area_km2"] = sum(
            province_summary[province_code][f"{slugify(class_name)}_area_km2"] for class_name in CLASS_NAMES
        )
        province_summary[province_code]["total_area_ha"] = province_summary[province_code]["total_area_km2"] * 100.0

    per_image_fields = [
        "file_name",
        "mask_name",
        "crs",
        "band_count",
        "band_descriptions",
        "width_px",
        "height_px",
        "left",
        "bottom",
        "right",
        "top",
        "image_area_km2",
        "image_area_ha",
        "known_area_km2",
        "known_area_ha",
        "unknown_area_km2",
        "unknown_area_ha",
        "known_pixels",
        "unknown_pixels",
        "mapbox_overlap_area_km2",
        "mapbox_overlap_area_ha",
        "outside_mapbox_area_km2",
        "outside_mapbox_area_ha",
        "province_allocated_area_km2",
        "province_allocated_area_ha",
        "inside_grid_unallocated_area_km2",
        "inside_grid_unallocated_area_ha",
        "overlapping_grid_cell_count",
    ]
    for class_name in CLASS_NAMES:
        key = slugify(class_name)
        per_image_fields.extend([f"{key}_pixels", f"{key}_area_km2", f"{key}_area_ha"])

    per_image_province_fields = [
        "file_name",
        "mask_name",
        "province_code",
        "crs",
        "left",
        "bottom",
        "right",
        "top",
        "grid_cells_hit",
        "allocated_area_km2",
        "allocated_area_ha",
    ]
    for class_name in CLASS_NAMES:
        key = slugify(class_name)
        per_image_province_fields.extend([f"{key}_pixels_equiv", f"{key}_area_km2", f"{key}_area_ha"])

    province_summary_fields = [
        "province_code",
        "image_count",
        "grid_cell_hits",
        "allocated_area_km2",
        "allocated_area_ha",
        "total_area_km2",
        "total_area_ha",
    ]
    for class_name in CLASS_NAMES:
        key = slugify(class_name)
        province_summary_fields.extend([f"{key}_pixels_equiv", f"{key}_area_km2", f"{key}_area_ha"])

    summary_fields = [
        "class_name",
        "image_count",
        "total_pixels",
        "pixel_ratio_over_known",
        "total_area_km2",
        "total_area_ha",
        "area_ratio_over_known",
        "area_ratio_over_image_extent",
    ]

    coverage_summary["dataset_total_area_ha"] = coverage_summary["dataset_total_area_km2"] * 100.0
    coverage_summary["dataset_known_area_ha"] = coverage_summary["dataset_known_area_km2"] * 100.0
    coverage_summary["dataset_unknown_area_ha"] = coverage_summary["dataset_unknown_area_km2"] * 100.0
    coverage_summary["mapbox_overlap_area_ha"] = coverage_summary["mapbox_overlap_area_km2"] * 100.0
    coverage_summary["mapbox_outside_area_ha"] = coverage_summary["mapbox_outside_area_km2"] * 100.0
    coverage_summary["province_allocated_area_ha"] = coverage_summary["province_allocated_area_km2"] * 100.0
    coverage_summary["inside_grid_unallocated_area_ha"] = coverage_summary["inside_grid_unallocated_area_km2"] * 100.0

    per_image_csv = output_dir / "per_image_area.csv"
    summary_csv = output_dir / "summary_area.csv"
    per_image_province_csv = output_dir / "per_image_province_area.csv"
    summary_province_csv = output_dir / "summary_province_area.csv"
    outside_coverage_csv = output_dir / "outside_mapbox_coverage.csv"
    coverage_summary_csv = output_dir / "coverage_summary.csv"

    write_csv(per_image_csv, per_image_fields, per_image_rows)
    write_csv(summary_csv, summary_fields, aggregate_summary_rows(per_image_rows))
    write_csv(per_image_province_csv, per_image_province_fields, per_image_province_rows)
    write_csv(
        summary_province_csv,
        province_summary_fields,
        [province_summary[code] for code in sorted(province_summary)],
    )
    write_csv(
        outside_coverage_csv,
        ["file_name", "mask_name", "left", "bottom", "right", "top", "image_area_km2", "image_area_ha"],
        outside_coverage_rows,
    )
    write_csv(coverage_summary_csv, list(coverage_summary.keys()), [coverage_summary])

    print(f"Processed {len(per_image_rows)} dataset TIFF images from {dataset_dir}")
    print(f"Images with Mapbox overlap: {coverage_summary['images_with_mapbox_overlap']}")
    print(f"Images without Mapbox overlap: {coverage_summary['images_without_mapbox_overlap']}")
    print(f"Per-image area CSV: {per_image_csv}")
    print(f"Summary area CSV: {summary_csv}")
    print(f"Per-image province CSV: {per_image_province_csv}")
    print(f"Summary province CSV: {summary_province_csv}")
    print(f"Outside-coverage CSV: {outside_coverage_csv}")
    print(f"Coverage summary CSV: {coverage_summary_csv}")
    if args.normalize_intersection_ratios:
        print("Province overlap ratios were normalized to sum to 1.0 for each grid cell")
    else:
        print("Province overlap ratios were used exactly as provided in the JSON file")


if __name__ == "__main__":
    main()
