import argparse
import json
from pathlib import Path

from area_calc.aggregator import ReportWriter, SummaryAggregator
from area_calc.calculator import AreaCalculator
from area_calc.config import RGB_TO_CLASS, SEASON_BY_DATE_TAG
from area_calc.geo import load_provinces
from area_calc.sources import DatasetSource, ImageSource, InferenceSource


DEFAULT_BOUNDARY = "mapbox/vietnam_adm1_7provinces_2024.geojson"
DEFAULT_BOUNDARY_METADATA = "mapbox/vietnam_adm1_7provinces_2024_metadata.json"


def _add_common_args(parser):
    parser.add_argument(
        "--boundary",
        type=str,
        default=DEFAULT_BOUNDARY,
        help="GeoJSON containing the target province boundaries.",
    )
    parser.add_argument(
        "--boundary_metadata",
        type=str,
        default=DEFAULT_BOUNDARY_METADATA,
        help="Optional metadata JSON for the boundary source.",
    )
    parser.add_argument(
        "--all_touched",
        action="store_true",
        help="Rasterize boundaries using all touched pixels (default: pixel center).",
    )


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="calculate_area",
        description=(
            "Calculate per-province landuse area by rasterizing ADM1 boundaries "
            "onto each TIF pixel grid and counting class pixels from the paired mask."
        ),
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    p_dataset = sub.add_parser(
        "dataset",
        help="Compute ground-truth area from dataset/ (*_sat.tif + *_mask.png).",
    )
    p_dataset.add_argument(
        "--dataset_dir",
        type=str,
        default="dataset",
        help="Folder containing *_sat.tif and *_mask.png pairs.",
    )
    p_dataset.add_argument(
        "--output",
        type=str,
        default="area_output/dataset_ground_truth_gadm_boundary",
    )
    _add_common_args(p_dataset)

    p_inf = sub.add_parser(
        "inference",
        help="Compute model-output area from inference TIF + PNG, split by season.",
    )
    p_inf.add_argument(
        "--inference_tif_dir",
        type=str,
        default="inference_tif/Resolution3x3",
        help="Folder of Sentinel-1 GeoTIFFs (georef source).",
    )
    p_inf.add_argument(
        "--inference_png_dir",
        type=str,
        required=True,
        help="Folder of model-inferred *_infered.png masks.",
    )
    p_inf.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output folder for CSV/JSON results.",
    )
    _add_common_args(p_inf)

    return parser


def _build_source(args) -> ImageSource:
    if args.mode == "dataset":
        return DatasetSource(args.dataset_dir)
    if args.mode == "inference":
        return InferenceSource(args.inference_tif_dir, args.inference_png_dir)
    raise ValueError(f"Unknown mode: {args.mode}")


def _build_method_metadata(args, source: ImageSource):
    boundary_metadata = None
    metadata_path = Path(args.boundary_metadata)
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as f:
            boundary_metadata = json.load(f)

    payload = {
        "mode": args.mode,
        "method": (
            "Rasterize ADM1 boundaries onto each TIF pixel grid, count class pixels "
            "from the paired mask, and convert to area using per-row geographic pixel size."
        ),
        "class_mapping": {str(rgb): name for rgb, name in RGB_TO_CLASS.items()},
        "boundary_metadata": boundary_metadata,
        "args": vars(args),
        "source": source.name,
    }
    if source.has_seasons:
        payload["season_mapping"] = SEASON_BY_DATE_TAG
    return payload


def main(argv=None):
    args = _build_parser().parse_args(argv)

    provinces = load_provinces(args.boundary)
    source = _build_source(args)
    calculator = AreaCalculator(provinces, all_touched=args.all_touched)
    aggregator = SummaryAggregator(provinces, has_seasons=source.has_seasons)

    processed = 0
    for pair in source.iter_pairs():
        tile = calculator.process(pair.tif_path, pair.mask_path)
        aggregator.add(pair, tile)
        processed += 1

    missing = source.collect_missing()
    writer = ReportWriter(Path(args.output), has_seasons=source.has_seasons)
    writer.write(aggregator, missing, _build_method_metadata(args, source))

    print(f"Mode: {args.mode}")
    print(f"Pairs processed: {processed}")
    print(f"Missing pairs: {len(missing)}")
    for cov_row in aggregator.coverage_rows():
        print(
            f"[{cov_row['bucket_label']}] images={cov_row['image_count']} "
            f"total_km2={cov_row['image_total_area_km2']:.2f} "
            f"inside_provinces_km2={cov_row['inside_any_province_area_km2']:.2f}"
        )
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
