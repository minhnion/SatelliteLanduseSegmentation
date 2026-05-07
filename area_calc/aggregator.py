import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

from area_calc.calculator import TileResult
from area_calc.config import CLASS_NAMES, SEASON_BY_DATE_TAG, slugify
from area_calc.geo import BOUNDARY_AREA_METHOD, Province
from area_calc.sources import ImagePair, MissingPair


def _class_fieldnames():
    fields = []
    for class_name in CLASS_NAMES:
        key = slugify(class_name)
        fields.extend([f"{key}_pixels", f"{key}_area_km2", f"{key}_area_ha"])
    return fields


def _province_summary_fields():
    return [
        "province_code",
        "province_name",
        "source_name",
        "boundary_area_km2",
        "boundary_area_ha",
        "covered_area_km2",
        "covered_area_ha",
        "coverage_ratio_over_boundary",
        "uncovered_area_km2",
        "uncovered_area_ha",
        "known_labeled_area_km2",
        "unknown_color_area_km2",
        "image_count",
        "province_pixel_count",
        "unknown_color_pixels",
    ] + _class_fieldnames()


class ProvinceAccumulator:
    def __init__(self, province: Province):
        self.province = province
        self.image_count = 0
        self.province_pixels = 0
        self.covered_m2 = 0.0
        self.known_m2 = 0.0
        self.unknown_m2 = 0.0
        self.unknown_pixels = 0
        self.class_pixels: Dict[str, float] = {n: 0.0 for n in CLASS_NAMES}
        self.class_area_m2: Dict[str, float] = {n: 0.0 for n in CLASS_NAMES}

    def add(self, p_result):
        self.image_count += 1
        self.province_pixels += p_result.province_pixels
        self.covered_m2 += p_result.province_area_m2
        self.known_m2 += p_result.known_area_m2
        self.unknown_m2 += p_result.unknown_area_m2
        self.unknown_pixels += p_result.unknown_pixels
        for class_name in CLASS_NAMES:
            self.class_pixels[class_name] += p_result.class_pixels[class_name]
            self.class_area_m2[class_name] += p_result.class_area_m2[class_name]

    def to_row(self):
        boundary_km2 = self.province.boundary_area_km2
        covered_km2 = self.covered_m2 / 1_000_000.0
        coverage_ratio = covered_km2 / boundary_km2 if boundary_km2 else 0.0
        uncovered_km2 = max(0.0, boundary_km2 - covered_km2)
        row = {
            "province_code": self.province.code,
            "province_name": self.province.name,
            "source_name": self.province.source_name,
            "boundary_area_km2": boundary_km2,
            "boundary_area_ha": self.province.boundary_area_ha,
            "covered_area_km2": covered_km2,
            "covered_area_ha": self.covered_m2 / 10_000.0,
            "coverage_ratio_over_boundary": coverage_ratio,
            "uncovered_area_km2": uncovered_km2,
            "uncovered_area_ha": uncovered_km2 * 100.0,
            "known_labeled_area_km2": self.known_m2 / 1_000_000.0,
            "unknown_color_area_km2": self.unknown_m2 / 1_000_000.0,
            "image_count": self.image_count,
            "province_pixel_count": self.province_pixels,
            "unknown_color_pixels": self.unknown_pixels,
        }
        for class_name in CLASS_NAMES:
            key = slugify(class_name)
            row[f"{key}_pixels"] = self.class_pixels[class_name]
            row[f"{key}_area_km2"] = self.class_area_m2[class_name] / 1_000_000.0
            row[f"{key}_area_ha"] = self.class_area_m2[class_name] / 10_000.0
        return row


class CoverageAccumulator:
    def __init__(self, key: str, label: str):
        self.key = key
        self.label = label
        self.image_count = 0
        self.image_total_m2 = 0.0
        self.inside_m2 = 0.0
        self.outside_m2 = 0.0
        self.overlap_m2 = 0.0
        self.known_inside_m2 = 0.0
        self.unknown_inside_m2 = 0.0

    def add(self, tile: TileResult):
        self.image_count += 1
        self.image_total_m2 += tile.image_area_m2
        self.inside_m2 += tile.inside_any_province_m2
        self.outside_m2 += tile.outside_provinces_m2
        self.overlap_m2 += tile.overlap_m2
        self.known_inside_m2 += sum(p.known_area_m2 for p in tile.province_results)
        self.unknown_inside_m2 += sum(p.unknown_area_m2 for p in tile.province_results)

    def to_row(self):
        return {
            "bucket_key": self.key,
            "bucket_label": self.label,
            "image_count": self.image_count,
            "image_total_area_km2": self.image_total_m2 / 1_000_000.0,
            "inside_any_province_area_km2": self.inside_m2 / 1_000_000.0,
            "outside_provinces_area_km2": self.outside_m2 / 1_000_000.0,
            "province_overlap_area_km2": self.overlap_m2 / 1_000_000.0,
            "known_labeled_inside_provinces_area_km2": self.known_inside_m2 / 1_000_000.0,
            "unknown_color_inside_provinces_area_km2": self.unknown_inside_m2 / 1_000_000.0,
        }


class SummaryAggregator:
    def __init__(self, provinces: List[Province], has_seasons: bool):
        self.provinces = provinces
        self.has_seasons = has_seasons
        self._buckets: Dict[str, Dict[int, ProvinceAccumulator]] = {}
        self._coverage: Dict[str, CoverageAccumulator] = {}
        self._combined: Dict[int, ProvinceAccumulator] = {}
        self.per_image_rows: List[dict] = []
        self.per_image_province_rows: List[dict] = []

        if has_seasons:
            for season in SEASON_BY_DATE_TAG.values():
                self._buckets[season["key"]] = {p.code: ProvinceAccumulator(p) for p in provinces}
                self._coverage[season["key"]] = CoverageAccumulator(season["key"], season["label"])
            self._combined = {p.code: ProvinceAccumulator(p) for p in provinces}
        else:
            self._buckets["all"] = {p.code: ProvinceAccumulator(p) for p in provinces}
            self._coverage["all"] = CoverageAccumulator("all", "All")

    def add(self, pair: ImagePair, tile: TileResult):
        bucket_key = pair.season_key if self.has_seasons else "all"
        bucket = self._buckets[bucket_key]
        self._coverage[bucket_key].add(tile)

        for p_result in tile.province_results:
            bucket[p_result.province_code].add(p_result)
            if self.has_seasons:
                self._combined[p_result.province_code].add(p_result)

        self._record_per_image(pair, tile)

    def _record_per_image(self, pair: ImagePair, tile: TileResult):
        image_row = {
            "stem": pair.stem,
            "tif_name": pair.tif_path.name,
            "mask_name": pair.mask_path.name,
            "crs": tile.crs,
            "band_count": tile.band_count,
            "width_px": tile.width,
            "height_px": tile.height,
            "left": tile.bounds[0],
            "bottom": tile.bounds[1],
            "right": tile.bounds[2],
            "top": tile.bounds[3],
            "image_area_km2": tile.image_area_m2 / 1_000_000.0,
            "inside_any_province_area_km2": tile.inside_any_province_m2 / 1_000_000.0,
            "outside_provinces_area_km2": tile.outside_provinces_m2 / 1_000_000.0,
            "province_overlap_area_km2": tile.overlap_m2 / 1_000_000.0,
            "overlapping_province_count": len(tile.province_results),
        }
        if self.has_seasons:
            image_row.update({
                "date_tag": pair.date_tag,
                "season_key": pair.season_key,
                "season_label": pair.season_label,
            })
        self.per_image_rows.append(image_row)

        for p_result in tile.province_results:
            province = next(p for p in self.provinces if p.code == p_result.province_code)
            row = {
                "stem": pair.stem,
                "tif_name": pair.tif_path.name,
                "mask_name": pair.mask_path.name,
                "province_code": province.code,
                "province_name": province.name,
                "source_name": province.source_name,
                "province_pixels": p_result.province_pixels,
                "province_area_km2": p_result.province_area_m2 / 1_000_000.0,
                "known_labeled_area_km2": p_result.known_area_m2 / 1_000_000.0,
                "unknown_color_pixels": p_result.unknown_pixels,
                "unknown_color_area_km2": p_result.unknown_area_m2 / 1_000_000.0,
            }
            for class_name in CLASS_NAMES:
                key = slugify(class_name)
                row[f"{key}_pixels"] = p_result.class_pixels[class_name]
                row[f"{key}_area_km2"] = p_result.class_area_m2[class_name] / 1_000_000.0
                row[f"{key}_area_ha"] = p_result.class_area_m2[class_name] / 10_000.0
            if self.has_seasons:
                row.update({
                    "date_tag": pair.date_tag,
                    "season_key": pair.season_key,
                    "season_label": pair.season_label,
                })
            self.per_image_province_rows.append(row)

    def coverage_rows(self):
        return [acc.to_row() for acc in self._coverage.values()]

    def bucket_summary_rows(self, bucket_key: str):
        return [self._buckets[bucket_key][p.code].to_row() for p in self.provinces]

    def combined_summary_rows(self):
        return [self._combined[p.code].to_row() for p in self.provinces]

    def by_season_rows(self):
        rows = []
        for season in SEASON_BY_DATE_TAG.values():
            for province in self.provinces:
                row = {
                    "season_key": season["key"],
                    "season_label": season["label"],
                    **self._buckets[season["key"]][province.code].to_row(),
                }
                rows.append(row)
        return rows


def _write_csv(path: Path, rows: List[dict], fieldnames: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)
        f.write("\n")


class ReportWriter:
    def __init__(self, output_dir: Path, has_seasons: bool):
        self.output_dir = Path(output_dir)
        self.has_seasons = has_seasons
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _per_image_fields(self):
        base = [
            "stem",
            "tif_name",
            "mask_name",
            "crs",
            "band_count",
            "width_px",
            "height_px",
            "left",
            "bottom",
            "right",
            "top",
            "image_area_km2",
            "inside_any_province_area_km2",
            "outside_provinces_area_km2",
            "province_overlap_area_km2",
            "overlapping_province_count",
        ]
        if self.has_seasons:
            base += ["date_tag", "season_key", "season_label"]
        return base

    def _per_image_province_fields(self):
        base = [
            "stem",
            "tif_name",
            "mask_name",
            "province_code",
            "province_name",
            "source_name",
            "province_pixels",
            "province_area_km2",
            "known_labeled_area_km2",
            "unknown_color_pixels",
            "unknown_color_area_km2",
        ]
        if self.has_seasons:
            base += ["date_tag", "season_key", "season_label"]
        return base + _class_fieldnames()

    def _coverage_fields(self):
        return [
            "bucket_key",
            "bucket_label",
            "image_count",
            "image_total_area_km2",
            "inside_any_province_area_km2",
            "outside_provinces_area_km2",
            "province_overlap_area_km2",
            "known_labeled_inside_provinces_area_km2",
            "unknown_color_inside_provinces_area_km2",
        ]

    def write(
        self,
        agg: SummaryAggregator,
        missing: List[MissingPair],
        method_metadata: dict,
    ):
        _write_csv(
            self.output_dir / "per_image_area.csv",
            agg.per_image_rows,
            self._per_image_fields(),
        )
        _write_csv(
            self.output_dir / "per_image_province_area.csv",
            agg.per_image_province_rows,
            self._per_image_province_fields(),
        )
        _write_csv(
            self.output_dir / "coverage_summary.csv",
            agg.coverage_rows(),
            self._coverage_fields(),
        )
        _write_csv(
            self.output_dir / "missing_pairs.csv",
            [{"mask_name": m.mask_name, "expected_tif": m.expected_tif} for m in missing],
            ["mask_name", "expected_tif"],
        )

        if self.has_seasons:
            _write_csv(
                self.output_dir / "summary_province_combined.csv",
                agg.combined_summary_rows(),
                _province_summary_fields(),
            )
            _write_csv(
                self.output_dir / "summary_province_by_season.csv",
                agg.by_season_rows(),
                ["season_key", "season_label"] + _province_summary_fields(),
            )
            for season in SEASON_BY_DATE_TAG.values():
                _write_csv(
                    self.output_dir / f"summary_province_{season['key']}.csv",
                    agg.bucket_summary_rows(season["key"]),
                    _province_summary_fields(),
                )
        else:
            _write_csv(
                self.output_dir / "summary_province_area.csv",
                agg.bucket_summary_rows("all"),
                _province_summary_fields(),
            )

        _write_json(
            self.output_dir / "method_metadata.json",
            {
                "boundary_area_method": BOUNDARY_AREA_METHOD,
                **method_metadata,
            },
        )
