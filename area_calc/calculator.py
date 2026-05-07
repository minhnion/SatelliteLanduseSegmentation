from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import rasterio

from area_calc.config import CLASS_NAMES
from area_calc.geo import Province, build_row_pixel_area_m2, bounds_intersect
from area_calc.masks import area_for_mask, build_class_masks, load_mask_rgb


@dataclass
class ProvinceTileResult:
    province_code: int
    province_pixels: int
    province_area_m2: float
    known_area_m2: float
    unknown_pixels: int
    unknown_area_m2: float
    class_pixels: Dict[str, int]
    class_area_m2: Dict[str, float]


@dataclass
class TileResult:
    width: int
    height: int
    crs: str
    band_count: int
    bounds: tuple
    image_area_m2: float
    inside_any_province_m2: float
    overlap_m2: float
    outside_provinces_m2: float
    province_results: List[ProvinceTileResult] = field(default_factory=list)


class AreaCalculator:
    def __init__(self, provinces: List[Province], all_touched: bool = False):
        self.provinces = provinces
        self.all_touched = all_touched

    def process(self, tif_path, mask_path) -> TileResult:
        with rasterio.open(tif_path) as src:
            width = src.width
            height = src.height
            transform = src.transform
            crs = src.crs
            bounds = (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
            band_count = src.count

        row_pixel_area = build_row_pixel_area_m2(transform, height, crs)
        image_area_m2 = float(row_pixel_area.sum() * width)

        mask_rgb = load_mask_rgb(mask_path, width, height)
        class_masks, unknown_mask = build_class_masks(mask_rgb)

        province_mask_sum = np.zeros((height, width), dtype=np.uint8)
        province_results: List[ProvinceTileResult] = []

        for province in self.provinces:
            if not bounds_intersect(bounds, province.bounds):
                continue

            province_mask = province.rasterize_to_grid(
                transform, width, height, all_touched=self.all_touched
            )
            province_pixels = int(province_mask.sum())
            if province_pixels == 0:
                continue

            province_area_m2 = area_for_mask(province_mask, row_pixel_area)
            unknown_pixels = int((province_mask & unknown_mask).sum())
            unknown_area_m2 = area_for_mask(province_mask & unknown_mask, row_pixel_area)
            known_area_m2 = max(0.0, province_area_m2 - unknown_area_m2)

            class_pixels: Dict[str, int] = {}
            class_area_m2: Dict[str, float] = {}
            for class_name in CLASS_NAMES:
                cls_mask = province_mask & class_masks[class_name]
                class_pixels[class_name] = int(cls_mask.sum())
                class_area_m2[class_name] = area_for_mask(cls_mask, row_pixel_area)

            province_mask_sum += province_mask.astype(np.uint8)
            province_results.append(
                ProvinceTileResult(
                    province_code=province.code,
                    province_pixels=province_pixels,
                    province_area_m2=province_area_m2,
                    known_area_m2=known_area_m2,
                    unknown_pixels=unknown_pixels,
                    unknown_area_m2=unknown_area_m2,
                    class_pixels=class_pixels,
                    class_area_m2=class_area_m2,
                )
            )

        inside_any = (province_mask_sum > 0)
        overlap = (province_mask_sum > 1)
        inside_m2 = area_for_mask(inside_any, row_pixel_area)
        overlap_m2 = area_for_mask(overlap, row_pixel_area)
        outside_m2 = max(0.0, image_area_m2 - inside_m2)

        return TileResult(
            width=width,
            height=height,
            crs=str(crs),
            band_count=band_count,
            bounds=bounds,
            image_area_m2=image_area_m2,
            inside_any_province_m2=inside_m2,
            overlap_m2=overlap_m2,
            outside_provinces_m2=outside_m2,
            province_results=province_results,
        )
