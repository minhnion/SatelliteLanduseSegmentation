import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

import numpy as np
from rasterio.features import rasterize

from area_calc.config import AUTHALIC_RADIUS_M

try:
    from pyproj import Geod
    _GEOD = Geod(ellps="WGS84")
    BOUNDARY_AREA_METHOD = "pyproj.Geod"
except Exception:
    _GEOD = None
    BOUNDARY_AREA_METHOD = "spherical_authalic_radius"


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


def build_row_pixel_area_m2(transform, height, crs):
    if transform.b != 0.0 or transform.d != 0.0:
        raise ValueError("Only north-up, axis-aligned GeoTIFFs are supported")

    if crs and crs.is_projected:
        return np.full(height, abs(transform.a * transform.e), dtype=np.float64)

    rows = np.arange(height, dtype=np.float64)
    y_centers = transform.f + transform.e * (rows + 0.5)
    pixel_w = abs(transform.a)
    pixel_h = abs(transform.e)
    return (
        pixel_w * meters_per_degree_lon(y_centers)
        * pixel_h * meters_per_degree_lat(y_centers)
    )


def geometry_bounds(geometry):
    xs = []
    ys = []

    def walk(coords):
        if isinstance(coords[0], (int, float)):
            xs.append(float(coords[0]))
            ys.append(float(coords[1]))
            return
        for item in coords:
            walk(item)

    walk(geometry["coordinates"])
    return min(xs), min(ys), max(xs), max(ys)


def bounds_intersect(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _ring_area_m2(coords):
    if _GEOD:
        xs = [pt[0] for pt in coords]
        ys = [pt[1] for pt in coords]
        area, _ = _GEOD.polygon_area_perimeter(xs, ys)
        return abs(float(area))

    closed = coords if coords[0] == coords[-1] else list(coords) + [coords[0]]
    total = 0.0
    for a, b in zip(closed[:-1], closed[1:]):
        lon1, lat1 = np.radians(a[0]), np.radians(a[1])
        lon2, lat2 = np.radians(b[0]), np.radians(b[1])
        total += (lon2 - lon1) * (np.sin(lat1) + np.sin(lat2))
    return abs(total) * (AUTHALIC_RADIUS_M ** 2) * 0.5


def _polygon_area_m2(rings):
    if not rings:
        return 0.0
    exterior = _ring_area_m2(rings[0])
    holes = sum(_ring_area_m2(ring) for ring in rings[1:])
    return max(0.0, exterior - holes)


def geometry_area_m2(geometry):
    geom_type = geometry["type"]
    coords = geometry["coordinates"]
    if geom_type == "Polygon":
        return _polygon_area_m2(coords)
    if geom_type == "MultiPolygon":
        return sum(_polygon_area_m2(polygon) for polygon in coords)
    raise ValueError(f"Unsupported geometry type: {geom_type}")


@dataclass
class Province:
    code: int
    name: str
    source_name: str
    geometry: dict
    bounds: Tuple[float, float, float, float]
    boundary_area_m2: float = field(init=False)

    def __post_init__(self):
        self.boundary_area_m2 = geometry_area_m2(self.geometry)

    @property
    def boundary_area_km2(self):
        return self.boundary_area_m2 / 1_000_000.0

    @property
    def boundary_area_ha(self):
        return self.boundary_area_m2 / 10_000.0

    def rasterize_to_grid(self, transform, width, height, all_touched=False):
        mask = rasterize(
            [(self.geometry, 1)],
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype="uint8",
            all_touched=all_touched,
        )
        return mask.astype(bool)


def _normalize_feature(feature, fallback_code):
    props = feature.get("properties") or {}
    code = props.get("province_code") or props.get("indexCode") or props.get("code") or fallback_code
    name = (
        props.get("province_name_ascii")
        or props.get("source_name")
        or props.get("NAME_1")
        or props.get("shapeName")
        or props.get("name")
        or f"province_{code}"
    )
    geometry = feature["geometry"]
    return Province(
        code=int(code),
        name=str(name),
        source_name=str(
            props.get("source_name") or props.get("NAME_1") or name
        ),
        geometry=geometry,
        bounds=geometry_bounds(geometry),
    )


def load_provinces(boundary_path: Path):
    with Path(boundary_path).open("r", encoding="utf-8") as f:
        geojson = json.load(f)

    provinces = []
    seen_codes = set()
    for idx, feature in enumerate(geojson.get("features", []), start=1):
        province = _normalize_feature(feature, idx)
        if province.code in seen_codes:
            raise ValueError(f"Duplicate province_code: {province.code}")
        seen_codes.add(province.code)
        provinces.append(province)

    if not provinces:
        raise ValueError(f"No province features found in {boundary_path}")

    provinces.sort(key=lambda p: p.code)
    return provinces
