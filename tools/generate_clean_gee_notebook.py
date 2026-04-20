#!/usr/bin/env python3
import json
from pathlib import Path
from textwrap import dedent


def _lines(text: str):
    text = dedent(text).strip("\n")
    if not text:
        return []
    return [line if line.endswith("\n") else f"{line}\n" for line in text.splitlines()]


def code_cell(text: str):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _lines(text),
    }


def markdown_cell(text: str):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _lines(text),
    }


def build_notebook():
    cells = [
        markdown_cell(
            """
            # Clean Google Earth Engine Crawl Notebook

            Notebook nay la ban refactor tu `get_gg_earth_engine_images_minh.ipynb`.

            Muc tieu:
            - tach ro 2 khoi lon: `Sentinel-1` va `Sentinel-2`
            - giu lai cac workflow crawl con gia tri van hanh
            - bo cac cell demo le, box test, va cac bien bi redefine nhieu lan
            - mac dinh cau hinh theo Drive layout hien tai

            Workflow giu lai:
            - `Sentinel-1`
              - crawl grid `3x3` tu `mapbox_grid_3km_filtered_7provinces.json`
              - crawl grid `1x1` tu folder `geojson_chunks`
              - crawl theo folder polygon `.txt`
              - crawl bo dataset train tu `MULTIPOLYGON WKT` thanh `Area_2024_N_o_<index>_sat`
            - `Sentinel-2`
              - crawl grid `3x3`
              - crawl grid `1x1`
              - crawl theo folder polygon `.txt`

            Luu y:
            - Earth Engine `Export.image.toDrive(folder=...)` dung ten folder tren Drive. Nested path co the hoat dong khong dong nhat tuy tai khoan.
            - Notebook nay dung nested path theo cau truc anh/chị muon. Neu GEE cua anh/chị khong resolve nested folder dung, doi sang ten folder phang.
            - Mac dinh notebook tap trung vao `Sentinel-1`, nhung `Sentinel-2` van duoc giu lai de dung sau.
            """
        ),
        code_cell(
            """
            # Fresh Colab runtime only
            !pip -q install earthengine-api shapely

            from google.colab import drive
            drive.mount("/content/drive")
            """
        ),
        code_cell(
            """
            from pathlib import Path
            import ast
            import json
            import os
            import re
            import unicodedata

            import ee
            from shapely import wkt
            from shapely.geometry import Polygon, box

            PROJECT_NAME = "linear-sight-456113-t3"

            ROOT_DIR = Path("/content/drive/MyDrive/Inest/3. Straw")
            MAPBOX_3KM_GEOJSON = ROOT_DIR / "mapbox_grid_3km_filtered_7provinces.json"
            GEOJSON_CHUNKS_DIR = ROOT_DIR / "geojson_chunks"
            POLYGON_DIR = ROOT_DIR / "polygon"

            DATA_CRAWL_DIR = ROOT_DIR / "data_crawl"
            LOCAL_OUTPUT_DIRS = {
                "3km": DATA_CRAWL_DIR / "3x3",
                "1km": DATA_CRAWL_DIR / "1x1",
            }
            for output_dir in LOCAL_OUTPUT_DIRS.values():
                output_dir.mkdir(parents=True, exist_ok=True)

            # GEE dung ten folder tren Drive. Neu nested folder khong hoat dong dung trong tai khoan cua ban,
            # doi sang ten phang nhu: "Inest_3_Straw_data_crawl_3x3"
            DRIVE_EXPORT_DIRS = {
                "3km": "Inest/3. Straw/data_crawl/3x3",
                "1km": "Inest/3. Straw/data_crawl/1x1",
            }

            def init_ee(project_name=PROJECT_NAME):
                try:
                    ee.Initialize(project=project_name)
                except Exception:
                    ee.Authenticate()
                    ee.Initialize(project=project_name)

            init_ee()
            print(f"Initialized Earth Engine with project: {PROJECT_NAME}")
            print(f"Root dir: {ROOT_DIR}")
            """
        ),
        code_cell(
            """
            S1_BANDS = ["VV", "VH"]
            S2_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]


            def strip_accents(text: str) -> str:
                text = text.replace("Đ", "D").replace("đ", "d")
                return "".join(
                    c for c in unicodedata.normalize("NFKD", text)
                    if not unicodedata.combining(c)
                )


            def sanitize_name(name: str, max_length: int = 100) -> str:
                cleaned = strip_accents(name)
                cleaned = re.sub(r"[^a-zA-Z0-9.,:_-]", "_", cleaned)
                return cleaned[:max_length]


            def build_date_tag(from_date: str, to_date: str | None = None, tag: str | None = None) -> str:
                if tag:
                    return sanitize_name(tag)
                year = int(from_date[:4])
                start_month = int(from_date[5:7])
                if to_date:
                    end_year = int(to_date[:4])
                    end_month = int(to_date[5:7]) - 1
                    if end_month == 0:
                        end_month = 12
                        end_year -= 1
                    if end_year != year:
                        return f"{start_month:02d}{end_month:02d}_{year}_{end_year}"
                    return f"{start_month:02d}{end_month:02d}_{year}"
                return f"{start_month:02d}{start_month:02d}_{year}"


            def read_geojson(path):
                path = Path(path)
                with path.open("r", encoding="utf-8") as f:
                    return json.load(f)


            def load_geojson_features(path):
                data = read_geojson(path)
                return data.get("features", [])


            def load_geojson_features_from_dir(directory):
                directory = Path(directory)
                candidates = sorted({*directory.glob("*.geojson"), *directory.glob("*.json")})
                features = []
                for path in candidates:
                    features.extend(load_geojson_features(path))
                return features


            def feature_property(feature, *names, default=None):
                props = feature.get("properties", {})
                for name in names:
                    if name in props:
                        return props[name]
                return default


            def feature_to_ee_geometry(feature):
                geometry = feature["geometry"]
                if geometry["type"] == "Polygon":
                    return ee.Geometry.Polygon(geometry["coordinates"])
                if geometry["type"] == "MultiPolygon":
                    return ee.Geometry.MultiPolygon(geometry["coordinates"])
                raise ValueError(f"Unsupported GeoJSON geometry type: {geometry['type']}")


            def grid_feature_description(feature, prefix: str, date_tag: str, fallback_idx: int | None = None):
                cell_code = feature_property(feature, "cell_code", "cellCode")
                row = feature_property(feature, "row")
                col = feature_property(feature, "col")
                if cell_code is not None and row is not None and col is not None:
                    return sanitize_name(f"{prefix}_{cell_code}_{row}_{col}_{date_tag}")
                if fallback_idx is not None:
                    return sanitize_name(f"{prefix}_{fallback_idx}_{date_tag}")
                return sanitize_name(f"{prefix}_{date_tag}")


            def polygon_txt_to_coords(path):
                path = Path(path)
                raw = path.read_text(encoding="utf-8").strip()
                coords = ast.literal_eval(raw)
                if coords[0] != coords[-1]:
                    coords.append(coords[0])
                return coords


            def make_drive_export_task(image, description: str, drive_folder: str, region, scale: int = 10, max_pixels: float = 1e13):
                return ee.batch.Export.image.toDrive(
                    image=image,
                    description=sanitize_name(description),
                    folder=drive_folder,
                    scale=scale,
                    region=region,
                    maxPixels=max_pixels,
                    fileFormat="GeoTIFF",
                    formatOptions={"cloudOptimized": True},
                )


            def start_tasks(tasks, limit: int | None = None, dry_run: bool = False):
                selected = tasks if limit is None else tasks[:limit]
                for task in selected:
                    description = task.config.get("description", "<no-description>")
                    if dry_run:
                        print(f"[DRY RUN] {description}")
                    else:
                        task.start()
                        print(f"Exporting {description}")
                print(f"Queued {len(selected)} task(s)")


            def build_rectangles_from_wkt(multipolygon_wkt: str, grid_size_deg: float = 0.05):
                geom = wkt.loads(multipolygon_wkt)
                minx, miny, maxx, maxy = geom.bounds
                x = minx
                rectangles = []
                while x < maxx:
                    y = miny
                    while y < maxy:
                        rect = box(x, y, x + grid_size_deg, y + grid_size_deg)
                        if not rect.intersection(geom).is_empty:
                            rectangles.append(
                                {
                                    "rectangle": ee.Geometry.Rectangle([x, y, x + grid_size_deg, y + grid_size_deg]),
                                    "bounds": (x, y, x + grid_size_deg, y + grid_size_deg),
                                }
                            )
                        y += grid_size_deg
                    x += grid_size_deg
                return rectangles
            """
        ),
        markdown_cell(
            """
            ## Sentinel-1 Module

            Day la khoi chinh de crawl:
            - grid `3x3`
            - grid `1x1` tu `geojson_chunks`
            - polygon folder `.txt`
            - dataset train tu `MULTIPOLYGON WKT`

            Dat ten file:
            - grid: `S1_<cell_code>_<row>_<col>_<date_tag>`
            - polygon: `<province>_S1_full_<year>`
            - dataset: `Area_<year>_N_o_<index>_sat`
            """
        ),
        code_cell(
            """
            def build_s1_image(region, from_date: str, to_date: str, orbit: str | None = "DESCENDING", reducer: str = "median"):
                collection = (
                    ee.ImageCollection("COPERNICUS/S1_GRD")
                    .filterBounds(region)
                    .filterDate(from_date, to_date)
                    .filter(ee.Filter.eq("instrumentMode", "IW"))
                    .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
                    .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
                )
                if orbit:
                    collection = collection.filter(ee.Filter.eq("orbitProperties_pass", orbit))

                if reducer == "median":
                    image = collection.median()
                elif reducer == "mean":
                    image = collection.mean()
                else:
                    raise ValueError(f"Unsupported Sentinel-1 reducer: {reducer}")

                return image.select(S1_BANDS).clip(region)


            def queue_s1_grid_exports_from_geojson(
                geojson_path,
                from_date: str,
                to_date: str,
                drive_folder: str,
                date_tag: str,
                orbit: str | None = "DESCENDING",
                reducer: str = "median",
                scale: int = 10,
                prefix: str = "S1",
            ):
                tasks = []
                features = load_geojson_features(geojson_path)
                for idx, feature in enumerate(features):
                    region = feature_to_ee_geometry(feature)
                    description = grid_feature_description(feature, prefix=prefix, date_tag=date_tag, fallback_idx=idx)
                    image = build_s1_image(region, from_date, to_date, orbit=orbit, reducer=reducer)
                    tasks.append(make_drive_export_task(image, description, drive_folder, region, scale=scale))
                return tasks


            def queue_s1_chunk_exports(
                chunks_dir,
                from_date: str,
                to_date: str,
                drive_folder: str,
                date_tag: str,
                orbit: str | None = "DESCENDING",
                reducer: str = "median",
                scale: int = 10,
                prefix: str = "S1",
            ):
                tasks = []
                features = load_geojson_features_from_dir(chunks_dir)
                for idx, feature in enumerate(features):
                    region = feature_to_ee_geometry(feature)
                    description = grid_feature_description(feature, prefix=prefix, date_tag=date_tag, fallback_idx=idx)
                    image = build_s1_image(region, from_date, to_date, orbit=orbit, reducer=reducer)
                    tasks.append(make_drive_export_task(image, description, drive_folder, region, scale=scale))
                return tasks


            def queue_s1_polygon_folder_exports(
                polygon_dir,
                from_date: str,
                to_date: str,
                drive_folder: str,
                orbit: str | None = "DESCENDING",
                reducer: str = "median",
                scale: int = 10,
            ):
                tasks = []
                polygon_dir = Path(polygon_dir)
                year = from_date[:4]
                for txt_path in sorted(polygon_dir.glob("*.txt")):
                    coords = polygon_txt_to_coords(txt_path)
                    region = ee.Geometry.Polygon([coords])
                    description = sanitize_name(f"{txt_path.stem}_S1_full_{year}")
                    image = build_s1_image(region, from_date, to_date, orbit=orbit, reducer=reducer)
                    tasks.append(make_drive_export_task(image, description, drive_folder, region, scale=scale))
                return tasks


            def queue_s1_dataset_exports_from_wkt(
                multipolygon_wkt: str,
                from_date: str,
                to_date: str,
                drive_folder: str,
                place_name: str = "Area",
                grid_size_deg: float = 0.05,
                orbit: str | None = "DESCENDING",
                reducer: str = "median",
                scale: int = 10,
            ):
                tasks = []
                rectangles = build_rectangles_from_wkt(multipolygon_wkt, grid_size_deg=grid_size_deg)
                year = from_date[:4]
                for idx, item in enumerate(rectangles):
                    region = item["rectangle"]
                    description = sanitize_name(f"{place_name}_{year}_N_o_{idx}_sat")
                    image = build_s1_image(region, from_date, to_date, orbit=orbit, reducer=reducer)
                    tasks.append(make_drive_export_task(image, description, drive_folder, region, scale=scale))
                return tasks
            """
        ),
        code_cell(
            """
            # =========================
            # Sentinel-1 run config
            # =========================

            RUN_S1_3KM = False
            RUN_S1_1KM = False
            RUN_S1_POLYGON_FOLDER = False
            RUN_S1_DATASET_FROM_WKT = False

            DRY_RUN = True
            START_ONLY_FIRST_N = 5

            S1_3KM_CONFIG = {
                "geojson_path": MAPBOX_3KM_GEOJSON,
                "from_date": "2023-03-01",
                "to_date": "2023-07-01",
                "date_tag": "0306_2023",
                "drive_folder": DRIVE_EXPORT_DIRS["3km"],
                "orbit": "DESCENDING",
                "reducer": "median",
                "scale": 10,
                "prefix": "S1",
            }

            S1_1KM_CONFIG = {
                "chunks_dir": GEOJSON_CHUNKS_DIR,
                "from_date": "2023-03-01",
                "to_date": "2023-07-01",
                "date_tag": "0306_2023",
                "drive_folder": DRIVE_EXPORT_DIRS["1km"],
                "orbit": "DESCENDING",
                "reducer": "median",
                "scale": 10,
                "prefix": "S1",
            }

            S1_POLYGON_CONFIG = {
                "polygon_dir": POLYGON_DIR,
                "from_date": "2024-01-01",
                "to_date": "2024-12-31",
                "drive_folder": DRIVE_EXPORT_DIRS["3km"],
                "orbit": "DESCENDING",
                "reducer": "median",
                "scale": 10,
            }

            # Dung workflow nay neu muon crawl ra bo train giong pattern dataset/:
            # Area_2024_N_o_<index>_sat
            S1_DATASET_CONFIG = {
                "multipolygon_wkt": \"\"\"PASTE_MULTIPOLYGON_WKT_HERE\"\"\",
                "from_date": "2024-01-01",
                "to_date": "2024-12-31",
                "drive_folder": DRIVE_EXPORT_DIRS["3km"],
                "place_name": "Area",
                "grid_size_deg": 0.05,
                "orbit": "DESCENDING",
                "reducer": "median",
                "scale": 10,
            }

            s1_tasks = []
            if RUN_S1_3KM:
                s1_tasks.extend(queue_s1_grid_exports_from_geojson(**S1_3KM_CONFIG))
            if RUN_S1_1KM:
                s1_tasks.extend(queue_s1_chunk_exports(**S1_1KM_CONFIG))
            if RUN_S1_POLYGON_FOLDER:
                s1_tasks.extend(queue_s1_polygon_folder_exports(**S1_POLYGON_CONFIG))
            if RUN_S1_DATASET_FROM_WKT:
                if "PASTE_MULTIPOLYGON_WKT_HERE" in S1_DATASET_CONFIG["multipolygon_wkt"]:
                    raise ValueError("Replace S1_DATASET_CONFIG['multipolygon_wkt'] before running dataset export.")
                s1_tasks.extend(queue_s1_dataset_exports_from_wkt(**S1_DATASET_CONFIG))

            print(f"Prepared {len(s1_tasks)} Sentinel-1 task(s)")
            start_tasks(s1_tasks, limit=START_ONLY_FIRST_N, dry_run=DRY_RUN)
            """
        ),
        markdown_cell(
            """
            ## Sentinel-2 Module

            Giu lai 3 workflow huu dung:
            - grid `3x3`
            - grid `1x1` tu `geojson_chunks`
            - polygon folder `.txt`

            Ho tro 2 kieu tong hop:
            - `median`
            - `least_cloud` theo QA60
            """
        ),
        code_cell(
            """
            def mask_s2_clouds(image):
                qa = image.select("QA60")
                cloud_bit_mask = 1 << 10
                cirrus_bit_mask = 1 << 11
                mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
                return image.updateMask(mask).divide(10000)


            def build_s2_image(
                region,
                from_date: str,
                to_date: str,
                composite: str = "median",
                max_cloud_pct: int = 20,
            ):
                collection = (
                    ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
                    .filterBounds(region)
                    .filterDate(from_date, to_date)
                    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud_pct))
                )

                if composite == "median":
                    image = collection.map(mask_s2_clouds).median()
                elif composite == "least_cloud":
                    def add_regional_cloudiness(img):
                        cloud_mask = img.select("QA60").bitwiseAnd(1 << 10).Or(img.select("QA60").bitwiseAnd(1 << 11))
                        cloud_fraction = cloud_mask.reduceRegion(
                            reducer=ee.Reducer.mean(),
                            geometry=region,
                            scale=10,
                            maxPixels=1e13,
                        ).get("QA60")
                        return img.set("regional_cloud", cloud_fraction)

                    image = ee.Image(collection.map(add_regional_cloudiness).sort("regional_cloud").first())
                    image = mask_s2_clouds(image)
                else:
                    raise ValueError(f"Unsupported Sentinel-2 composite mode: {composite}")

                return image.select(S2_BANDS).clip(region)


            def queue_s2_grid_exports_from_geojson(
                geojson_path,
                from_date: str,
                to_date: str,
                drive_folder: str,
                date_tag: str,
                composite: str = "median",
                max_cloud_pct: int = 20,
                scale: int = 10,
                prefix: str = "S2",
            ):
                tasks = []
                features = load_geojson_features(geojson_path)
                for idx, feature in enumerate(features):
                    region = feature_to_ee_geometry(feature)
                    description = grid_feature_description(feature, prefix=prefix, date_tag=date_tag, fallback_idx=idx)
                    image = build_s2_image(region, from_date, to_date, composite=composite, max_cloud_pct=max_cloud_pct)
                    tasks.append(make_drive_export_task(image, description, drive_folder, region, scale=scale))
                return tasks


            def queue_s2_chunk_exports(
                chunks_dir,
                from_date: str,
                to_date: str,
                drive_folder: str,
                date_tag: str,
                composite: str = "median",
                max_cloud_pct: int = 20,
                scale: int = 10,
                prefix: str = "S2",
            ):
                tasks = []
                features = load_geojson_features_from_dir(chunks_dir)
                for idx, feature in enumerate(features):
                    region = feature_to_ee_geometry(feature)
                    description = grid_feature_description(feature, prefix=prefix, date_tag=date_tag, fallback_idx=idx)
                    image = build_s2_image(region, from_date, to_date, composite=composite, max_cloud_pct=max_cloud_pct)
                    tasks.append(make_drive_export_task(image, description, drive_folder, region, scale=scale))
                return tasks


            def queue_s2_polygon_folder_exports(
                polygon_dir,
                from_date: str,
                to_date: str,
                drive_folder: str,
                composite: str = "median",
                max_cloud_pct: int = 20,
                scale: int = 10,
            ):
                tasks = []
                polygon_dir = Path(polygon_dir)
                year = from_date[:4]
                for txt_path in sorted(polygon_dir.glob("*.txt")):
                    coords = polygon_txt_to_coords(txt_path)
                    region = ee.Geometry.Polygon([coords])
                    description = sanitize_name(f"{txt_path.stem}_S2_full_{year}")
                    image = build_s2_image(region, from_date, to_date, composite=composite, max_cloud_pct=max_cloud_pct)
                    tasks.append(make_drive_export_task(image, description, drive_folder, region, scale=scale))
                return tasks
            """
        ),
        code_cell(
            """
            # =========================
            # Sentinel-2 run config
            # =========================

            RUN_S2_3KM = False
            RUN_S2_1KM = False
            RUN_S2_POLYGON_FOLDER = False

            DRY_RUN = True
            START_ONLY_FIRST_N = 5

            S2_3KM_CONFIG = {
                "geojson_path": MAPBOX_3KM_GEOJSON,
                "from_date": "2024-01-01",
                "to_date": "2024-12-31",
                "date_tag": "0112_2024",
                "drive_folder": DRIVE_EXPORT_DIRS["3km"],
                "composite": "median",
                "max_cloud_pct": 100,
                "scale": 10,
                "prefix": "S2",
            }

            S2_1KM_CONFIG = {
                "chunks_dir": GEOJSON_CHUNKS_DIR,
                "from_date": "2023-03-01",
                "to_date": "2023-07-01",
                "date_tag": "0306_2023",
                "drive_folder": DRIVE_EXPORT_DIRS["1km"],
                "composite": "least_cloud",
                "max_cloud_pct": 20,
                "scale": 10,
                "prefix": "S2",
            }

            S2_POLYGON_CONFIG = {
                "polygon_dir": POLYGON_DIR,
                "from_date": "2024-01-01",
                "to_date": "2024-12-31",
                "drive_folder": DRIVE_EXPORT_DIRS["3km"],
                "composite": "median",
                "max_cloud_pct": 100,
                "scale": 10,
            }

            s2_tasks = []
            if RUN_S2_3KM:
                s2_tasks.extend(queue_s2_grid_exports_from_geojson(**S2_3KM_CONFIG))
            if RUN_S2_1KM:
                s2_tasks.extend(queue_s2_chunk_exports(**S2_1KM_CONFIG))
            if RUN_S2_POLYGON_FOLDER:
                s2_tasks.extend(queue_s2_polygon_folder_exports(**S2_POLYGON_CONFIG))

            print(f"Prepared {len(s2_tasks)} Sentinel-2 task(s)")
            start_tasks(s2_tasks, limit=START_ONLY_FIRST_N, dry_run=DRY_RUN)
            """
        ),
        markdown_cell(
            """
            ## Cach dung de tap trung Sentinel-1 ngay bay gio

            Truong hop crawl `3x3` S1:
            1. Chay cell setup va config.
            2. O cell `Sentinel-1 run config`, dat:
               - `RUN_S1_3KM = True`
               - cac toggle khac = `False`
               - `DRY_RUN = False`
               - `START_ONLY_FIRST_N = None`
            3. Kiem tra:
               - `S1_3KM_CONFIG["from_date"]`
               - `S1_3KM_CONFIG["to_date"]`
               - `S1_3KM_CONFIG["date_tag"]`
               - `S1_3KM_CONFIG["orbit"]`
            4. Chay cell do de queue task len Earth Engine.

            Truong hop crawl `1x1` S1:
            - dat `RUN_S1_1KM = True`
            - `chunks_dir = GEOJSON_CHUNKS_DIR`

            Truong hop crawl lai bo dataset train:
            - dat `RUN_S1_DATASET_FROM_WKT = True`
            - thay `PASTE_MULTIPOLYGON_WKT_HERE`
            - giu `place_name = "Area"` va `grid_size_deg = 0.05` neu muon giong pattern `dataset/`
            """
        ),
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main():
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / "notebooks" / "get_gg_earth_engine_images_minh_refactored.ipynb"
    notebook = build_notebook()
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {target}")


if __name__ == "__main__":
    main()
