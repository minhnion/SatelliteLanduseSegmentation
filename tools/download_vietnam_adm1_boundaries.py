#!/usr/bin/env python3
"""Download and filter Vietnam ADM1 boundaries for the 7-province dataset QC.

The default source is GADM 4.1 because it represents the pre-2025 Vietnam
province layout more closely than the current geoBoundaries VNM ADM1 layer.
"""

import argparse
import json
import re
import tempfile
import unicodedata
import urllib.request
import zipfile
from pathlib import Path


GADM41_VNM_ADM1_URL = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_VNM_1.json.zip"
GEOB_API_URL = "https://www.geoboundaries.org/api/current/gbOpen/VNM/ADM1/"

TARGET_PROVINCES = [
    {"code": 1, "name_ascii": "Bac Ninh", "aliases": ["Bac Ninh", "BacNinh"]},
    {"code": 2, "name_ascii": "Ha Noi", "aliases": ["Ha Noi", "HaNoi", "Hanoi"]},
    {"code": 3, "name_ascii": "Hai Duong", "aliases": ["Hai Duong", "HaiDuong"]},
    {"code": 4, "name_ascii": "Hai Phong", "aliases": ["Hai Phong", "HaiPhong"]},
    {"code": 5, "name_ascii": "Hung Yen", "aliases": ["Hung Yen", "HungYen"]},
    {"code": 6, "name_ascii": "Quang Ninh", "aliases": ["Quang Ninh", "QuangNinh"]},
    {"code": 7, "name_ascii": "Vinh Phuc", "aliases": ["Vinh Phuc", "VinhPhuc"]},
]

NAME_FIELDS = [
    "NAME_1",
    "shapeName",
    "name",
    "Name",
    "NAME",
    "ADM1_EN",
    "ADM1_VI",
    "province_name",
]


def normalize_name(value):
    value = unicodedata.normalize("NFKD", str(value))
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def read_json_url(url):
    with urllib.request.urlopen(url, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def download_file(url, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, output_path)
    return output_path


def load_gadm_geojson(cache_dir):
    zip_path = cache_dir / "gadm41_VNM_1.json.zip"
    if not zip_path.exists():
        print(f"Downloading GADM ADM1: {GADM41_VNM_ADM1_URL}")
        download_file(GADM41_VNM_ADM1_URL, zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        json_members = [name for name in zf.namelist() if name.lower().endswith(".json")]
        if not json_members:
            raise ValueError(f"No JSON file found inside {zip_path}")
        with zf.open(json_members[0]) as f:
            return json.load(f), {
                "source": "GADM 4.1",
                "download_url": GADM41_VNM_ADM1_URL,
                "cache_file": str(zip_path),
                "license_note": (
                    "GADM is freely available for academic and other non-commercial use; "
                    "check https://gadm.org/license.html for redistribution/commercial use."
                ),
            }


def load_geoboundaries_geojson(cache_dir):
    print(f"Fetching geoBoundaries metadata: {GEOB_API_URL}")
    metadata = read_json_url(GEOB_API_URL)
    geojson_url = metadata["gjDownloadURL"]
    geojson_path = cache_dir / "geoBoundaries-VNM-ADM1.geojson"
    if not geojson_path.exists():
        print(f"Downloading geoBoundaries ADM1: {geojson_url}")
        download_file(geojson_url, geojson_path)
    with geojson_path.open("r", encoding="utf-8") as f:
        return json.load(f), {
            "source": "geoBoundaries gbOpen current",
            "download_url": geojson_url,
            "metadata_url": GEOB_API_URL,
            "metadata": metadata,
            "cache_file": str(geojson_path),
        }


def load_input_geojson(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f), {
            "source": "local input",
            "input_path": str(Path(path).resolve()),
        }


def feature_display_name(feature):
    props = feature.get("properties") or {}
    for field in NAME_FIELDS:
        value = props.get(field)
        if value:
            return str(value), field
    return "", None


def filter_target_provinces(geojson, source_name):
    target_by_alias = {}
    for target in TARGET_PROVINCES:
        for alias in target["aliases"] + [target["name_ascii"]]:
            target_by_alias[normalize_name(alias)] = target

    matched = {}
    available_names = []

    for feature in geojson.get("features", []):
        source_display_name, name_field = feature_display_name(feature)
        if source_display_name:
            available_names.append(source_display_name)

        target = target_by_alias.get(normalize_name(source_display_name))
        if not target:
            continue

        code = target["code"]
        if code in matched:
            raise ValueError(
                f"Multiple features matched province {target['name_ascii']}: "
                f"{matched[code]['properties'].get('source_name')} and {source_display_name}"
            )

        out_feature = json.loads(json.dumps(feature))
        out_feature.setdefault("properties", {})
        out_feature["properties"].update(
            {
                "province_code": code,
                "province_name_ascii": target["name_ascii"],
                "source_name": source_display_name,
                "source_name_field": name_field,
                "boundary_source": source_name,
                "boundary_target_year": 2024,
            }
        )
        matched[code] = out_feature

    missing = [target["name_ascii"] for target in TARGET_PROVINCES if target["code"] not in matched]
    if missing:
        unique_available = sorted(set(available_names))
        raise ValueError(
            "Could not match all target provinces. "
            f"Missing: {missing}. Available names include: {unique_available[:120]}"
        )

    return {
        "type": "FeatureCollection",
        "name": "vietnam_adm1_7provinces_2024",
        "features": [matched[target["code"]] for target in TARGET_PROVINCES],
    }


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)
        f.write("\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download Vietnam ADM1 boundaries and extract the 7 provinces used by dataset/."
    )
    parser.add_argument(
        "--source",
        choices=["gadm", "geoboundaries"],
        default="gadm",
        help="Boundary source to download. Default: gadm.",
    )
    parser.add_argument(
        "--input_geojson",
        type=str,
        default=None,
        help="Optional local ADM1 GeoJSON to filter instead of downloading.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="mapbox/vietnam_adm1_7provinces_2024.geojson",
        help="Filtered 7-province GeoJSON output.",
    )
    parser.add_argument(
        "--raw_output",
        type=str,
        default="mapbox/vietnam_adm1_raw.geojson",
        help="Raw downloaded ADM1 GeoJSON copy.",
    )
    parser.add_argument(
        "--metadata_output",
        type=str,
        default="mapbox/vietnam_adm1_7provinces_2024_metadata.json",
        help="Metadata JSON output.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="mapbox/.boundary_cache",
        help="Download cache directory.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cache_dir = Path(args.cache_dir)

    if args.input_geojson:
        geojson, metadata = load_input_geojson(args.input_geojson)
    elif args.source == "gadm":
        geojson, metadata = load_gadm_geojson(cache_dir)
    else:
        geojson, metadata = load_geoboundaries_geojson(cache_dir)

    source_name = metadata["source"]
    filtered = filter_target_provinces(geojson, source_name)

    raw_output = Path(args.raw_output)
    output = Path(args.output)
    metadata_output = Path(args.metadata_output)

    write_json(raw_output, geojson)
    write_json(output, filtered)

    summary = {
        "target_year": 2024,
        "province_count": len(filtered["features"]),
        "provinces": [
            {
                "province_code": feature["properties"]["province_code"],
                "province_name_ascii": feature["properties"]["province_name_ascii"],
                "source_name": feature["properties"]["source_name"],
            }
            for feature in filtered["features"]
        ],
        "outputs": {
            "raw_geojson": str(raw_output.resolve()),
            "filtered_geojson": str(output.resolve()),
        },
        "source_metadata": metadata,
    }
    write_json(metadata_output, summary)

    print(f"Wrote filtered boundary: {output}")
    print(f"Wrote raw boundary copy: {raw_output}")
    print(f"Wrote metadata: {metadata_output}")
    print("Matched provinces:")
    for item in summary["provinces"]:
        print(f"  {item['province_code']}: {item['province_name_ascii']} <- {item['source_name']}")


if __name__ == "__main__":
    main()
