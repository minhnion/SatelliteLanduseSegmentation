import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from area_calc.config import SEASON_BY_DATE_TAG


_PNG_NAME_RE = re.compile(r"^(?P<stem>.+)_infered\.png$")
_DATE_TAG_RE = re.compile(r"_(\d{4}_\d{4})$")


@dataclass
class ImagePair:
    stem: str
    tif_path: Path
    mask_path: Path
    season_key: Optional[str] = None
    season_label: Optional[str] = None
    date_tag: Optional[str] = None


@dataclass
class MissingPair:
    mask_name: str
    expected_tif: str


def detect_season(stem):
    match = _DATE_TAG_RE.search(stem)
    if not match:
        return None
    tag = match.group(1)
    season = SEASON_BY_DATE_TAG.get(tag)
    if season is None:
        return None
    return tag, season["key"], season["label"]


class ImageSource(ABC):
    name: str
    has_seasons: bool

    @abstractmethod
    def iter_pairs(self) -> Iterable[ImagePair]:
        ...

    @abstractmethod
    def collect_missing(self) -> list:
        ...


class DatasetSource(ImageSource):
    name = "dataset"
    has_seasons = False

    def __init__(self, dataset_dir: Path):
        self.dataset_dir = Path(dataset_dir)
        self._missing: list = []

    def iter_pairs(self):
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset folder not found: {self.dataset_dir}")
        tif_paths = sorted(self.dataset_dir.glob("*_sat.tif"))
        if not tif_paths:
            raise FileNotFoundError(f"No *_sat.tif files in {self.dataset_dir}")

        for tif_path in tif_paths:
            mask_path = tif_path.with_name(tif_path.name.replace("_sat.tif", "_mask.png"))
            if not mask_path.exists():
                self._missing.append(MissingPair(tif_path.name, mask_path.name))
                continue
            yield ImagePair(
                stem=tif_path.stem.replace("_sat", ""),
                tif_path=tif_path,
                mask_path=mask_path,
            )

    def collect_missing(self):
        return list(self._missing)


class InferenceSource(ImageSource):
    name = "inference"
    has_seasons = True

    def __init__(self, tif_dir: Path, png_dir: Path):
        self.tif_dir = Path(tif_dir)
        self.png_dir = Path(png_dir)
        self._missing: list = []

    def _tif_index(self):
        tif_paths = sorted(self.tif_dir.rglob("*.tif"))
        index = {}
        for tif_path in tif_paths:
            existing = index.get(tif_path.name)
            if existing is not None:
                raise ValueError(
                    f"Duplicate TIF filename found: {tif_path.name} in {existing.parent} and {tif_path.parent}"
                )
            index[tif_path.name] = tif_path
        return index

    def iter_pairs(self):
        if not self.tif_dir.exists():
            raise FileNotFoundError(f"TIF folder not found: {self.tif_dir}")
        if not self.png_dir.exists():
            raise FileNotFoundError(f"PNG folder not found: {self.png_dir}")

        tif_index = self._tif_index()
        if not tif_index:
            raise FileNotFoundError(f"No *.tif files in {self.tif_dir}")

        png_paths = sorted(self.png_dir.rglob("*_infered.png"))
        if not png_paths:
            raise FileNotFoundError(f"No *_infered.png files in {self.png_dir}")

        seen_stems = set()
        for png_path in png_paths:
            match = _PNG_NAME_RE.match(png_path.name)
            if not match:
                raise ValueError(f"Unexpected PNG name (expected *_infered.png): {png_path.name}")
            stem = match.group("stem")
            if stem in seen_stems:
                raise ValueError(f"Duplicate inferred PNG stem found: {stem}")
            seen_stems.add(stem)

            tif_name = f"{stem}.tif"
            tif_path = tif_index.get(tif_name)
            if tif_path is None:
                self._missing.append(MissingPair(png_path.name, tif_name))
                continue
            season = detect_season(stem)
            if season is None:
                raise ValueError(
                    f"Cannot detect season tag in {stem}. "
                    f"Expected one of: {list(SEASON_BY_DATE_TAG.keys())}"
                )
            date_tag, season_key, season_label = season
            yield ImagePair(
                stem=stem,
                tif_path=tif_path,
                mask_path=png_path,
                season_key=season_key,
                season_label=season_label,
                date_tag=date_tag,
            )

    def collect_missing(self):
        return list(self._missing)
