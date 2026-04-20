"""Area-of-interest (AOI) loading and conversion helpers.

An AOI is defined in a YAML config (see ``configs/kerala_2018.yaml``). This
module lets the rest of the pipeline stay region-agnostic: any code that
needs an AOI accepts an :class:`AOIConfig` dataclass rather than a hard-coded
bbox.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DateWindow:
    start: str  # ISO date, e.g. "2018-07-06"
    end: str

    def as_tuple(self) -> tuple[str, str]:
        return self.start, self.end


@dataclass(frozen=True)
class AOIConfig:
    name: str
    region: str
    description: str
    bbox: tuple[float, float, float, float]  # [W, S, E, N] in WGS84 degrees
    crs: str                                 # projected CRS for exports
    pixel_size_m: int
    bands: tuple[str, ...]
    max_cloud_cover_pct: int
    pre_event: DateWindow
    post_event: DateWindow
    composite_reducer: str
    collections: dict[str, str]
    ground_truth: dict[str, Any] = field(default_factory=dict)

    def bbox_geojson(self) -> dict[str, Any]:
        """Return the bbox as a GeoJSON Polygon geometry."""
        w, s, e, n = self.bbox
        return {
            "type": "Polygon",
            "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]],
        }

    def to_ee_geometry(self) -> Any:
        """Return an ``ee.Geometry.Rectangle`` for this AOI.

        Imported lazily so that modules that only need the dataclass (e.g. the
        Sen1Floods11 loader, the tests) do not pull in the Earth Engine SDK.
        """
        import ee  # noqa: PLC0415

        w, s, e, n = self.bbox
        return ee.Geometry.Rectangle([w, s, e, n], proj="EPSG:4326", geodesic=False)


def load_aoi(config_path: str | Path) -> AOIConfig:
    """Parse an AOI YAML config into an :class:`AOIConfig`."""
    path = Path(config_path)
    with path.open() as f:
        raw = yaml.safe_load(f)

    return AOIConfig(
        name=raw["name"],
        region=raw["region"],
        description=raw.get("description", "").strip(),
        bbox=tuple(raw["bbox"]),
        crs=raw["crs"],
        pixel_size_m=int(raw["pixel_size_m"]),
        bands=tuple(raw["bands"]),
        max_cloud_cover_pct=int(raw["max_cloud_cover_pct"]),
        pre_event=DateWindow(**raw["pre_event"]),
        post_event=DateWindow(**raw["post_event"]),
        composite_reducer=raw["composite_reducer"],
        collections=dict(raw["collections"]),
        ground_truth=dict(raw.get("ground_truth", {})),
    )
