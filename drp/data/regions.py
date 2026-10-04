"""Static geographical attributes per region.

The bundled `data/regions.csv` describes the 14 districts of Kerala, India
(a state with recurring monsoon floods and landslides). Coordinates are
district headquarters; elevation, slope, soil, land cover, NDVI and the
other attributes are approximate district-level values intended as
placeholders - replace them with values derived from USGS SRTM (elevation,
slope, aspect), Copernicus land cover and Sentinel/MODIS NDVI for real use.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import resolve

SOIL_TYPES = ["clay", "laterite", "loam", "sandy"]
LAND_COVERS = ["cropland", "forest", "mixed", "urban", "wetland"]

REQUIRED = [
    "region_id", "name", "lat", "lon", "elevation_m", "slope_deg", "aspect_deg",
    "soil_type", "land_cover", "ndvi_base", "dist_river_km", "impervious_frac", "rain_factor",
]


def load_regions(cfg: dict) -> pd.DataFrame:
    regions = pd.read_csv(resolve(cfg["data"]["regions_file"]))
    missing = [c for c in REQUIRED if c not in regions.columns]
    if missing:
        raise ValueError(f"regions file is missing columns: {missing}")
    bad_soil = set(regions["soil_type"]) - set(SOIL_TYPES)
    bad_cover = set(regions["land_cover"]) - set(LAND_COVERS)
    if bad_soil or bad_cover:
        raise ValueError(f"unknown soil types {bad_soil} / land covers {bad_cover}")
    max_regions = cfg["data"].get("max_regions")
    if max_regions and max_regions < len(regions):
        # keep a mix of lowland and highland districts in small runs
        by_elev = regions.sort_values("elevation_m").reset_index(drop=True)
        picks = np.unique(np.linspace(0, len(by_elev) - 1, max_regions).round().astype(int))
        regions = by_elev.iloc[picks]
    return regions.sort_values("region_id").reset_index(drop=True)
