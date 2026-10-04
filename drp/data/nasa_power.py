"""Client for the NASA POWER hourly point API (free, no key required).

https://power.larc.nasa.gov/docs/services/api/temporal/hourly/
Responses are cached per region-year as CSV under `data/raw/`.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

log = logging.getLogger(__name__)

URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"
PARAMETERS = {
    "PRECTOTCORR": "rainfall_mm",     # mm/hour, bias-corrected precipitation
    "T2M": "temperature_c",
    "RH2M": "humidity_pct",
    "WS2M": "wind_speed_ms",
}


def _fetch_year(lat: float, lon: float, year: int, retries: int = 4) -> pd.DataFrame:
    params = {
        "parameters": ",".join(PARAMETERS),
        "community": "AG",
        "latitude": lat,
        "longitude": lon,
        "start": f"{year}0101",
        "end": f"{year}1231",
        "format": "JSON",
        "time-standard": "LST",
    }
    for attempt in range(retries):
        try:
            resp = requests.get(URL, params=params, timeout=120)
            resp.raise_for_status()
            data = resp.json()["properties"]["parameter"]
            frame = pd.DataFrame({PARAMETERS[k]: pd.Series(v) for k, v in data.items()})
            frame.index = pd.to_datetime(frame.index, format="%Y%m%d%H")
            return frame.replace(-999.0, np.nan).sort_index()
        except (requests.RequestException, KeyError, ValueError) as exc:
            wait = 5 * 2**attempt
            log.warning("NASA POWER request failed (%s); retrying in %ss", exc, wait)
            time.sleep(wait)
    raise RuntimeError(f"NASA POWER download failed for ({lat}, {lon}) {year}")


def fetch_point(lat: float, lon: float, index: pd.DatetimeIndex, cache_dir: Path) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for year in sorted(set(index.year)):
        path = cache_dir / f"nasa_power_{lat:.2f}_{lon:.2f}_{year}.csv"
        if path.exists():
            frame = pd.read_csv(path, index_col=0, parse_dates=True)
        else:
            log.info("Downloading NASA POWER %s (%.2f, %.2f)", year, lat, lon)
            frame = _fetch_year(lat, lon, year)
            frame.to_csv(path)
        frames.append(frame)
    return pd.concat(frames).reindex(index)


def fetch_regions(regions: pd.DataFrame, index: pd.DatetimeIndex, cache_dir: Path) -> dict[str, np.ndarray]:
    per_region = [fetch_point(r.lat, r.lon, index, cache_dir) for r in regions.itertuples()]
    return {col: np.column_stack([f[col].values for f in per_region]) for col in PARAMETERS.values()}
