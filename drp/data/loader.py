"""Data acquisition + integration (pipeline steps 1-2a).

Produces one long, hourly, region-wise table:

    region_id, timestamp,
    rainfall_mm, temperature_c, humidity_pct, wind_speed_ms,   # meteorology (IMD / NASA POWER)
    soil_moisture, river_level_m,                              # hydrology (river stations)
    ndvi,                                                      # earth observation (Sentinel / MODIS)
    flood_event, landslide_event,                              # disaster inventory (NDMA)
    + static geography columns from regions.csv               # USGS / Copernicus
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..config import resolve
from .hazard import simulate_events
from .hydrology import simulate_hydrology
from .inventory import load_inventory
from .regions import load_regions
from .synthetic import generate_weather, hourly_index, seasonal_ndvi

log = logging.getLogger(__name__)

OBSERVED_VARS = [
    "rainfall_mm", "temperature_c", "humidity_pct", "wind_speed_ms",
    "soil_moisture", "river_level_m", "ndvi",
]
STATIC_COLS = [
    "lat", "lon", "elevation_m", "slope_deg", "aspect_deg", "soil_type", "land_cover",
    "dist_river_km", "impervious_frac",
]


def _to_long(arrays: dict[str, np.ndarray], regions: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    R, T = len(regions), len(index)
    out = {
        "region_id": np.repeat(regions["region_id"].values, T),
        "timestamp": np.tile(index.values, R),
    }
    for name, arr in arrays.items():
        out[name] = np.asarray(arr).T.reshape(-1)  # region-major order
    return pd.DataFrame(out)


def inject_sensor_issues(df: pd.DataFrame, missing_rate: float, outlier_rate: float, rng) -> pd.DataFrame:
    """Add realistic gaps (blocks of 1-6 h) and gross errors to observed variables."""
    df = df.copy()
    n = len(df)
    bad_values = {
        "rainfall_mm": lambda x: np.where(rng.random(x.size) < 0.5, 999.0, -5.0),
        "temperature_c": lambda x: x + rng.choice([-40.0, 35.0], x.size),
        "humidity_pct": lambda x: np.full(x.size, 150.0),
        "wind_speed_ms": lambda x: np.full(x.size, 85.0),
        "soil_moisture": lambda x: np.full(x.size, 1.8),
        "river_level_m": lambda x: x * 3 + 5,
        "ndvi": lambda x: np.full(x.size, -3.0),
    }
    for col in OBSERVED_VARS:
        values = df[col].to_numpy(dtype=float, copy=True)
        n_blocks = int(missing_rate * n / 3.5)
        starts = rng.integers(0, n - 6, n_blocks)
        lengths = rng.integers(1, 7, n_blocks)
        for s, l in zip(starts, lengths):
            values[s : s + l] = np.nan
        pos = rng.choice(n, int(outlier_rate * n), replace=False)
        values[pos] = bad_values[col](values[pos])
        df[col] = values
    return df


def integrate_static(df: pd.DataFrame, regions: pd.DataFrame) -> pd.DataFrame:
    return df.merge(regions[["region_id"] + STATIC_COLS], on="region_id", how="left", validate="many_to_one")


def load_dataset(cfg: dict, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    dcfg = cfg["data"]
    source = dcfg["source"]
    hazards = cfg["targets"]["hazards"]
    regions = load_regions(cfg)
    meta = {"source": source, "regions": regions["region_id"].tolist()}

    if source == "csv":
        df = pd.read_csv(resolve(dcfg["csv_path"]), parse_dates=["timestamp"])
        needed = {"region_id", "timestamp", *OBSERVED_VARS, *(f"{h}_event" for h in hazards)}
        if needed - set(df.columns):
            raise ValueError(f"CSV dataset is missing columns: {sorted(needed - set(df.columns))}")
        df = df[df["region_id"].isin(regions["region_id"])]
        meta["label_source"] = "csv"
        return integrate_static(df, regions), regions, meta

    index = hourly_index(dcfg["start"], dcfg["end"])
    if source == "synthetic":
        weather = generate_weather(regions, index, rng)
        meta["weather_source"] = "synthetic"
    elif source == "nasa_power":
        from .nasa_power import fetch_regions

        weather = fetch_regions(regions, index, resolve(dcfg["cache_dir"]))
        weather["ndvi"] = seasonal_ndvi(regions, index, rng)
        meta["weather_source"] = "NASA POWER hourly (real)"
    else:
        raise ValueError(f"unknown data source: {source}")

    rain_filled = np.nan_to_num(weather["rainfall_mm"], nan=0.0).clip(0, None)
    temp_filled = pd.DataFrame(weather["temperature_c"]).interpolate(limit_direction="both").values
    hydro = simulate_hydrology(
        rain_filled, temp_filled, regions["soil_type"].tolist(), regions["impervious_frac"].values
    )

    if dcfg.get("event_inventory"):
        events = load_inventory(resolve(dcfg["event_inventory"]), regions, index, hazards)
        meta["label_source"] = "inventory"
    else:
        events = simulate_events(
            rain_filled, hydro["soil_moisture"], hydro["river_level_m"], regions,
            dcfg["events_per_region_year"], dcfg["refractory_hours"], rng,
            sharpness=dcfg.get("hazard_sharpness", 3.0), latent_noise=dcfg.get("hazard_latent_noise", 0.3),
        )
        meta["label_source"] = "simulated"

    arrays = {
        **{k: weather[k] for k in ["rainfall_mm", "temperature_c", "humidity_pct", "wind_speed_ms", "ndvi"]},
        "soil_moisture": hydro["soil_moisture"],
        "river_level_m": hydro["river_level_m"],
        **{f"{hz}_event": events[hz] for hz in hazards},
    }
    df = _to_long(arrays, regions, index)
    if source == "synthetic" and dcfg.get("inject_sensor_issues", False):
        df = inject_sensor_issues(df, dcfg["missing_rate"], dcfg["outlier_rate"], rng)
    for hz in hazards:
        log.info("%s onsets: %d (%.2f per region-year)", hz, df[f"{hz}_event"].sum(),
                 df[f"{hz}_event"].sum() / len(regions) / (len(index) / 8760))
    return integrate_static(df, regions), regions, meta
