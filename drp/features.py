"""Feature engineering + target construction (pipeline step 4).

All features at time t use information available up to and including t.
Targets for horizon h are 1 if an event onset occurs in (t, t+h].
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data.regions import LAND_COVERS, SOIL_TYPES

FEATURE_GROUPS: dict[str, list[str]] = {}  # filled by build_features; used for reporting


def _future_any(event: pd.Series, h: int) -> pd.Series:
    """1 if any event in (t, t+h]; NaN where the window runs past the data."""
    rev = event[::-1].rolling(h, min_periods=h).max()[::-1]
    return rev.shift(-1)


def _region_features(g: pd.DataFrame, hazards: list[str], horizons: list[int]) -> pd.DataFrame:
    f = pd.DataFrame(index=g.index)
    rain = g["rainfall_mm"]
    sm = g["soil_moisture"]
    river = g["river_level_m"]
    groups = FEATURE_GROUPS

    # meteorological (current)
    meteo = ["rainfall_mm", "temperature_c", "humidity_pct", "wind_speed_ms"]
    for c in meteo:
        f[c] = g[c]

    # lag features: 1 h, and previous 1 / 7 / 30 day rainfall totals
    f["rain_lag_1h"] = rain.shift(1)
    rain_24h = rain.rolling(24, min_periods=1).sum()
    f["rain_prev_day"] = rain_24h.shift(24)
    f["rain_prev_7d"] = rain.rolling(168, min_periods=1).sum().shift(24)
    f["rain_prev_30d"] = rain.rolling(720, min_periods=1).sum().shift(24)
    lag = ["rain_lag_1h", "rain_prev_day", "rain_prev_7d", "rain_prev_30d"]

    # rolling statistics
    for w in (3, 6, 24, 72, 168, 720):
        f[f"rain_sum_{w}h"] = rain.rolling(w, min_periods=1).sum()
    f["rain_max_3h"] = rain.rolling(3, min_periods=1).max()
    f["rain_max_24h"] = rain.rolling(24, min_periods=1).max()
    f["rain_std_24h"] = rain.rolling(24, min_periods=2).std()
    f["wet_hours_24h"] = (rain > 0.5).rolling(24, min_periods=1).sum()
    f["temp_mean_24h"] = g["temperature_c"].rolling(24, min_periods=1).mean()
    f["humidity_mean_24h"] = g["humidity_pct"].rolling(24, min_periods=1).mean()
    f["wind_max_24h"] = g["wind_speed_ms"].rolling(24, min_periods=1).max()
    rolling = [c for c in f.columns if c.startswith(("rain_sum", "rain_max", "rain_std", "wet_hours"))] + [
        "temp_mean_24h", "humidity_mean_24h", "wind_max_24h"]

    # hydrological
    f["soil_moisture"] = sm
    f["sm_lag_24h"] = sm.shift(24)
    f["sm_change_24h"] = sm - sm.shift(24)
    f["sm_mean_7d"] = sm.rolling(168, min_periods=1).mean()
    f["river_level_m"] = river
    for w in (1, 6, 24):
        f[f"river_change_{w}h"] = river - river.shift(w)
    f["river_mean_24h"] = river.rolling(24, min_periods=1).mean()
    f["river_std_24h"] = river.rolling(24, min_periods=2).std()
    f["river_max_72h"] = river.rolling(72, min_periods=1).max()
    f["river_anom_30d"] = river - river.rolling(720, min_periods=1).mean()
    hydro = ["soil_moisture", "sm_lag_24h", "sm_change_24h", "sm_mean_7d", "river_level_m",
             "river_change_1h", "river_change_6h", "river_change_24h", "river_mean_24h",
             "river_std_24h", "river_max_72h", "river_anom_30d"]

    # geographical (static + vegetation)
    f["elevation_m"] = g["elevation_m"]
    f["slope_deg"] = g["slope_deg"]
    f["aspect_sin"] = np.sin(np.deg2rad(g["aspect_deg"]))
    f["aspect_cos"] = np.cos(np.deg2rad(g["aspect_deg"]))
    f["dist_river_km"] = g["dist_river_km"]
    f["impervious_frac"] = g["impervious_frac"]
    for s in SOIL_TYPES:
        f[f"soil_{s}"] = (g["soil_type"] == s).astype(np.int8)
    for lc in LAND_COVERS:
        f[f"cover_{lc}"] = (g["land_cover"] == lc).astype(np.int8)
    f["ndvi"] = g["ndvi"]
    f["ndvi_anom_30d"] = g["ndvi"] - g["ndvi"].rolling(720, min_periods=1).mean()
    geo = ["elevation_m", "slope_deg", "aspect_sin", "aspect_cos", "dist_river_km", "impervious_frac",
           *[f"soil_{s}" for s in SOIL_TYPES], *[f"cover_{lc}" for lc in LAND_COVERS], "ndvi", "ndvi_anom_30d"]

    # interaction indices (domain-motivated)
    f["rain72_x_slope"] = f["rain_sum_72h"] * g["slope_deg"] / 10.0
    f["rain24_x_sm"] = f["rain_sum_24h"] * sm
    interaction = ["rain72_x_slope", "rain24_x_sm"]

    # temporal / seasonal
    ts = g["timestamp"]
    f["hour_sin"] = np.sin(2 * np.pi * ts.dt.hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * ts.dt.hour / 24)
    f["doy_sin"] = np.sin(2 * np.pi * ts.dt.dayofyear / 365.25)
    f["doy_cos"] = np.cos(2 * np.pi * ts.dt.dayofyear / 365.25)
    f["sw_monsoon"] = ts.dt.month.between(6, 9).astype(np.int8)
    f["ne_monsoon"] = ts.dt.month.between(10, 12).astype(np.int8)
    temporal = ["hour_sin", "hour_cos", "doy_sin", "doy_cos", "sw_monsoon", "ne_monsoon"]

    # historical disaster record features (past only)
    history = []
    for hz in hazards:
        ev = g[f"{hz}_event"].astype(float)
        f[f"{hz}_events_past_365d"] = ev.rolling(8760, min_periods=1).sum()
        last = pd.Series(np.where(ev > 0, np.arange(len(ev)), np.nan), index=ev.index).ffill()
        f[f"hours_since_{hz}"] = np.minimum(np.arange(len(ev)) - last.fillna(-10**6), 8760)
        history += [f"{hz}_events_past_365d", f"hours_since_{hz}"]

    # targets
    for hz in hazards:
        for h in horizons:
            f[f"y_{hz}_{h}h"] = _future_any(g[f"{hz}_event"].astype(float), h)

    groups.clear()
    groups.update({"meteorological": meteo, "lag": lag, "rolling": rolling, "hydrological": hydro,
                   "geographical": geo, "interaction": interaction, "temporal": temporal,
                   "historical": history})
    return f


def build_features(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, list[str], list[str]]:
    hazards = cfg["targets"]["hazards"]
    horizons = cfg["targets"]["horizons"]
    warmup = cfg["features"]["warmup_hours"]
    parts = []
    for _, g in df.groupby("region_id", sort=True):
        g = g.sort_values("timestamp").reset_index(drop=True)
        f = _region_features(g, hazards, horizons)
        f.insert(0, "timestamp", g["timestamp"])
        f.insert(0, "region_id", g["region_id"])
        for hz in hazards:
            f[f"{hz}_event"] = g[f"{hz}_event"].astype(np.int8)
        parts.append(f.iloc[warmup:])
    out = pd.concat(parts, ignore_index=True)

    target_cols = [f"y_{hz}_{h}h" for hz in hazards for h in horizons]
    out = out.dropna(subset=target_cols).reset_index(drop=True)
    out[target_cols] = out[target_cols].astype(np.int8)
    feature_cols = [c for grp in FEATURE_GROUPS.values() for c in grp]
    out[feature_cols] = out[feature_cols].fillna(0.0).astype(np.float32)
    return out, feature_cols, target_cols


def assign_splits(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """Chronological train/val/test split with an embargo before each boundary.

    Rows within max(horizon) hours before a boundary are dropped ('gap') so
    no training target window overlaps the next period.
    """
    embargo = pd.Timedelta(hours=max(cfg["targets"]["horizons"]))
    train_end = pd.Timestamp(cfg["split"]["train_end"])
    val_end = pd.Timestamp(cfg["split"]["val_end"])
    ts = df["timestamp"]
    split = pd.Series("test", index=df.index)
    split[ts <= val_end] = "val"
    split[ts <= train_end] = "train"
    split[(ts > train_end - embargo) & (ts <= train_end)] = "gap"
    split[(ts > val_end - embargo) & (ts <= val_end)] = "gap"
    return split
