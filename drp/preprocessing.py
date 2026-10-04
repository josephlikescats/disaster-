"""Preprocessing (pipeline step 2b): temporal alignment, outliers, missing values.

Scaling is deliberately NOT done here: scalers are fitted on the training
split only, inside each model, so no test-period statistics leak in.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .data.loader import OBSERVED_VARS

log = logging.getLogger(__name__)

PHYSICAL_BOUNDS = {
    "rainfall_mm": (0.0, 300.0),     # mm/h; world record hourly totals are ~300 mm
    "temperature_c": (-10.0, 50.0),
    "humidity_pct": (0.0, 100.0),
    "wind_speed_ms": (0.0, 60.0),
    "soil_moisture": (0.0, 1.0),
    "river_level_m": (0.0, 30.0),
    "ndvi": (-1.0, 1.0),
}


def align_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure each region has exactly one row per hour over its time span."""
    df = df.drop_duplicates(["region_id", "timestamp"], keep="last")
    parts = []
    for rid, g in df.groupby("region_id", sort=True):
        full = pd.date_range(g["timestamp"].min(), g["timestamp"].max(), freq="h")
        g = g.set_index("timestamp").reindex(full)
        g.index.name = "timestamp"
        g["region_id"] = rid
        static = [c for c in g.columns if g[c].dtype == object or c.endswith("_event")]
        g[static] = g[static].ffill().bfill()
        parts.append(g.reset_index())
    return pd.concat(parts, ignore_index=True)


def _isolated_spikes(x: pd.Series, threshold: float) -> pd.Series:
    """A value that jumps away from BOTH neighbours by > threshold in opposite directions."""
    d_prev = x - x.shift(1)
    d_next = x - x.shift(-1)
    return (d_prev.abs() > threshold) & (d_next.abs() > threshold) & (np.sign(d_prev) == np.sign(d_next))


def preprocess(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    pcfg = cfg["preprocessing"]
    report: dict = {"rows_in": int(len(df))}
    df = align_hourly(df).sort_values(["region_id", "timestamp"]).reset_index(drop=True)
    report["rows_after_alignment"] = int(len(df))
    report["missing_before"] = {c: int(df[c].isna().sum()) for c in OBSERVED_VARS}

    # 1. physical-bounds check
    report["out_of_bounds"] = {}
    for col, (lo, hi) in PHYSICAL_BOUNDS.items():
        bad = (df[col] < lo) | (df[col] > hi)
        report["out_of_bounds"][col] = int(bad.sum())
        df.loc[bad, col] = np.nan

    # 2. isolated spike detection (per region)
    report["spikes"] = {}
    for col, thr in pcfg["spike_thresholds"].items():
        mask = df.groupby("region_id", group_keys=False)[col].apply(lambda s: _isolated_spikes(s, thr))
        report["spikes"][col] = int(mask.sum())
        df.loc[mask, col] = np.nan

    # 3. imputation: short gaps interpolated, remaining gaps carried forward/back
    limit = pcfg["interpolation_limit_hours"]

    for _, idx in df.groupby("region_id").groups.items():
        block = df.loc[idx, OBSERVED_VARS].interpolate(limit=limit, limit_area="inside")
        block["rainfall_mm"] = block["rainfall_mm"].fillna(0.0)
        df.loc[idx, OBSERVED_VARS] = block.ffill().bfill()
    report["missing_after"] = {c: int(df[c].isna().sum()) for c in OBSERVED_VARS}
    log.info("Preprocessing: out-of-bounds=%s spikes=%s", report["out_of_bounds"], report["spikes"])
    return df.reset_index(drop=True), report
