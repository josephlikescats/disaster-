"""Simulated flood & landslide event inventory.

When no real disaster inventory is supplied, event onsets are drawn from a
hazard model whose hourly onset probability depends on the (true, noise-free)
environmental state plus an unobserved slowly-varying factor. Intercepts are
solved so each hazard reaches a target average number of onsets per region
per year, and a refractory period prevents one storm producing a run of
duplicate onsets.

Floods respond to river level relative to the region's danger level, short
intense rain and saturated soils, modulated by low elevation, impervious
cover, wetlands and clay soils. Landslides respond to multi-day cumulative
rain, short-burst intensity and soil saturation, modulated mostly by slope.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .synthetic import ar1, rolling_sum


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def hazard_scores(rain, soil_moisture, river_level, regions: pd.DataFrame) -> dict[str, np.ndarray]:
    """Return intercept-free log-odds per hazard, each (T, R)."""
    elev = regions["elevation_m"].values.astype(float)
    slope = regions["slope_deg"].values.astype(float)
    imperv = regions["impervious_frac"].values.astype(float)
    soil = regions["soil_type"].values
    cover = regions["land_cover"].values
    dist_river = regions["dist_river_km"].values.astype(float)

    rain_3h = rolling_sum(rain, 3)
    rain_6h = rolling_sum(rain, 6)
    rain_72h = rolling_sum(rain, 72)

    danger = np.quantile(river_level, 0.995, axis=0)
    median = np.median(river_level, axis=0)
    excess = np.clip((river_level - danger) / (danger - median + 1e-6), -2.0, 3.0)

    flood_susc = (
        -0.6 * np.log1p(elev / 30.0)
        + 1.5 * imperv
        + 0.8 * (cover == "wetland")
        + 0.6 * (soil == "clay")
        - 0.04 * slope
        + 0.3 * (dist_river < 1.5)
    )
    flood = 2.5 * excess + 0.5 * np.log1p(rain_6h) + 2.0 * (soil_moisture - 0.5) + flood_susc[None, :]

    slide_susc = (
        0.15 * slope
        + 0.4 * np.isin(soil, ["loam", "laterite"])
        + 0.5 * np.isin(cover, ["mixed", "cropland"])
    )
    landslide = (
        0.9 * np.log1p(rain_72h) + 0.5 * np.log1p(rain_3h) + 3.0 * (soil_moisture - 0.6) + slide_susc[None, :]
    )
    return {"flood": flood, "landslide": landslide}


def _sample_onsets(prob: np.ndarray, u: np.ndarray, refractory: int) -> np.ndarray:
    events = np.zeros(prob.shape, dtype=np.int8)
    for r in range(prob.shape[1]):
        last = -10**9
        for t in np.flatnonzero(u[:, r] < prob[:, r]):
            if t - last >= refractory:
                events[t, r] = 1
                last = t
    return events


def simulate_events(
    rain: np.ndarray,
    soil_moisture: np.ndarray,
    river_level: np.ndarray,
    regions: pd.DataFrame,
    targets_per_region_year: dict[str, float],
    refractory: int,
    rng: np.random.Generator,
    sharpness: float = 2.0,
    latent_noise: float = 0.3,
) -> dict[str, np.ndarray]:
    T, R = rain.shape
    years = T / 8760.0
    scores = hazard_scores(rain, soil_moisture, river_level, regions)
    events = {}
    for hazard, z in scores.items():
        # sharpness concentrates onsets in extreme conditions; latent noise = unobserved
        # factors (e.g. dam releases, drainage blockage) no model can see
        z = sharpness * z + latent_noise * ar1(T, 0.99, rng, R)
        u = rng.random((T, R))
        target = targets_per_region_year[hazard] * R * years
        lo, hi = -80.0, 20.0
        for _ in range(40):                          # bisection on the intercept
            mid = 0.5 * (lo + hi)
            n = _sample_onsets(_sigmoid(mid + z), u, refractory).sum()
            lo, hi = (mid, hi) if n < target else (lo, mid)
        events[hazard] = _sample_onsets(_sigmoid(0.5 * (lo + hi) + z), u, refractory)
    return events
