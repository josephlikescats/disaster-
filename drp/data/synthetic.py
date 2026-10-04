"""Physically-motivated synthetic hourly meteorology for monsoon regions.

The generator reproduces the main statistical features of Kerala weather:
a south-west monsoon (Jun-Sep) and north-east monsoon (Oct-Nov), wet/dry
spell persistence, spatially-correlated synoptic systems, orographic
enhancement, afternoon convection, rare multi-day extreme episodes (such
as August 2018), and coupled temperature / humidity / wind / NDVI.

It is a stand-in for IMD gridded observations so the full pipeline can run
offline; it is NOT a substitute for real data when drawing conclusions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import lfilter
from scipy.stats import norm

# Approximate Kerala state-average climatology (mm / month) and wet-day probability.
MONTHLY_RAIN_MM = np.array([12, 18, 35, 110, 230, 650, 700, 450, 300, 310, 170, 40], dtype=float)
P_WET = np.array([0.08, 0.10, 0.15, 0.38, 0.55, 0.88, 0.90, 0.82, 0.68, 0.62, 0.48, 0.16])
# Share of rain falling as afternoon convection (vs. spread through the day), per month.
CONVECTIVE_SHARE = np.array([0.5, 0.5, 0.7, 0.85, 0.75, 0.3, 0.25, 0.3, 0.45, 0.75, 0.7, 0.5])


def ar1(n: int, phi: float, rng: np.random.Generator, cols: int | None = None) -> np.ndarray:
    """Unit-variance AR(1) noise of length n (optionally n x cols)."""
    shape = (n,) if cols is None else (n, cols)
    e = rng.standard_normal(shape)
    return lfilter([np.sqrt(1 - phi**2)], [1, -phi], e, axis=0)


def rolling_sum(x: np.ndarray, window: int) -> np.ndarray:
    """Trailing rolling sum over axis 0 (partial windows at the start)."""
    c = np.vstack([np.zeros((1,) + x.shape[1:]), np.cumsum(x, axis=0)])
    idx = np.arange(1, x.shape[0] + 1)
    return c[idx] - c[np.maximum(idx - window, 0)]


def _diurnal_profiles() -> np.ndarray:
    h = np.arange(24)
    convective = np.exp(-0.5 * ((h - 16) / 2.5) ** 2)
    convective = convective / convective.mean()
    nocturnal = 1.0 + 0.25 * np.cos(2 * np.pi * (h - 4) / 24)
    prof = CONVECTIVE_SHARE[:, None] * convective + (1 - CONVECTIVE_SHARE[:, None]) * nocturnal
    return prof / prof.mean(axis=1, keepdims=True)  # (12, 24), mean 1


def hourly_index(start: str, end: str) -> pd.DatetimeIndex:
    start_ts = pd.Timestamp(start).floor("D")
    end_ts = pd.Timestamp(end).floor("D") + pd.Timedelta(hours=23)
    return pd.date_range(start_ts, end_ts, freq="h")


def generate_weather(regions: pd.DataFrame, index: pd.DatetimeIndex, rng: np.random.Generator) -> dict:
    R = len(regions)
    days = pd.date_range(index[0].floor("D"), index[-1].floor("D"), freq="D")
    D = len(days)
    month = days.month.values - 1
    elev = regions["elevation_m"].values.astype(float)
    lat = regions["lat"].values.astype(float)
    rain_factor = regions["rain_factor"].values.astype(float)

    # --- daily rainfall: Gaussian-copula wet/dry occurrence with shared synoptic driver
    shared = ar1(D, 0.75, rng)
    local = ar1(D, 0.5, rng, R)
    latent = 0.75 * shared[:, None] + np.sqrt(1 - 0.75**2) * local
    wet = latent > norm.ppf(1 - P_WET[month])[:, None]
    mean_wet_mm = MONTHLY_RAIN_MM[month] / (days.days_in_month.values * P_WET[month])
    orographic = 1.0 + 0.0002 * elev
    amount = rng.gamma(0.9, (mean_wet_mm / 0.9)[:, None], size=(D, R))
    amount *= np.exp(0.5 * latent - 0.125) * (rain_factor * orographic)[None, :]

    # --- rare multi-day extreme episodes (depressions / atmospheric rivers)
    for year in np.unique(days.year):
        for _ in range(rng.poisson(1.2)):
            season = (days >= f"{year}-06-01") & (days <= f"{year}-11-10")
            candidates = np.flatnonzero(season)
            if len(candidates) == 0:
                continue
            d0 = rng.choice(candidates)
            dur = int(rng.integers(2, 7))
            centre = rng.uniform(lat.min(), lat.max())
            affected = np.abs(lat - centre) < rng.uniform(1.0, 2.5)
            mult = rng.uniform(2.0, 3.5)
            sl = slice(d0, min(d0 + dur, D))
            amount[sl, affected] *= mult
            wet[sl, affected] = True

    daily = np.where(wet, amount, 0.0)

    # --- disaggregate to hours with bursty Dirichlet weights
    profiles = _diurnal_profiles()[month]                 # (D, 24)
    alpha = 0.6 * profiles[:, None, :]                    # (D, 1, 24)
    w = rng.gamma(np.broadcast_to(alpha, (D, R, 24)))
    w /= w.sum(axis=2, keepdims=True) + 1e-12
    hourly = daily[:, :, None] * w                        # (D, R, 24)
    rain = np.round(hourly.transpose(0, 2, 1).reshape(D * 24, R), 1)

    full_index = pd.date_range(days[0], periods=D * 24, freq="h")
    keep = full_index.isin(index)
    rain = rain[keep]
    T = len(index)
    hour = index.hour.values[:, None].astype(float)
    doy = index.dayofyear.values[:, None].astype(float)

    monsoon = np.exp(-0.5 * ((doy - 200) / 40) ** 2)
    diurnal = np.cos(2 * np.pi * (hour - 14) / 24)
    rain_24h = rolling_sum(rain, 24)

    temp = (
        27.3
        + 1.8 * np.cos(2 * np.pi * (doy - 110) / 365)
        - 0.0065 * elev[None, :]
        + 3.2 * diurnal * (1 - 0.5 * monsoon)
        - 0.7 * np.sqrt(np.minimum(rain, 25.0))
        + 0.7 * ar1(T, 0.95, rng, R)
    )
    humidity = np.clip(
        66 + 16 * monsoon + 12 * (1 - np.exp(-rain_24h / 25)) - 9 * diurnal + 3 * ar1(T, 0.9, rng, R),
        25, 100,
    )
    coastal = (elev < 100).astype(float)[None, :]
    wind = np.clip(1.8 + 2.4 * monsoon + 0.35 * np.sqrt(rain) + 0.6 * coastal + 0.8 * ar1(T, 0.9, rng, R), 0.1, None)

    ndvi_daily = (
        regions["ndvi_base"].values[None, :]
        + 0.10 * np.cos(2 * np.pi * (days.dayofyear.values[:, None] - 275) / 365)
        + 0.015 * ar1(D, 0.9, rng, R)
    )
    ndvi = np.clip(np.repeat(ndvi_daily, 24, axis=0)[keep], 0.05, 0.95)

    return {
        "rainfall_mm": rain,
        "temperature_c": temp,
        "humidity_pct": humidity,
        "wind_speed_ms": wind,
        "ndvi": ndvi,
    }


def seasonal_ndvi(regions: pd.DataFrame, index: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
    """NDVI placeholder for real-weather runs (replace with MODIS/Sentinel NDVI)."""
    doy = index.dayofyear.values[:, None].astype(float)
    base = regions["ndvi_base"].values[None, :]
    noise = 0.01 * ar1(len(index), 0.999, rng, len(regions))
    return np.clip(base + 0.10 * np.cos(2 * np.pi * (doy - 275) / 365) + noise, 0.05, 0.95)
