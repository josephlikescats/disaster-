"""Conceptual hydrology: soil-moisture bucket + Nash-cascade river routing.

Used to derive soil moisture and river level from rainfall when direct
observations are unavailable (simulation, or NASA POWER meteorology where
river gauges are not openly accessible). All arrays are shaped (T, R):
hours x regions.
"""
from __future__ import annotations

import numpy as np

SOIL_CAPACITY_MM = {"sandy": 70.0, "laterite": 110.0, "loam": 130.0, "clay": 160.0}
SOIL_PERCOLATION = {"sandy": 0.006, "laterite": 0.003, "loam": 0.0025, "clay": 0.0012}


def simulate_hydrology(
    rain: np.ndarray,
    temp: np.ndarray,
    soil_types: list[str],
    impervious: np.ndarray,
    sm0: float = 0.35,
    tau_fast: float = 6.0,
    tau_slow: float = 96.0,
) -> dict[str, np.ndarray]:
    T, R = rain.shape
    cap = np.array([SOIL_CAPACITY_MM[s] for s in soil_types])
    perc_rate = np.array([SOIL_PERCOLATION[s] for s in soil_types])
    impervious = np.asarray(impervious, dtype=float)

    k_fast = np.exp(-1.0 / tau_fast)
    k_slow = np.exp(-1.0 / tau_slow)

    sm = np.full(R, sm0)
    fast1 = np.zeros(R)
    fast2 = np.zeros(R)
    slow = np.full(R, 5.0)

    soil_out = np.empty((T, R))
    q_out = np.empty((T, R))

    for t in range(T):
        p = rain[t]
        # saturation-excess + infiltration-excess runoff on pervious area
        pervious_runoff = np.minimum(p, p * sm**3 + np.maximum(p - 25.0, 0.0) * 0.5)
        runoff = impervious * p + (1.0 - impervious) * pervious_runoff
        infil = p - runoff
        et = 0.17 * np.clip(temp[t] / 27.0, 0.0, 1.6) * sm          # mm/h
        perc = perc_rate * cap * sm**2                                # mm/h
        sm = sm + (infil - et - perc) / cap
        overflow = np.maximum(sm - 1.0, 0.0) * cap
        sm = np.clip(sm, 0.02, 1.0)
        runoff = runoff + overflow

        fast1 = k_fast * fast1 + runoff
        out1 = (1.0 - k_fast) * fast1
        fast1 -= out1
        fast2 = k_fast * fast2 + out1
        out2 = (1.0 - k_fast) * fast2
        fast2 -= out2
        slow = k_slow * slow + perc
        q = out2 + (1.0 - k_slow) * slow

        soil_out[t] = sm
        q_out[t] = q

    river_level = 1.0 + 1.5 * np.power(np.maximum(q_out, 0.0), 0.6)
    return {"soil_moisture": soil_out, "discharge": q_out, "river_level_m": river_level}
