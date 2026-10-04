"""Load a real historical disaster inventory (e.g. compiled from NDMA / state portals).

Expected CSV columns: region_id, timestamp, hazard   (hazard in {flood, landslide}).
Each row is one event onset; timestamps are floored to the hour.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def load_inventory(path, regions: pd.DataFrame, index: pd.DatetimeIndex, hazards: list[str]) -> dict[str, np.ndarray]:
    inv = pd.read_csv(path)
    missing = {"region_id", "timestamp", "hazard"} - set(inv.columns)
    if missing:
        raise ValueError(f"event inventory is missing columns: {missing}")
    inv["timestamp"] = pd.to_datetime(inv["timestamp"]).dt.floor("h")
    inv["hazard"] = inv["hazard"].str.lower().str.strip()

    region_pos = {rid: i for i, rid in enumerate(regions["region_id"])}
    unknown = set(inv["region_id"]) - set(region_pos)
    if unknown:
        log.warning("Ignoring inventory rows for unknown regions: %s", sorted(unknown))
    time_pos = pd.Series(np.arange(len(index)), index=index)

    events = {hz: np.zeros((len(index), len(regions)), dtype=np.int8) for hz in hazards}
    for row in inv.itertuples():
        if row.hazard in events and row.region_id in region_pos and row.timestamp in time_pos.index:
            events[row.hazard][time_pos[row.timestamp], region_pos[row.region_id]] = 1
    for hz, arr in events.items():
        log.info("Inventory: %d %s onsets inside the study window", arr.sum(), hz)
    return events
