"""Probability -> risk level translation and alerting.

    p < 0.25          Low      routine monitoring
    0.25 <= p < 0.50  Medium   increased observation and preparedness
    0.50 <= p <= 0.75 High     pre-alert and response readiness
    p > 0.75          Severe   urgent warning and intervention planning
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RISK_LEVELS = ["Low", "Medium", "High", "Severe"]
DECISIONS = {
    "Low": "Routine monitoring",
    "Medium": "Increased observation and preparedness",
    "High": "Pre-alert and response readiness",
    "Severe": "Urgent warning and intervention planning",
}


def categorize(prob, thresholds=(0.25, 0.50, 0.75), labels=RISK_LEVELS) -> np.ndarray:
    p = np.asarray(prob, dtype=float)
    t1, t2, t3 = thresholds
    idx = np.select([p < t1, p < t2, p <= t3], [0, 1, 2], default=3)
    return np.asarray(labels, dtype=object)[idx]


def level_index(levels) -> np.ndarray:
    order = {lvl: i for i, lvl in enumerate(RISK_LEVELS)}
    return np.array([order[l] for l in levels])


def alerts(frame: pd.DataFrame, prob_col: str, alert_level: str = "High",
           thresholds=(0.25, 0.50, 0.75)) -> pd.DataFrame:
    """Rows whose risk is at or above `alert_level`, newest first."""
    levels = categorize(frame[prob_col], thresholds)
    mask = level_index(levels) >= RISK_LEVELS.index(alert_level)
    out = frame.loc[mask].copy()
    out["risk_level"] = levels[mask]
    out["action"] = out["risk_level"].map(DECISIONS)
    return out.sort_values("timestamp", ascending=False)
