"""Evaluation: classification quality, probabilistic quality, horizon
reliability, risk-level behaviour, warning lead time and regional stability."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, brier_score_loss, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score,
)

from .risk import RISK_LEVELS, categorize


def expected_calibration_error(y, p, bins: int = 10) -> float:
    """ECE with equal-mass bins (equal-width bins are uninformative for rare events)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    order = np.argsort(p)
    ece = 0.0
    for chunk in np.array_split(order, bins):
        if len(chunk):
            ece += len(chunk) / len(p) * abs(y[chunk].mean() - p[chunk].mean())
    return float(ece)


def reliability_curve(y, p, bins: int = 10) -> pd.DataFrame:
    y, p = np.asarray(y, float), np.asarray(p, float)
    order = np.argsort(p)
    rows = [{"mean_predicted": p[c].mean(), "observed_rate": y[c].mean(), "n": len(c)}
            for c in np.array_split(order, bins) if len(c)]
    return pd.DataFrame(rows)


def best_f1_threshold(y, p) -> float:
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return float(thr[np.argmax(f1[:-1])]) if len(thr) else 0.5


def classification_metrics(y, p, threshold: float) -> dict:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, float)
    pred = (p >= threshold).astype(int)
    base = y.mean()
    brier = brier_score_loss(y, p)
    brier_ref = base * (1 - base)
    has_both = 0 < y.sum() < len(y)
    return {
        "positives": int(y.sum()),
        "base_rate": float(base),
        "roc_auc": float(roc_auc_score(y, p)) if has_both else np.nan,
        "pr_auc": float(average_precision_score(y, p)) if has_both else np.nan,
        "brier": float(brier),
        "brier_skill": float(1 - brier / brier_ref) if brier_ref > 0 else np.nan,
        "ece": expected_calibration_error(y, p),
        "threshold": float(threshold),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
    }


def risk_level_table(y, p, thresholds) -> pd.DataFrame:
    levels = categorize(p, thresholds)
    frame = pd.DataFrame({"level": levels, "y": np.asarray(y)})
    out = frame.groupby("level")["y"].agg(hours="size", events="sum", observed_rate="mean")
    out = out.reindex(RISK_LEVELS).fillna(0)
    out[["hours", "events"]] = out[["hours", "events"]].astype(int)
    out["share_of_hours"] = out["hours"] / out["hours"].sum()
    out["share_of_events"] = out["events"] / max(out["events"].sum(), 1)
    return out.reset_index()


def lead_time_analysis(frame: pd.DataFrame, prob_col: str, event_col: str, threshold: float,
                       horizon: int) -> dict:
    """For each event onset, how many hours before it did the alarm first fire?

    The alarm counts if p >= threshold at some hour in [onset - horizon, onset - 1];
    lead time is measured from the earliest such hour.
    """
    leads = []
    for _, g in frame.groupby("region_id"):
        g = g.sort_values("timestamp")
        p = g[prob_col].to_numpy()
        onsets = np.flatnonzero(g[event_col].to_numpy() == 1)
        for o in onsets:
            lo = max(o - horizon, 0)
            hit = np.flatnonzero(p[lo:o] >= threshold)
            leads.append(o - (lo + hit[0]) if len(hit) else 0)
    leads = np.array(leads)
    detected = leads > 0
    return {
        "events": int(len(leads)),
        "detected": int(detected.sum()),
        "detection_rate": float(detected.mean()) if len(leads) else np.nan,
        "mean_lead_hours": float(leads[detected].mean()) if detected.any() else 0.0,
        "median_lead_hours": float(np.median(leads[detected])) if detected.any() else 0.0,
    }


def false_alarms_per_region_month(frame: pd.DataFrame, prob_col: str, target_col: str,
                                  threshold: float) -> float:
    """Alarm episodes (contiguous runs above threshold) not followed by an event."""
    episodes = 0
    for _, g in frame.groupby("region_id"):
        above = g[prob_col].to_numpy() >= threshold
        y = g[target_col].to_numpy()
        starts = np.flatnonzero(above & ~np.r_[False, above[:-1]])
        ends = np.flatnonzero(above & ~np.r_[above[1:], False])
        episodes += sum(1 for s, e in zip(starts, ends) if y[s : e + 1].max() == 0)
    months = frame["timestamp"].dt.to_period("M").nunique()
    return episodes / max(frame["region_id"].nunique() * months, 1)


def regional_stability(frame: pd.DataFrame, prob_col: str, target_col: str) -> pd.DataFrame:
    rows = []
    for rid, g in frame.groupby("region_id"):
        y = g[target_col].to_numpy()
        if 0 < y.sum() < len(y):
            rows.append({"region_id": rid, "positives": int(y.sum()),
                         "roc_auc": roc_auc_score(y, g[prob_col]), "pr_auc": average_precision_score(y, g[prob_col])})
        else:
            rows.append({"region_id": rid, "positives": int(y.sum()), "roc_auc": np.nan, "pr_auc": np.nan})
    return pd.DataFrame(rows)
