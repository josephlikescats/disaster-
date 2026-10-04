"""Data validation (pipeline step 4 of the Phase-I design).

Checks timestamp consistency, identifiers, residual missingness, class
imbalance per split, and potential leakage, and records everything in a
JSON report so preprocessing decisions stay reproducible.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


class ValidationError(RuntimeError):
    pass


def validate(df: pd.DataFrame, feature_cols: list[str], target_cols: list[str], split: pd.Series,
             known_regions: list[str]) -> dict:
    issues: list[str] = []
    warnings: list[str] = []
    report: dict = {"n_rows": int(len(df)), "n_features": len(feature_cols)}

    # identifiers and timestamps
    unknown = sorted(set(df["region_id"]) - set(known_regions))
    if unknown:
        issues.append(f"unknown region ids: {unknown}")
    dups = int(df.duplicated(["region_id", "timestamp"]).sum())
    report["duplicate_rows"] = dups
    if dups:
        issues.append(f"{dups} duplicate (region, timestamp) rows")
    gaps = {}
    for rid, g in df.groupby("region_id"):
        steps = g["timestamp"].diff().dropna()
        gaps[rid] = int((steps != pd.Timedelta(hours=1)).sum())
    report["timestamp_discontinuities"] = gaps
    if any(gaps.values()):
        warnings.append("non-hourly steps found (expected only where dropped target rows end a series)")
    report["time_range"] = [str(df["timestamp"].min()), str(df["timestamp"].max())]

    # missingness / finiteness
    values = df[feature_cols].to_numpy(dtype=np.float64)
    nonfinite = int((~np.isfinite(values)).sum())
    report["non_finite_feature_values"] = nonfinite
    if nonfinite:
        issues.append(f"{nonfinite} non-finite feature values")

    # chronology of splits
    ranges = {}
    for name in ["train", "val", "test"]:
        ts = df.loc[split == name, "timestamp"]
        if ts.empty:
            issues.append(f"split '{name}' is empty")
            continue
        ranges[name] = [str(ts.min()), str(ts.max())]
    report["split_ranges"] = ranges
    if len(ranges) == 3 and not (ranges["train"][1] < ranges["val"][0] <= ranges["val"][1] < ranges["test"][0]):
        issues.append("splits are not chronologically ordered")
    report["split_rows"] = split.value_counts().to_dict()

    # class balance
    balance = {}
    for name in ["train", "val", "test"]:
        part = df.loc[split == name, target_cols]
        balance[name] = {c: {"positives": int(part[c].sum()), "rate": round(float(part[c].mean()), 5)}
                         for c in target_cols}
        for c in target_cols:
            if part[c].sum() == 0:
                issues.append(f"no positive examples of {c} in {name}")
    report["class_balance"] = balance

    # leakage: target names in features, or a feature nearly identical to a target
    leaked = [c for c in feature_cols if c.startswith("y_")]
    if leaked:
        issues.append(f"target columns used as features: {leaked}")
    sample = df.loc[split == "train"].sample(min(50000, int((split == "train").sum())), random_state=0)
    suspicious = {}
    for t in target_cols:
        y = sample[t].astype(float)
        if y.std() == 0:
            continue
        corr = sample[feature_cols].astype(float).corrwith(y).abs()
        top = corr.sort_values(ascending=False).head(1)
        if len(top) and top.iloc[0] > 0.95:
            suspicious[t] = {top.index[0]: round(float(top.iloc[0]), 3)}
    report["leakage_suspects"] = suspicious
    if suspicious:
        issues.append(f"features almost perfectly correlated with targets: {suspicious}")

    report["issues"] = issues
    report["warnings"] = warnings
    report["passed"] = not issues
    for w in warnings:
        log.warning("Validation: %s", w)
    if issues:
        raise ValidationError("; ".join(issues))
    log.info("Validation passed (%d rows, %d features)", len(df), len(feature_cols))
    return report
