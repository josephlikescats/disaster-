"""End-to-end pipeline:

Data Sources -> Integration -> Preprocessing -> Validation -> Feature Engineering
-> Model Training -> Probability Estimation (+calibration) -> Risk Classification
-> Evaluation & Visualization -> Early-warning artefacts for the dashboard.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import visualize as viz
from .config import resolve, target_names
from .data import load_dataset
from .evaluation import (
    best_f1_threshold, classification_metrics, false_alarms_per_region_month, lead_time_analysis,
    regional_stability, risk_level_table,
)
from .features import FEATURE_GROUPS, assign_splits, build_features
from .models import DISPLAY_NAMES, SEQUENCE_MODELS, TABULAR_MODELS
from .models.sequence import SequenceRiskModel, sequence_training_endpoints
from .models.tabular import TabularRiskModel, training_rows
from .preprocessing import preprocess
from .risk import RISK_LEVELS
from .validation import validate

log = logging.getLogger(__name__)


def _parse_target(t: str) -> tuple[str, int]:
    _, hazard, h = t.split("_")
    return hazard, int(h.rstrip("h"))


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return str(o)


def _dump(obj, path: Path):
    path.write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")


class Timer:
    def __init__(self):
        self.t = time.time()
        self.steps = {}

    def lap(self, name):
        now = time.time()
        self.steps[name] = round(now - self.t, 1)
        log.info("== %s done in %.1fs", name, now - self.t)
        self.t = now


def run(cfg: dict) -> dict:
    seed = cfg["project"]["seed"]
    rng = np.random.default_rng(seed)
    out = resolve(cfg["project"]["output_dir"])
    dirs = {k: out / k for k in ["data", "models", "reports", "figures"]}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    hazards, horizons = cfg["targets"]["hazards"], cfg["targets"]["horizons"]
    thresholds = tuple(cfg["risk"]["thresholds"])
    alert_p = thresholds[RISK_LEVELS.index(cfg["risk"]["alert_level"]) - 1]
    timer = Timer()

    # 1-2. acquisition + integration ----------------------------------------
    raw, regions, meta = load_dataset(cfg, rng)
    regions.to_csv(dirs["data"] / "regions.csv", index=False)
    timer.lap("data acquisition & integration")

    # 2b. preprocessing --------------------------------------------------------
    clean, prep_report = preprocess(raw, cfg)
    _dump(prep_report, dirs["reports"] / "preprocessing_report.json")
    del raw
    timer.lap("preprocessing")

    # 3. exploratory analysis --------------------------------------------------
    viz.eda_monthly(clean, hazards, dirs["figures"] / "eda_monthly.png")
    flood_region = clean.groupby("region_id")[f"{hazards[0]}_event"].sum().idxmax()
    busiest_year = int(clean.groupby(clean["timestamp"].dt.year)[f"{hazards[0]}_event"].sum().idxmax())
    viz.eda_timeseries(clean, flood_region, busiest_year, hazards,
                       dirs["figures"] / f"eda_timeseries_{flood_region}_{busiest_year}.png")
    timer.lap("exploratory analysis")

    # 4. feature engineering + validation --------------------------------------
    feats, feature_cols, target_cols = build_features(clean, cfg)
    assert target_cols == target_names(cfg)
    del clean
    split = assign_splits(feats, cfg)
    val_report = validate(feats, feature_cols, target_cols, split, regions["region_id"].tolist())
    val_report["feature_groups"] = FEATURE_GROUPS
    _dump(val_report, dirs["reports"] / "validation_report.json")
    corr_cols = ["rainfall_mm", "rain_sum_24h", "rain_sum_72h", "rain_prev_7d", "soil_moisture",
                 "river_level_m", "river_change_6h", "humidity_pct", "temperature_c", "slope_deg",
                 "elevation_m", "ndvi"] + [f"y_{hz}_24h" for hz in hazards if 24 in horizons]
    viz.correlation_heatmap(feats.loc[split == "train"], corr_cols, dirs["figures"] / "eda_correlation.png")
    _dump({"feature_cols": feature_cols, "target_cols": target_cols, "feature_groups": FEATURE_GROUPS},
          dirs["models"] / "features.json")
    timer.lap("feature engineering & validation")

    tr, va, te = (split == "train").to_numpy(), (split == "val").to_numpy(), (split == "test").to_numpy()
    X = feats[feature_cols]
    keep_cols = ["region_id", "timestamp", *[f"{hz}_event" for hz in hazards], *target_cols]
    preds = {"val": feats.loc[va, keep_cols].reset_index(drop=True),
             "test": feats.loc[te, keep_cols].reset_index(drop=True)}
    enabled = cfg["models"]["enabled"]
    calibration = cfg["models"]["calibration"]
    importances = []

    # 5a. tabular models --------------------------------------------------------
    for name in [m for m in enabled if m in TABULAR_MODELS]:
        for target in target_cols:
            y_tr = feats.loc[tr, target].to_numpy()
            rows = training_rows(y_tr, cfg["models"]["negative_stride"], rng)
            model = TabularRiskModel(name, cfg["models"].get(name, {}), calibration, seed)
            model.fit(X[tr].iloc[rows], y_tr[rows], X[va], feats.loc[va, target].to_numpy())
            preds["val"][f"p_{name}__{target}"] = model.predict_proba(X[va])
            preds["test"][f"p_{name}__{target}"] = model.predict_proba(X[te])
            imp = model.feature_importance()
            importances.append(pd.DataFrame({"model": name, "target": target, "feature": imp.index,
                                             "importance": imp.values}))
            joblib.dump(model, dirs["models"] / f"{name}__{target}.joblib")
        timer.lap(f"train {name}")

    # 5b. sequence models ------------------------------------------------------
    seq_names = [m for m in enabled if m in SEQUENCE_MODELS]
    histories = {}
    if seq_names:
        X_all = X.to_numpy(dtype=np.float32)
        Y_all = feats[target_cols].to_numpy(dtype=np.float32)
        codes = pd.factorize(feats["region_id"])[0]
        for name in seq_names:
            model = SequenceRiskModel(name, cfg["models"]["sequence"], calibration, seed)
            ok = model.valid_endpoints(codes)
            train_idx = sequence_training_endpoints(Y_all, np.flatnonzero(tr & ok),
                                                    cfg["models"]["sequence"]["stride"], rng)
            val_idx, test_idx = np.flatnonzero(va & ok), np.flatnonzero(te & ok)
            if len(test_idx) != te.sum() or len(val_idx) != va.sum():
                raise RuntimeError("look-back window does not fit before the val/test periods")
            model.fit(X_all, Y_all, train_idx, val_idx, target_cols)
            pv, pt = model.predict_proba(val_idx), model.predict_proba(test_idx)
            for k, target in enumerate(target_cols):
                preds["val"][f"p_{name}__{target}"] = pv[:, k]
                preds["test"][f"p_{name}__{target}"] = pt[:, k]
            histories[name] = model.history
            model.save(dirs["models"] / f"{name}.pt")
            timer.lap(f"train {name}")
        viz.training_curves(histories, dirs["figures"] / "sequence_training.png")

    models_run = [m for m in enabled if m in TABULAR_MODELS + SEQUENCE_MODELS]

    # 6. evaluation -------------------------------------------------------------
    rows = []
    for name in models_run:
        for target in target_cols:
            hazard, h = _parse_target(target)
            col = f"p_{name}__{target}"
            y_val, p_val = preds["val"][target], preds["val"][col]
            thr = best_f1_threshold(y_val, p_val)
            m = classification_metrics(preds["test"][target], preds["test"][col], thr)
            at_alert = classification_metrics(preds["test"][target], preds["test"][col], alert_p)
            m.update({f"{k}_at_alert": at_alert[k] for k in ["precision", "recall", "f1"]})
            m["val_pr_auc"] = classification_metrics(y_val, p_val, thr)["pr_auc"]
            rows.append({"model": name, "hazard": hazard, "horizon": h, "target": target, **m})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(dirs["reports"] / "metrics.csv", index=False)

    # best model per hazard chosen on VALIDATION PR-AUC (test stays untouched)
    best = (metrics.groupby(["hazard", "model"])["val_pr_auc"].mean().reset_index()
            .sort_values("val_pr_auc", ascending=False).groupby("hazard").head(1)
            .set_index("hazard")["model"].to_dict())

    lt_h = cfg["evaluation"]["lead_time_horizon"] if cfg["evaluation"]["lead_time_horizon"] in horizons else max(horizons)
    lead_rows, risk_tables, stability = [], [], []
    for hazard in hazards:
        target = f"y_{hazard}_{lt_h}h"
        for name in models_run:
            col = f"p_{name}__{target}"
            thr_f1 = float(metrics.query("model == @name and target == @target")["threshold"].iloc[0])
            for rule, thr in [("alert_level", alert_p), ("best_f1", thr_f1)]:
                lt = lead_time_analysis(preds["test"], col, f"{hazard}_event", thr, lt_h)
                fa = false_alarms_per_region_month(preds["test"], col, target, thr)
                lead_rows.append({"hazard": hazard, "model": name, "rule": rule, "threshold": thr, **lt,
                                  "false_alarm_episodes_per_region_month": fa})
        for h in horizons:
            t = f"y_{hazard}_{h}h"
            rt = risk_level_table(preds["test"][t], preds["test"][f"p_{best[hazard]}__{t}"], thresholds)
            rt.insert(0, "target", t)
            rt.insert(0, "model", best[hazard])
            risk_tables.append(rt)
        rs = regional_stability(preds["test"], f"p_{best[hazard]}__{target}", target)
        rs.insert(0, "target", target)
        rs.insert(0, "model", best[hazard])
        stability.append(rs)
    lead = pd.DataFrame(lead_rows)
    lead.to_csv(dirs["reports"] / "lead_time.csv", index=False)
    risk_tab = pd.concat(risk_tables)
    risk_tab.to_csv(dirs["reports"] / "risk_levels.csv", index=False)
    stab = pd.concat(stability)
    stab.to_csv(dirs["reports"] / "regional_stability.csv", index=False)
    imp_df = pd.concat(importances) if importances else pd.DataFrame()
    if not imp_df.empty:
        imp_df.to_csv(dirs["reports"] / "feature_importance.csv", index=False)
    timer.lap("evaluation")

    # 7. figures ----------------------------------------------------------------
    viz.horizon_reliability(metrics, dirs["figures"] / "horizon_reliability.png")
    for hazard in hazards:
        target = f"y_{hazard}_{lt_h}h"
        viz.roc_pr_curves(preds["test"], models_run, target, dirs["figures"] / f"roc_pr_{target}.png")
        viz.reliability_diagram(preds["test"], models_run, target, dirs["figures"] / f"reliability_{target}.png",
                                cfg["evaluation"]["calibration_bins"])
        for imp_model in [m for m in ["xgboost", "random_forest"] if m in models_run]:
            imp = imp_df.query("model == @imp_model and target == @target").set_index("feature")["importance"]
            viz.feature_importance(imp, f"{DISPLAY_NAMES[imp_model]} - top drivers of {target}",
                                   dirs["figures"] / f"importance_{imp_model}_{target}.png")
        col = f"p_{best[hazard]}__{target}"
        region = preds["test"].groupby("region_id")[f"{hazard}_event"].sum().idxmax()
        test_r = preds["test"][preds["test"]["region_id"] == region]
        monsoon = test_r[test_r["timestamp"].dt.month.between(5, 11)]
        viz.risk_timeline(monsoon, col, f"{hazard}_event", region, thresholds,
                          f"{region}: {lt_h} h {hazard} risk ({DISPLAY_NAMES[best[hazard]]}), test monsoon",
                          dirs["figures"] / f"risk_timeline_{hazard}_{region}.png")
        peak_ts = preds["test"].groupby("timestamp")[col].mean().idxmax()
        snap = preds["test"][preds["test"]["timestamp"] == peak_ts].merge(regions, on="region_id")
        viz.risk_map(snap, col, thresholds, f"{hazard.capitalize()} {lt_h} h risk\n{peak_ts:%d %b %Y %H:%M}",
                     dirs["figures"] / f"risk_map_{hazard}.png")
    timer.lap("figures")

    # 8. artefacts for the dashboard ------------------------------------------
    preds["test"].to_parquet(dirs["data"] / "predictions_test.parquet", index=False)
    test_feats = feats.loc[te, ["region_id", "timestamp", *feature_cols]].reset_index(drop=True)
    test_feats.to_parquet(dirs["data"] / "features_test.parquet", index=False)

    summary = {
        "meta": meta,
        "config": cfg,
        "best_model_per_hazard": best,
        "alert_probability": alert_p,
        "timings_s": timer.steps,
        "n_features": len(feature_cols),
        "split_rows": val_report["split_rows"],
    }
    _dump(summary, dirs["reports"] / "summary.json")
    write_markdown_summary(metrics, lead, risk_tab, stab, best, meta, dirs["reports"] / "results_summary.md")
    log.info("All outputs written to %s", out)
    return summary


def write_markdown_summary(metrics, lead, risk_tab, stab, best, meta, path: Path):
    def table(df, cols, fmt=".3f"):
        head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
        body = ""
        for _, r in df[cols].iterrows():
            body += "| " + " | ".join(f"{v:{fmt}}" if isinstance(v, float) else str(v) for v in r) + " |\n"
        return head + body

    lines = ["# Results summary", ""]
    lines.append(f"- Weather source: **{meta.get('weather_source', meta['source'])}**; "
                 f"event labels: **{meta.get('label_source')}**")
    if meta.get("label_source") == "simulated":
        lines.append("- **Event labels are simulated.** Scores show that the pipeline works end to end. "
                     "They are not evidence of real-world skill.")
    lines += ["- Best model per hazard (selected on validation PR-AUC): "
              + ", ".join(f"{h}: **{DISPLAY_NAMES[m]}**" for h, m in best.items()), ""]
    m = metrics.copy()
    m["model"] = m["model"].map(DISPLAY_NAMES)
    lines += ["## Test-set metrics", "",
              table(m, ["hazard", "horizon", "model", "positives", "roc_auc", "pr_auc", "brier_skill", "ece",
                        "precision", "recall", "f1"]), ""]
    lt = lead.copy()
    lt["model"] = lt["model"].map(DISPLAY_NAMES)
    lines += ["## Warning lead time (24 h models)", "",
              table(lt, ["hazard", "model", "rule", "threshold", "events", "detected", "detection_rate",
                         "mean_lead_hours", "false_alarm_episodes_per_region_month"]), ""]
    lines += ["## Risk levels vs. observed event rate (best model)", "",
              table(risk_tab, ["target", "level", "hours", "events", "observed_rate", "share_of_events"]), ""]
    lines += ["## Stability across regions (best model, 24 h)", "",
              table(stab, ["target", "region_id", "positives", "roc_auc", "pr_auc"]), ""]
    path.write_text("\n".join(lines), encoding="utf-8")
