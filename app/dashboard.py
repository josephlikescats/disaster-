"""Early-warning dashboard.

    streamlit run app/dashboard.py
    streamlit run app/dashboard.py -- --outputs outputs_quick
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from drp.models import DISPLAY_NAMES  # noqa: E402
from drp.risk import DECISIONS, RISK_LEVELS, categorize, level_index  # noqa: E402
from drp.visualize import MODEL_COLORS, RISK_COLORS  # noqa: E402

FEATURE_LABELS = {
    "rain_sum_72h": ("72 h rainfall", "mm"), "rain_sum_24h": ("24 h rainfall", "mm"),
    "rain_sum_6h": ("6 h rainfall", "mm"), "rain_sum_3h": ("3 h rainfall", "mm"),
    "rain_sum_168h": ("7 day rainfall", "mm"), "rain_sum_720h": ("30 day rainfall", "mm"),
    "rain_prev_7d": ("previous 7 day rainfall", "mm"), "rain_prev_day": ("previous day rainfall", "mm"),
    "rain_max_3h": ("max hourly rain (3 h)", "mm/h"), "rain_max_24h": ("max hourly rain (24 h)", "mm/h"),
    "rainfall_mm": ("current rainfall", "mm/h"), "soil_moisture": ("soil moisture", ""),
    "sm_mean_7d": ("7 day soil moisture", ""), "river_level_m": ("river level", "m"),
    "river_max_72h": ("72 h max river level", "m"), "river_change_6h": ("6 h river rise", "m"),
    "river_change_24h": ("24 h river rise", "m"), "river_mean_24h": ("24 h mean river level", "m"),
    "river_anom_30d": ("river level vs 30 day mean", "m"), "rain72_x_slope": ("rain x slope index", ""),
    "rain24_x_sm": ("rain x soil-moisture index", ""), "slope_deg": ("slope", "deg"),
    "elevation_m": ("elevation", "m"), "humidity_pct": ("humidity", "%"),
}

st.set_page_config(page_title="Disaster Risk Early Warning", layout="wide")


def _outputs_dir() -> Path:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default=os.environ.get("DRP_OUTPUTS", "outputs"))
    args, _ = ap.parse_known_args()
    return ROOT / args.outputs


OUT = _outputs_dir()


@st.cache_data
def load_artifacts(out: str):
    out = Path(out)
    preds = pd.read_parquet(out / "data" / "predictions_test.parquet")
    regions = pd.read_csv(out / "data" / "regions.csv")
    metrics = pd.read_csv(out / "reports" / "metrics.csv")
    lead = pd.read_csv(out / "reports" / "lead_time.csv")
    risk_tab = pd.read_csv(out / "reports" / "risk_levels.csv")
    summary = json.loads((out / "reports" / "summary.json").read_text(encoding="utf-8"))
    validation = json.loads((out / "reports" / "validation_report.json").read_text(encoding="utf-8"))
    return preds, regions, metrics, lead, risk_tab, summary, validation


@st.cache_data
def load_test_features(out: str) -> pd.DataFrame:
    return pd.read_parquet(Path(out) / "data" / "features_test.parquet")


@st.cache_resource
def load_explainer(out: str, target: str):
    for name in ["xgboost", "lightgbm", "logistic_regression"]:
        path = Path(out) / "models" / f"{name}__{target}.joblib"
        if path.exists():
            return joblib.load(path)
    return None


def describe_driver(feature: str, value: float) -> str:
    label, unit = FEATURE_LABELS.get(feature, (feature.replace("_", " "), ""))
    return f"{label} {value:,.1f} {unit}".strip()


def top_drivers(out: str, target: str, rows: pd.DataFrame, k: int = 3) -> list[str]:
    model = load_explainer(out, target)
    if model is None or rows.empty:
        return [""] * len(rows)
    contrib = model.contributions(rows[model.feature_names])
    if contrib is None:
        return [""] * len(rows)
    texts = []
    for idx, c in contrib.iterrows():
        top = c[c > 0].sort_values(ascending=False).head(k)
        texts.append(", ".join(describe_driver(f, rows.loc[idx, f]) for f in top.index) or "no strong drivers")
    return texts


if not (OUT / "reports" / "summary.json").exists():
    st.error(f"No pipeline outputs found in `{OUT}`. Run `python run_pipeline.py` first.")
    st.stop()

preds, regions, metrics, lead, risk_tab, summary, validation = load_artifacts(str(OUT))
cfg = summary["config"]
thresholds = tuple(cfg["risk"]["thresholds"])
hazards, horizons = cfg["targets"]["hazards"], cfg["targets"]["horizons"]
models = [m for m in cfg["models"]["enabled"] if any(c.startswith(f"p_{m}__") for c in preds.columns)]
best = summary["best_model_per_hazard"]
names = regions.set_index("region_id")["name"].to_dict()

# ------------------------------------------------------------------ sidebar
st.sidebar.title("Disaster Risk Early Warning")
hazard = st.sidebar.selectbox("Hazard", hazards, format_func=str.capitalize)
horizon = st.sidebar.selectbox("Forecast horizon", horizons, index=len(horizons) - 1, format_func=lambda h: f"{h} hours")
model = st.sidebar.selectbox("Model", models, index=models.index(best[hazard]) if best[hazard] in models else 0,
                             format_func=lambda m: DISPLAY_NAMES[m] + (" (best on validation)" if m == best[hazard] else ""))
target = f"y_{hazard}_{horizon}h"
pcol = f"p_{model}__{target}"

label_source = summary["meta"].get("label_source")
st.sidebar.markdown("---")
st.sidebar.caption(f"Weather: **{summary['meta'].get('weather_source', summary['meta']['source'])}**  \n"
                   f"Event labels: **{label_source}**  \nOutputs: `{OUT.name}`")
if label_source == "simulated":
    st.sidebar.warning("Event labels are simulated, so scores only show that the pipeline works. "
                       "They are not real-world performance.")
st.sidebar.info("This is a decision-support tool. Official disaster-management authorities remain "
                "responsible for issuing warnings and acting on them.")

tab_board, tab_trend, tab_alerts, tab_eval, tab_data = st.tabs(
    ["Risk board", "Trends", "Alerts", "Model evaluation", "Data & validation"])

# --------------------------------------------------------------- risk board
with tab_board:
    times = preds["timestamp"].drop_duplicates().sort_values()
    default_ts = preds.groupby("timestamp")[pcol].max().idxmax()
    c1, c2 = st.columns([1, 1])
    day = c1.date_input("Date", value=default_ts.date(), min_value=times.min().date(), max_value=times.max().date())
    hour = c2.slider("Hour", 0, 23, value=int(default_ts.hour))
    ts = pd.Timestamp(day) + pd.Timedelta(hours=hour)
    snap = preds[preds["timestamp"] == ts].merge(regions, on="region_id")
    if snap.empty:
        st.warning("No predictions at this time.")
    else:
        snap["probability"] = snap[pcol]
        snap["risk_level"] = categorize(snap["probability"], thresholds)
        snap["action"] = snap["risk_level"].map(DECISIONS)
        feats = load_test_features(str(OUT))
        rows = feats[feats["timestamp"] == ts].set_index("region_id").loc[snap["region_id"]].reset_index()
        snap["key_drivers"] = top_drivers(str(OUT), target, rows)
        snap["event_in_window"] = np.where(snap[target] == 1, "yes", "")

        counts = snap["risk_level"].value_counts()
        cols = st.columns(4)
        for col, lvl in zip(cols, RISK_LEVELS):
            col.metric(f"{lvl} risk regions", int(counts.get(lvl, 0)))

        left, right = st.columns([1.1, 1])
        with left:
            fig = px.scatter_map(
                snap, lat="lat", lon="lon", color="risk_level", size=np.clip(snap["probability"], 0.05, 1) * 30 + 8,
                color_discrete_map=RISK_COLORS, category_orders={"risk_level": RISK_LEVELS},
                hover_name="name", hover_data={"probability": ":.3f", "risk_level": True, "lat": False, "lon": False},
                zoom=6.2, height=560, map_style="carto-positron",
            )
            fig.update_layout(margin=dict(l=0, r=0, t=30, b=0), legend_title_text="Risk level",
                              title=f"{hazard.capitalize()} risk, next {horizon} h, {ts:%d %b %Y %H:%M}")
            st.plotly_chart(fig, width="stretch")
        with right:
            table = snap.sort_values("probability", ascending=False)[
                ["name", "probability", "risk_level", "action", "key_drivers", "event_in_window"]]
            st.dataframe(table, hide_index=True, width="stretch", height=560,
                         column_config={"probability": st.column_config.ProgressColumn(
                             "Probability", min_value=0.0, max_value=1.0, format="%.3f"),
                             "event_in_window": "Event occurred",
                             "key_drivers": st.column_config.TextColumn("Key drivers (model explanation)", width="large")})
        st.caption("Key drivers come from additive per-feature contributions of the gradient-boosting model for this "
                   "target. 'Event occurred' shows the actual outcome; it is available here only because this is "
                   "a historical test period.")

# ------------------------------------------------------------------- trends
with tab_trend:
    c1, c2 = st.columns([1, 2])
    rid = c1.selectbox("Region", regions["region_id"], format_func=lambda r: f"{names[r]} ({r})",
                       index=int(np.argmax([preds.loc[preds.region_id == r, f"{hazard}_event"].sum()
                                            for r in regions["region_id"]])))
    g = preds[preds["region_id"] == rid].sort_values("timestamp")
    lo, hi = g["timestamp"].min().date(), g["timestamp"].max().date()
    rng_sel = c2.date_input("Period", value=(pd.Timestamp(f"{lo.year}-06-01").date(), pd.Timestamp(f"{lo.year}-09-30").date()),
                            min_value=lo, max_value=hi)
    if isinstance(rng_sel, tuple) and len(rng_sel) == 2:
        g = g[(g["timestamp"] >= pd.Timestamp(rng_sel[0])) & (g["timestamp"] < pd.Timestamp(rng_sel[1]) + pd.Timedelta(days=1))]
    compare = st.multiselect("Models", models, default=[model], format_func=DISPLAY_NAMES.get)

    fig = go.Figure()
    edges = [0, *thresholds, 1]
    for lvl, a, b in zip(RISK_LEVELS, edges[:-1], edges[1:]):
        fig.add_hrect(y0=a, y1=b, fillcolor=RISK_COLORS[lvl], opacity=0.08, line_width=0,
                      annotation_text=lvl, annotation_position="right")
    for m in compare:
        fig.add_trace(go.Scatter(x=g["timestamp"], y=g[f"p_{m}__{target}"], mode="lines", name=DISPLAY_NAMES[m],
                                 line=dict(color=MODEL_COLORS[m], width=2)))
    ev = g[g[f"{hazard}_event"] == 1]
    fig.add_trace(go.Scatter(x=ev["timestamp"], y=np.full(len(ev), 0.97), mode="markers", name="Event onset",
                             marker=dict(symbol="triangle-down", size=12, color="#0b0b0b")))
    fig.update_layout(height=420, yaxis=dict(range=[0, 1], title="Probability"), hovermode="x unified",
                      title=f"{names[rid]}: probability of a {hazard} within {horizon} h",
                      margin=dict(l=10, r=60, t=50, b=10), legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, width="stretch")

    feats = load_test_features(str(OUT))
    fr = feats[feats["region_id"] == rid].set_index("timestamp").loc[g["timestamp"]]
    c1, c2 = st.columns(2)
    for col, (fname, title) in zip([c1, c2], [("rain_sum_24h", "24 h rainfall (mm)"), ("river_level_m", "River level (m)")]):
        f2 = go.Figure(go.Scatter(x=fr.index, y=fr[fname], mode="lines", line=dict(color="#2a78d6", width=1.5)))
        f2.update_layout(height=250, title=title, margin=dict(l=10, r=10, t=40, b=10))
        col.plotly_chart(f2, width="stretch")

# ------------------------------------------------------------------- alerts
with tab_alerts:
    alert_level = st.select_slider("Minimum level", RISK_LEVELS[1:], value=cfg["risk"]["alert_level"])
    a = preds[["region_id", "timestamp", pcol, target]].copy()
    a["risk_level"] = categorize(a[pcol], thresholds)
    a = a[level_index(a["risk_level"]) >= RISK_LEVELS.index(alert_level)].sort_values(["region_id", "timestamp"])
    if a.empty:
        st.success(f"No {hazard} alerts at or above {alert_level} in the test period for this model.")
    else:
        # collapse consecutive alert hours into alert episodes
        new_ep = (a["region_id"] != a["region_id"].shift()) | (a["timestamp"].diff() != pd.Timedelta(hours=1))
        a["episode"] = new_ep.cumsum()
        eps = a.groupby("episode").agg(region_id=("region_id", "first"), start=("timestamp", "min"),
                                       end=("timestamp", "max"), peak_probability=(pcol, "max"),
                                       event_followed=(target, "max")).reset_index(drop=True)
        eps["region"] = eps["region_id"].map(names)
        eps["peak_level"] = categorize(eps["peak_probability"], thresholds)
        eps["hours"] = ((eps["end"] - eps["start"]).dt.total_seconds() / 3600 + 1).astype(int)
        eps["outcome"] = np.where(eps["event_followed"] == 1, "event occurred", "false alarm")
        c1, c2, c3 = st.columns(3)
        c1.metric("Alert episodes", len(eps))
        c2.metric("Followed by an event", int(eps["event_followed"].sum()))
        c3.metric("False alarms", int((eps["event_followed"] == 0).sum()))
        st.dataframe(eps.sort_values("start", ascending=False)[
            ["region", "start", "end", "hours", "peak_probability", "peak_level", "outcome"]],
            hide_index=True, width="stretch",
            column_config={"peak_probability": st.column_config.NumberColumn(format="%.3f")})

# --------------------------------------------------------------- evaluation
with tab_eval:
    st.subheader("Horizon-based reliability (held-out test year)")
    measure = st.radio("Metric", ["pr_auc", "roc_auc", "brier_skill", "f1", "ece"], horizontal=True,
                       format_func={"pr_auc": "PR-AUC", "roc_auc": "ROC-AUC", "brier_skill": "Brier skill",
                                    "f1": "F1 (best-F1 threshold)", "ece": "Calibration error"}.get)
    m = metrics[metrics["hazard"] == hazard]
    fig = go.Figure()
    for name in models:
        mm = m[m["model"] == name].sort_values("horizon")
        fig.add_trace(go.Scatter(x=[f"{h} h" for h in mm["horizon"]], y=mm[measure], mode="lines+markers",
                                 name=DISPLAY_NAMES[name], line=dict(color=MODEL_COLORS[name], width=2),
                                 marker=dict(size=9)))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10), yaxis_title=measure,
                      legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch")
    show = m.copy()
    show["model"] = show["model"].map(DISPLAY_NAMES)
    st.dataframe(show[["model", "horizon", "positives", "roc_auc", "pr_auc", "brier", "brier_skill", "ece",
                       "threshold", "precision", "recall", "f1", "f1_at_alert"]].round(4),
                 hide_index=True, width="stretch")

    st.subheader("Warning lead time and false alarms")
    lt = lead[lead["hazard"] == hazard].copy()
    lt["model"] = lt["model"].map(DISPLAY_NAMES)
    st.dataframe(lt.round(3), hide_index=True, width="stretch")

    st.subheader(f"Risk levels against observed outcomes ({DISPLAY_NAMES[best[hazard]]})")
    st.dataframe(risk_tab[risk_tab["target"] == target].round(4), hide_index=True, width="stretch")

    st.subheader("Figures")
    figs = sorted((OUT / "figures").glob("*.png"))
    pick = st.selectbox("Figure", figs, format_func=lambda p: p.stem)
    if pick:
        st.image(str(pick))

# ---------------------------------------------------------- data/validation
with tab_data:
    st.subheader("Data provenance")
    st.json(summary["meta"])
    st.subheader("Validation report")
    c1, c2 = st.columns(2)
    c1.write({"passed": validation["passed"], "rows": validation["n_rows"], "features": validation["n_features"],
              "split_ranges": validation["split_ranges"], "duplicates": validation["duplicate_rows"]})
    c2.write({"warnings": validation["warnings"], "leakage_suspects": validation["leakage_suspects"]})
    st.subheader("Engineered feature groups")
    st.json(validation.get("feature_groups", {}))
    st.subheader("Regions")
    st.dataframe(regions, hide_index=True, width="stretch")
