"""Static figures for the report (matplotlib, PNG)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

from .evaluation import reliability_curve  # noqa: E402
from .models import DISPLAY_NAMES  # noqa: E402
from .risk import RISK_LEVELS, categorize  # noqa: E402

# Validated categorical order (adjacent-pair CVD safe), fixed per model - never cycled.
MODEL_COLORS = {
    "logistic_regression": "#2a78d6", "random_forest": "#eb6834", "xgboost": "#1baf7a",
    "lightgbm": "#eda100", "lstm": "#e87ba4", "gru": "#008300",
}
HAZARD_COLORS = {"flood": "#2a78d6", "landslide": "#eb6834"}
# Status palette: risk levels always shipped with their text label.
RISK_COLORS = {"Low": "#0ca30c", "Medium": "#fab219", "High": "#ec835a", "Severe": "#d03b3b"}
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", ["#104281", "#5598e7", "#f0efec", "#e66767", "#a32626"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "legend.frameon": False, "lines.linewidth": 2,
})


def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------- EDA
def eda_monthly(df: pd.DataFrame, hazards: list[str], out: Path):
    month = df["timestamp"].dt.month
    years = df["timestamp"].dt.year.nunique()
    n_regions = df["region_id"].nunique()
    rain = df.groupby(month)["rainfall_mm"].sum() / years / n_regions
    fig, axes = plt.subplots(1 + len(hazards), 1, figsize=(8, 2.4 * (1 + len(hazards))), sharex=True)
    axes[0].bar(rain.index, rain.values, color="#2a78d6", width=0.7)
    axes[0].set_title("Mean monthly rainfall per region (mm)")
    for ax, hz in zip(axes[1:], hazards):
        ev = df.groupby(month)[f"{hz}_event"].sum()
        ax.bar(ev.index, ev.values, color=HAZARD_COLORS.get(hz, "#2a78d6"), width=0.7)
        ax.set_title(f"{hz.capitalize()} onsets by month (all regions, all years)")
    axes[-1].set_xticks(range(1, 13), ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    return _save(fig, out)


def eda_timeseries(df: pd.DataFrame, region: str, year: int, hazards: list[str], out: Path):
    g = df[(df["region_id"] == region) & (df["timestamp"].dt.year == year)]
    fig, axes = plt.subplots(3, 1, figsize=(10, 6.5), sharex=True)
    panels = [("rainfall_mm", "Rainfall (mm/h)"), ("soil_moisture", "Soil moisture (fraction)"),
              ("river_level_m", "River level (m)")]
    for ax, (col, title) in zip(axes, panels):
        ax.plot(g["timestamp"], g[col], color="#2a78d6", linewidth=1)
        ax.set_title(title, loc="left")
    for hz in hazards:
        for ts in g.loc[g[f"{hz}_event"] == 1, "timestamp"]:
            for ax in axes:
                ax.axvline(ts, color=HAZARD_COLORS[hz], linewidth=1, alpha=0.8)
    handles = [plt.Line2D([], [], color=HAZARD_COLORS[hz], label=f"{hz} onset") for hz in hazards]
    axes[0].legend(handles=handles, loc="upper left")
    fig.suptitle(f"{region} - {year}: drivers and recorded events", x=0.01, ha="left", fontweight="bold")
    return _save(fig, out)


def correlation_heatmap(df: pd.DataFrame, cols: list[str], out: Path):
    corr = df[cols].astype(float).corr()
    fig, ax = plt.subplots(figsize=(0.45 * len(cols) + 2, 0.45 * len(cols) + 1.5))
    im = ax.imshow(corr.values, cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(cols)), cols, fontsize=8)
    ax.grid(False)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                    color="#ffffff" if abs(v) > 0.6 else INK)
    fig.colorbar(im, ax=ax, shrink=0.7, label="Pearson r")
    ax.set_title("Correlation of key drivers with 24 h targets")
    return _save(fig, out)


# --------------------------------------------------------------------- evaluation
def horizon_reliability(metrics: pd.DataFrame, out: Path):
    hazards = list(metrics["hazard"].unique())
    measures = [("pr_auc", "PR-AUC (higher is better)"), ("roc_auc", "ROC-AUC (higher is better)"),
                ("brier_skill", "Brier skill score (higher is better)")]
    fig, axes = plt.subplots(len(hazards), 3, figsize=(13, 3.4 * len(hazards)), squeeze=False)
    for r, hz in enumerate(hazards):
        for c, (m, title) in enumerate(measures):
            ax = axes[r, c]
            sub = metrics[metrics["hazard"] == hz]
            for model, g in sub.groupby("model", sort=False):
                g = g.sort_values("horizon")
                ax.plot(g["horizon"], g[m], marker="o", markersize=6, color=MODEL_COLORS.get(model),
                        label=DISPLAY_NAMES.get(model, model))
            ax.set_xticks(sorted(sub["horizon"].unique()), [f"{h} h" for h in sorted(sub["horizon"].unique())])
            ax.set_title(f"{hz.capitalize()} - {title}", loc="left", fontsize=10)
    axes[0, 0].legend(loc="best", fontsize=8)
    return _save(fig, out)


def roc_pr_curves(frame: pd.DataFrame, models: list[str], target: str, out: Path):
    y = frame[target].to_numpy()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for m in models:
        p = frame[f"p_{m}__{target}"].to_numpy()
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        a1.plot(fpr, tpr, color=MODEL_COLORS.get(m), label=DISPLAY_NAMES.get(m, m))
        a2.plot(rec, prec, color=MODEL_COLORS.get(m), label=DISPLAY_NAMES.get(m, m))
    a1.plot([0, 1], [0, 1], color=AXIS, linestyle="--", linewidth=1)
    a2.axhline(y.mean(), color=AXIS, linestyle="--", linewidth=1)
    a1.set(xlabel="False positive rate", ylabel="True positive rate", title=f"ROC - {target}")
    a2.set(xlabel="Recall", ylabel="Precision", title=f"Precision-recall - {target}")
    a1.legend(loc="lower right", fontsize=8)
    return _save(fig, out)


def reliability_diagram(frame: pd.DataFrame, models: list[str], target: str, out: Path, bins: int = 10):
    y = frame[target].to_numpy()
    fig, ax = plt.subplots(figsize=(5.5, 5))
    top = 0.0
    for m in models:
        rc = reliability_curve(y, frame[f"p_{m}__{target}"], bins)
        top = max(top, rc["mean_predicted"].max(), rc["observed_rate"].max())
        ax.plot(rc["mean_predicted"], rc["observed_rate"], marker="o", markersize=6,
                color=MODEL_COLORS.get(m), label=DISPLAY_NAMES.get(m, m))
    top = min(1.0, top * 1.1 + 1e-3)
    ax.plot([0, top], [0, top], color=AXIS, linestyle="--", linewidth=1, label="Perfect calibration")
    ax.set(xlim=(0, top), ylim=(0, top), xlabel="Mean predicted probability",
           ylabel="Observed event rate", title=f"Reliability - {target}")
    ax.legend(fontsize=8)
    return _save(fig, out)


def feature_importance(imp: pd.Series, title: str, out: Path, top: int = 15):
    s = imp.head(top)[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(s) + 1.2))
    ax.barh(s.index, s.values, color="#2a78d6", height=0.7)
    ax.set_title(title, loc="left")
    ax.set_xlabel("Relative importance")
    ax.grid(axis="y", visible=False)
    return _save(fig, out)


def risk_timeline(frame: pd.DataFrame, prob_col: str, event_col: str, region: str, thresholds, title: str,
                  out: Path):
    g = frame[frame["region_id"] == region].sort_values("timestamp")
    fig, ax = plt.subplots(figsize=(11, 3.6))
    edges = [0, *thresholds, 1.0]
    for lvl, lo, hi in zip(RISK_LEVELS, edges[:-1], edges[1:]):
        ax.axhspan(lo, hi, color=RISK_COLORS[lvl], alpha=0.10, linewidth=0)
        ax.text(1.005, (lo + hi) / 2, lvl, transform=ax.get_yaxis_transform(), va="center", fontsize=8, color=INK2)
    ax.plot(g["timestamp"], g[prob_col], color="#2a78d6", linewidth=1.2, label="Predicted probability")
    ev = g.loc[g[event_col] == 1, "timestamp"]
    ax.scatter(ev, np.full(len(ev), 0.97), marker="v", s=60, color=INK, label="Recorded event onset", zorder=3)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    ax.set_title(title, loc="left")
    ax.legend(loc="upper left", fontsize=8)
    return _save(fig, out)


def risk_map(snapshot: pd.DataFrame, prob_col: str, thresholds, title: str, out: Path):
    levels = categorize(snapshot[prob_col], thresholds)
    fig, ax = plt.subplots(figsize=(5.5, 7))
    for lvl in RISK_LEVELS:
        m = levels == lvl
        ax.scatter(snapshot.loc[m, "lon"], snapshot.loc[m, "lat"], s=260, color=RISK_COLORS[lvl],
                   edgecolor=SURFACE, linewidth=2, label=lvl, zorder=3)
    for _, r in snapshot.iterrows():
        ax.annotate(f"{r['region_id']}  {r[prob_col]:.2f}", (r["lon"], r["lat"]), xytext=(10, -3),
                    textcoords="offset points", fontsize=8, color=INK2)
    ax.set(xlabel="Longitude", ylabel="Latitude", title=title)
    ax.set_aspect("equal")
    ax.legend(title="Risk level", loc="lower left", fontsize=8)
    return _save(fig, out)


def training_curves(histories: dict[str, list[dict]], out: Path):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for name, hist in histories.items():
        h = pd.DataFrame(hist)
        ax.plot(h["epoch"], h["val_pr_auc"], marker="o", markersize=6, color=MODEL_COLORS.get(name),
                label=DISPLAY_NAMES.get(name, name))
    ax.set(xlabel="Epoch", ylabel="Validation mean PR-AUC", title="Sequence model training")
    ax.legend(fontsize=8)
    return _save(fig, out)
