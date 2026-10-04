# AI-Based Disaster Risk Prediction System

A machine-learning system that estimates the probability of a **flood** or **landslide** starting in a
district within the next **1, 6 or 24 hours**. It turns that probability into a
**Low / Medium / High / Severe** risk level with a recommended action, and shows the results on an
interactive early-warning dashboard.

The case study covers the 14 districts of **Kerala, India**, hour by hour, from 2016 to 2020.

> **Read this first:** by default, both the weather and the disaster events are **simulated**, so the
> project runs offline with no downloads. The results show that the method and software work end to
> end. **They are not evidence that the system can forecast real disasters.** See
> [Using real data](#using-real-data) for how to switch to real inputs.

---

## Contents

1. [What the project does](#what-the-project-does)
2. [How it works, step by step](#how-it-works-step-by-step)
3. [Installation](#installation)
4. [Running the project](#running-the-project)
5. [The dashboard](#the-dashboard)
6. [What gets produced (outputs)](#what-gets-produced-outputs)
7. [Understanding the results](#understanding-the-results)
8. [Data sources](#data-sources)
9. [Using real data](#using-real-data)
10. [Configuration reference](#configuration-reference)
11. [Tests](#tests)
12. [Project layout](#project-layout)
13. [Troubleshooting](#troubleshooting)
14. [Limitations](#limitations)

---

## What the project does

For every district and every hour, the system answers these questions:

> *"How likely is a flood (or landslide) to begin here in the next 1 h / 6 h / 24 h, and what should
> be done about it?"*

To answer them, it:

1. **Collects** hourly weather (rainfall, temperature, humidity, wind), hydrology (soil moisture,
   river level) and fixed geography (elevation, slope, soil type, land cover, vegetation) for each
   district.
2. **Cleans and checks** the data: it fills gaps, removes sensor spikes and checks for data leakage.
3. **Builds 62 features**, such as "rain over the last 72 hours", "how fast the river is rising" and
   "rain × slope".
4. **Trains six models** and compares them:

   | Model | Type | Why it is included |
   |---|---|---|
   | Logistic Regression | Linear baseline | A simple, interpretable reference point |
   | Random Forest | Tree ensemble | Strong general-purpose model |
   | XGBoost | Gradient-boosted trees | Usually the best on tabular data |
   | LightGBM | Gradient-boosted trees | A fast alternative to XGBoost |
   | LSTM | Recurrent neural network | Learns from the raw 48-hour history |
   | GRU | Recurrent neural network | A lighter alternative to the LSTM |

5. **Calibrates** each model's output so that "30 % risk" really means that events follow about
   30 % of the time.
6. **Converts** the probability into a risk level and an action:

   | Probability | Level | Recommended action |
   |---|---|---|
   | below 0.25 | **Low** | Routine monitoring |
   | 0.25 – 0.50 | **Medium** | Increased observation and preparedness |
   | 0.50 – 0.75 | **High** | Pre-alert and response readiness |
   | above 0.75 | **Severe** | Urgent warning and intervention planning |

   An alert is raised at **High** or above. You can change this level in `config.yaml`.

7. **Evaluates** every model on a held-out year it never saw during training, and writes reports,
   figures and a dashboard.

---

## How it works, step by step

```
Data Sources -> Integration -> Preprocessing -> Validation -> Feature Engineering -> Model Training
-> Probability Estimation (+calibration) -> Risk Classification -> Evaluation/Visualization -> Early Warning
```

All of the logic is in the `drp/` package ("disaster risk prediction"). `run_pipeline.py` runs the
stages in order.

| Step | Module | What happens |
|---|---|---|
| 1. Acquisition | `data/loader.py`, `data/synthetic.py`, `data/nasa_power.py` | Builds one long hourly table: one row per district per hour. |
| 2. Integration | `data/loader.py`, `data/regions.py` | Attaches each district's fixed geography (from `data/regions.csv`) to its hourly rows. |
| 3. Preprocessing | `preprocessing.py` | Re-indexes to a complete hourly grid and removes duplicates. Discards values outside physical limits (e.g. negative rain, humidity above 100 %) and isolated one-hour spikes. Fills gaps of up to 12 h by interpolation, then forward/back-fills the rest. |
| 4. Exploratory analysis | `visualize.py` | Plots monthly rainfall and events, example time series and a correlation heatmap. |
| 5. Features | `features.py` | Creates 62 features (listed below) and the prediction targets. |
| 6. Validation | `validation.py` | Checks timestamp continuity, IDs, missing or infinite values, split order, class balance per split, and **leakage** (features named after targets, features suspiciously correlated with the answer). Serious problems stop the run. |
| 7. Models | `models/tabular.py`, `models/sequence.py` | The four tabular models are trained once per hazard × horizon (6 models each). The LSTM and GRU each predict all 6 targets at once from a 48-hour look-back window. |
| 8. Calibration | `models/calibration.py` | Fits Platt (sigmoid) or isotonic calibration on the validation year. |
| 9. Risk levels | `risk.py` | Converts probabilities into Low / Medium / High / Severe and alerts. |
| 10. Evaluation | `evaluation.py` | Computes all metrics, lead times and regional stability on the test year. |
| 11. Figures and reports | `visualize.py`, `pipeline.py` | Writes PNG figures, CSV/JSON reports and a Markdown summary. |

### Feature groups (62 features)

- **Lags:** rainfall over the previous 1 hour, 1 day, 7 days and 30 days.
- **Rolling statistics:** rainfall sums, maxima and standard deviation over windows from 3 hours to
  30 days.
- **Hydrological:** soil moisture, river level, river rise over 6 h and 24 h, and river level compared
  with its 30-day mean.
- **Geographical:** elevation, slope, aspect, soil type, land cover, NDVI (vegetation index) and NDVI
  anomaly.
- **Interaction indices:** rain × slope (landslide trigger) and rain × soil moisture (runoff trigger).
- **Temporal and seasonal:** hour of day, day of year, south-west and north-east monsoon flags.
- **Historical disasters:** number of events in the past 365 days and hours since the last event.

### Prediction targets

`y_{hazard}_{h}h` equals 1 if an event **starts** in that district within the next *h* hours
(the interval (t, t+h]), and 0 otherwise. There are six targets: `y_flood_1h`, `y_flood_6h`,
`y_flood_24h`, `y_landslide_1h`, `y_landslide_6h` and `y_landslide_24h`.

### Train / validation / test split

The data is split **by time** so that the models never see the future:

| Split | Full run | Used for |
|---|---|---|
| Train | 2016 – 2018 | Fitting the models |
| Validation | 2019 | Early stopping, calibration, alert-threshold tuning and choosing the best model |
| Test | 2020 | **Only** for the reported results |

A 24-hour gap (embargo) is left before each boundary so that no prediction window crosses into the
next period. The first 30 days are dropped because the long rolling windows are incomplete there.

### Handling rare events

Disasters are rare. Fewer than 1 % of hours are followed by an event within 24 hours, so a model that
always says "no" would be right 99 % of the time. To deal with this, the system:

- keeps every positive hour but only every 3rd negative hour for tabular training,
- re-weights the classes during training,
- recalibrates the output afterwards so that the probabilities are realistic again, and
- reports **PR-AUC**, which is sensitive to rare-event performance, rather than only accuracy or
  ROC-AUC.

---

## Installation

### Requirements

- **Python 3.10 or newer.** Tested on Python 3.13 on Windows 11.
- About **2 GB** of disk space for the dependencies (PyTorch is the largest).
- No GPU needed. The neural networks run on the CPU, and use a GPU automatically if PyTorch can
  find one.
- Internet access is needed only for `pip install` and the optional `--source nasa_power` mode.

### Steps

```bash
# 1. Get the code
git clone https://github.com/josephlikescats/disaster-.git
cd disaster-

# 2. (Recommended) create a virtual environment
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

# 3. Install the dependencies
pip install -r requirements.txt
```

`requirements.txt` installs: numpy, pandas, pyarrow, scikit-learn, scipy, xgboost, lightgbm, torch,
matplotlib, plotly, streamlit, pyyaml, requests, joblib and pytest.

---

## Running the project

### 1. Quick run (about 2 minutes), to start with

```bash
python run_pipeline.py --config config_quick.yaml
```

This runs every stage on a smaller problem: 6 districts, 2017–2019, fewer trees and 2 training epochs
for the neural networks. Results go to `outputs_quick/`. You will see one log line per stage, for
example:

```
INFO drp.validation: Validation passed (153216 rows, 62 features)
INFO drp.pipeline: == train xgboost done in 11.3s
...
INFO drp.pipeline: All outputs written to .../outputs_quick
```

### 2. Full run (about 20 minutes on a laptop CPU)

```bash
python run_pipeline.py
```

This uses all 14 districts, 2016–2020 and all 6 models. Results go to `outputs/`. Most of the time is
spent training the LSTM and GRU.

### 3. Command-line options

| Flag | Example | Effect |
|---|---|---|
| `--config` | `--config config_quick.yaml` | Which configuration file to use (default `config.yaml`) |
| `--models` | `--models xgboost,lstm` | Train only these models. Choices: `logistic_regression, random_forest, xgboost, lightgbm, lstm, gru` |
| `--source` | `--source nasa_power` | Where the weather comes from: `synthetic` (default), `nasa_power` or `csv` |
| `--inventory` | `--inventory data/my_events.csv` | Use a real list of past disasters instead of simulated ones |
| `--output-dir` | `--output-dir outputs_test` | Write the results to a different folder |

For example, to skip the slow neural networks:

```bash
python run_pipeline.py --models logistic_regression,random_forest,xgboost,lightgbm
```

---

## The dashboard

Once the pipeline has run, start the dashboard:

```bash
streamlit run app/dashboard.py                              # reads ./outputs
streamlit run app/dashboard.py -- --outputs outputs_quick   # reads ./outputs_quick
```

It opens in your browser at <http://localhost:8501>. In the **sidebar**, pick the hazard (flood or
landslide), the forecast horizon (1 / 6 / 24 h) and the model. The best model is pre-selected.

| Tab | What it shows |
|---|---|
| **Risk board** | A map of the districts coloured by risk level, plus a ranked table with each district's probability, risk level, recommended action and the **key drivers** behind the risk (e.g. "72 h rainfall 310 mm"). You can choose any hour of the test year. It opens at the hour with the highest risk. |
| **Trends** | Probability over time for a district, against the Low/Medium/High/Severe bands, with actual event onsets marked. Several models can be overlaid. |
| **Alerts** | Every alert episode at or above a chosen level, labelled as *followed by an event* (a hit) or as a *false alarm*. |
| **Model evaluation** | Interactive comparison of models across horizons for any metric, lead-time and false-alarm tables, the risk-level table, and all generated figures. |
| **Data & validation** | Where the data came from, the full validation report, the feature groups and the district table. |

---

## What gets produced (outputs)

Each run writes a folder (`outputs/` or `outputs_quick/`) with this structure:

```
outputs/
├── reports/
│   ├── results_summary.md        ← start here: all key tables in one readable file
│   ├── metrics.csv               every model × hazard × horizon, all metrics
│   ├── lead_time.csv             how many hours of warning each model gives
│   ├── risk_levels.csv           how often events actually follow each risk level
│   ├── regional_stability.csv    performance per district
│   ├── feature_importance.csv    which features matter most
│   ├── preprocessing_report.json what cleaning was done (values removed, gaps filled)
│   ├── validation_report.json    results of the data checks
│   └── summary.json              run metadata, config used, best models
├── figures/                      PNG charts (see below)
├── models/                       trained models: *.joblib (tabular) and *.pt (LSTM/GRU)
├── data/                         test-year predictions and features, used by the dashboard
└── run.log                       full log of the run
```

**Figures in `figures/`:**

| File | Shows |
|---|---|
| `eda_monthly.png` | Monthly rainfall and event counts (the monsoon pattern) |
| `eda_timeseries_*.png` | Rain, soil moisture and river level for one district, with events marked |
| `eda_correlation.png` | Correlations between the main variables |
| `roc_pr_*.png` | ROC and precision-recall curves for each model |
| `reliability_*.png` | Calibration: predicted probability against how often events actually happened |
| `horizon_reliability.png` | How performance changes from 1 h to 6 h to 24 h |
| `importance_*.png` | Most important features for Random Forest and XGBoost |
| `risk_timeline_*.png` | Risk level over time for one district, with events |
| `risk_map_*.png` | Risk by district at the peak hour |
| `sequence_training.png` | LSTM/GRU training loss and validation score per epoch |

---

## Understanding the results

### What the metrics mean

| Metric | Meaning | Good value |
|---|---|---|
| **ROC-AUC** | How well the model ranks risky hours above safe ones. 0.5 is random guessing. | Close to 1 |
| **PR-AUC** | Like ROC-AUC, but focused on the rare positive class. Its baseline equals the event rate, so here it is about 0.003–0.008. **This is the main score** and is used to pick the best model. | Higher is better. Compare it with the base rate. |
| **Precision** | Of the hours the model flagged, the share that really were followed by an event. | High means few false alarms |
| **Recall** | Of the hours that were followed by an event, the share the model flagged. | High means few missed events |
| **F1** | A balance of precision and recall | Higher is better |
| **Brier score** | Mean squared error of the probabilities | Lower is better |
| **Brier skill** | Improvement in Brier score over always predicting the average rate. 0 means no skill and negative means worse. | Above 0 |
| **ECE** | Expected calibration error: how far predicted probabilities are from the frequencies actually observed | Close to 0 |
| **Detection rate** | Share of real events that had an alert beforehand | Higher is better |
| **Lead time** | How many hours before the event the alert started | Higher is better |
| **False-alarm episodes** | Alerts not followed by an event, per district per month | Lower is better |

Precision, recall and F1 are reported at two thresholds:

- **best-F1 threshold:** tuned on the validation year to balance hits and false alarms.
- **alert level:** the fixed "High" threshold (p ≥ 0.5) from the risk scale (`*_at_alert` columns).

### Current results (full run, synthetic data, test year 2020)

- **Best model:** XGBoost for both floods and landslides, chosen on validation PR-AUC.
- **Ranking ability is strong.** ROC-AUC is 0.95–0.99 for most models and horizons.
- **Precision on rare events is modest.** The best PR-AUC is about 0.19 for floods and about 0.32 for
  landslides (24 h). That is 25–40× better than chance, but most alerts are still false alarms.
- **Longer horizons are easier.** 24 h predictions beat 1 h predictions because "will something start
  in the next hour?" has very few positive examples.
- **Lead time:** at the best-F1 threshold, the 24 h models detect about 65–90 % of events, with
  13–19 hours of warning on average and about 0.3–2.3 false-alarm episodes per district per month.
- **The fixed "High" alert level is conservative.** Calibrated probabilities rarely reach 0.5, so at
  that level many models raise few or no alerts. In practice the alert threshold should be tuned
  (see `risk.thresholds` and `risk.alert_level`).
- **The LSTM and GRU overfit quickly.** Their validation score peaks after 1–3 epochs. Early stopping
  (`patience: 3`) keeps the best epoch.

Full tables are in `outputs/reports/results_summary.md`.

---

## Data sources

| Real-world source (design) | In this implementation |
|---|---|
| IMD / NASA POWER meteorology | `--source nasa_power` downloads **real hourly** rainfall, temperature, humidity and wind for each district from the free [NASA POWER API](https://power.larc.nasa.gov/) (no API key needed). Downloads are cached in `data/raw/`. The default `synthetic` source simulates the same variables offline. |
| River monitoring stations | River level and soil moisture come from a simple hydrological model (`drp/data/hydrology.py`: a soil "bucket" that fills and drains, feeding a Nash-cascade river) driven by the rainfall. Real gauge data can be supplied through the `csv` source. |
| USGS / Copernicus / Sentinel | Elevation, slope, aspect, soil type, land cover and NDVI per district are in `data/regions.csv`. **These values are approximate placeholders.** Replace them with values derived from SRTM, Copernicus and MODIS. |
| NDMA / state disaster inventories | Supply a CSV of real events with `--inventory`. Without one, events are **simulated** by `drp/data/hazard.py`. |

### How the simulation works

The synthetic mode exists so that the whole pipeline can run offline and be tested. It is designed to
behave like the real system:

- **Weather** (`drp/data/synthetic.py`) reproduces Kerala's climate: the south-west monsoon
  (Jun–Sep) and north-east monsoon (Oct–Nov), wet and dry spells, storms that hit neighbouring
  districts together, more rain in the hills, afternoon thunderstorms, and rare multi-day extreme
  episodes like August 2018.
- **Sensor problems** are deliberately injected (1 % missing values, 0.1 % outliers) so that the
  cleaning stage has realistic work to do.
- **Floods** (`drp/data/hazard.py`) become more likely when the river nears its danger level, after
  intense short rain, and on saturated soil. They are more likely in low-lying, urbanised and
  clay-soil districts.
- **Landslides** become more likely with multi-day rain totals, intense bursts and saturated soil,
  mostly on steep slopes.
- A hidden random factor is added so that events are not perfectly predictable from the inputs, as in
  reality.

The districts are `TVM` Thiruvananthapuram, `KLM` Kollam, `PTA` Pathanamthitta, `ALP` Alappuzha,
`KTM` Kottayam, `IDK` Idukki, `EKM` Ernakulam, `TSR` Thrissur, `PKD` Palakkad, `MLP` Malappuram,
`KKD` Kozhikode, `WYD` Wayanad, `KNR` Kannur and `KSD` Kasaragod.

---

## Using real data

For results you can report, use real weather **and** a real event list.

### Option A: real weather plus a real event inventory (recommended)

1. Make a CSV of past disaster onsets, one row per event. Start from
   `data/event_inventory_template.csv`:

   ```csv
   region_id,timestamp,hazard
   IDK,2018-08-15 06:00,landslide
   EKM,2018-08-16 02:00,flood
   ```

   - `region_id` must be one of the IDs in `data/regions.csv`. Rows for unknown IDs are skipped with a
     warning.
   - `timestamp` is local time. It is rounded down to the hour.
   - `hazard` is `flood` or `landslide`.

2. Run:

   ```bash
   python run_pipeline.py --source nasa_power --inventory data/my_events.csv
   ```

   The first run downloads one file per district per year from NASA POWER, so it needs internet and
   takes a few minutes. Later runs use the cache in `data/raw/`.

### Option B: your own complete dataset

Set `data.source: csv` and `data.csv_path: path/to/file.csv` in `config.yaml`. The file needs one row
per district per hour with these columns:

```
region_id, timestamp, rainfall_mm, temperature_c, humidity_pct, wind_speed_ms,
soil_moisture, river_level_m, ndvi, flood_event, landslide_event
```

`flood_event` and `landslide_event` are 1 in the hour an event starts and 0 otherwise. The
`region_id` values must match `data/regions.csv`.

### Updating the geography

Edit `data/regions.csv`. It has one row per district:

| Column | Meaning |
|---|---|
| `region_id`, `name`, `state` | Identifiers |
| `lat`, `lon` | District centre, used for the NASA POWER download and the map |
| `elevation_m`, `slope_deg`, `aspect_deg` | Terrain |
| `soil_type` | `sandy`, `laterite`, `loam` or `clay` |
| `land_cover` | `urban`, `mixed`, `forest`, `cropland` or `wetland` |
| `ndvi_base` | Typical vegetation index (0–1) |
| `dist_river_km` | Distance to the main river |
| `impervious_frac` | Fraction of sealed or urban surface (0–1) |
| `rain_factor` | Relative rainfall multiplier, used by the simulator |

To study a different area, replace the rows with your own regions, and update the dates in
`config.yaml` as needed.

---

## Configuration reference

All settings are in `config.yaml`. `config_quick.yaml` starts with `extends: config.yaml` and only
overrides what changes. Paths are relative to the project root.

| Section | Key settings |
|---|---|
| `project` | `seed` (makes runs repeatable), `output_dir` |
| `data` | `source`, `start` / `end` dates, `regions_file`, `event_inventory`, `csv_path`, `max_regions` (limit the number of districts), simulator settings (`events_per_region_year`, `missing_rate`, `outlier_rate`, …) |
| `preprocessing` | `interpolation_limit_hours` (longest gap to interpolate), `spike_thresholds` per variable |
| `features` | `warmup_hours` (dropped at the start, default 720 = 30 days) |
| `targets` | `hazards` (`[flood, landslide]`), `horizons` (`[1, 6, 24]` hours) |
| `split` | `train_end`, `val_end`. Everything after `val_end` is the test set. |
| `models` | `enabled` list, `negative_stride`, `calibration` (`sigmoid` or `isotonic`), and hyperparameters for each model, including `sequence` (window, hidden size, epochs, patience, …) |
| `risk` | `thresholds` (`[0.25, 0.50, 0.75]`), `labels`, `alert_level` |
| `evaluation` | `calibration_bins`, `lead_time_horizon` |

---

## Tests

```bash
python -m pytest -q
```

12 tests, about 10 seconds. They check that:

- the risk thresholds match the design (0.25 / 0.50 / 0.75),
- targets only look **forward** and features never use future information,
- the train/validation/test splits are in time order with the 24 h gap,
- preprocessing removes the injected sensor problems,
- the hydrology model stays within physical limits,
- calibration, ECE, lead-time analysis and risk-level tables compute correctly, and
- LSTM/GRU windows never mix data from two districts.

---

## Project layout

```
config.yaml              main configuration
config_quick.yaml        fast configuration (extends config.yaml)
run_pipeline.py          command-line entry point
requirements.txt         Python dependencies

drp/                     the main package
├── config.py            loads YAML config (with `extends`)
├── pipeline.py          runs all stages in order and writes outputs
├── preprocessing.py     cleaning: bounds, spikes, gaps
├── features.py          62 features + prediction targets
├── validation.py        data-quality and leakage checks
├── risk.py              probability → Low/Medium/High/Severe, alerts
├── evaluation.py        metrics, calibration, lead time, regional stability
├── visualize.py         all figures
├── data/
│   ├── loader.py        builds the integrated hourly dataset
│   ├── regions.py       reads regions.csv
│   ├── synthetic.py     simulated weather
│   ├── hydrology.py     soil moisture + river level model
│   ├── hazard.py        simulated flood/landslide events
│   ├── nasa_power.py    NASA POWER API client (with caching)
│   └── inventory.py     reads a real event CSV
└── models/
    ├── tabular.py       Logistic Regression, Random Forest, XGBoost, LightGBM
    ├── sequence.py      LSTM and GRU (PyTorch)
    └── calibration.py   Platt / isotonic calibration

app/dashboard.py         Streamlit early-warning dashboard
data/regions.csv         district geography
data/event_inventory_template.csv   template for real event data
data/raw/                cached NASA POWER downloads
tests/                   pytest suite
outputs/, outputs_quick/ results of the full and quick runs
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'drp'` | Run commands from the project root folder, not from inside `drp/` or `app/`. |
| Dashboard says files are missing | Run the pipeline first, and point the dashboard at the right folder (`-- --outputs outputs_quick`). Note the extra `--`. |
| `torch` fails to install | Install PyTorch for your platform from <https://pytorch.org/get-started/locally/>, then run `pip install -r requirements.txt` again. To skip the neural networks, run with `--models logistic_regression,random_forest,xgboost,lightgbm`. |
| NASA POWER download fails or times out | The client retries automatically. Check your internet connection and run again. Already-downloaded years are cached in `data/raw/`. |
| Full run is too slow | Use `config_quick.yaml`, drop `lstm,gru` with `--models`, or lower `models.sequence.epochs`. |
| `Validation failed ...` and the run stops | Open `outputs/reports/validation_report.json` to see which check failed. Usually it is a problem with a custom CSV (missing hours, unknown region IDs, NaNs). |
| PowerShell will not activate the venv | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once. |

---

## Limitations

- In the default run, the weather and the event labels are **simulated**, and the district geography
  is approximate.
- River level and soil moisture are **modelled** from rainfall, not measured.
- The resolution is one point per district per hour. Very local or sudden hazards (cloudbursts,
  dam releases) cannot be anticipated.
- Calibrated probabilities rarely reach the "High" level, so the alert threshold needs tuning with
  real data before any operational use.
- The output is **decision support only**. Official disaster-management authorities remain
  responsible for issuing warnings and taking emergency action.
