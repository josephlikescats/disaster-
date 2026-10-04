import numpy as np
import pandas as pd
import pytest

from drp.config import load_config, target_names
from drp.data.hydrology import simulate_hydrology
from drp.data.loader import inject_sensor_issues, load_dataset
from drp.data.synthetic import rolling_sum
from drp.evaluation import expected_calibration_error, lead_time_analysis, risk_level_table
from drp.features import _future_any, assign_splits, build_features
from drp.models.calibration import PlattCalibrator
from drp.models.sequence import SequenceRiskModel
from drp.preprocessing import preprocess
from drp.risk import categorize


@pytest.fixture(scope="module")
def cfg():
    return load_config("config_quick.yaml", {"data": {"start": "2017-01-01", "end": "2018-12-31", "max_regions": 3},
                                             "split": {"train_end": "2017-09-30 23:00", "val_end": "2018-03-31 23:00"}})


@pytest.fixture(scope="module")
def dataset(cfg):
    return load_dataset(cfg, np.random.default_rng(0))


def test_risk_thresholds_match_phase1_design():
    p = [0.0, 0.2499, 0.25, 0.4999, 0.5, 0.75, 0.7501, 1.0]
    assert list(categorize(p)) == ["Low", "Low", "Medium", "Medium", "High", "High", "Severe", "Severe"]


def test_future_any_looks_only_ahead():
    ev = pd.Series([0, 0, 1, 0, 0, 0, 0], dtype=float)
    y = _future_any(ev, 2)
    # t=0 -> window (1,2] contains the event; t=2 itself is excluded
    assert list(y[:5]) == [1, 1, 0, 0, 0]
    assert y.iloc[-2:].isna().all()


def test_rolling_sum():
    x = np.arange(1, 6, dtype=float)[:, None]
    assert rolling_sum(x, 2)[:, 0].tolist() == [1, 3, 5, 7, 9]


def test_hydrology_bounds():
    rain = np.zeros((500, 2))
    rain[100:130] = 20.0
    h = simulate_hydrology(rain, np.full((500, 2), 27.0), ["clay", "sandy"], np.array([0.1, 0.0]))
    assert (h["soil_moisture"] >= 0).all() and (h["soil_moisture"] <= 1).all()
    assert h["river_level_m"][140].min() > h["river_level_m"][90].max()  # river responds to rain


def test_dataset_schema_and_events(dataset, cfg):
    df, regions, meta = dataset
    assert meta["label_source"] == "simulated"
    assert df.groupby("region_id").size().nunique() == 1
    for hz in cfg["targets"]["hazards"]:
        assert df[f"{hz}_event"].sum() > 0


def test_preprocessing_removes_injected_issues(dataset, cfg):
    df = inject_sensor_issues(dataset[0], 0.02, 0.005, np.random.default_rng(1))
    clean, report = preprocess(df, cfg)
    assert sum(report["missing_after"].values()) == 0
    assert clean["rainfall_mm"].between(0, 300).all()
    assert clean["humidity_pct"].between(0, 100).all()
    assert clean["soil_moisture"].between(0, 1).all()
    assert report["spikes"]["river_level_m"] > 0


def test_features_have_no_future_information(dataset, cfg):
    clean, _ = preprocess(dataset[0], cfg)
    feats, feature_cols, target_cols = build_features(clean, cfg)
    assert target_cols == target_names(cfg)
    assert not any(c.startswith("y_") for c in feature_cols)
    # perturbing the future must not change features at time t
    g = clean[clean["region_id"] == clean["region_id"].iloc[0]].reset_index(drop=True)
    cut = len(g) // 2
    g2 = g.copy()
    g2.loc[cut + 1:, ["rainfall_mm", "river_level_m", "soil_moisture"]] *= 3
    f1, _, _ = build_features(g, cfg)
    f2, _, _ = build_features(g2, cfg)
    t = g.loc[cut, "timestamp"]
    r1 = f1.loc[f1.timestamp == t, feature_cols].to_numpy()
    r2 = f2.loc[f2.timestamp == t, feature_cols].to_numpy()
    np.testing.assert_allclose(r1, r2)


def test_splits_are_chronological_with_embargo(dataset, cfg):
    clean, _ = preprocess(dataset[0], cfg)
    feats, _, _ = build_features(clean, cfg)
    split = assign_splits(feats, cfg)
    tr = feats.loc[split == "train", "timestamp"].max()
    va = feats.loc[split == "val", "timestamp"]
    assert tr + pd.Timedelta(hours=max(cfg["targets"]["horizons"])) <= va.min()
    assert va.max() < feats.loc[split == "test", "timestamp"].min()


def test_calibration_and_ece():
    rng = np.random.default_rng(0)
    p_true = rng.uniform(0, 0.3, 20000)
    y = rng.random(20000) < p_true
    distorted = np.sqrt(p_true)  # over-confident scores
    cal = PlattCalibrator().fit(distorted, y)
    assert expected_calibration_error(y, cal.transform(distorted)) < expected_calibration_error(y, distorted)


def test_lead_time_analysis():
    ts = pd.date_range("2020-01-01", periods=48, freq="h")
    frame = pd.DataFrame({"region_id": "A", "timestamp": ts, "p": 0.0, "ev": 0})
    frame.loc[30, "ev"] = 1
    frame.loc[20:29, "p"] = 0.9
    out = lead_time_analysis(frame, "p", "ev", 0.5, 24)
    assert out["detected"] == 1 and out["mean_lead_hours"] == 10


def test_risk_level_table_counts():
    tab = risk_level_table(np.array([0, 1, 1, 0]), np.array([0.1, 0.3, 0.9, 0.6]), (0.25, 0.5, 0.75))
    assert tab["hours"].sum() == 4 and tab["events"].sum() == 2


def test_sequence_windows_never_cross_regions():
    m = SequenceRiskModel("gru", {"window": 3})
    codes = np.array([0, 0, 0, 0, 1, 1, 1])
    assert m.valid_endpoints(codes).tolist() == [False, False, True, True, False, False, True]
