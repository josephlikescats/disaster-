"""Baseline and tree-ensemble classifiers (one model per hazard x horizon)."""
from __future__ import annotations

import logging
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .calibration import make_calibrator

log = logging.getLogger(__name__)
N_JOBS = os.cpu_count() or 4


def _imbalance_weight(y) -> float:
    pos = max(int(np.sum(y)), 1)
    return float(np.sqrt((len(y) - pos) / pos))


class TabularRiskModel:
    """Wraps an estimator with validation-period probability calibration."""

    def __init__(self, name: str, params: dict, calibration: str = "sigmoid", seed: int = 42):
        self.name = name
        self.params = dict(params or {})
        self.calibration = calibration
        self.seed = seed
        self.feature_names: list[str] = []

    def _build(self, y):
        p = self.params
        if self.name == "logistic_regression":
            return make_pipeline(
                StandardScaler(),
                LogisticRegression(C=p.get("C", 1.0), class_weight="balanced", max_iter=3000),
            )
        if self.name == "random_forest":
            return RandomForestClassifier(
                n_estimators=p.get("n_estimators", 200), max_depth=p.get("max_depth", 14),
                min_samples_leaf=p.get("min_samples_leaf", 20), max_features=p.get("max_features", "sqrt"),
                class_weight="balanced_subsample", n_jobs=N_JOBS, random_state=self.seed,
            )
        if self.name == "xgboost":
            from xgboost import XGBClassifier

            return XGBClassifier(
                n_estimators=p.get("n_estimators", 600), max_depth=p.get("max_depth", 5),
                learning_rate=p.get("learning_rate", 0.05), subsample=p.get("subsample", 0.8),
                colsample_bytree=p.get("colsample_bytree", 0.8), min_child_weight=p.get("min_child_weight", 5),
                scale_pos_weight=_imbalance_weight(y), tree_method="hist", eval_metric="aucpr",
                early_stopping_rounds=p.get("early_stopping_rounds", 50), n_jobs=N_JOBS,
                random_state=self.seed, verbosity=0,
            )
        if self.name == "lightgbm":
            from lightgbm import LGBMClassifier

            return LGBMClassifier(
                n_estimators=p.get("n_estimators", 600), num_leaves=p.get("num_leaves", 31),
                learning_rate=p.get("learning_rate", 0.05), subsample=p.get("subsample", 0.8),
                subsample_freq=p.get("subsample_freq", 1), colsample_bytree=p.get("colsample_bytree", 0.8),
                min_child_samples=p.get("min_child_samples", 40), scale_pos_weight=_imbalance_weight(y),
                metric="average_precision", n_jobs=N_JOBS, random_state=self.seed, verbose=-1,
            )
        raise ValueError(f"unknown tabular model {self.name}")

    def fit(self, X_train: pd.DataFrame, y_train, X_val: pd.DataFrame, y_val):
        self.feature_names = list(X_train.columns)
        self.estimator = self._build(y_train)
        if self.name == "xgboost":
            self.estimator.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        elif self.name == "lightgbm":
            from lightgbm import early_stopping

            warnings.filterwarnings("ignore", message=".*eval_set.*deprecated")
            self.estimator.fit(
                X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="average_precision",
                callbacks=[early_stopping(self.params.get("early_stopping_rounds", 50), first_metric_only=True,
                                         verbose=False)],
            )
        else:
            self.estimator.fit(X_train, y_train)
        self.calibrator = make_calibrator(self.calibration).fit(self.predict_raw(X_val), np.asarray(y_val))
        return self

    def predict_raw(self, X) -> np.ndarray:
        return self.estimator.predict_proba(X)[:, 1]

    def predict_proba(self, X) -> np.ndarray:
        return self.calibrator.transform(self.predict_raw(X))

    def feature_importance(self) -> pd.Series:
        est = self.estimator
        if self.name == "logistic_regression":
            values = np.abs(est[-1].coef_[0])  # standardized coefficients
        elif self.name == "xgboost":
            gain = est.get_booster().get_score(importance_type="gain")
            values = np.array([gain.get(f, 0.0) for f in self.feature_names])
        elif self.name == "lightgbm":
            values = est.booster_.feature_importance(importance_type="gain").astype(float)
        else:
            values = est.feature_importances_
        s = pd.Series(values, index=self.feature_names, dtype=float)
        return (s / s.sum() if s.sum() > 0 else s).sort_values(ascending=False)

    def contributions(self, X: pd.DataFrame) -> pd.DataFrame | None:
        """Per-row additive feature contributions (log-odds), where the model supports them."""
        if self.name == "xgboost":
            import xgboost as xgb

            contrib = self.estimator.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]
            return pd.DataFrame(contrib, columns=self.feature_names, index=X.index)
        if self.name == "lightgbm":
            contrib = self.estimator.predict(X, pred_contrib=True)[:, :-1]
            return pd.DataFrame(contrib, columns=self.feature_names, index=X.index)
        if self.name == "logistic_regression":
            scaler, lr = self.estimator[0], self.estimator[-1]
            return pd.DataFrame(scaler.transform(X) * lr.coef_[0], columns=self.feature_names, index=X.index)
        return None


def training_rows(y: np.ndarray, stride: int, rng: np.random.Generator) -> np.ndarray:
    """All positive rows + every `stride`-th negative row (random phase)."""
    idx = np.arange(len(y))
    keep = (y == 1) | ((idx + int(rng.integers(stride))) % stride == 0)
    return idx[keep]
