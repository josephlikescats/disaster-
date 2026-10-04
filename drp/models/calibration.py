"""Post-hoc probability calibration fitted on the validation period.

Models are trained with class re-weighting and negative subsampling, which
distorts raw scores; calibration maps them back to real event frequencies so
the Low/Medium/High/Severe thresholds mean what they say.
"""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

EPS = 1e-6


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


class PlattCalibrator:
    def fit(self, scores, y):
        self.lr = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(scores).reshape(-1, 1), y)
        return self

    def transform(self, scores):
        return self.lr.predict_proba(_logit(scores).reshape(-1, 1))[:, 1]


class IsotonicCalibrator:
    def fit(self, scores, y):
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(scores, y)
        return self

    def transform(self, scores):
        return self.iso.predict(scores)


def make_calibrator(kind: str):
    return {"sigmoid": PlattCalibrator, "isotonic": IsotonicCalibrator}[kind]()
