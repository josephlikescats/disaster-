"""LSTM / GRU multi-task sequence models (PyTorch).

One network per architecture predicts every hazard x horizon target at once
from a look-back window of hourly feature vectors. Windows are gathered on
the fly from one contiguous feature matrix, never crossing region borders.
"""
from __future__ import annotations

import copy
import logging
import os
import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from torch import nn

from .calibration import make_calibrator

log = logging.getLogger(__name__)


class SequenceNet(nn.Module):
    def __init__(self, n_features: int, n_outputs: int, cell: str, hidden: int, layers: int, dropout: float):
        super().__init__()
        rnn = {"lstm": nn.LSTM, "gru": nn.GRU}[cell]
        self.rnn = rnn(n_features, hidden, num_layers=layers, batch_first=True,
                       dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, n_outputs))

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1])


class SequenceRiskModel:
    def __init__(self, cell: str, params: dict, calibration: str = "sigmoid", seed: int = 42):
        self.cell = cell
        self.name = cell
        self.p = params
        self.calibration = calibration
        self.seed = seed
        self.window = int(params["window"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_num_threads(os.cpu_count() or 4)

    # ------------------------------------------------------------------ data
    def _windows(self, idx: np.ndarray) -> torch.Tensor:
        offsets = np.arange(-self.window + 1, 1)
        return torch.from_numpy(self.X[idx[:, None] + offsets[None, :]])

    def valid_endpoints(self, region_codes: np.ndarray) -> np.ndarray:
        """Row positions whose full look-back window lies inside the same region."""
        pos = np.arange(len(region_codes))
        start = pos - self.window + 1
        ok = start >= 0
        ok[ok] = region_codes[start[ok]] == region_codes[ok]
        return ok

    # --------------------------------------------------------------- training
    def fit(self, X_all: np.ndarray, Y_all: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray,
            target_names: list[str]):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.target_names = target_names
        self.scaler = StandardScaler().fit(X_all[train_idx])
        self.X = np.clip(self.scaler.transform(X_all), -10, 10).astype(np.float32)
        Y = Y_all.astype(np.float32)

        pos = Y[train_idx].sum(0)
        pos_weight = np.clip(np.sqrt((len(train_idx) - pos) / np.maximum(pos, 1)), 1.0, 50.0)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=self.device))

        self.net = SequenceNet(X_all.shape[1], Y.shape[1], self.cell, self.p["hidden"], self.p["layers"],
                               self.p["dropout"]).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.p["lr"])

        val_eval = val_idx
        if len(val_eval) > self.p["max_val_windows"]:
            val_eval = np.sort(rng.choice(val_idx, self.p["max_val_windows"], replace=False))

        best, best_state, bad_epochs = -np.inf, None, 0
        bs = self.p["batch_size"]
        self.history = []
        for epoch in range(self.p["epochs"]):
            t0 = time.time()
            self.net.train()
            order = rng.permutation(train_idx)
            total = 0.0
            for i in range(0, len(order), bs):
                b = order[i : i + bs]
                xb = self._windows(b).to(self.device)
                yb = torch.from_numpy(Y[b]).to(self.device)
                opt.zero_grad()
                loss = loss_fn(self.net(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                total += loss.item() * len(b)
            logits = self._logits(val_eval)
            ap = np.mean([average_precision_score(Y[val_eval, k], logits[:, k])
                          for k in range(Y.shape[1]) if Y[val_eval, k].sum() > 0])
            self.history.append({"epoch": epoch + 1, "train_loss": total / len(order), "val_pr_auc": float(ap)})
            log.info("%s epoch %d: loss=%.4f val mean PR-AUC=%.4f (%.0fs)", self.cell.upper(), epoch + 1,
                     total / len(order), ap, time.time() - t0)
            if ap > best:
                best, best_state, bad_epochs = ap, copy.deepcopy(self.net.state_dict()), 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.p["patience"]:
                    break
        self.net.load_state_dict(best_state)

        val_logits = self._logits(val_idx)
        self.calibrators = [make_calibrator(self.calibration).fit(_sigmoid(val_logits[:, k]), Y[val_idx, k])
                            for k in range(Y.shape[1])]
        return self

    @torch.no_grad()
    def _logits(self, idx: np.ndarray, bs: int = 4096) -> np.ndarray:
        self.net.eval()
        out = [self.net(self._windows(idx[i : i + bs]).to(self.device)).cpu().numpy()
               for i in range(0, len(idx), bs)]
        return np.concatenate(out) if out else np.zeros((0, len(self.target_names)))

    def predict_proba(self, idx: np.ndarray, X_all: np.ndarray | None = None) -> np.ndarray:
        """Calibrated probabilities (N, n_targets) for window endpoints `idx`."""
        if X_all is not None:
            self.X = np.clip(self.scaler.transform(X_all), -10, 10).astype(np.float32)
        raw = _sigmoid(self._logits(idx))
        return np.column_stack([c.transform(raw[:, k]) for k, c in enumerate(self.calibrators)])

    # ------------------------------------------------------------ persistence
    def save(self, path):
        torch.save({"cell": self.cell, "params": self.p, "calibration": self.calibration,
                    "state_dict": self.net.state_dict(), "scaler": self.scaler,
                    "calibrators": self.calibrators, "target_names": self.target_names,
                    "n_features": self.scaler.n_features_in_, "history": self.history}, path)

    @classmethod
    def load(cls, path):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        m = cls(ck["cell"], ck["params"], ck["calibration"])
        m.net = SequenceNet(ck["n_features"], len(ck["target_names"]), ck["cell"], ck["params"]["hidden"],
                            ck["params"]["layers"], ck["params"]["dropout"]).to(m.device)
        m.net.load_state_dict(ck["state_dict"])
        m.scaler, m.calibrators, m.target_names = ck["scaler"], ck["calibrators"], ck["target_names"]
        m.history = ck.get("history", [])
        return m


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sequence_training_endpoints(Y: np.ndarray, candidates: np.ndarray, stride: int,
                                rng: np.random.Generator) -> np.ndarray:
    """All windows ending at a positive hour + every `stride`-th negative one."""
    any_pos = Y[candidates].max(axis=1) > 0
    phase = int(rng.integers(stride))
    keep = any_pos | ((np.arange(len(candidates)) + phase) % stride == 0)
    return candidates[keep]
