"""Configuration loading (YAML with single-level `extends` inheritance)."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _deep_update(base: dict, update: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def resolve(path: str | Path) -> Path:
    """Resolve a config path relative to the project root."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    path = resolve(path or "config.yaml")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("extends", None)
    if parent:
        cfg = _deep_update(load_config(path.parent / parent), cfg)
    if overrides:
        cfg = _deep_update(cfg, overrides)
    return cfg


def target_names(cfg: dict) -> list[str]:
    return [f"y_{hz}_{h}h" for hz in cfg["targets"]["hazards"] for h in cfg["targets"]["horizons"]]
