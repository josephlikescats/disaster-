"""Run the full disaster-risk prediction pipeline.

    python run_pipeline.py                         # full run (config.yaml)
    python run_pipeline.py --config config_quick.yaml
    python run_pipeline.py --source nasa_power     # real NASA POWER meteorology
    python run_pipeline.py --models xgboost,lstm
"""
from __future__ import annotations

import argparse
import logging
import warnings

from drp.config import load_config
from drp.pipeline import run


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--source", choices=["synthetic", "nasa_power", "csv"])
    ap.add_argument("--inventory", help="CSV of real event onsets (region_id,timestamp,hazard)")
    ap.add_argument("--models", help="comma-separated subset of models to train")
    ap.add_argument("--output-dir")
    args = ap.parse_args()

    overrides: dict = {}
    if args.source:
        overrides.setdefault("data", {})["source"] = args.source
    if args.inventory:
        overrides.setdefault("data", {})["event_inventory"] = args.inventory
    if args.models:
        overrides["models"] = {"enabled": [m.strip() for m in args.models.split(",")]}
    if args.output_dir:
        overrides["project"] = {"output_dir": args.output_dir}

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    warnings.filterwarnings("ignore", category=UserWarning)
    run(load_config(args.config, overrides))


if __name__ == "__main__":
    main()
