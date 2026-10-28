"""Configuration loading and shared paths."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "default.yaml"
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
PROCESSED_CSV = PROCESSED_DIR / "france_2019_hourly.csv"


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    """Load the YAML configuration. Raises a readable error if invalid or missing."""
    p = Path(path) if path else CONFIG_PATH
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with open(p, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    required = ["data", "site", "tariff", "observation", "reward", "train_dist", "training", "evaluation"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Config is missing sections: {missing}")
    return cfg
