"""Data loading helpers for viz: CSV/YAML readers and color lookups."""

import pandas as pd
import yaml

from ..utils import posix_hours_to_dt
from .theme import AGG_COLORS, COUNTRY_COLORS, TEC_COLORS


def load_hourly(run_dir, filename, col_names):
    path = run_dir / "inputs" / filename
    if not path.exists():
        return None
    df = pd.read_csv(path, header=None, names=col_names)
    if "hour" in df.columns:
        df["datetime"] = posix_hours_to_dt(df["hour"])
    return df


def load_actual_prices(run_dir):
    """Load historical day-ahead prices from validation/ directory, if available."""
    path = run_dir / "validation" / "actual_prices.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "hour" in df.columns:
        df["datetime"] = posix_hours_to_dt(df["hour"])
    return df


def load_actual_production(run_dir):
    """Load historical generation data from validation/actual_production.csv.

    Returns None if the file is absent (e.g. data collection was skipped).
    """
    path = run_dir / "validation" / "actual_production.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_metadata(run_dir):
    meta_path = run_dir / "run.yaml"
    if meta_path.exists():
        with open(meta_path) as f:
            return yaml.safe_load(f)
    return {}


def country_color(area, idx=0):
    return COUNTRY_COLORS.get(area, f"hsl({(idx * 51) % 360}, 70%, 50%)")


def _tec_color(tec):
    return TEC_COLORS.get(tec, AGG_COLORS.get(tec, "#888888"))
