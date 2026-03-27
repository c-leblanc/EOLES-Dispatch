"""Data loading helpers for viz: CSV/YAML readers and color lookups."""

import pandas as pd
import yaml

from .theme import AGG_COLORS, COUNTRY_COLORS, TEC_COLORS


def _posix_hours_to_dt(hours_series):
    return pd.to_datetime(hours_series * 3600, unit="s", origin="unix", utc=True)


def _load_hourly(run_dir, filename, col_names):
    path = run_dir / "inputs" / filename
    if not path.exists():
        return None
    df = pd.read_csv(path, header=None, names=col_names)
    if "hour" in df.columns:
        df["datetime"] = _posix_hours_to_dt(df["hour"])
    return df


def _load_actual_prices(run_dir):
    """Load historical day-ahead prices from validation/ directory, if available."""
    path = run_dir / "validation" / "actual_prices.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "hour" in df.columns:
        df["datetime"] = _posix_hours_to_dt(df["hour"])
    return df


def _load_actual_production(run_dir):
    """Load historical generation data from validation/, building it on-the-fly if absent.

    Falls back to aggregating data/<year>/production_<area>.csv directly when
    actual_production.csv was not written at run creation (e.g. runs created before
    this feature was added). Caches the result for subsequent calls.
    """
    path = run_dir / "validation" / "actual_production.csv"
    if path.exists():
        return pd.read_csv(path)

    # Not cached — try to build from raw data using run metadata
    meta = _load_metadata(run_dir)
    year = meta.get("year")
    areas = meta.get("areas")
    months_raw = meta.get("months")
    if not year or not areas:
        return None

    months = None
    if months_raw:
        ms = str(months_raw)
        if "-" in ms:
            a, b = ms.split("-", 1)
            months = (int(a), int(b))
        else:
            months = (int(ms), int(ms))

    data_dir = run_dir.parent.parent / "data"
    df = _aggregate_actual_production(data_dir, year, areas, months)
    if df is None:
        return None

    # Cache for future calls
    path.parent.mkdir(exist_ok=True)
    df.to_csv(path, index=False)
    return df


def _aggregate_actual_production(data_dir, year, areas, months):
    """Aggregate raw production CSVs to agg level (GW) for the given period.

    Returns a DataFrame with columns [area, hour, <agg_tec>...] using POSIX hours,
    or None if no source files are found.
    """
    from datetime import datetime as _dt

    from ..config import RAW_TO_AGG
    from ..utils import cet_to_utc, cet_year_bounds

    if months:
        start_m, end_m = months
        start = cet_to_utc(_dt(year, start_m, 1))
        end = cet_to_utc(_dt(year, end_m + 1, 1)) if end_m < 12 else cet_to_utc(_dt(year + 1, 1, 1))
    else:
        start, end = cet_year_bounds(year)

    area_frames = []
    for area in areas:
        src = data_dir / str(year) / f"production_{area}.csv"
        if not src.exists():
            continue
        raw = pd.read_csv(src, parse_dates=["hour"])
        raw = raw[(raw["hour"] >= start) & (raw["hour"] < end)].copy()
        if raw.empty:
            continue

        agg_cols = {}
        for raw_col, agg_name in RAW_TO_AGG.items():
            if raw_col not in raw.columns:
                continue
            if agg_name not in agg_cols:
                agg_cols[agg_name] = raw[raw_col].values.copy()
            else:
                agg_cols[agg_name] = agg_cols[agg_name] + raw[raw_col].values

        result = pd.DataFrame({"area": area, "hour": raw["hour"]})
        for agg_name, vals in agg_cols.items():
            result[agg_name] = vals / 1000.0  # MW → GW
        result["hour"] = (
            (result["hour"] - pd.Timestamp("1970-01-01")).dt.total_seconds() / 3600
        ).astype(int)
        area_frames.append(result)

    if not area_frames:
        return None

    combined = pd.concat(area_frames, ignore_index=True)
    tec_cols = [c for c in combined.columns if c not in ("area", "hour")]
    return combined[["area", "hour"] + tec_cols]


def _load_metadata(run_dir):
    meta_path = run_dir / "run.yaml"
    if meta_path.exists():
        with open(meta_path) as f:
            return yaml.safe_load(f)
    return {}


def _country_color(area, idx=0):
    return COUNTRY_COLORS.get(area, f"hsl({(idx * 51) % 360}, 70%, 50%)")


def _tec_color(tec):
    return TEC_COLORS.get(tec, AGG_COLORS.get(tec, "#888888"))
