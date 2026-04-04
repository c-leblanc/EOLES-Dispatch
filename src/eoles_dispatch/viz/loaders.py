"""Data loading and validation-data preparation for viz.

High-level entry point:
    prepare_validation_data  — called by generate_report when validate=True;
        ensures validation/ files exist (downloading source data if needed).

Loaders (called by chart builders):
    load_inputs              — load a formatted input CSV from inputs/.
    load_actual_prices       — load historical prices from validation/.
    load_actual_production   — load historical production from validation/.
    load_metadata            — load run.yaml metadata.

Preparation helpers (private):
    _prepare_actual_prices      — format raw price CSVs into validation/.
    _prepare_actual_production  — format raw production CSVs into validation/.
"""

import logging

import numpy as np
import pandas as pd
import yaml

from ..config import RAW_TO_AGG
from ..utils import cet_period_bounds, posix_hours_to_dt, to_posix_hours

logger = logging.getLogger(__name__)


# ── High-level entry point ──


def prepare_validation_data(run_dir, meta):
    """Ensure validation data is ready for a --validate report.

    Checks whether validation/actual_prices.csv and
    validation/actual_production.csv already exist (idempotent for runs
    created before this refactoring).  If absent, downloads missing source
    price files via collect and then formats both validation CSVs.

    Args:
        run_dir: Path to the run directory.
        meta: Run metadata dict (from run.yaml).
    """
    validation_dir = run_dir / "validation"
    prices_ok = (validation_dir / "actual_prices.csv").exists()
    production_ok = (validation_dir / "actual_production.csv").exists()

    if prices_ok and production_ok:
        logger.info("Validation data already present, skipping preparation.")
        return

    # Extract run parameters from metadata
    year = meta["year"]
    areas = meta["areas"]
    months = _parse_months(meta.get("months"))
    data_dir = run_dir.parent.parent / "data"

    # Download missing price files if needed
    _ensure_price_data(data_dir, year, areas)

    logger.info("Preparing validation data...")
    _prepare_actual_prices(data_dir, run_dir, year, areas, months)
    _prepare_actual_production(data_dir, run_dir, year, areas, months)


# ── Loaders (called by chart builders) ──


def load_inputs(run_dir, areas, filename, col_names):
    """Load a formatted input CSV and filter by area.

    Reads from runs/<name>/inputs/<filename>, applies datetime conversions
    based on column names (hour → datetime, week → date, month → date),
    and filters rows to the requested areas.

    Returns a DataFrame, or None (with a warning) if the file is absent
    or contains no data for the requested areas.
    """
    path = run_dir / "inputs" / filename
    if not path.exists():
        logger.warning(f"{filename}: file not found.")
        return None
    df = pd.read_csv(path, header=None, names=col_names)

    if "hour" in col_names:
        df["datetime"] = posix_hours_to_dt(df["hour"])
    if "week" in col_names:
        week_str = df["week"].astype(str).str.zfill(6)
        df["date"] = pd.to_datetime(
            week_str.str[:4] + "-W" + week_str.str[4:] + "-1",
            format="%Y-W%W-%w",
            errors="coerce",
        )
    if "month" in col_names:
        df["date"] = pd.to_datetime(df["month"], format="%Y%m", errors="coerce")

    if "area" in col_names:
        df = df[df["area"].isin(areas)]
    if "importer" in col_names and "exporter" in col_names:
        df = df[df["exporter"].isin(areas) | df["importer"].isin(areas)]

    if df.empty:
        logger.warning(f"{filename}: no relevant data found.")
        return None
    return df


def load_actual_prices(run_dir):
    """Load historical day-ahead prices from validation/actual_prices.csv.

    Returns a DataFrame with a ``datetime`` column, or None (with a
    warning) if the file is absent.
    """
    path = run_dir / "validation" / "actual_prices.csv"
    if not path.exists():
        logger.warning("actual_prices.csv: file not found in validation/.")
        return None
    df = pd.read_csv(path)
    if "hour" in df.columns:
        df["datetime"] = posix_hours_to_dt(df["hour"])
    return df


def load_actual_production(run_dir):
    """Load historical generation data from validation/actual_production.csv.

    Returns a DataFrame, or None (with a warning) if the file is absent.
    """
    path = run_dir / "validation" / "actual_production.csv"
    if not path.exists():
        logger.warning("actual_production.csv: file not found in validation/.")
        return None
    return pd.read_csv(path)


def load_metadata(run_dir):
    """Load run metadata from run.yaml.

    Raises FileNotFoundError if run.yaml is absent.
    """
    meta_path = run_dir / "run.yaml"
    if not meta_path.exists():
        raise FileNotFoundError(f"run.yaml not found in {run_dir}. Is this a valid run directory?")
    with open(meta_path) as f:
        return yaml.safe_load(f)


# ── Preparation helpers ──


def _parse_months(months_raw):
    """Parse the months field from run.yaml into a tuple or None.

    ``"1-3"`` → ``(1, 3)``, ``"5"`` → ``(5, 5)``, ``None`` → ``None``.
    """
    if months_raw is None:
        return None
    ms = str(months_raw)
    if "-" in ms:
        a, b = ms.split("-", 1)
        return (int(a), int(b))
    return (int(ms), int(ms))


def _ensure_price_data(data_dir, year, areas):
    """Download missing price files for modeled areas.

    Checks whether ``data/<year>/prices_<area>.csv`` exists for every
    area.  If any are missing, triggers an ENTSO-E collection targeting
    only the absent areas (passed as *exo_areas* so that
    ``collect_history`` downloads their price files).
    """
    year_dir = data_dir / str(year)
    missing = [a for a in areas if not (year_dir / f"prices_{a}.csv").exists()]
    if not missing:
        return

    logger.info(f"Missing price data for {missing}, triggering download...")
    from ..collect._main_collect import collect_all

    collect_all(data_dir, year, year + 1, areas=[], exo_areas=missing, source="entsoe")


def _prepare_actual_prices(data_dir, run_dir, year, areas, months):
    """Format raw price CSVs into validation/actual_prices.csv.

    Reads ``data/<year>/prices_<area>.csv`` for each area, filters to the
    requested time period, converts timestamps to POSIX hours, pivots to
    wide format, and saves to ``runs/<name>/validation/actual_prices.csv``.

    Raises FileNotFoundError if no price file is found for any area.
    """
    start, end = cet_period_bounds(year, months)
    area_frames = []
    missing_areas = []

    for area in areas:
        src = data_dir / str(year) / f"prices_{area}.csv"
        if not src.exists():
            missing_areas.append(area)
            continue

        df = pd.read_csv(src, parse_dates=["hour"])
        df = df[(df["hour"] >= start) & (df["hour"] < end)].copy()
        if df.empty:
            continue
        result = pd.DataFrame({"area": area, "hour": df["hour"], "price": df["prices"]})
        result["hour"] = to_posix_hours(result["hour"])
        area_frames.append(result)

    if missing_areas:
        raise FileNotFoundError(
            f"Price data missing for areas {missing_areas}. "
            f"Expected files at data/{year}/prices_<area>.csv."
        )

    if not area_frames:
        raise FileNotFoundError(
            f"No price data found for any area in the requested period ({start} to {end})."
        )

    combined_long = pd.concat(area_frames, ignore_index=True)
    combined_wide = combined_long.pivot_table(index="hour", columns="area", values="price")

    validation_dir = run_dir / "validation"
    validation_dir.mkdir(exist_ok=True)
    combined_wide.to_csv(validation_dir / "actual_prices.csv", index=True)


def _prepare_actual_production(data_dir, run_dir, year, areas, months):
    """Format raw production CSVs into validation/actual_production.csv.

    Reads ``data/<year>/production_<area>.csv`` for each area, filters to
    the requested time period, aggregates raw technology columns to display
    groups (via ``RAW_TO_AGG``), converts timestamps to POSIX hours, joins
    demand from the run inputs, and derives net imports/exports.

    The resulting CSV mirrors the column layout of ``outputs/production.csv``.

    Raises FileNotFoundError if no production file is found for any area.
    """
    start, end = cet_period_bounds(year, months)
    area_frames = []
    missing_areas = []

    for area in areas:
        src = data_dir / str(year) / f"production_{area}.csv"
        if not src.exists():
            missing_areas.append(area)
            continue

        df = pd.read_csv(src, parse_dates=["hour"])
        df = df[(df["hour"] >= start) & (df["hour"] < end)].copy()
        if df.empty:
            continue

        agg_cols = {}
        for raw_col, agg_name in RAW_TO_AGG.items():
            if raw_col not in df.columns:
                continue
            if agg_name not in agg_cols:
                agg_cols[agg_name] = df[raw_col].values.copy()
            else:
                agg_cols[agg_name] = agg_cols[agg_name] + df[raw_col].values

        result = pd.DataFrame({"hour": df["hour"], "area": area})
        for agg_name, vals in agg_cols.items():
            result[agg_name] = vals

        result["hour"] = to_posix_hours(result["hour"])
        area_frames.append(result)

    if missing_areas:
        raise FileNotFoundError(
            f"Production data missing for areas {missing_areas}. "
            f"Expected files at data/{year}/production_<area>.csv."
        )

    if not area_frames:
        raise FileNotFoundError(
            f"No production data found for any area in the requested period ({start} to {end})."
        )

    combined = pd.concat(area_frames, ignore_index=True)

    # Join demand from run inputs (already in GW / POSIX hours)
    demand_path = run_dir / "inputs" / "demand.csv"
    demand = pd.read_csv(demand_path, header=None, names=["area", "hour", "demand"])
    combined = combined.merge(demand, on=["area", "hour"], how="left")

    # Derive net imports / exports = demand - total production
    tec_cols = [c for c in combined.columns if c not in ("area", "hour", "demand")]
    total_prod = combined[tec_cols].sum(axis=1)
    net = combined["demand"] - total_prod
    combined["net_imports"] = np.maximum(net, 0)
    combined["net_exports"] = np.minimum(net, 0)

    # Reorder to match outputs/production.csv layout
    combined = combined[["area", "hour"] + tec_cols + ["net_imports", "net_exports", "demand"]]

    validation_dir = run_dir / "validation"
    validation_dir.mkdir(exist_ok=True)
    combined.to_csv(validation_dir / "actual_production.csv", index=False)
