"""Orchestrator for data collection from all sources.

Coordinates downloads from ENTSO-E, Elexon BMRS, and Renewables.ninja,
then sanitizes and saves the results as CSV files organized by year.
This module does NOT contain source-specific logic (API calls, column
parsing) — that lives in entsoe.py, elexon.py, and rninja.py.

The collection stores *intermediate* data: harmonized and gap-filled, but
not transformed into model inputs. Derived variables (capacity factors,
nuclear availability, lake inflows, hydro limits) are computed later at
run creation time by format_inputs.py.

Delegates to:
    - entsoe.py         ENTSO-E API calls and format normalization.
    - elexon.py         Elexon BMRS API calls (UK fallback).
    - gap_filling.py    Temporal interpolation of missing data.
    - rninja.py         Renewables.ninja capacity-factor downloads.
    - config.py         Area lists, coverage thresholds.
    - utils.py          Timezone conversion (cet_year_bounds, expected_hours).

Called from:
    - __main__.py       CLI entry point (eoles-dispatch collect).
    - run/_main_run.py  Auto-download when creating a run with missing data.

Data structure:
    data/<year>/
        production_<area>.csv         - hourly generation by production type (GW)
        demand_<area>.csv             - hourly demand for one area (GW)
        installed_capacity_<area>.csv - installed capacity: ['tec', 'value'] (GW)
        prices_<area>.csv             - hourly day-ahead prices for one area (EUR/MWh)
                                        exo areas (model input) always collected;
                                        modeled areas (validation only) collected on
                                        demand by viz --validate or eoles-dispatch collect
        _gap_fill_report.csv/txt      - gap-filling audit trail
    data/renewable_ninja/
        solar.csv, onshore_current.csv, ...  - capacity factor profiles

Functions:
    collect_all(output_dir, start_year, end_year, ...)
        Top-level orchestrator: loops over years, calls collect_history,
        then sanitize_year. Called from __main__.py and run/_main_run.py.

    collect_history(output_dir, client, year, areas, exo_areas)
        Download all ENTSO-E data for a single year. Loops over demand,
        production, and prices via a config-driven call to _collect_timeseries,
        then handles installed_capacity separately. Skips areas that already
        have files on disk.
        Called from collect_all.

    sanitize_year(year_dir, year)
        Check integrity of CSV files in a year directory. Renames corrupt
        files to *_corrupt so they are re-downloaded on next collection.
        Area-agnostic.

    collect_installed_capacity(client, areas, year)
        Download installed generation capacity (MW). Returns a dict of
        per-area DataFrames with columns ['tec', 'value'].
        ENTSO-E primary, Elexon fallback for UK.
        Called from collect_history.

Usage:
    eoles-dispatch collect --start 2020 --end 2024
    eoles-dispatch collect --start 2023 --end 2024 --source entsoe
    eoles-dispatch collect --start 2020 --end 2024 --source ninja
    eoles-dispatch collect --start 2021 --end 2022 --force
"""

import logging
import shutil
from pathlib import Path

import pandas as pd

from ..config import DEFAULT_AREAS, DEFAULT_EXO_AREAS, ENTSOE_MIN_COVERAGE
from ..utils import canonical_index, cet_year_bounds, expected_hours
from . import elexon, entsoe
from .gap_filling import Report, interpolate_gaps
from .rninja import collect_ninja

logger = logging.getLogger(__name__)


# ── Main orchestrator ──


def collect_all(
    output_dir,
    start_year,
    end_year,
    areas=None,
    exo_areas=None,
    source="all",
    force=False,
    include_area_prices=True,
):
    if areas is None:
        areas = list(DEFAULT_AREAS)
    if exo_areas is None:
        exo_areas = list(DEFAULT_EXO_AREAS)

    output_dir = Path(output_dir)

    if source in ("all", "entsoe"):
        logger.info("=== STARTING DOWNLOADING HISTORY DATA ===")
        # Validate API key upfront (fail fast before starting a long run)
        try:
            client = entsoe.set_client()
        except (EnvironmentError, RuntimeError) as e:
            raise SystemExit(f"Error: {e}") from None

        for year in range(start_year, end_year):
            year_dir = output_dir / str(year)

            if force and year_dir.exists():
                logger.info(f"Data for {year} present locally, force removing and redownloading...")
                shutil.rmtree(year_dir)

            year_dir.mkdir(parents=True, exist_ok=True)

            # collect_history skips areas that already have files
            collect_history(
                output_dir=year_dir,
                client=client,
                year=year,
                areas=areas,
                exo_areas=exo_areas,
                include_area_prices=include_area_prices,
            )

            # Sanitize: flag bad files so next collection re-downloads them
            issues = sanitize_year(year_dir, year)
            if issues:
                logger.warning(f"{year}: validation issues:")
                for issue in issues:
                    logger.warning(f"    - {issue}")
            else:
                logger.info(f"{year}: all files validated")

    if source in ("all", "ninja"):
        ninja_dir = output_dir / "renewable_ninja"
        ninja_files = [
            "solar.csv",
            "onshore_current.csv",
            "onshore_future.csv",
            "offshore_current.csv",
            "offshore_future.csv",
        ]
        ninja_missing = not ninja_dir.exists() or not all(
            (ninja_dir / f).exists() for f in ninja_files
        )

        logger.info("=== Collecting Renewables.ninja profiles ===")

        if ninja_missing:
            logger.info(f"Renewable Ninja data not found in {ninja_dir}, downloading...")
            collect_ninja(ninja_dir, areas=areas)
        elif force:
            logger.info(
                "Renewable Ninja data already available locally, force remove and redownload..."
            )
            shutil.rmtree(ninja_dir)
            collect_ninja(ninja_dir, areas=areas)
        else:
            logger.info("Renewable Ninja data already available locally, skipping download.")

        # Verify download succeeded
        still_missing = [f for f in ninja_files if not (ninja_dir / f).exists()]
        if still_missing:
            raise RuntimeError(
                f"Failed to download Renewables.ninja data. "
                f"Missing files: {still_missing}. "
                f"Check your internet connection, or provide the data manually in {ninja_dir}/"
            )

    logger.info("=== Collection complete ===")
    return output_dir


# ── Intermediate-level orchestration ──


def collect_history(
    output_dir,
    client,
    year,
    areas=None,
    exo_areas=None,
    include_area_prices=True,
):
    """Download all time-varying ENTSO-E data for a single year and save to CSV.

    Fetches demand, generation by production type, prices, and installed
    capacity for all areas. Each data type is saved as one file per area:
    demand_<area>.csv, production_<area>.csv, prices_<area>.csv,
    installed_capacity_<area>.csv. Per-area file checks allow resuming
    interrupted collections without re-downloading already-collected areas.

    Args:
        output_dir: Directory to write CSV files into (e.g. data/<year>/).
        client: EntsoePandasClient (created by entsoe.set_client()).
        year: Calendar year to download.
        areas: Modeled country codes (default: DEFAULT_AREAS).
        exo_areas: Non-modeled country codes for price data (default: DEFAULT_EXO_AREAS).
        include_area_prices: If True (default), also download prices for
            modeled areas. Set to False during ``create`` to skip validation-only
            data; prices for modeled areas are then fetched on demand by
            ``viz --validate``.
    """
    if areas is None:
        areas = list(DEFAULT_AREAS)
    if exo_areas is None:
        exo_areas = list(DEFAULT_EXO_AREAS)

    gap_report = (
        Report.load(output_dir / "_gap_fill_report.csv")
        if (output_dir / "_gap_fill_report.csv").exists()
        else Report(output_dir)
    )
    start, end = cet_year_bounds(year)
    canon_idx = canonical_index(year)
    n_exp = expected_hours(year)

    # Time series: config-driven loop over demand, production, prices
    ts_configs = [
        (
            "demand",
            areas,
            dict(
                entsoe_fetch=lambda area: entsoe.fetch_demand(client, area, start, end),
                elexon_fetch=lambda: elexon.fetch_demand(start, end),
                usable_fn=lambda raw: entsoe.is_usable(raw, n_exp),
            ),
        ),
        (
            "production",
            areas,
            dict(
                entsoe_fetch=lambda area: entsoe.fetch_generation(client, area, start, end),
                elexon_fetch=lambda: elexon.fetch_generation(start, end),
                usable_fn=lambda raw: _is_production_usable(raw, n_exp),
            ),
        ),
        (
            "prices",
            list(exo_areas) + (list(areas) if include_area_prices else []),
            dict(
                entsoe_fetch=lambda area: entsoe.fetch_day_ahead_prices(client, area, start, end),
                elexon_fetch=lambda: elexon.fetch_day_ahead_prices(start, end),
                usable_fn=lambda raw: entsoe.is_usable(raw, n_exp),
            ),
        ),
    ]

    for ts_type, area_list, config in ts_configs:
        logger.info(f"=== {ts_type.capitalize()} (year: {year}) ===")
        missing = [a for a in area_list if not (output_dir / f"{ts_type}_{a}.csv").exists()]
        if not missing:
            logger.info(f"  → all {ts_type} files already exist, skipping")
            continue
        existing = [a for a in area_list if a not in missing]
        if existing:
            logger.info(
                f"  → {ts_type} already available for {existing}, downloading missing: {missing}"
            )

        _collect_timeseries(
            ts_type=ts_type,
            areas=missing,
            canon_idx=canon_idx,
            gap_report=gap_report,
            output_dir=output_dir,
            **config,
        )

    # Installed capacity (not a time series — separate handling)
    logger.info("=== Installed capacity ===")
    areas_ic_missing = [
        a for a in areas if not (output_dir / f"installed_capacity_{a}.csv").exists()
    ]
    if not areas_ic_missing:
        logger.info("  → all installed_capacity files already exist, skipping")
    else:
        existing_ic = [a for a in areas if a not in areas_ic_missing]
        if existing_ic:
            logger.info(
                f"  → installed_capacity already available for {existing_ic}, downloading missing: {areas_ic_missing}"
            )
        installed = collect_installed_capacity(client, areas_ic_missing, year)
        for area, df in installed.items():
            path = output_dir / f"installed_capacity_{area}.csv"
            df.to_csv(path, index=False)
            logger.info(f"  → installed_capacity_{area}.csv ({len(df)} technologies)")

    gap_report.save()


# ── Time series collection helper ──


def _is_production_usable(raw, n_expected):
    """Check whether raw production data has sufficient non-NaN coverage."""
    if raw is None:
        return False
    if isinstance(raw, pd.DataFrame):
        data_cols = raw.drop(columns=["hour"], errors="ignore")
        return data_cols.notna().any(axis=1).sum() > n_expected * ENTSOE_MIN_COVERAGE
    return hasattr(raw, "__len__") and len(raw) > n_expected * ENTSOE_MIN_COVERAGE


def _collect_timeseries(
    ts_type,
    areas,
    canon_idx,
    gap_report,
    output_dir,
    entsoe_fetch,
    elexon_fetch=None,
    usable_fn=None,
    transform=None,
):
    """Fetch, gap-fill, and return time series data for a list of areas.

    Handles the common pattern for demand, production, and prices:
    ENTSO-E as primary source, optional Elexon fallback for UK, reindexing
    onto the canonical hourly index, gap-filling, and optional transform.

    For Series results (demand, prices), each area's DataFrame has columns
    ['hour', label]. For DataFrame results (production), the existing
    production-type columns are preserved.

    Args:
        areas: List of area codes to fetch.
        canon_idx: DatetimeIndex from canonical_index(year).
        gap_report: Report instance for gap-filling audit trail.
        label: String used as column name for scalar series (e.g. "demand",
            "prices") and in progress/log messages.
        entsoe_fetch: Callable[(area)] -> pd.Series|pd.DataFrame|None.
        elexon_fetch: Callable[()] -> pd.Series|pd.DataFrame|None.
            Called only for UK when ENTSO-E data is not usable. None means
            no Elexon fallback.
        usable_fn: Callable[(raw)] -> bool. Returns True when ENTSO-E data
            is sufficient without a fallback. None means any non-empty
            result is considered usable.
        transform: Callable applied to the filled series/DataFrame before
            saving (e.g. unit conversion). None means no transform.

    Returns:
        dict {area: pd.DataFrame} with an 'hour' column.
        Areas for which no data is available are absent from the dict.
    """
    if usable_fn is None:

        def usable_fn(raw):
            return raw is not None and (not hasattr(raw, "__len__") or len(raw) > 0)

    result = {}
    for area in areas:
        data = None
        print(f"{ts_type.capitalize()} {area}... ", end="", flush=True)

        # Try ENTSO-E
        try:
            raw = entsoe_fetch(area)
            is_empty = raw is None or (hasattr(raw, "__len__") and len(raw) == 0)
            if not is_empty:
                data = raw
        except Exception as e:
            if area == "UK":
                print("no data at ENTSO-E, try Elexon...", end="", flush=True)
            else:
                print(f"FAILED ({type(e).__name__})")
                logger.warning("%s %s error: %s", ts_type.capitalize(), area, e)
                continue

        # Elexon fallback for UK when ENTSO-E data is absent or insufficient
        if elexon_fetch is not None and area == "UK" and not usable_fn(data):
            if data is not None:
                print(" partial ENTSO-E, filling gaps with Elexon...", end="", flush=True)
            try:
                elexon_data = elexon_fetch()
                is_elexon_empty = elexon_data is None or (
                    hasattr(elexon_data, "__len__") and len(elexon_data) == 0
                )
                if not is_elexon_empty:
                    if data is not None:
                        # Merge: keep ENTSO-E where available, Elexon fills gaps
                        if isinstance(data, pd.DataFrame):
                            ep = data.set_index("hour") if "hour" in data.columns else data
                            el = (
                                elexon_data.set_index("hour")
                                if "hour" in elexon_data.columns
                                else elexon_data
                            )
                            data = ep.combine_first(el)
                        else:
                            data = data.combine_first(elexon_data)
                    else:
                        data = elexon_data
            except Exception as e:
                print(f" Elexon fallback FAILED ({type(e).__name__})", end="", flush=True)
                logger.warning("%s %s Elexon fallback error: %s", ts_type.capitalize(), area, e)

        if data is None or (hasattr(data, "__len__") and len(data) == 0):
            print("no data available (KO)")
            continue
        if not usable_fn(data):
            print("insufficient data (KO)")
            continue

        print("OK", end="", flush=True)

        # Reindex onto canonical index and gap-fill
        gaps_filled = 0
        gaps_not_filled = 0
        if isinstance(data, pd.DataFrame):
            if "hour" in data.columns:
                data = data.set_index("hour")
            indexed = data.reindex(canon_idx)
            for col in indexed.columns:
                indexed[col], col_filled, col_not_filled = interpolate_gaps(
                    indexed[col], report=gap_report, variable=col, area=area
                )
                gaps_filled += col_filled
                gaps_not_filled += col_not_filled
            if transform is not None:
                indexed = transform(indexed)
            indexed.index.name = "hour"
            result[area] = indexed.reset_index()
        else:
            # Series (demand, prices)
            series = data.reindex(canon_idx)
            series, gaps_filled, gaps_not_filled = interpolate_gaps(
                series, report=gap_report, variable=ts_type, area=area
            )
            if transform is not None:
                series = transform(series)
            result[area] = series.to_frame(name=ts_type).rename_axis("hour").reset_index()

        if gaps_filled > 0:
            print(f" [Gaps in data: {gaps_filled} data points filled]", end="", flush=True)
        if gaps_not_filled > 0:
            print(
                f" [❌ Gaps in data that could not be filled: {gaps_filled} data points]",
                end="",
                flush=True,
            )

        path = output_dir / f"{ts_type}_{area}.csv"
        result[area].to_csv(path, index=False)
        print(f" → {ts_type}_{area}.csv ({len(result[area])} rows)")

    return result


# ── Installed capacity ──


def collect_installed_capacity(client, areas, year):
    """Collect installed generation capacity per production type for each area.

    ENTSO-E primary, Elexon fallback for UK.

    Args:
        client: EntsoePandasClient.
        areas: List of area codes.
        year: Calendar year.

    Returns:
        dict {area: pd.DataFrame} with columns ['tec', 'value'] (GW).
        Areas for which no data is available are absent from the dict.
    """
    result = {}
    for area in areas:
        capa = None
        print(f"Installed capacity {area}... ", end="", flush=True)

        # Try ENTSO-E
        try:
            capa = entsoe.fetch_installed_capacity(client, area, year)
        except Exception as e:
            if area != "UK":
                print(f"FAILED ({type(e).__name__})")
                logger.warning("Installed capacity %s error: %s", area, e)

        # Elexon fallback for UK
        if not capa and area == "UK":
            try:
                print(" try Elexon...", end="", flush=True)
                capa = elexon.fetch_installed_capacity(year)
            except Exception as e:
                print(f" Elexon fallback FAILED ({type(e).__name__})", end="", flush=True)
                logger.warning("Installed capacity %s Elexon fallback error: %s", area, e)

        if capa:
            result[area] = pd.DataFrame([{"tec": tec, "value": gw} for tec, gw in capa.items()])
            print(f"OK ({len(capa)} types)")
        else:
            print("no data available (KO)")

    return result


# ── Sanitization ──


def sanitize_year(year_dir, year):
    """Check integrity of CSV files in a year directory, rename corrupt ones.

    Scans all *.csv files (excluding _gap_fill_report and already-flagged
    *_corrupt* files). For timeseries files (identified by an 'hour' column),
    validates row count and absence of NaN. Corrupt files are renamed to
    *_corrupt so they are ignored by collect_history's skip logic and
    re-downloaded on next collection.

    Area-agnostic: validates whatever files exist, does not check completeness.

    Args:
        year_dir: Path to data/<year>/ directory.
        year: Calendar year (for computing expected row count).

    Returns:
        List of issue strings (empty if all files OK).
    """
    if not year_dir.exists():
        return []

    issues = []
    n_expected = expected_hours(year)

    for path in sorted(year_dir.glob("*.csv")):
        if path.name.startswith("_gap_fill_report") or "_corrupt" in path.name:
            continue

        df = pd.read_csv(path)

        # Only validate timeseries files (have an 'hour' column)
        if "hour" not in df.columns:
            continue

        if len(df) != n_expected:
            issue = f"{path.name}: {len(df)} rows, expected {n_expected}"
            issues.append(issue)
            logger.warning(issue)
            path.rename(path.with_stem(path.stem + "_corrupt"))
        elif df.drop(columns=["hour"]).isna().any().any():
            n_nan = int(df.drop(columns=["hour"]).isna().sum().sum())
            issue = f"{path.name}: {n_nan} NaN values remain"
            issues.append(issue)
            logger.warning(issue)
            path.rename(path.with_stem(path.stem + "_corrupt"))

    return issues
