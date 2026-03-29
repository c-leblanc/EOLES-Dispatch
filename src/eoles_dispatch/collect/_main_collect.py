"""Orchestrator for data collection from all sources.

Coordinates downloads from ENTSO-E, Elexon BMRS, and Renewables.ninja,
then validates and saves the results as CSV files organized by year.
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
    - run.py            Auto-download when creating a run with missing data.

Data structure:
    data/<year>/
        production_<area>.csv   - hourly generation by production type (MW)
        demand.csv              - hourly demand per area (MW)
        installed_capacity.csv  - installed capacity: technologies in rows, areas in columns (MW)
        exo_prices.csv          - hourly day-ahead prices for exo areas (EUR/MWh)
        gap_fill_report.csv/txt - gap-filling audit trail
    data/renewable_ninja/
        solar.csv, onshore_current.csv, ...  - capacity factor profiles

Functions:
    collect_all(output_dir, start_year, end_year, ...)
        Top-level orchestrator: loops over years, calls the three collect_*
        functions below, validates, and saves.
        Called from __main__.py and run.py.

    collect_demand(client, areas, start, end, gap_report)
        Download hourly demand per area (GW). ENTSO-E primary, Elexon
        fallback for UK. Gap-fills then returns a DataFrame.
        Called from collect_all.

    collect_production(client, areas, start, end, gap_report)
        Download hourly generation by production type per area (MW). ENTSO-E
        primary, Elexon fallback for UK. PHS split into phs/phs_in.
        Called from collect_all.

    collect_installed_capacity(client, areas, year, out_dir)
        Download installed generation capacity (MW). Saves wide-format CSV
        with technologies in rows and areas in columns.
        ENTSO-E primary, Elexon fallback for UK. Static yearly data.
        Called from collect_all.

    collect_exo_prices(client, exo_areas, start, end, gap_report)
        Download day-ahead prices for non-modeled areas (EUR/MWh).
        Called from collect_all.

    _validate_year(year_dir, year, areas, exo_areas)
        Check row counts, NaN, and expected columns after collection.
        Called from collect_all.

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
):
    if areas is None:
        areas = list(DEFAULT_AREAS)
    if exo_areas is None:
        exo_areas = list(DEFAULT_EXO_AREAS)

    output_dir = Path(output_dir)

    if source in ("all", "entsoe"):
        logger.info("=== STARTING DOWNLOADING HISTORY DATA ===")
        # Validate API key upfront (fail fast before starting a long run)
        client = entsoe.set_client()

        for year in range(start_year, end_year):
            year_dir = output_dir / str(year)
            corrupt_dir = output_dir / f"{year}_corrupt"
            partial_dir = output_dir / f"{year}_partial"

            # Skip if already valid (unless force) or clean up any previous partial/corrupt directories
            if year_dir.exists() and force:
                logger.info(f"Data for {year} present locally, force removing and redownloading...")
                shutil.rmtree(year_dir)
            elif year_dir.exists():
                logger.info(
                    f"Data for {year} already available locally, "
                    f"skipping (use --force to re-download)"
                )
                continue
            elif corrupt_dir.exists():
                logger.info(f"Data for {year} present locally is corrupt: removing corrupt files and redownloading them...")
                # Remove only files with 'corrupt' in the name
                for f in corrupt_dir.glob("*corrupt*"):
                    f.unlink()
                corrupt_dir.rename(partial_dir)
            elif partial_dir.exists(): #Case were a collect_all was interrupted before completion and validation
                logger.info(f"Data for {year} was not fully downloaded: checking if present files are valid...")
                _validate_year(partial_dir, year, areas, exo_areas)
                # Remove corrupt files (renamed by _validate_year) so they get re-downloaded
                for f in partial_dir.glob("*corrupt*"):
                    f.unlink()
            else:
                logger.info(f"No data for {year}: downloading...")
                partial_dir.mkdir(parents=True)

            # Launch download of history data for <year>
            collect_history(output_dir=partial_dir, client=client, year=year, areas=areas, exo_areas=exo_areas)
            
            # Validate history data for <year>
            is_valid, issues = _validate_year(partial_dir, year, areas, exo_areas)
            if is_valid:
                partial_dir.rename(year_dir)
                logger.info(f"{year}: validated and saved to {year_dir}")
            else:
                target = output_dir / f"{year}_corrupt"
                partial_dir.rename(target)
                logger.warning(f"{year}: VALIDATION FAILED, marked as corrupt")
                for issue in issues:
                    logger.warning(f"    - {issue}")

    if source in ("all", "ninja"):
        ninja_dir = output_dir / "renewable_ninja"
        ninja_files = ["solar.csv", "onshore_current.csv", "onshore_future.csv", "offshore_current.csv", "offshore_future.csv"]
        ninja_missing = not ninja_dir.exists() or not all((ninja_dir / f).exists() for f in ninja_files)

        logger.info("=== Collecting Renewables.ninja profiles ===")

        if ninja_missing:
            logger.info(f"Renewable Ninja data not found in {ninja_dir}, downloading...")
            collect_ninja(ninja_dir, areas=areas)
        elif force:
            logger.info("Renewable Ninja data already available locally, force remove and redownload...")
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
):
    """Download all time-varying ENTSO-E data for a single year and save to CSV.

    Fetches demand, generation by production type, installed capacity, exogenous
    prices, and actual prices for modeled areas. Writes directly into
    output_dir — does not create partial/corrupt directories (that lifecycle
    is managed by collect_all).

    Args:
        output_dir: Directory to write CSV files into (e.g. data/<year>_partial/).
        client: EntsoePandasClient (created by entsoe.set_client()).
        year: Calendar year to download.
        areas: Modeled country codes (default: DEFAULT_AREAS).
        exo_areas: Non-modeled country codes for price data (default: DEFAULT_EXO_AREAS).
    """
    # Initialize gap-fill report for this year
    gap_report = Report()

    # CET year bounds (naive UTC — entsoe module handles tz conversion)
    start, end = cet_year_bounds(year)
    canon_idx = canonical_index(year)

    # 1. Demand (collect_demand returns GW, store as MW for raw data)
    logger.info("=== Demand ===")
    demand_path = output_dir / "demand.csv"
    if not demand_path.exists():
        demand = collect_demand(client, areas, start, end, gap_report, canon_idx)
        demand_mw = demand.copy()
        area_cols = [c for c in demand_mw.columns if c != "hour"]
        demand_mw[area_cols] = demand_mw[area_cols] * 1000  # GW → MW
        demand_mw.to_csv(demand_path, index=False)
        logger.info(f"  → demand.csv ({len(demand_mw)} rows)")
    else:
        logger.info("  → demand.csv already exists, skipping")

    # 2. Raw production per area
    logger.info("=== Production ===")
    # Filter out areas that already have production files
    areas_prod_missing = [a for a in areas if not (output_dir / f"production_{a}.csv").exists()]
    if areas_prod_missing:
        if areas_prod_missing != areas:
            areas_prod_existing = [a for a in areas if a not in areas_prod_missing]
            logger.info(f"  → production already available for {areas_prod_existing}, downloading missing: {areas_prod_missing}")
        production = collect_production(client, areas_prod_missing, start, end, gap_report, canon_idx)
        for area, prod_df in production.items():
            prod_df.to_csv(output_dir / f"production_{area}.csv", index=False)
            logger.info(f"  → production_{area}.csv ({len(prod_df)} rows, {len(prod_df.columns)-1} production types)")
    else:
        logger.info("  → all production files already exist, skipping")

    # 3. Installed capacity per area
    logger.info("=== Installed capacity ===")
    installed_capacity_path = output_dir / "installed_capacity.csv"
    if not installed_capacity_path.exists():
        installed_capacity = collect_installed_capacity(client, areas, year)
        installed_capacity.to_csv(installed_capacity_path, index=True)
        logger.info(f"  → installed_capacity.csv ({len(installed_capacity)} technologies, {len(installed_capacity.columns)} areas)")
    else:
        logger.info("  → installed_capacity.csv already exists, skipping")

    # 4. Exogenous prices
    logger.info("=== Exogenous prices ===")
    exo_prices_path = output_dir / "exo_prices.csv"
    if not exo_prices_path.exists():
        exo_prices = collect_prices(client, exo_areas, start, end, gap_report, canon_idx)
        exo_prices.to_csv(exo_prices_path, index=False)
        logger.info(f"  → exo_prices.csv ({len(exo_prices)} rows)")
    else:
        logger.info("  → exo_prices.csv already exists, skipping")

    # 5. Actual prices for modeled areas (validation, not model input)
    logger.info("=== Actual prices (modeled areas) ===")
    actual_prices_path = output_dir / "actual_prices.csv"
    if not actual_prices_path.exists():
        actual_prices = collect_prices(client, areas, start, end, gap_report, canon_idx)
        actual_prices.to_csv(actual_prices_path, index=False)
        logger.info(f"  → actual_prices.csv ({len(actual_prices)} rows)")
    else:
        logger.info("  → actual_prices.csv already exists, skipping")

    # Save gap-fill report in the year directory
    gap_report.save(output_dir)



# ── Demand ──

def collect_demand(client, areas, start, end, gap_report, canon_idx):
    """Collect actual load for each area, in GW.

    For GB/UK, falls back to the Elexon BMRS API when ENTSO-E data is
    unavailable or too sparse (post-Brexit).

    Both entsoe.fetch_demand() and elexon.fetch_demand() return hourly
    naive UTC Series in MW. The orchestrator handles fallback logic,
    reindexing onto the canonical index, and gap-filling.

    Args:
        client: EntsoePandasClient.
        areas: List of area codes.
        start, end: Period bounds (naive UTC).
        gap_report: Report instance for gap-filling audit trail.
        canon_idx: DatetimeIndex from canonical_index(year).

    Returns a DataFrame with columns ['hour', area1, area2, ...].
    """
    frames = {}
    for area in areas:
        series = None
        print(f"Demand {area}... ", end='', flush=True)
        # Try ENTSO-E first
        entsoe_partial = None
        try:
            raw = entsoe.fetch_demand(client, area, start, end)
            if raw is not None and entsoe.is_usable(raw, start, end):
                series = raw
            elif area == "UK" and raw is not None:
                print(" partial ENTSO-E, filling gaps with Elexon...", end='', flush=True)
                entsoe_partial = raw
        except Exception as e:
            if area == "UK":
                print(" no data at ENTSO-E, try Elexon...", end='', flush=True)
            else:
                print(f"FAILED ({type(e).__name__})")
                logger.warning("Demand %s error: %s", area, e)
                continue

        # Elexon fallback for UK: fill gaps in partial ENTSO-E data
        if series is None and area == "UK":
            try:
                elexon_demand = elexon.fetch_demand(start, end)
                if elexon_demand is not None and len(elexon_demand) > 0:
                    if entsoe_partial is not None:
                        series = entsoe_partial.combine_first(elexon_demand)
                    else:
                        series = elexon_demand
            except Exception as e:
                print(f" Elexon fallback FAILED ({type(e).__name__})", end='', flush=True)
                logger.warning("Demand %s Elexon fallback error: %s", area, e)
            # If Elexon also failed, use whatever ENTSO-E partial data we have
            if series is None and entsoe_partial is not None:
                series = entsoe_partial

        if series is None:
            print("no data available (KO)")
            continue

        print("OK")
        series = series.reindex(canon_idx)
        series = interpolate_gaps(series, report=gap_report, variable="demand", area=area)
        frames[area] = series / 1000  # MW → GW
        

    df = pd.DataFrame(frames)
    df.index.name = "hour"
    return df.reset_index()


# ── Production data (raw generation by production type) ──

def collect_production(client, areas, start, end, gap_report, canon_idx):
    """Collect raw hourly production by production type for each area.

    Downloads generation data from ENTSO-E (or Elexon for UK fallback),
    reindexes onto the canonical index, then gap-fills each production series.
    Both sources return the same format: DataFrame with 'hour' column
    (naive UTC) and production columns including 'phs' and 'phs_in'.

    Args:
        client: EntsoePandasClient.
        areas: List of area codes.
        start, end: Period bounds (naive UTC from cet_year_bounds).
        gap_report: Report instance for gap-filling audit trail.
        canon_idx: DatetimeIndex from canonical_index(year).

    Returns:
        dict {area: pd.DataFrame} with columns ['hour', production1, ..., phs, phs_in].
    """
    result = {}
    for area in areas:
        production_df = None
        entsoe_usable = False
        print(f"Production {area}... ", end='', flush=True)

        # Try ENTSO-E first
        try:
            production_df = entsoe.fetch_generation(client, area, start, end)
            if production_df is not None:
                n_expected = expected_hours(start.year)
                if len(production_df) > n_expected * ENTSOE_MIN_COVERAGE:
                    entsoe_usable = True
        except Exception as e:
            if area == "UK":
                print(" no data at ENTSO-E, try Elexon...", end='', flush=True)
            else:
                print(f"FAILED ({type(e).__name__})")
                logger.warning("Production %s error: %s", area, e)
                continue

        # Elexon fallback for UK
        if not entsoe_usable and area == "UK":
            try:
                elexon_df = elexon.fetch_generation(start, end)
                if elexon_df is not None and len(elexon_df) > 0:
                    if production_df is not None and len(production_df) > 0:
                        # Merge: keep ENTSO-E where available, fill gaps with Elexon
                        entsoe_indexed = production_df.set_index("hour")
                        elexon_indexed = elexon_df.set_index("hour")
                        merged = entsoe_indexed.combine_first(elexon_indexed)
                        production_df = merged.reset_index()
                    else:
                        production_df = elexon_df
            except Exception as e:
                print(f" Elexon fallback FAILED ({type(e).__name__})", end='', flush=True)
                logger.warning("Production %s Elexon fallback error: %s", area, e)

        if production_df is not None and len(production_df) > 0:
            print("OK")
            # Reindex onto canonical index, then gap-fill each production column
            indexed = production_df.set_index("hour")
            indexed = indexed.reindex(canon_idx)
            for col in indexed.columns:
                indexed[col] = interpolate_gaps(
                    indexed[col], report=gap_report, variable=col, area=area
                )
            indexed.index.name = "hour"
            result[area] = indexed.reset_index()
        else:
            print("no data available (KO)")

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
        pd.DataFrame in wide format: technologies in rows, areas in columns (MW).
    """
    rows = []
    for area in areas:
        capa = None
        print(f"Installed capacity {area}... ", end='', flush=True)

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
                print(" try Elexon...", end='', flush=True)
                capa = elexon.fetch_installed_capacity(year)
            except Exception as e:
                print(f" Elexon fallback FAILED ({type(e).__name__})", end='', flush=True)
                logger.warning("Installed capacity %s Elexon fallback error: %s", area, e)

        if capa:
            for tec, mw in capa.items():
                rows.append({"area": area, "tec": tec, "value": mw})
            print(f"OK ({len(capa)} types)")
        else:
            print("no data available (KO)")

    long = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["area", "tec", "value"])
    df = long.pivot(index="tec", columns="area", values="value").fillna(0)
    df.columns.name = None  # drop "area" header above column names
    return df


# ── Prices ──

def collect_prices(client, areas, start, end, gap_report, canon_idx):
    """Collect day-ahead prices, in EUR/MWh.

    Uses entsoe.fetch_day_ahead_prices() which handles tz conversion and
    resampling. The orchestrator reindexes onto the canonical index and
    handles gap-filling.

    Args:
        client: EntsoePandasClient.
        areas: List of area codes.
        start, end: Period bounds (naive UTC).
        gap_report: Report instance for gap-filling audit trail.
        canon_idx: DatetimeIndex from canonical_index(year).

    Returns a DataFrame with columns ['hour', area1, area2, ...].
    """
    frames = {}
    for area in areas:
        print(f"Prices {area}... ", end='', flush=True)
        try:
            prices = entsoe.fetch_day_ahead_prices(client, area, start, end)
            if prices is None:
                print("no data available (KO)")
                continue
            print("OK")
            prices = prices.reindex(canon_idx)
            prices = interpolate_gaps(prices, report=gap_report, max_gap=24, variable="price", area=area)
            frames[area] = prices
        except Exception as e:
            print(f"FAILED ({type(e).__name__})")
            logger.warning("Prices %s error: %s", area, e)
            continue

    df = pd.DataFrame(frames)
    df.index.name = "hour"
    return df.reset_index()



# ── Validation helper ──

def _validate_year(year_dir, year, areas, exo_areas):
    """Validate completeness of a year data directory.

    Checks:
        - All required files exist (demand.csv, exo_prices.csv, production_<area>.csv)
        - Each hourly file has the expected number of rows
        - No NaN values remain

    Args:
        year_dir: Path to the year directory (e.g. data/2021_partial).
        year: The year (for computing expected hours).
        areas: List of modeled area codes.
        exo_areas: List of exogenous area codes.

    Returns:
        (is_valid, issues) tuple. issues is a list of error strings.
    """
    issues = []
    n_expected = expected_hours(year)

    # Check demand.csv
    demand_path = year_dir / "demand.csv"
    if not demand_path.exists():
        issues.append("demand.csv missing")
    else:
        df = pd.read_csv(demand_path)
        if len(df) != n_expected:
            issues.append(f"demand.csv: {len(df)} rows, expected {n_expected}")
            demand_path.rename(demand_path.with_name("demand_corrupt.csv"))
        if df.drop(columns=["hour"], errors="ignore").isna().any().any():
            n_nan = df.drop(columns=["hour"], errors="ignore").isna().sum().sum()
            issues.append(f"demand.csv: {n_nan} NaN values remain")
            demand_path.rename(demand_path.with_name("demand_corrupt.csv"))

    # Check exo_prices.csv
    exo_path = year_dir / "exo_prices.csv"
    if not exo_path.exists():
        issues.append("exo_prices.csv missing")
    else:
        df = pd.read_csv(exo_path)
        if len(df) != n_expected:
            issues.append(f"exo_prices.csv: {len(df)} rows, expected {n_expected}")
            exo_path.rename(exo_path.with_name("exo_prices_corrupt.csv"))
        missing_exo = set(exo_areas) - (set(df.columns) - {"hour"})
        if missing_exo:
            issues.append(f"exo_prices.csv: missing areas {missing_exo}")
            exo_path.rename(exo_path.with_name("exo_prices_corrupt.csv"))
        if df.drop(columns=["hour"], errors="ignore").isna().any().any():
            n_nan = df.drop(columns=["hour"], errors="ignore").isna().sum().sum()
            issues.append(f"exo_prices.csv: {n_nan} NaN values remain")
            exo_path.rename(exo_path.with_name("exo_prices_corrupt.csv"))

    # Check production files
    for area in areas:
        prod_path = year_dir / f"production_{area}.csv"
        if not prod_path.exists():
            issues.append(f"production_{area}.csv missing")
        else:
            df = pd.read_csv(prod_path)
            if len(df) != n_expected:
                issues.append(f"production_{area}.csv: {len(df)} rows, expected {n_expected}")
                prod_path.rename(prod_path.with_name(f"production_{area}_corrupt.csv"))
            if df.drop(columns=["hour"], errors="ignore").isna().any().any():
                n_nan = df.drop(columns=["hour"], errors="ignore").isna().sum().sum()
                issues.append(f"production_{area}.csv: {n_nan} NaN values remain")
                prod_path.rename(prod_path.with_name(f"production_{area}_corrupt.csv"))

    # Check actual_prices.csv (soft validation — warnings only, does not block)
    actual_path = year_dir / "actual_prices.csv"
    if not actual_path.exists():
        logger.warning("actual_prices.csv missing (validation data, not critical)")
    else:
        df = pd.read_csv(actual_path)
        if len(df) != n_expected:
            logger.warning(f"actual_prices.csv: {len(df)} rows, expected {n_expected}")
        missing_areas = set(areas) - (set(df.columns) - {"hour"})
        if missing_areas:
            logger.warning(f"actual_prices.csv: missing areas {missing_areas}")

    return len(issues) == 0, issues


