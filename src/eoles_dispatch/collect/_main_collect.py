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
        production_<area>.csv   - hourly generation by fuel type (MW)
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
        Download hourly generation by fuel type per area (MW). ENTSO-E
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



# ── Main collection orchestrator ──

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
            elif partial_dir.exists():
                logger.info(f"Data for {year} present locally is partial: removing and redownloading...")
                shutil.rmtree(partial_dir)
            elif corrupt_dir.exists():
                logger.info(f"Data for {year} present locally is corrupt: removing and redownloading...")
                shutil.rmtree(corrupt_dir)
            else:
                logger.info(f"No data for {year}: downloading...")

            # Launch download of history data for <year>
            partial_dir.mkdir(parents=True)
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


# ── Collection of full history data for 1 year ──

def collect_history(
    output_dir,
    client,
    year,
    areas=None,
    exo_areas=None,
):
    """Download all time-varying ENTSO-E data for a single year and save to CSV.

    Fetches demand, generation by fuel type, installed capacity, exogenous
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
    demand = collect_demand(client, areas, start, end, gap_report, canon_idx)
    demand_mw = demand.copy()
    area_cols = [c for c in demand_mw.columns if c != "hour"]
    demand_mw[area_cols] = demand_mw[area_cols] * 1000  # GW → MW
    demand_mw.to_csv(output_dir / "demand.csv", index=False)
    logger.info(f"  → demand.csv ({len(demand_mw)} rows)")

    # 2. Raw production per area
    logger.info("=== Production ===")
    production = collect_production(client, areas, start, end, gap_report, canon_idx)
    for area, prod_df in production.items():
        prod_df.to_csv(output_dir / f"production_{area}.csv", index=False)
        logger.info(f"  → production_{area}.csv ({len(prod_df)} rows, {len(prod_df.columns)-1} fuel types)")

    # 3. Installed capacity per area
    logger.info("=== Installed capacity ===")
    collect_installed_capacity(client, areas, year, output_dir)

    # 4. Exogenous prices
    logger.info("=== Exogenous prices ===")
    exo_prices = collect_exo_prices(client, exo_areas, start, end, gap_report, canon_idx)
    exo_prices.to_csv(output_dir / "exo_prices.csv", index=False)
    logger.info(f"  → exoPrices.csv ({len(exo_prices)} rows)")

    # 5. Actual prices for modeled areas (validation, not model input)
    logger.info("=== Actual prices (modeled areas) ===")
    actual_prices = collect_actual_prices(client, areas, start, end, gap_report, canon_idx)
    actual_prices.to_csv(output_dir / "actual_prices.csv", index=False)
    logger.info(f"  → actual_prices.csv ({len(actual_prices)} rows)")

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
                logger.info("downloaded from ENTSO-E (OK)")
            elif area == "UK" and raw is not None:
                print("partially downloaded demand from ENTSO-E, filling gaps with Elexon... ", end='', flush=True)
                entsoe_partial = raw
        except Exception as e:
            if area == "UK":
                print("no data at ENTSO-E, try Elexon... ", end='', flush=True)
            else:
                logger.warning(f"Failed (KO)\nERROR:{e}")
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
                logger.info("completed from Elexon (OK)")    
            except Exception as e:
                logger.warning(f"Elexon fallback failed (KO)\nERROR: {e}")
            # If Elexon also failed, use whatever ENTSO-E partial data we have
            if series is None and entsoe_partial is not None:
                series = entsoe_partial

        if series is None:
            logger.warning("no data available (KO)")
            continue

        series = series.reindex(canon_idx)
        series = interpolate_gaps(series, report=gap_report, variable="demand", area=area)
        frames[area] = series / 1000  # MW → GW

    df = pd.DataFrame(frames)
    df.index.name = "hour"
    return df.reset_index()


# ── Production data (raw generation by fuel type) ──

def collect_production(client, areas, start, end, gap_report, canon_idx):
    """Collect raw hourly production by fuel type for each area.

    Downloads generation data from ENTSO-E (or Elexon for UK fallback),
    reindexes onto the canonical index, then gap-fills each fuel series.
    Both sources return the same format: DataFrame with 'hour' column
    (naive UTC) and fuel columns including 'phs' and 'phs_in'.

    Args:
        client: EntsoePandasClient.
        areas: List of area codes.
        start, end: Period bounds (naive UTC from cet_year_bounds).
        gap_report: Report instance for gap-filling audit trail.
        canon_idx: DatetimeIndex from canonical_index(year).

    Returns:
        dict {area: pd.DataFrame} with columns ['hour', fuel1, ..., phs, phs_in].
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
                    logger.info("downloaded from ENTSO-E (OK)")
        except Exception as e:
            if area == "UK":
                logger.info("no data at ENTSO-E, try Elexon... ")
            else:
                logger.warning(f"Failed (KO)\nERROR:{e}")
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
                    logger.info("completed from Elexon (OK)")   
            except Exception as e:
                logger.warning(f"Elexon fallback failed (KO)\nERROR: {e}")

        if production_df is not None and len(production_df) > 0:
            # Reindex onto canonical index, then gap-fill each fuel column
            indexed = production_df.set_index("hour")
            indexed = indexed.reindex(canon_idx)
            for col in indexed.columns:
                indexed[col] = interpolate_gaps(
                    indexed[col], report=gap_report, variable=col, area=area
                )
            indexed.index.name = "hour"
            result[area] = indexed.reset_index()

    return result


# ── Installed capacity ──

def collect_installed_capacity(client, areas, year, out_dir):
    """Collect installed generation capacity per fuel type for each area.

    ENTSO-E primary, Elexon fallback for UK. Saves installed_capacity.csv
    in wide format: technologies in rows, areas in columns (MW).

    Args:
        client: EntsoePandasClient.
        areas: List of area codes.
        year: Calendar year.
        out_dir: Path to write installed_capacity.csv.
    """
    rows = []
    for area in areas:
        capa = None
        print(f"Installed capacity {area}... ", end='', flush=True)

        # Try ENTSO-E
        try:
            capa = entsoe.fetch_installed_capacity(client, area, year)
            if capa:
                logger.info(f"from ENTSO-E ({len(capa)} fuel types) (OK)")
        except Exception as e:
            if area != "UK":
                logger.warning(f"Failed (KO)\nERROR: {e}")

        # Elexon fallback for UK
        if not capa and area == "UK":
            try:
                print("try Elexon... ", end='', flush=True)
                capa = elexon.fetch_installed_capacity(year)
                if capa:
                    logger.info(f"from Elexon ({len(capa)} fuel types) (OK)")
            except Exception as e:
                logger.warning(f"Elexon fallback failed (KO)\nERROR: {e}")

        if capa:
            for tec, mw in capa.items():
                rows.append({"area": area, "tec": tec, "value": mw})
        else:
            logger.warning("no data available (KO)")

    long = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["area", "tec", "value"])
    df = long.pivot(index="tec", columns="area", values="value").fillna(0)
    df.columns.name = None  # drop "area" header above column names
    df.to_csv(out_dir / "installed_capacity.csv")
    logger.info(f"  → installed_capacity.csv ({len(df)} technologies, {len(df.columns)} areas)")


# ── Exogenous prices ──

def collect_exo_prices(client, exo_areas, start, end, gap_report, canon_idx):
    """Collect day-ahead prices for exogenous (non-modeled) areas, in EUR/MWh.

    Uses entsoe.fetch_day_ahead_prices() which handles tz conversion and
    resampling. The orchestrator reindexes onto the canonical index and
    handles gap-filling.

    Args:
        client: EntsoePandasClient.
        exo_areas: List of exogenous area codes.
        start, end: Period bounds (naive UTC).
        gap_report: Report instance for gap-filling audit trail.
        canon_idx: DatetimeIndex from canonical_index(year).

    Returns a DataFrame with columns ['hour', area1, area2, ...].
    """
    frames = {}
    for area in exo_areas:
        print(f"Exogenous prices {area}... ", end='', flush=True)
        try:
            prices = entsoe.fetch_day_ahead_prices(client, area, start, end)
            if prices is None:
                logger.warning("no data available (KO)")
                continue
            prices = prices.reindex(canon_idx)
            prices = interpolate_gaps(prices, report=gap_report, max_gap=24, variable="exo_price", area=area)
            frames[area] = prices
            logger.info("downloaded (OK)")
        except Exception as e:
            logger.warning(f"Failed (KO)\nERROR: {e}")
            continue

    df = pd.DataFrame(frames)
    df.index.name = "hour"
    return df.reset_index()


# ── Actual prices (modeled areas, for validation) ──

def collect_actual_prices(client, areas, start, end, gap_report, canon_idx):
    """Collect day-ahead prices for modeled areas, in EUR/MWh.

    These prices are NOT used as model inputs — they serve as a reference
    for comparing simulated prices against observed market outcomes.
    Stored separately from exo_prices.csv to maintain a clean separation.

    Args:
        client: EntsoePandasClient.
        areas: List of modeled area codes (e.g. FR, BE, DE, ...).
        start, end: Period bounds (naive UTC).
        gap_report: Report instance for gap-filling audit trail.
        canon_idx: DatetimeIndex from canonical_index(year).

    Returns a DataFrame with columns ['hour', area1, area2, ...].
    """
    frames = {}
    for area in areas:
        print(f"Actual prices {area}... ", end='', flush=True)
        try:
            prices = entsoe.fetch_day_ahead_prices(client, area, start, end)
            if prices is None:
                logger.warning("no data available (KO)")
                continue
            prices = prices.reindex(canon_idx)
            prices = interpolate_gaps(prices, report=gap_report, max_gap=24, variable="actual_price", area=area)
            frames[area] = prices
            logger.info("downloaded (OK)")
        except Exception as e:
            logger.warning(f"Failed (KO)\nERROR: {e}")
            continue

    df = pd.DataFrame(frames)
    df.index.name = "hour"
    return df.reset_index()



# ── Year-based validation ──

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
        if df.drop(columns=["hour"], errors="ignore").isna().any().any():
            n_nan = df.drop(columns=["hour"], errors="ignore").isna().sum().sum()
            issues.append(f"demand.csv: {n_nan} NaN values remain")

    # Check exo_prices.csv
    exo_path = year_dir / "exo_prices.csv"
    if not exo_path.exists():
        issues.append("exo_prices.csv missing")
    else:
        df = pd.read_csv(exo_path)
        if len(df) != n_expected:
            issues.append(f"exo_prices.csv: {len(df)} rows, expected {n_expected}")
        missing_exo = set(exo_areas) - (set(df.columns) - {"hour"})
        if missing_exo:
            issues.append(f"exo_prices.csv: missing areas {missing_exo}")
        if df.drop(columns=["hour"], errors="ignore").isna().any().any():
            n_nan = df.drop(columns=["hour"], errors="ignore").isna().sum().sum()
            issues.append(f"exo_prices.csv: {n_nan} NaN values remain")

    # Check production files
    for area in areas:
        prod_path = year_dir / f"production_{area}.csv"
        if not prod_path.exists():
            issues.append(f"production_{area}.csv missing")
        else:
            df = pd.read_csv(prod_path)
            if len(df) != n_expected:
                issues.append(f"production_{area}.csv: {len(df)} rows, expected {n_expected}")
            if df.drop(columns=["hour"], errors="ignore").isna().any().any():
                n_nan = df.drop(columns=["hour"], errors="ignore").isna().sum().sum()
                issues.append(f"production_{area}.csv: {n_nan} NaN values remain")

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


