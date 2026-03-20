"""Collect input data from ENTSO-E and Renewables.ninja for EOLES-Dispatch.

Replaces the R-based data collection scripts (collect_entsoe_data.R).
Uses the entsoe-py client library for robust API access and public
Renewables.ninja country downloads for VRE capacity factor profiles.

Requires:
    pip install entsoe-py
    ENTSOE_API_KEY environment variable (register at https://transparency.entsoe.eu/)

Usage:
    python -m eoles_dispatch collect --start 2020 --end 2024
    python -m eoles_dispatch collect --start 2023 --end 2024 --source entsoe
    python -m eoles_dispatch collect --start 2020 --end 2024 --source ninja
"""

import io
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DEFAULT_AREAS, DEFAULT_EXO_AREAS

logger = logging.getLogger(__name__)

# ── ENTSO-E area codes ──
# Maps our internal country codes to entsoe-py area identifiers.
#
# Perimeter choices:
#   DE → "DE_LU" (bidding zone, includes Luxembourg since Oct 2018).
#        LU is ~0.6 GW peak vs DE ~80 GW, so the impact is negligible.
#        We use the bidding zone rather than control area because ENTSO-E
#        data availability is better at bidding-zone level, and prices are
#        only published for DE_LU (not DE alone).
#        Renewables.ninja uses DE-only data, which is consistent since LU
#        wind/solar capacity is negligible relative to DE.
#   IT → "IT" (whole country, control area). NOT IT_NORD or other sub-zones.
#        The model treats Italy as a single node, so we use the national
#        aggregate. ENTSO-E publishes load/generation at this level.
#   UK → "GB" (Great Britain = England + Scotland + Wales).
#        Excludes Northern Ireland which is part of IE_SEM (all-island market).
#        This matches the R scripts which used EIC 10YGB----------A.
#        Renewables.ninja also uses GB.
AREA_CODES = {
    "FR": "FR",
    "BE": "BE",
    "DE": "DE_LU",
    "CH": "CH",
    "IT": "IT",
    "ES": "ES",
    "UK": "GB",
    "NL": "NL",
    "DK1": "DK_1",
    "DK2": "DK_2",
    "SE4": "SE_4",
    "PL": "PL",
    "CZ": "CZ",
    "AT": "AT",
    "GR": "GR",
    "SI": "SI",
    "PT": "PT",
    "IE": "IE_SEM",
}

# For day-ahead prices, some areas need a different code than for load/generation.
# DE prices were published under DE-AT-LU until Oct 2018, then DE-LU.
# entsoe-py handles this automatically when using DE_LU.
# IT prices use IT_NORD (the reference price zone), not IT (which has no price).
AREA_CODES_PRICE = {
    "IT": "IT_NORD",
}

# PSR type codes for generation by fuel
PSR_TYPES = {
    "biomass": "B01",
    "lignite": "B02",
    "coal_gas": "B03",
    "gas": "B04",
    "hard_coal": "B05",
    "oil": "B06",
    "oil_shale": "B07",
    "peat": "B08",
    "geothermal": "B09",
    "phs": "B10",
    "river": "B11",
    "lake": "B12",
    "marine": "B13",
    "nuclear": "B14",
    "other_renew": "B15",
    "solar": "B16",
    "waste": "B17",
    "offshore": "B18",
    "onshore": "B19",
    "other": "B20",
}

# NMD (non-market-dependent) fuel types to aggregate
NMD_TYPES = ["biomass", "geothermal", "marine", "other_renew", "waste", "other"]


def _get_client():
    """Create an EntsoePandasClient from the ENTSOE_API_KEY env variable."""
    from entsoe import EntsoePandasClient

    api_key = os.environ.get("ENTSOE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ENTSOE_API_KEY environment variable not set. "
            "Register at https://transparency.entsoe.eu/ to get an API key."
        )
    return EntsoePandasClient(api_key=api_key)


def _area_code(area):
    """Map our area code to an entsoe-py country code string for load/generation."""
    code = AREA_CODES.get(area)
    if code is None:
        raise ValueError(f"Unknown area code: {area}. Known: {list(AREA_CODES.keys())}")
    return code


def _area_code_price(area):
    """Map our area code to an entsoe-py code for day-ahead prices.

    Some areas need a different code for prices (e.g. IT → IT_NORD).
    """
    return AREA_CODES_PRICE.get(area, _area_code(area))


def _to_hourly(series):
    """Resample a time series to hourly frequency (mean aggregation)."""
    if series.index.freq is not None and series.index.freq <= pd.Timedelta("1h"):
        return series.resample("h").mean()
    return series


def _interpolate_gaps(series, max_gap=6):
    """Interpolate NaN gaps up to max_gap consecutive hours, fill remainder with 0."""
    interpolated = series.interpolate(method="linear", limit=max_gap)
    return interpolated.fillna(0)


# ── Demand ──

def collect_demand(client, areas, start, end):
    """Collect actual load for each area, in GW.

    Returns a DataFrame with columns ['hour', area1, area2, ...].
    """
    frames = {}
    for area in areas:
        logger.info(f"Downloading demand for {area}")
        try:
            series = client.query_load(_area_code(area), start=start, end=end)
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]  # take first column if DataFrame
            series = _to_hourly(series)
            series = _interpolate_gaps(series)
            frames[area] = series / 1000  # MW → GW
        except Exception as e:
            logger.warning(f"Failed to download demand for {area}: {e}")
            continue

    df = pd.DataFrame(frames)
    df.index.name = "hour"
    return df.reset_index()


# ── Non-market-dependent production ──

def collect_nmd(client, areas, start, end):
    """Collect NMD production (biomass, geothermal, marine, waste, other) in GW.

    Returns a DataFrame with columns ['hour', area1, area2, ...].
    """
    frames = {}
    for area in areas:
        logger.info(f"Downloading NMD production for {area}")
        area_total = None
        try:
            gen = client.query_generation(_area_code(area), start=start, end=end, psr_type=None)
            if isinstance(gen, pd.DataFrame):
                # entsoe-py returns MultiIndex columns (type, production/consumption)
                # We need production only for NMD types
                nmd_cols = []
                for nmd_type in NMD_TYPES:
                    psr = PSR_TYPES[nmd_type]
                    # Try to find this PSR type in the columns
                    for col in gen.columns:
                        col_name = col[0] if isinstance(col, tuple) else col
                        if psr in str(col_name) and (not isinstance(col, tuple) or col[1] == "Actual Aggregated"):
                            nmd_cols.append(col)
                if nmd_cols:
                    area_total = gen[nmd_cols].sum(axis=1)
                else:
                    area_total = pd.Series(0, index=gen.index)
            else:
                area_total = pd.Series(0, index=pd.date_range(start, end, freq="h")[:-1])
        except Exception as e:
            logger.warning(f"Failed to download NMD for {area}: {e}. Using generation query fallback.")
            # Fallback: try individual PSR types
            area_total = pd.Series(0, index=pd.date_range(start, end, freq="h")[:-1])
            for nmd_type in NMD_TYPES:
                try:
                    gen = client.query_generation(
                        _area_code(area), start=start, end=end, psr_type=PSR_TYPES[nmd_type]
                    )
                    if isinstance(gen, pd.DataFrame):
                        gen = gen.iloc[:, 0]
                    area_total = area_total.add(gen, fill_value=0)
                except Exception:
                    pass

        if area_total is not None:
            area_total = _to_hourly(area_total)
            area_total = _interpolate_gaps(area_total)
            frames[area] = area_total / 1000  # MW → GW

    df = pd.DataFrame(frames)
    df.index.name = "hour"
    return df.reset_index()


# ── VRE capacity factors ──

def collect_capacity_factors(client, areas, start, end, technologies=None):
    """Collect actual capacity factors for VRE technologies.

    Returns dict of DataFrames: {tec: DataFrame with ['hour', area1, area2, ...]}.
    Capacity factors are production / installed capacity, clipped to [0, 1].
    """
    if technologies is None:
        technologies = ["offshore", "onshore", "pv", "river"]

    # Map our names to PSR types
    tec_psr = {"offshore": "B18", "onshore": "B19", "pv": "B16", "river": "B11"}

    result = {}
    for tec in technologies:
        psr = tec_psr[tec]
        frames = {}
        for area in areas:
            logger.info(f"Downloading {tec} CF for {area}")
            try:
                gen = client.query_generation(
                    _area_code(area), start=start, end=end, psr_type=psr
                )
                if isinstance(gen, pd.DataFrame):
                    # Get production (not consumption)
                    prod_cols = [c for c in gen.columns
                                 if not isinstance(c, tuple) or c[1] == "Actual Aggregated"]
                    gen = gen[prod_cols[0]] if prod_cols else gen.iloc[:, 0]
                gen = _to_hourly(gen)
                gen = _interpolate_gaps(gen)

                # Get installed capacity
                try:
                    capa = client.query_installed_generation_capacity(
                        _area_code(area), start=start, end=end, psr_type=psr
                    )
                    if isinstance(capa, pd.DataFrame):
                        capa_mw = capa.iloc[-1].sum()  # last available value
                    else:
                        capa_mw = float(capa.iloc[-1]) if len(capa) > 0 else gen.max()
                except Exception:
                    capa_mw = gen.max()

                if capa_mw <= 0:
                    capa_mw = gen.max() if gen.max() > 0 else 1

                cf = (gen / capa_mw).clip(0, 1)
                cf = cf.fillna(0)
                frames[area] = cf
            except Exception as e:
                logger.warning(f"Failed to download {tec} for {area}: {e}")
                continue

        if frames:
            df = pd.DataFrame(frames)
            df.index.name = "hour"
            result[tec] = df.reset_index()

    return result


# ── Exogenous prices ──

def collect_exo_prices(client, exo_areas, start, end):
    """Collect day-ahead prices for exogenous (non-modeled) areas, in EUR/MWh.

    Returns a DataFrame with columns ['hour', area1, area2, ...].
    """
    frames = {}
    for area in exo_areas:
        logger.info(f"Downloading prices for {area}")
        try:
            prices = client.query_day_ahead_prices(_area_code_price(area), start=start, end=end)
            if isinstance(prices, pd.DataFrame):
                prices = prices.iloc[:, 0]
            prices = _to_hourly(prices)
            prices = _interpolate_gaps(prices, max_gap=24)
            frames[area] = prices
        except Exception as e:
            logger.warning(f"Failed to download prices for {area}: {e}")
            continue

    df = pd.DataFrame(frames)
    df.index.name = "hour"
    return df.reset_index()


# ── Lake inflows (monthly budget) ──

def collect_lake_inflows(client, areas, start, end):
    """Collect monthly lake + PHS net production as a proxy for hydro inflows, in TWh.

    Lake inflows are estimated from net hydro production: lake_prod + phs_prod - η * phs_cons.
    Aggregated to monthly sums and converted to TWh.

    Returns a DataFrame with columns ['month', area1, area2, ...].
    """
    eta_phs = 0.9 * 0.95  # round-trip efficiency for PHS

    frames = {}
    for area in areas:
        logger.info(f"Downloading lake inflows for {area}")
        try:
            gen = client.query_generation(_area_code(area), start=start, end=end, psr_type=None)
            if not isinstance(gen, pd.DataFrame):
                continue

            lake_prod = pd.Series(0, index=gen.index)
            phs_net = pd.Series(0, index=gen.index)

            # Find lake production
            for col in gen.columns:
                col_name = col[0] if isinstance(col, tuple) else col
                if "B12" in str(col_name):
                    if isinstance(col, tuple) and col[1] == "Actual Aggregated":
                        lake_prod = lake_prod.add(gen[col], fill_value=0)
                    elif not isinstance(col, tuple):
                        lake_prod = lake_prod.add(gen[col], fill_value=0)

            # Find PHS production and consumption
            for col in gen.columns:
                col_name = col[0] if isinstance(col, tuple) else col
                if "B10" in str(col_name):
                    if isinstance(col, tuple) and col[1] == "Actual Aggregated":
                        phs_net = phs_net.add(gen[col], fill_value=0)
                    elif isinstance(col, tuple) and col[1] == "Actual Consumption":
                        phs_net = phs_net.subtract(gen[col].abs() * eta_phs, fill_value=0)

            total = _to_hourly(lake_prod + phs_net)
            total = _interpolate_gaps(total)

            # Aggregate to monthly, convert MWh → TWh (sum of hourly MW values = MWh)
            monthly = total.resample("MS").sum() / 1e6
            monthly = monthly.clip(lower=0)
            frames[area] = monthly
        except Exception as e:
            logger.warning(f"Failed to download lake inflows for {area}: {e}")
            continue

    df = pd.DataFrame(frames)
    df.index.name = "month_dt"
    df = df.reset_index()
    df["month"] = df["month_dt"].dt.strftime("%Y%m")
    df = df.drop(columns=["month_dt"])
    cols = ["month"] + [c for c in df.columns if c != "month"]
    return df[cols]


# ── Hydro max in/out (monthly) ──

def collect_hydro_limits(client, areas, start, end):
    """Collect monthly max hydro charge/discharge power as a fraction of installed capacity.

    Returns (hMaxIn, hMaxOut) DataFrames with columns ['month', area1, area2, ...].
    Values in GW.
    """
    frames_in = {}
    frames_out = {}
    for area in areas:
        logger.info(f"Downloading hydro limits for {area}")
        try:
            gen = client.query_generation(_area_code(area), start=start, end=end, psr_type=None)
            if not isinstance(gen, pd.DataFrame):
                continue

            # PHS production (out) and consumption (in)
            phs_prod = pd.Series(0, index=gen.index)
            phs_cons = pd.Series(0, index=gen.index)
            lake_prod = pd.Series(0, index=gen.index)

            for col in gen.columns:
                col_name = col[0] if isinstance(col, tuple) else col
                if "B10" in str(col_name):
                    if isinstance(col, tuple) and col[1] == "Actual Aggregated":
                        phs_prod = phs_prod.add(gen[col].clip(lower=0), fill_value=0)
                    elif isinstance(col, tuple) and col[1] == "Actual Consumption":
                        phs_cons = phs_cons.add(gen[col].abs(), fill_value=0)
                if "B12" in str(col_name):
                    if isinstance(col, tuple) and col[1] == "Actual Aggregated":
                        lake_prod = lake_prod.add(gen[col].clip(lower=0), fill_value=0)

            total_out = _to_hourly(lake_prod + phs_prod)
            total_in = _to_hourly(phs_cons)

            # Monthly max, convert MW → GW
            monthly_out = total_out.resample("MS").max() / 1000
            monthly_in = total_in.resample("MS").max() / 1000

            frames_out[area] = monthly_out
            frames_in[area] = monthly_in
        except Exception as e:
            logger.warning(f"Failed to download hydro limits for {area}: {e}")
            continue

    def _format_monthly(frames):
        df = pd.DataFrame(frames)
        df.index.name = "month_dt"
        df = df.reset_index()
        df["month"] = df["month_dt"].dt.strftime("%Y%m")
        df = df.drop(columns=["month_dt"])
        cols = ["month"] + [c for c in df.columns if c != "month"]
        return df[cols]

    return _format_monthly(frames_in), _format_monthly(frames_out)


# ── Nuclear weekly availability ──

def collect_nuclear_availability(client, areas, start, end):
    """Collect weekly max nuclear availability factor (proxy for maintenance schedule).

    Returns a DataFrame with columns ['week', area1, area2, ...].
    Values in [0, 1].
    """
    frames = {}
    for area in areas:
        logger.info(f"Downloading nuclear availability for {area}")
        try:
            gen = client.query_generation(
                _area_code(area), start=start, end=end, psr_type=PSR_TYPES["nuclear"]
            )
            if isinstance(gen, pd.DataFrame):
                prod_cols = [c for c in gen.columns
                             if not isinstance(c, tuple) or c[1] == "Actual Aggregated"]
                gen = gen[prod_cols[0]] if prod_cols else gen.iloc[:, 0]
            gen = _to_hourly(gen)
            gen = _interpolate_gaps(gen)

            # Get installed nuclear capacity
            try:
                capa = client.query_installed_generation_capacity(
                    _area_code(area), start=start, end=end, psr_type=PSR_TYPES["nuclear"]
                )
                if isinstance(capa, pd.DataFrame):
                    capa_mw = capa.iloc[-1].sum()
                else:
                    capa_mw = float(capa.iloc[-1])
            except Exception:
                capa_mw = gen.max()

            if capa_mw <= 0:
                capa_mw = gen.max() if gen.max() > 0 else 1

            af = (gen / capa_mw).clip(0, 1).fillna(0)

            # Weekly max
            weekly = af.resample("W-MON").max()
            frames[area] = weekly
        except Exception as e:
            logger.warning(f"Failed to download nuclear data for {area}: {e}")
            continue

    df = pd.DataFrame(frames)
    df.index.name = "week_dt"
    df = df.reset_index()
    df["week"] = df["week_dt"].dt.strftime("%Y%W")
    df = df.drop(columns=["week_dt"])
    cols = ["week"] + [c for c in df.columns if c != "week"]
    return df[cols]


# ── Renewables.ninja country downloads ──

# URL template for Renewables.ninja public country downloads.
# Format: ninja-{type}-country-{ISO2}-{variant}-merra2.csv
NINJA_BASE_URL = "https://www.renewables.ninja/country_downloads/{iso2}"

# Maps our area codes to Renewables.ninja ISO2 codes.
# Perimeter consistency with ENTSO-E:
#   DE → "DE" (Germany only). Ninja has no DE_LU aggregate, but LU capacity
#        is negligible so DE-only profiles are representative of the DE_LU zone.
#   IT → "IT" (whole country). Matches ENTSO-E IT control area.
#   UK → "GB" (Great Britain). Matches ENTSO-E GB bidding zone.
NINJA_ISO2 = {
    "FR": "FR", "BE": "BE", "DE": "DE", "CH": "CH",
    "IT": "IT", "ES": "ES", "UK": "GB", "LU": "LU",
}

# File definitions: (our_name, url_filename_template)
# {iso2} is replaced with the country code.
NINJA_FILES = {
    "pv": "ninja-pv-country-{iso2}-national-merra2.csv",
    "onshore_CU": "ninja-wind-country-{iso2}-current_onshore-merra2.csv",
    "onshore_NT": "ninja-wind-country-{iso2}-future_onshore-merra2.csv",
    "offshore_CU": "ninja-wind-country-{iso2}-current_offshore-merra2.csv",
    "offshore_NT": "ninja-wind-country-{iso2}-future_offshore-merra2.csv",
    "offshore_LT": "ninja-wind-country-{iso2}-future_offshore-merra2.csv",  # same as NT for now
}

# Countries that have no offshore data (landlocked)
NO_OFFSHORE = {"CH", "LU"}


def _download_ninja_csv(iso2, filename):
    """Download a single CSV from Renewables.ninja and return the NATIONAL column as a Series."""
    import urllib.request

    url = f"{NINJA_BASE_URL.format(iso2=iso2)}/{filename.format(iso2=iso2)}"
    logger.info(f"  Downloading {url}")
    try:
        with urllib.request.urlopen(url) as resp:
            raw = resp.read().decode("utf-8")
    except Exception as e:
        logger.warning(f"  Failed to download {url}: {e}")
        return None

    # Skip comment lines (start with #)
    lines = raw.split("\n")
    data_lines = [l for l in lines if not l.startswith('"#') and not l.startswith("#")]
    csv_text = "\n".join(data_lines)

    df = pd.read_csv(io.StringIO(csv_text), parse_dates=["time"])
    df = df.set_index("time")

    if "NATIONAL" not in df.columns:
        logger.warning(f"  No NATIONAL column in {filename} for {iso2}")
        return None

    return df["NATIONAL"]


def collect_ninja(output_dir, areas=None):
    """Download Renewables.ninja capacity factor profiles for all areas.

    Downloads PV, onshore (current/future), and offshore (current/future)
    from the public country downloads. Produces CSVs in the same format as
    the existing renewable_ninja/ directory: columns ['hour', area1, area2, ...].

    Args:
        output_dir: Path to data/renewable_ninja/ directory.
        areas: List of area codes (default: DEFAULT_AREAS).
    """
    if areas is None:
        areas = list(DEFAULT_AREAS)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for file_key, filename_template in NINJA_FILES.items():
        logger.info(f"=== Collecting {file_key} ===")
        is_offshore = "offshore" in file_key
        series_dict = {}

        for area in areas:
            iso2 = NINJA_ISO2.get(area)
            if iso2 is None:
                logger.warning(f"  No Renewables.ninja ISO2 mapping for {area}, skipping")
                continue
            if is_offshore and area in NO_OFFSHORE:
                logger.info(f"  Skipping offshore for {area} (landlocked)")
                continue

            series = _download_ninja_csv(iso2, filename_template)
            if series is not None:
                series_dict[area] = series

        if not series_dict:
            logger.warning(f"  No data collected for {file_key}")
            continue

        # Align all countries on the same time index (intersection)
        df = pd.DataFrame(series_dict)
        df.index.name = "hour"

        # Fill missing offshore countries with 0
        if is_offshore:
            for area in areas:
                if area not in df.columns:
                    df[area] = 0.0
            df = df[areas]  # reorder columns

        df = df.reset_index()
        # Convert timezone-aware timestamps to naive UTC
        df["hour"] = pd.to_datetime(df["hour"]).dt.tz_localize(None)

        out_path = output_dir / f"{file_key}.csv"
        df.to_csv(out_path, index=False)
        logger.info(f"  → {file_key}.csv ({len(df)} rows, {len(df.columns)-1} areas)")

    logger.info("=== Ninja collection complete ===")
    return output_dir


# ── Main collection orchestrator ──

def collect_all(
    output_dir,
    start_year,
    end_year,
    areas=None,
    exo_areas=None,
    source="all",
):
    """Download all time-varying data and save to CSV files.

    Args:
        output_dir: Path to data/ directory (will write to time_varying_inputs/ and renewable_ninja/).
        start_year: First year to download (e.g. 2020).
        end_year: Last year to download, exclusive (e.g. 2025 to get 2020-2024).
        areas: Modeled country codes (default: FR, BE, DE, CH, IT, ES, UK).
        exo_areas: Non-modeled country codes for price data.
        source: "all", "entsoe", or "ninja".
    """
    if areas is None:
        areas = list(DEFAULT_AREAS)
    if exo_areas is None:
        exo_areas = list(DEFAULT_EXO_AREAS)

    output_dir = Path(output_dir)

    if source in ("all", "entsoe"):
        tv_dir = output_dir / "time_varying_inputs"
        tv_dir.mkdir(parents=True, exist_ok=True)

        start = pd.Timestamp(f"{start_year}-01-01", tz="Europe/Brussels")
        end = pd.Timestamp(f"{end_year}-01-01", tz="Europe/Brussels")

        client = _get_client()

        # 1. Demand
        logger.info("=== Collecting demand ===")
        demand = collect_demand(client, areas, start, end)
        demand.to_csv(tv_dir / "demand.csv", index=False)
        logger.info(f"  → demand.csv ({len(demand)} rows)")

        # 2. NMD production
        logger.info("=== Collecting NMD production ===")
        nmd = collect_nmd(client, areas, start, end)
        nmd.to_csv(tv_dir / "nmd.csv", index=False)
        logger.info(f"  → nmd.csv ({len(nmd)} rows)")

        # 3. VRE capacity factors (from ENTSO-E historical data)
        logger.info("=== Collecting VRE capacity factors (ENTSO-E) ===")
        cfs = collect_capacity_factors(client, areas, start, end)
        for tec, df in cfs.items():
            df.to_csv(tv_dir / f"{tec}.csv", index=False)
            logger.info(f"  → {tec}.csv ({len(df)} rows)")

        # 4. Exogenous prices
        logger.info("=== Collecting exogenous prices ===")
        exo_prices = collect_exo_prices(client, exo_areas, start, end)
        exo_prices.to_csv(tv_dir / "exoPrices.csv", index=False)
        logger.info(f"  → exoPrices.csv ({len(exo_prices)} rows)")

        # 5. Lake inflows
        logger.info("=== Collecting lake inflows ===")
        lake_inflows = collect_lake_inflows(client, areas, start, end)
        lake_inflows.to_csv(tv_dir / "lake_inflows.csv", index=False)
        logger.info(f"  → lake_inflows.csv ({len(lake_inflows)} rows)")

        # 6. Hydro limits
        logger.info("=== Collecting hydro limits ===")
        h_max_in, h_max_out = collect_hydro_limits(client, areas, start, end)
        h_max_in.to_csv(tv_dir / "hMaxIn.csv", index=False)
        h_max_out.to_csv(tv_dir / "hMaxOut.csv", index=False)
        logger.info(f"  → hMaxIn.csv, hMaxOut.csv ({len(h_max_in)} rows)")

        # 7. Nuclear availability
        logger.info("=== Collecting nuclear availability ===")
        nuc = collect_nuclear_availability(client, areas, start, end)
        nuc.to_csv(tv_dir / "nucMaxAF.csv", index=False)
        logger.info(f"  → nucMaxAF.csv ({len(nuc)} rows)")

    if source in ("all", "ninja"):
        ninja_dir = output_dir / "renewable_ninja"
        logger.info("=== Collecting Renewables.ninja profiles ===")
        collect_ninja(ninja_dir, areas)

    logger.info("=== Collection complete ===")
    return output_dir
