"""ENTSO-E Transparency Platform API client.

Handles all interactions with the ENTSO-E API via the entsoe-py library:
API key validation, area code mapping, and data extraction (demand,
generation, prices). Converts ENTSO-E's multi-level DataFrames
(CET/CEST, sub-hourly) into our standard format: hourly naive UTC.

All public fetch_* functions handle the normalization pipeline:
    naive UTC → CET (for entsoe-py) → download → resample_to_hourly
Clipping to the year range is done by the caller via .reindex(canonical_index(year)).

Delegates to:
    - config.py     ENTSOE_API_KEY, AREA_CODES, AREA_CODES_PRICE, ENTSOE_MIN_COVERAGE.
    - utils.py      resample_to_hourly (tz + resample).

Called from:
    - main_collect.py   All public functions are called from there.

Functions:
    set_client()
        Validate ENTSOE_API_KEY and return an EntsoePandasClient.
        Called from main_collect.collect_all.

    is_usable(series, start, end)
        Check if a series has sufficient coverage (>= ENTSOE_MIN_COVERAGE).
        Called from main_collect.collect_demand.

    fetch_demand(client, area, start, end)
        Download actual load, return as hourly naive UTC Series (MW).
        Called from main_collect.collect_demand.

    fetch_day_ahead_prices(client, area, start, end)
        Download day-ahead prices, return as hourly naive UTC Series (EUR/MWh).
        Called from main_collect.collect_exo_prices.

    fetch_generation(client, area, start, end)
        Download all fuel types, extract per-fuel production, split PHS
        into phs_prod/phs_cons, return as hourly naive UTC DataFrame.
        Called from main_collect.collect_production.

Internal helpers:
    _to_api_timestamps(start, end)
        Convert naive UTC to tz-aware CET for entsoe-py.
    col_matches(col, fuel_type)
        Match an ENTSO-E column name to our internal fuel type key.
    area_code(area)
        Map our area code to an entsoe-py bidding zone code (static, current default).
    area_code_price(area)
        Map our area code to an entsoe-py price zone code (static).
    _resolve_area(area, start, end)
        Time-dependent area resolution (handles DE_AT_LU → DE_LU transition).
    _resolve_area_price(area, start, end)
        Time-dependent price-area resolution (adds IT → IT_NORD override).

Constants:
    ENTSOE_COL_NAMES    Human-readable column name mapping.
    PRODUCTION_FUELS    List of fuel types to extract (excl. PHS).
"""

import logging
import pandas as pd
from entsoe import EntsoePandasClient

from ..config import ENTSOE_MIN_COVERAGE, ENTSOE_API_KEY, AREA_CODES, AREA_CODES_PRICE
from ..utils import resample_to_hourly

logger = logging.getLogger(__name__)


def _to_api_timestamps(start, end):
    """Convert naive UTC timestamps to tz-aware CET for entsoe-py.

    entsoe-py requires tz-aware pandas Timestamps. Its @year_limited
    decorator splits multi-year queries based on timestamp.year, so
    timestamps must be expressed in CET (Europe/Brussels) to align
    with CET year boundaries used by cet_year_bounds().

    Our internal convention is naive UTC everywhere. This helper
    bridges the two: naive UTC → aware UTC → CET.
    """
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    if start.tzinfo is None:
        start = start.tz_localize("UTC").tz_convert("Europe/Brussels")
    if end.tzinfo is None:
        end = end.tz_localize("UTC").tz_convert("Europe/Brussels")
    return start, end


# ── Preliminary ──

def set_client():
    """Check that the ENTSO-E API key is set and looks valid and set the client

    Performs a lightweight test query (FR load for 1 hour) to catch invalid keys
    early, before starting a long collection run.

    Raises EnvironmentError if the key is missing, or RuntimeError if the test
    query fails (wrong key, network issue, etc.).
    """
    if not ENTSOE_API_KEY:
        raise EnvironmentError(
            "ENTSOE_API_KEY environment variable not set.\n"
            "  1. Register at https://transparency.entsoe.eu/\n"
            "  2. Copy your API key from My Account > Web API Security Token\n"
            "  3. Set it via:\n"
            "       export ENTSOE_API_KEY=your-key-here\n"
            "     or add it to a .env file (see .env.example)"
        )

    if len(ENTSOE_API_KEY.strip()) < 10:
        raise EnvironmentError(
            f"ENTSOE_API_KEY looks too short ({len(ENTSOE_API_KEY.strip())} chars). "
            "Check your .env file or environment variable."
        )

    # Set the entsoe client
    client = EntsoePandasClient(api_key=ENTSOE_API_KEY)
    
    # Quick smoke test: query 1 hour of FR load
    try:
        test_start = pd.Timestamp("2023-01-01", tz="Europe/Brussels")
        test_end = pd.Timestamp("2023-01-01T01:00:00", tz="Europe/Brussels")
        result = client.query_load("FR", start=test_start, end=test_end)
        if result is None or (hasattr(result, "__len__") and len(result) == 0):
            raise RuntimeError("ENTSO-E returned empty data for the test query")
    except Exception as e:
        raise RuntimeError(
            f"ENTSO-E API key validation failed: {e}\n"
            "Check that your ENTSOE_API_KEY is correct and that "
            "https://transparency.entsoe.eu/ is reachable."
        ) from e

    logger.info("ENTSO-E API key validated successfully")
    return client

def is_usable(series, start, end):
    """Check whether an ENTSO-E series has sufficient coverage.

    Returns True if the series is non-empty and covers at least
    _ENTSOE_MIN_COVERAGE of the expected hourly range.
    """
    if series is None or (hasattr(series, "__len__") and len(series) == 0):
        return False
    if isinstance(series, pd.Series) and series.isna().all():
        return False
    expected_hours = (end - start).total_seconds() / 3600
    if expected_hours <= 0:
        return False
    valid_count = series.notna().sum() if isinstance(series, pd.Series) else len(series)
    return (valid_count / expected_hours) >= ENTSOE_MIN_COVERAGE



# Human-readable column names returned by entsoe-py when querying with psr_type=None.
# Maps our internal type names → set of possible ENTSO-E column name prefixes.
ENTSOE_COL_NAMES = {
    "biomass": {"Biomass"},
    "lignite": {"Fossil Brown coal/Lignite"},
    "coal_gas": {"Fossil Coal-derived gas"},
    "gas": {"Fossil Gas"},
    "hard_coal": {"Fossil Hard coal"},
    "oil": {"Fossil Oil"},
    "oil_shale": {"Fossil Oil shale"},
    "peat": {"Fossil Peat"},
    "geothermal": {"Geothermal"},
    "phs": {"Hydro Pumped Storage"},
    "river": {"Hydro Run-of-river and poundage"},
    "lake": {"Hydro Water Reservoir"},
    "marine": {"Marine"},
    "nuclear": {"Nuclear"},
    "other_renew": {"Other renewable"},
    "solar": {"Solar"},
    "waste": {"Waste"},
    "offshore": {"Wind Offshore"},
    "onshore": {"Wind Onshore"},
    "other": {"Other"},
}


def col_matches(col, fuel_type):
    """Check if an ENTSO-E DataFrame column matches a given fuel type.

    Args:
        col: Column name — either a string or a tuple (name, aggregation_type).
        fuel_type: Our internal fuel type key (e.g. 'biomass', 'phs', 'lake').

    Returns:
        True if the column matches the fuel type.
    """
    col_name = col[0] if isinstance(col, tuple) else col
    col_str = str(col_name)
    # Check human-readable names
    for name in ENTSOE_COL_NAMES.get(fuel_type, set()):
        if name in col_str:
            return True
    return False

def area_code(area):
    """Map our area code to an entsoe-py country code string for load/generation."""
    code = AREA_CODES.get(area)
    if code is None:
        raise ValueError(f"Unknown area code: {area}. Known: {list(AREA_CODES.keys())}")
    return code


def area_code_price(area):
    """Map our area code to an entsoe-py code for day-ahead prices.

    Some areas need a different code for prices (e.g. IT → IT_NORD).
    """
    return AREA_CODES_PRICE.get(area, area_code(area))


# Germany switched from DE_AT_LU to DE_LU bidding zone on 1 October 2018.
# Naive UTC equivalent: 2018-09-30 22:00 (Oct 1 00:00 CET = UTC+2 in CEST).
_DE_TRANSITION = pd.Timestamp("2018-09-30 22:00:00")


def _resolve_area(area, start, end):
    """Return [(entsoe_code, period_start, period_end), ...] for an area.

    Handles the DE bidding zone transition (DE_AT_LU → DE_LU, Oct 2018).
    For all other areas, returns a single period with the standard code.
    """
    if area == "DE":
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        if e <= _DE_TRANSITION:
            return [("DE_AT_LU", s, e)]
        elif s >= _DE_TRANSITION:
            return [("DE_LU", s, e)]
        else:
            return [("DE_AT_LU", s, _DE_TRANSITION),
                    ("DE_LU", _DE_TRANSITION, e)]
    code = AREA_CODES.get(area)
    if code is None:
        raise ValueError(f"Unknown area code: {area}. Known: {list(AREA_CODES.keys())}")
    return [(code, pd.Timestamp(start), pd.Timestamp(end))]


def _resolve_area_price(area, start, end):
    """Like _resolve_area but with price-specific overrides (IT → IT_NORD).

    IT prices always use IT_NORD (time-independent).
    DE prices follow the same zone transition as load/generation.
    """
    if area in AREA_CODES_PRICE:
        code = AREA_CODES_PRICE[area]
        return [(code, pd.Timestamp(start), pd.Timestamp(end))]
    return _resolve_area(area, start, end)


# ── Demand ──

def fetch_demand(client, area, start, end):
    """Fetch hourly actual load from ENTSO-E for a single area, in MW.

    Handles tz conversion (naive UTC → CET for entsoe-py) and resampling
    to hourly naive UTC. For DE, splits at the Oct 2018 zone transition.
    Caller is responsible for reindexing onto canonical_index.

    Args:
        client: EntsoePandasClient.
        area: Our area code (e.g. 'FR', 'DE', 'UK').
        start, end: Period bounds (naive UTC from cet_year_bounds).

    Returns:
        pd.Series indexed by hourly naive UTC timestamps, values in MW.
        Returns None if the download fails or returns empty data.
    """
    periods = _resolve_area(area, start, end)
    parts = []
    for code, p_start, p_end in periods:
        api_start, api_end = _to_api_timestamps(p_start, p_end)
        raw = client.query_load(code, start=api_start, end=api_end)
        if raw is not None and (not hasattr(raw, "__len__") or len(raw) > 0):
            if isinstance(raw, pd.DataFrame):
                raw = raw.iloc[:, 0]
            parts.append(resample_to_hourly(raw))
    if not parts:
        return None
    return pd.concat(parts).sort_index()


# ── Day-ahead prices ──

def fetch_day_ahead_prices(client, area, start, end):
    """Fetch hourly day-ahead prices from ENTSO-E for a single area, in EUR/MWh.

    Handles tz conversion and resampling to hourly naive UTC.
    For DE, splits at the Oct 2018 zone transition.

    Args:
        client: EntsoePandasClient.
        area: Our area code (e.g. 'FR', 'DE', 'IT').
        start, end: Period bounds (naive UTC from cet_year_bounds).

    Returns:
        pd.Series indexed by hourly naive UTC timestamps, values in EUR/MWh.
        Returns None if the download fails or returns empty data.
    """
    periods = _resolve_area_price(area, start, end)
    parts = []
    for code, p_start, p_end in periods:
        api_start, api_end = _to_api_timestamps(p_start, p_end)
        prices = client.query_day_ahead_prices(code, start=api_start, end=api_end)
        if prices is not None and (not hasattr(prices, "__len__") or len(prices) > 0):
            if isinstance(prices, pd.DataFrame):
                prices = prices.iloc[:, 0]
            parts.append(resample_to_hourly(prices))
    if not parts:
        return None
    return pd.concat(parts).sort_index()


# ── Generation by fuel type ──

# Fuel types to extract from ENTSO-E multi-level generation columns.
# PHS is handled separately (needs both production and consumption).
PRODUCTION_FUELS = [
    "biomass", "lignite", "coal_gas", "gas", "hard_coal", "oil", "oil_shale",
    "peat", "geothermal", "river", "lake", "marine", "nuclear", "other_renew",
    "solar", "waste", "offshore", "onshore", "other",
]


def fetch_generation(client, area, start, end):
    """Fetch hourly generation by fuel type from ENTSO-E.

    Downloads all PSR types at once and extracts per-fuel production.
    PHS is split into phs_prod (generation) and phs_cons (pumping, positive).
    All series are converted to hourly naive UTC.
    For DE, splits at the Oct 2018 zone transition.

    Args:
        client: EntsoePandasClient.
        area: Our area code (e.g. 'FR', 'DE', 'UK').
        start, end: pd.Timestamps for clipping (naive UTC from cet_year_bounds).

    Returns:
        pd.DataFrame with 'hour' column (naive UTC) and one column per
        fuel type found. PHS appears as 'phs_prod' and 'phs_cons'.
        Returns None if the download fails or returns empty data.
    """
    periods = _resolve_area(area, start, end)
    raw_parts = []
    for code, p_start, p_end in periods:
        api_start, api_end = _to_api_timestamps(p_start, p_end)
        part = client.query_generation(code, start=api_start, end=api_end, psr_type=None)
        if isinstance(part, pd.DataFrame) and not part.empty:
            raw_parts.append(part)
    if not raw_parts:
        return None
    raw = pd.concat(raw_parts).sort_index()

    result = {}

    for fuel in PRODUCTION_FUELS:
        prod_series = pd.Series(0, index=raw.index, dtype=float)
        found = False
        for col in raw.columns:
            if col_matches(col, fuel):
                if isinstance(col, tuple) and col[1] == "Actual Aggregated":
                    prod_series = prod_series.add(raw[col], fill_value=0)
                    found = True
                elif not isinstance(col, tuple):
                    prod_series = prod_series.add(raw[col], fill_value=0)
                    found = True
        if found:
            result[fuel] = resample_to_hourly(prod_series)

    # PHS: extract both production and consumption
    phs_prod = pd.Series(0, index=raw.index, dtype=float)
    phs_cons = pd.Series(0, index=raw.index, dtype=float)
    for col in raw.columns:
        if col_matches(col, "phs"):
            if isinstance(col, tuple) and col[1] == "Actual Aggregated":
                phs_prod = phs_prod.add(raw[col], fill_value=0)
            elif isinstance(col, tuple) and col[1] == "Actual Consumption":
                phs_cons = phs_cons.add(raw[col].abs(), fill_value=0)
            elif not isinstance(col, tuple):
                # Single column = net production; approximate prod/cons split
                phs_prod = phs_prod.add(raw[col].clip(lower=0), fill_value=0)
                phs_cons = phs_cons.add((-raw[col]).clip(lower=0), fill_value=0)

    result["phs_prod"] = resample_to_hourly(phs_prod)
    result["phs_cons"] = resample_to_hourly(phs_cons)

    df = pd.DataFrame(result)
    df.index.name = "hour"
    return df.reset_index()

