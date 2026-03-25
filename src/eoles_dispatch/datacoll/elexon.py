"""Elexon BMRS Insights API client for GB electricity data.

Since Brexit (~mid-2021), Great Britain no longer transmits complete data to
ENTSO-E. This module provides equivalent data from the Elexon BMRS Insights
API, used as an automatic fallback for UK in main_collect.py.

The Insights API is free, public, and requires no API key or registration.
All returned DataFrames follow the project convention: hourly naive UTC.

API documentation: https://bmrs.elexon.co.uk/api-documentation
Base URL: https://data.elexon.co.uk/bmrs/api/v1

Delegates to:
    - utils.py      resample_to_hourly (tz normalization + resample).

Called from:
    - main_collect.py   fetch_demand (from collect_demand, UK fallback),
                        fetch_generation (from collect_production, UK fallback).

Functions:
    fetch_demand(start, end)
        Fetch half-hourly GB demand from /demand/outturn, resample to
        hourly naive UTC. Returns a Series (MW).
        Called from main_collect.collect_demand.

    fetch_generation(start, end)
        Fetch generation by fuel type from /generation/actual/per-type.
        Splits PHS into phs_prod/phs_cons. Returns a DataFrame with
        'hour' column (naive UTC) and one column per fuel.
        Called from main_collect.collect_production, fetch_generation_for_fuel.

    fetch_generation_for_fuel(start, end, fuel)
        Convenience wrapper: fetch_generation then extract a single fuel.
        Not currently called internally (available for ad-hoc use).

    fetch_day_ahead_prices(start, end, gbp_to_eur=1.18)
        Fetch GB day-ahead prices (N2EX/APX), convert GBP to EUR.
        Not currently called internally (available for ad-hoc use).
        TODO: fetch live GBP/EUR rate instead of hardcoded value.

Internal helpers:
    _fetch_json(endpoint, params)   - Raw API call.
    _date_chunks(start, end)        - Split range into 7-day chunks.
    _settlement_period_to_time()    - Settlement period -> UTC timestamp.
    _to_hourly_utc(df, value_col)   - Extract column + resample_to_hourly.
"""

import json
import logging
import urllib.request
from datetime import timedelta

import numpy as np
import pandas as pd

from ..utils import resample_to_hourly

logger = logging.getLogger(__name__)

BASE_URL = "https://data.elexon.co.uk/bmrs/api/v1"

# Maximum date range per request. The API does not document an explicit limit,
# but empirically large ranges may time out. 7 days is safe and fast.
_CHUNK_DAYS = 7

# Elexon psrType → our internal fuel category mapping.
# Used to extract the right generation columns for NMD, VRE capacity factors, etc.
PSR_MAP = {
    "Biomass": "biomass",
    "Fossil Gas": "gas",
    "Fossil Hard coal": "hard_coal",
    "Fossil Oil": "oil",
    "Hydro Pumped Storage": "phs",
    "Hydro Run-of-river and poundage": "river",
    "Nuclear": "nuclear",
    "Other": "other",
    "Solar": "solar",
    "Wind Offshore": "offshore",
    "Wind Onshore": "onshore",
}



def _fetch_json(endpoint, params):
    """Fetch JSON from the Elexon Insights API.

    Args:
        endpoint: API path after the base URL (e.g. "/demand/actual/total").
        params: Dict of query parameters.

    Returns:
        Parsed JSON response, or None on failure.
    """
    params["format"] = "json"
    query = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{BASE_URL}{endpoint}?{query}"

    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        logger.warning(f"  Elexon API request failed: {url} — {e}")
        return None


def _date_chunks(start, end, chunk_days=_CHUNK_DAYS):
    """Yield (chunk_start, chunk_end) pairs covering [start, end)."""
    current = start
    while current < end:
        chunk_end = min(current + timedelta(days=chunk_days), end)
        yield current, chunk_end
        current = chunk_end


def _settlement_period_to_time(date_str, period):
    """Convert a settlement date + period (1–48) to a UTC timestamp.

    Each settlement period is 30 minutes. Period 1 starts at 00:00 UTC.
    """
    base = pd.Timestamp(date_str, tz="UTC")
    return base + pd.Timedelta(minutes=30 * (period - 1))


def _to_hourly_utc(df, value_col="value"):
    """Extract a column from a timestamped DataFrame and convert to hourly naive UTC.

    Uses the shared resample_to_hourly helper for tz handling and resampling.
    """
    if df.empty:
        return pd.Series(dtype=float)
    series = df.set_index("timestamp")[value_col].sort_index()
    return resample_to_hourly(series)


# ── Demand ──────────────────────────────────────────────────────────────────

def fetch_demand(start, end):
    """Fetch actual total load for GB from Elexon, in MW.

    Uses the /demand/outturn endpoint (INDO/ITSDO datasets) which provides
    half-hourly demand outturn by settlement date and period.

    Args:
        start: Start datetime (pd.Timestamp or datetime).
        end: End datetime (pd.Timestamp or datetime).

    Returns:
        pd.Series indexed by hourly UTC timestamps, values in MW.
    """
    records = []

    for chunk_start, chunk_end in _date_chunks(start, end):
        data = _fetch_json("/demand/outturn", {
            "settlementDateFrom": chunk_start.strftime("%Y-%m-%d"),
            "settlementDateTo": chunk_end.strftime("%Y-%m-%d"),
        })
        if data is None or "data" not in data:
            continue

        for rec in data["data"]:
            ts = _settlement_period_to_time(
                rec["settlementDate"], rec["settlementPeriod"]
            )
            records.append({
                "timestamp": ts,
                "value": rec.get("initialDemandOutturn", 0) or 0,
            })

    if not records:
        logger.warning("  No demand data returned from Elexon")
        return pd.Series(dtype=float)

    df = pd.DataFrame(records)
    series = _to_hourly_utc(df)
    return series


# ── Generation by fuel type ─────────────────────────────────────────────────

def fetch_generation(start, end):
    """Fetch actual generation by fuel type for GB from Elexon, in MW.

    PHS is split into 'phs_prod' (generation, positive) and 'phs_cons'
    (pumping, positive). All other fuel types appear as individual columns.

    Args:
        start: Start datetime.
        end: End datetime.

    Returns:
        pd.DataFrame with 'hour' column (naive UTC) and one column per
        fuel type. Values in MW. Returns None if no data.
    """

    records = []

    for chunk_start, chunk_end in _date_chunks(start, end):
        data = _fetch_json("/generation/actual/per-type", {
            "from": chunk_start.strftime("%Y-%m-%dT%H:%MZ"),
            "to": chunk_end.strftime("%Y-%m-%dT%H:%MZ"),
        })
        if data is None or "data" not in data:
            continue

        for period_block in data["data"]:
            ts = pd.Timestamp(period_block["startTime"])
            for gen in period_block.get("data", []):
                psr_type = gen.get("psrType", "")
                our_name = PSR_MAP.get(psr_type)
                if our_name is None:
                    logger.debug(f"  Unmapped Elexon psrType: {psr_type!r}")
                    continue
                records.append({
                    "timestamp": ts,
                    "fuel": our_name,
                    "value": gen.get("quantity", 0) or 0,
                })

    if not records:
        logger.warning("  No generation data returned from Elexon")
        return None

    df = pd.DataFrame(records)
    # Pivot to wide format: one column per fuel type
    pivot = df.pivot_table(
        index="timestamp", columns="fuel", values="value", aggfunc="mean"
    )
    # Normalize each column to hourly naive UTC via shared helper
    result = pd.DataFrame({
        col: resample_to_hourly(pivot[col])
        for col in pivot.columns
    })

    # PHS: split net value into production and consumption (both positive)
    if "phs" in result.columns:
        result["phs_prod"] = result["phs"].clip(lower=0).fillna(0)
        result["phs_cons"] = (-result["phs"]).clip(lower=0).fillna(0)
        result = result.drop(columns=["phs"])
    else:
        result["phs_prod"] = 0.0
        result["phs_cons"] = 0.0

    result.index.name = "hour"
    return result.reset_index()



def fetch_generation_for_fuel(start, end, fuel):
    """Fetch generation for a single fuel type for GB, in MW.

    Args:
        fuel: One of our internal fuel names (e.g. "onshore", "solar", "nuclear").

    Returns:
        pd.Series indexed by hourly UTC timestamps, values in MW.
    """
    gen = fetch_generation(start, end)
    if gen is None or fuel not in gen.columns:
        return pd.Series(dtype=float)
    return gen.set_index("hour")[fuel]


# ── Day-ahead prices ────────────────────────────────────────────────────────

def fetch_day_ahead_prices(start, end, gbp_to_eur=1.18):
    """Fetch day-ahead market index prices for GB from Elexon, in EUR/MWh.

    Uses the N2EXMIDP (N2EX) data provider as the primary price reference.
    Falls back to APXMIDP if N2EX is unavailable.

    Elexon prices are natively in GBP/MWh. They are converted to EUR/MWh
    using the gbp_to_eur rate (default: 1.18, approximate long-term average).

    Args:
        start: Start datetime.
        end: End datetime.
        gbp_to_eur: GBP to EUR conversion rate.

    Returns:
        pd.Series indexed by hourly UTC timestamps, values in EUR/MWh.
    """
    logger.info("  Fetching GB day-ahead prices from Elexon BMRS")
    records = []

    for chunk_start, chunk_end in _date_chunks(start, end):
        data = _fetch_json("/balancing/pricing/market-index", {
            "from": chunk_start.strftime("%Y-%m-%d"),
            "to": chunk_end.strftime("%Y-%m-%d"),
        })
        if data is None or "data" not in data:
            continue

        for rec in data["data"]:
            records.append({
                "timestamp": _settlement_period_to_time(
                    rec["settlementDate"], rec["settlementPeriod"]
                ),
                "provider": rec.get("dataProvider", ""),
                "price": rec.get("price", np.nan),
            })

    if not records:
        logger.warning("  No price data returned from Elexon")
        return pd.Series(dtype=float)

    df = pd.DataFrame(records)

    # Prefer N2EX prices, fall back to APX
    n2ex = df[df["provider"] == "N2EXMIDP"]
    if len(n2ex) > len(df) * 0.3:
        df = n2ex
    else:
        apx = df[df["provider"] == "APXMIDP"]
        if not apx.empty:
            df = apx

    prices = _to_hourly_utc(df, value_col="price")
    return prices * gbp_to_eur
