"""Collect input data from ENTSO-E, Elexon BMRS, and Renewables.ninja.

Downloads hourly time series for electricity demand, generation by fuel type,
day-ahead prices, and renewable capacity factor profiles. Data is cleaned,
gap-filled, and saved as CSV files ready for the EOLES-Dispatch model.

Sources:
    - ENTSO-E Transparency Platform (requires ENTSOE_API_KEY env variable)
    - Elexon BMRS Insights API (automatic fallback for GB post-Brexit, no key)
    - Renewables.ninja public country downloads (no key)

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

from . import elexon
from .config import DEFAULT_AREAS, DEFAULT_EXO_AREAS

logger = logging.getLogger(__name__)


def _load_dotenv():
    """Load environment variables from .env file if python-dotenv is available.

    Searches for .env in the current directory and up to 3 parent directories
    (covers running from src/, project root, or a subdirectory).
    """
    try:
        from dotenv import load_dotenv
        # find_dotenv walks up the directory tree
        from dotenv import find_dotenv
        env_path = find_dotenv(usecwd=True)
        if env_path:
            load_dotenv(env_path)
            logger.debug(f"Loaded environment from {env_path}")
    except ImportError:
        # python-dotenv not installed — rely on shell environment
        pass

# Minimum valid-data ratio to accept an ENTSO-E series before falling back to
# an alternative source. Below this threshold, the series is considered too
# sparse and the Elexon fallback is triggered for GB.
_ENTSOE_MIN_COVERAGE = 0.5


# ── Gap-fill report ──

class GapFillReport:
    """Accumulates gap-filling operations and writes a summary CSV + text report."""

    def __init__(self):
        self.entries = []  # list of dicts

    def add(self, variable, area, gap_start, gap_hours, method, scaling_ratio=None):
        self.entries.append({
            "variable": variable,
            "area": area,
            "gap_start": str(gap_start),
            "gap_end": str(gap_start + pd.Timedelta(hours=gap_hours))
                if isinstance(gap_start, pd.Timestamp) else "",
            "gap_hours": gap_hours,
            "method": method,
            "scaling_ratio": round(scaling_ratio, 4) if scaling_ratio is not None else "",
        })

    def save(self, output_dir):
        """Write the report to output_dir/gap_fill_report.csv and .txt."""
        if not self.entries:
            # No gaps at all — still write a minimal report
            report_path = Path(output_dir) / "gap_fill_report.txt"
            report_path.write_text(
                "Gap-fill report\n"
                "===============\n\n"
                "No missing values were detected. No gap-filling was needed.\n"
            )
            logger.info(f"  → gap_fill_report.txt (no gaps)")
            return

        output_dir = Path(output_dir)
        df = pd.DataFrame(self.entries)

        # CSV — detailed log of every gap
        csv_path = output_dir / "gap_fill_report.csv"
        df.to_csv(csv_path, index=False)

        # TXT — human-readable summary
        txt_path = output_dir / "gap_fill_report.txt"
        lines = [
            "Gap-fill report",
            "===============",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]

        # Summary stats
        total_gaps = len(df)
        total_hours = df["gap_hours"].sum()
        lines.append(f"Total gaps filled: {total_gaps}")
        lines.append(f"Total hours filled: {int(total_hours)}")
        lines.append("")

        # Breakdown by method
        lines.append("By method:")
        for method, group in df.groupby("method"):
            lines.append(f"  {method}: {len(group)} gaps, {int(group['gap_hours'].sum())}h")
        lines.append("")

        # Breakdown by variable × area
        lines.append("By variable / area:")
        for (var, area), group in df.groupby(["variable", "area"]):
            n = len(group)
            h = int(group["gap_hours"].sum())
            max_gap = int(group["gap_hours"].max())
            lines.append(f"  {var:20s} {area:5s}: {n:3d} gaps, {h:6d}h total, max {max_gap}h")
        lines.append("")

        # Flag large gaps (>24h) as warnings
        large = df[df["gap_hours"] > 24]
        if not large.empty:
            lines.append("⚠ Large gaps (>24h) — review recommended:")
            for _, row in large.iterrows():
                lines.append(
                    f"  {row['variable']:20s} {row['area']:5s}: "
                    f"{row['gap_start']} → {row['gap_end']} "
                    f"({int(row['gap_hours'])}h, {row['method']})"
                )
            lines.append("")

        txt_path.write_text("\n".join(lines))
        logger.info(
            f"  → gap_fill_report.csv ({total_gaps} entries), "
            f"gap_fill_report.txt ({int(total_hours)}h filled)"
        )


# Module-level report instance, reset at the start of each collect_all() call.
_gap_report = None  # type: GapFillReport | None


def _get_report():
    """Return the current gap-fill report, or None if not in a collection context."""
    return _gap_report


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


def _validate_entsoe_key():
    """Check that the ENTSO-E API key is set and looks valid.

    Performs a lightweight test query (FR load for 1 hour) to catch invalid keys
    early, before starting a long collection run.

    Raises EnvironmentError if the key is missing, or RuntimeError if the test
    query fails (wrong key, network issue, etc.).
    """
    _load_dotenv()

    api_key = os.environ.get("ENTSOE_API_KEY")
    if not api_key:
        # Check if the user has a .env file but python-dotenv is not installed
        dotenv_hint = ""
        env_candidates = [Path.cwd() / ".env"]
        # Also check a few parent dirs
        for parent in list(Path.cwd().parents)[:3]:
            env_candidates.append(parent / ".env")
        has_env_file = any(p.exists() for p in env_candidates)
        try:
            import dotenv  # noqa: F401
            dotenv_installed = True
        except ImportError:
            dotenv_installed = False

        if has_env_file and not dotenv_installed:
            dotenv_hint = (
                "\n\n  A .env file was found but python-dotenv is not installed.\n"
                "  Install it with:  pip install eoles-dispatch[collect]"
            )

        raise EnvironmentError(
            "ENTSOE_API_KEY environment variable not set.\n"
            "  1. Register at https://transparency.entsoe.eu/\n"
            "  2. Copy your API key from My Account > Web API Security Token\n"
            "  3. Set it via:\n"
            "       export ENTSOE_API_KEY=your-key-here\n"
            "     or add it to a .env file (see .env.example)"
            + dotenv_hint
        )

    if len(api_key.strip()) < 10:
        raise EnvironmentError(
            f"ENTSOE_API_KEY looks too short ({len(api_key.strip())} chars). "
            "Check your .env file or environment variable."
        )

    # Quick smoke test: query 1 hour of FR load
    from entsoe import EntsoePandasClient
    client = EntsoePandasClient(api_key=api_key)
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


def _get_client():
    """Create an EntsoePandasClient from the ENTSOE_API_KEY env variable."""
    _load_dotenv()

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
    """Resample a time series to hourly frequency (mean) and normalize to UTC tz-naive.

    ENTSO-E returns data in CET/CEST (Europe/Brussels). Elexon returns UTC.
    Sub-hourly data (15min, 30min) is aggregated to hourly means.
    The output is always UTC tz-naive for consistency across all sources.
    """
    # Normalize timezone: convert to UTC then drop tz info
    if series.index.tz is not None:
        series = series.tz_convert("UTC")
        series.index = series.index.tz_localize(None)

    # Resample to hourly if sub-hourly
    # Check median diff instead of relying on .freq (which is often None)
    if len(series) > 1:
        median_diff = series.index.to_series().diff().median()
        if median_diff < pd.Timedelta("1h"):
            series = series.resample("h").mean()

    return series


def _find_gaps(series):
    """Identify contiguous NaN gaps in a series.

    Returns a list of (start_idx, length) tuples for each gap.
    """
    is_nan = series.isna()
    gaps = []
    i = 0
    while i < len(is_nan):
        if is_nan.iloc[i]:
            start = i
            while i < len(is_nan) and is_nan.iloc[i]:
                i += 1
            gaps.append((start, i - start))
        else:
            i += 1
    return gaps


def _fill_from_analogue(series, gap_start, gap_length, offset):
    """Try to fill a gap using data from a time offset (e.g. ±1 week, ±1 year).

    Looks at the analogue period at `gap_start + offset` for `gap_length` hours.
    If the analogue period has enough valid data (>80%), uses it scaled to match
    the level of observed data around the gap.

    Returns the filled values as a Series, or None if the analogue is unsuitable.
    """
    idx = series.index
    gap_end = gap_start + gap_length

    # Analogue period indices
    analogue_start = gap_start + offset
    analogue_end = gap_end + offset

    # Check bounds
    if analogue_start < 0 or analogue_end > len(series):
        return None

    analogue = series.iloc[analogue_start:analogue_end]
    valid_ratio = analogue.notna().mean()
    if valid_ratio < 0.8:
        return None

    # Compute scaling ratio from context around the gap (±24h)
    ctx_start = max(0, gap_start - 24)
    ctx_end = min(len(series), gap_end + 24)
    ctx_observed = series.iloc[ctx_start:gap_start].dropna()
    ctx_after = series.iloc[gap_end:ctx_end].dropna()
    ctx_all = pd.concat([ctx_observed, ctx_after])

    ana_ctx_start = max(0, analogue_start - 24)
    ana_ctx_end = min(len(series), analogue_end + 24)
    ana_ctx = series.iloc[ana_ctx_start:analogue_start].dropna()
    ana_ctx_after = series.iloc[analogue_end:ana_ctx_end].dropna()
    ana_ctx_all = pd.concat([ana_ctx, ana_ctx_after])

    # Scale if we have enough context on both sides
    if len(ctx_all) > 6 and len(ana_ctx_all) > 6:
        ctx_mean = ctx_all.mean()
        ana_ctx_mean = ana_ctx_all.mean()
        if ana_ctx_mean > 0:
            ratio = ctx_mean / ana_ctx_mean
        else:
            ratio = 1.0
    else:
        ratio = 1.0

    filled = analogue.values * ratio
    # Interpolate any remaining NaNs within the analogue itself
    filled_series = pd.Series(filled, index=idx[gap_start:gap_end])
    filled_series = filled_series.interpolate(method="linear")
    return filled_series


def _interpolate_gaps(series, max_gap=3, variable="", area=""):
    """Fill NaN gaps using a cascade of temporal analogues.

    Strategy by gap size:
      - ≤ max_gap: linear interpolation (signal barely changes)
      - max_gap-48h: same weekday ±1 week (preserves daily + weekly cycle)
      - 48h-7d: same week ±1 year (preserves seasonality)
      - > 7d:   same period from other available years, with scaling

    All filled gaps are logged with their size and the method used.
    If a GapFillReport is active, each operation is recorded in it.

    Args:
        series: Time series with potential NaN gaps.
        max_gap: Maximum gap size (hours) for linear interpolation.
        variable: Name of the variable (for reporting, e.g. "demand").
        area: Area code (for reporting, e.g. "FR").
    """
    if series.isna().sum() == 0:
        return series

    report = _get_report()
    result = series.copy()
    gaps = _find_gaps(result)
    hours_per_week = 7 * 24
    hours_per_year = 365 * 24

    for gap_start, gap_length in gaps:
        gap_hours = gap_length
        gap_time = series.index[gap_start] if gap_start < len(series.index) else "?"

        # Strategy 1: linear interpolation for small gaps
        if gap_hours <= max_gap:
            lo = max(0, gap_start - 1)
            hi = min(len(result), gap_start + gap_length + 1)
            chunk = result.iloc[lo:hi].copy()
            chunk = chunk.interpolate(method="linear")
            offset_in_chunk = gap_start - lo
            result.iloc[gap_start:gap_start + gap_length] = (
                chunk.iloc[offset_in_chunk:offset_in_chunk + gap_length].values
            )
            logger.debug(f"  Gap at {gap_time} ({gap_hours}h): linear interpolation")
            if report:
                report.add(variable, area, gap_time, gap_hours, "linear_interpolation")
            continue

        filled = False
        method = ""
        scaling_ratio = None

        # Strategy 2: same weekday ±1 week (for gaps up to 48h)
        if gap_hours <= 48:
            for sign in [1, -1]:
                offset = sign * hours_per_week
                fill = _fill_from_analogue(result, gap_start, gap_length, offset)
                if fill is not None:
                    result.iloc[gap_start:gap_start + gap_length] = fill.values
                    direction = "next" if sign > 0 else "previous"
                    method = f"weekly_analogue_{direction}"
                    logger.info(f"  Gap at {gap_time} ({gap_hours}h): filled from {direction} week")
                    filled = True
                    break

            if not filled:
                for sign in [1, -1]:
                    offset = sign * 2 * hours_per_week
                    fill = _fill_from_analogue(result, gap_start, gap_length, offset)
                    if fill is not None:
                        result.iloc[gap_start:gap_start + gap_length] = fill.values
                        direction = "next" if sign > 0 else "previous"
                        method = f"weekly_analogue_{direction}_±2"
                        logger.info(f"  Gap at {gap_time} ({gap_hours}h): filled from {direction} week (±2)")
                        filled = True
                        break

        # Strategy 3: same week ±1 year (for gaps 48h-7d, or fallback)
        if not filled and gap_hours <= hours_per_week:
            for sign in [-1, 1]:
                offset = sign * hours_per_year
                fill = _fill_from_analogue(result, gap_start, gap_length, offset)
                if fill is not None:
                    result.iloc[gap_start:gap_start + gap_length] = fill.values
                    direction = "next" if sign > 0 else "previous"
                    method = f"yearly_analogue_{direction}"
                    logger.info(f"  Gap at {gap_time} ({gap_hours}h): filled from {direction} year")
                    filled = True
                    break

        # Strategy 4: multi-year average (for gaps > 7d, or fallback)
        if not filled:
            candidates = []
            for year_offset in [-1, 1, -2, 2]:
                offset = year_offset * hours_per_year
                fill = _fill_from_analogue(result, gap_start, gap_length, offset)
                if fill is not None:
                    candidates.append(fill.values)
            if candidates:
                avg = np.mean(candidates, axis=0)
                result.iloc[gap_start:gap_start + gap_length] = avg
                method = f"multi_year_average_{len(candidates)}y"
                logger.info(
                    f"  Gap at {gap_time} ({gap_hours}h): "
                    f"filled from {len(candidates)}-year average"
                )
                filled = True

        # Last resort: linear interpolation (better than zeros)
        if not filled:
            lo = max(0, gap_start - 1)
            hi = min(len(result), gap_start + gap_length + 1)
            chunk = result.iloc[lo:hi].copy()
            interpolated = chunk.interpolate(method="linear")
            offset_in_chunk = gap_start - lo
            result.iloc[gap_start:gap_start + gap_length] = (
                interpolated.iloc[offset_in_chunk:offset_in_chunk + gap_length].values
            )
            method = "linear_interpolation_fallback"
            logger.warning(
                f"  Gap at {gap_time} ({gap_hours}h): "
                f"no analogue found, used linear interpolation as last resort"
            )

        if report:
            report.add(variable, area, gap_time, gap_hours, method, scaling_ratio)

    # Final safety net: no NaN should remain
    remaining_nans = result.isna().sum()
    if remaining_nans > 0:
        logger.warning(
            f"  {remaining_nans} NaN values remain after gap-filling, "
            f"forward-filling then back-filling"
        )
        if report:
            report.add(variable, area, "various", remaining_nans, "ffill_bfill_safety_net")
        result = result.ffill().bfill()

    return result


def _entsoe_is_usable(series, start, end):
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
    return (valid_count / expected_hours) >= _ENTSOE_MIN_COVERAGE


# ── Demand ──

def collect_demand(client, areas, start, end):
    """Collect actual load for each area, in GW.

    For GB/UK, falls back to the Elexon BMRS API when ENTSO-E data is
    unavailable or too sparse (post-Brexit).

    Returns a DataFrame with columns ['hour', area1, area2, ...].
    """
    frames = {}
    for area in areas:
        logger.info(f"Downloading demand for {area}")
        series = None

        # Try ENTSO-E first
        try:
            raw = client.query_load(_area_code(area), start=start, end=end)
            if isinstance(raw, pd.DataFrame):
                raw = raw.iloc[:, 0]
            raw = _to_hourly(raw)
            if _entsoe_is_usable(raw, start, end):
                series = raw
            elif area == "UK":
                logger.info(f"  ENTSO-E data for UK too sparse, falling back to Elexon")
        except Exception as e:
            if area == "UK":
                logger.info(f"  ENTSO-E unavailable for UK ({e}), falling back to Elexon")
            else:
                logger.warning(f"Failed to download demand for {area}: {e}")
                continue

        # Elexon fallback for UK
        if series is None and area == "UK":
            try:
                series = elexon.fetch_demand(start, end)
                if series is not None and len(series) == 0:
                    series = None
            except Exception as e:
                logger.warning(f"Elexon fallback also failed for UK demand: {e}")

        if series is None:
            logger.warning(f"No demand data available for {area}")
            continue

        series = _interpolate_gaps(series, variable="demand", area=area)
        frames[area] = series / 1000  # MW → GW

    df = pd.DataFrame(frames)
    df.index.name = "hour"
    return df.reset_index()


# ── Non-market-dependent production ──

def collect_nmd(client, areas, start, end):
    """Collect NMD production (biomass, geothermal, marine, waste, other) in GW.

    For GB/UK, falls back to Elexon BMRS when ENTSO-E data is unavailable.

    Returns a DataFrame with columns ['hour', area1, area2, ...].
    """
    frames = {}
    for area in areas:
        logger.info(f"Downloading NMD production for {area}")
        area_total = None
        entsoe_ok = False

        try:
            gen = client.query_generation(_area_code(area), start=start, end=end, psr_type=None)
            if isinstance(gen, pd.DataFrame) and not gen.empty:
                nmd_cols = []
                for nmd_type in NMD_TYPES:
                    psr = PSR_TYPES[nmd_type]
                    for col in gen.columns:
                        col_name = col[0] if isinstance(col, tuple) else col
                        if psr in str(col_name) and (not isinstance(col, tuple) or col[1] == "Actual Aggregated"):
                            nmd_cols.append(col)
                if nmd_cols:
                    area_total = gen[nmd_cols].sum(axis=1)
                else:
                    area_total = pd.Series(0, index=gen.index)
                entsoe_ok = _entsoe_is_usable(area_total, start, end)
            else:
                area_total = pd.Series(0, index=pd.date_range(start, end, freq="h")[:-1])
        except Exception as e:
            if area == "UK":
                logger.info(f"  ENTSO-E unavailable for UK NMD ({e}), falling back to Elexon")
            else:
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
                entsoe_ok = True  # accept whatever we got from fallback

        # Elexon fallback for UK
        if not entsoe_ok and area == "UK":
            logger.info(f"  ENTSO-E NMD data for UK too sparse, falling back to Elexon")
            try:
                elexon_nmd = elexon.fetch_nmd(start, end)
                if elexon_nmd is not None and len(elexon_nmd) > 0:
                    area_total = elexon_nmd
            except Exception as e:
                logger.warning(f"Elexon fallback also failed for UK NMD: {e}")

        if area_total is not None:
            area_total = _to_hourly(area_total)
            area_total = _interpolate_gaps(area_total, variable="nmd", area=area)
            frames[area] = area_total / 1000  # MW → GW

    df = pd.DataFrame(frames)
    df.index.name = "hour"
    return df.reset_index()


# ── VRE capacity factors ──

def _collect_cf_from_elexon(start, end, tec):
    """Fetch a VRE capacity factor series for GB from Elexon.

    Maps our technology name to Elexon's fuel type, fetches generation, and
    divides by the observed maximum to estimate a capacity factor.

    Returns a pd.Series (hourly, dimensionless 0–1), or None on failure.
    """
    # Map our technology names to Elexon generation column names
    tec_to_elexon = {"offshore": "offshore", "onshore": "onshore", "pv": "solar", "river": "river"}
    elexon_fuel = tec_to_elexon.get(tec)
    if elexon_fuel is None:
        return None
    try:
        gen = elexon.fetch_generation_for_fuel(start, end, elexon_fuel)
        if gen is None or len(gen) == 0:
            return None
        capa_mw = gen.max()
        if capa_mw <= 0:
            capa_mw = 1
        return (gen / capa_mw).clip(0, 1).fillna(0)
    except Exception as e:
        logger.warning(f"  Elexon CF fallback failed for UK {tec}: {e}")
        return None


def collect_capacity_factors(client, areas, start, end, technologies=None):
    """Collect actual capacity factors for VRE technologies.

    For GB/UK, falls back to Elexon BMRS when ENTSO-E data is unavailable.

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
            gen = None
            capa_mw = None

            try:
                raw = client.query_generation(
                    _area_code(area), start=start, end=end, psr_type=psr
                )
                if isinstance(raw, pd.DataFrame):
                    prod_cols = [c for c in raw.columns
                                 if not isinstance(c, tuple) or c[1] == "Actual Aggregated"]
                    raw = raw[prod_cols[0]] if prod_cols else raw.iloc[:, 0]
                raw = _to_hourly(raw)

                if _entsoe_is_usable(raw, start, end):
                    gen = raw
                    # Get installed capacity
                    try:
                        capa = client.query_installed_generation_capacity(
                            _area_code(area), start=start, end=end, psr_type=psr
                        )
                        if isinstance(capa, pd.DataFrame):
                            capa_mw = capa.iloc[-1].sum()
                        else:
                            capa_mw = float(capa.iloc[-1]) if len(capa) > 0 else gen.max()
                    except Exception:
                        capa_mw = gen.max()
                elif area == "UK":
                    logger.info(f"  ENTSO-E {tec} data for UK too sparse, falling back to Elexon")
            except Exception as e:
                if area == "UK":
                    logger.info(f"  ENTSO-E unavailable for UK {tec} ({e}), falling back to Elexon")
                else:
                    logger.warning(f"Failed to download {tec} for {area}: {e}")
                    continue

            # Elexon fallback for UK
            if gen is None and area == "UK":
                cf = _collect_cf_from_elexon(start, end, tec)
                if cf is not None:
                    cf = _interpolate_gaps(cf, variable=f"cf_{tec}", area=area)
                    frames[area] = cf
                    continue

            if gen is None:
                continue

            gen = _interpolate_gaps(gen, variable=f"cf_{tec}", area=area)

            if capa_mw is None or capa_mw <= 0:
                capa_mw = gen.max() if gen.max() > 0 else 1

            cf = (gen / capa_mw).clip(0, 1)
            cf = cf.fillna(0)
            frames[area] = cf

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
            prices = _interpolate_gaps(prices, max_gap=24, variable="exo_price", area=area)
            frames[area] = prices
        except Exception as e:
            logger.warning(f"Failed to download prices for {area}: {e}")
            continue

    df = pd.DataFrame(frames)
    df.index.name = "hour"
    return df.reset_index()


# ── Lake inflows (monthly budget) ──

def _collect_lake_inflows_elexon(start, end):
    """Fetch lake inflows for GB from Elexon generation-by-type data.

    Uses PHS generation as a proxy (GB has no significant lake hydro separate
    from pumped storage). Returns a monthly Series in TWh, or None on failure.
    """
    eta_phs = 0.9 * 0.95
    try:
        gen = elexon.fetch_generation(start, end)
        if gen.empty:
            return None

        phs = gen["phs"] if "phs" in gen.columns else pd.Series(0, index=gen.index)
        river = gen["river"] if "river" in gen.columns else pd.Series(0, index=gen.index)
        # Approximate: Elexon does not separate PHS production/consumption in the
        # per-type endpoint, so we use gross production as a rough upper bound.
        total = (river + phs).clip(lower=0)
        total = _interpolate_gaps(total, variable="lake_inflows", area="UK")
        monthly = total.resample("MS").sum() / 1e6  # MWh → TWh
        return monthly.clip(lower=0)
    except Exception as e:
        logger.warning(f"  Elexon fallback failed for UK lake inflows: {e}")
        return None


def collect_lake_inflows(client, areas, start, end):
    """Collect monthly lake + PHS net production as a proxy for hydro inflows, in TWh.

    Lake inflows are estimated from net hydro production: lake_prod + phs_prod - η * phs_cons.
    Aggregated to monthly sums and converted to TWh.

    For GB/UK, falls back to Elexon BMRS when ENTSO-E data is unavailable.

    Returns a DataFrame with columns ['month', area1, area2, ...].
    """
    eta_phs = 0.9 * 0.95  # round-trip efficiency for PHS

    frames = {}
    for area in areas:
        logger.info(f"Downloading lake inflows for {area}")
        entsoe_ok = False

        try:
            gen = client.query_generation(_area_code(area), start=start, end=end, psr_type=None)
            if isinstance(gen, pd.DataFrame) and not gen.empty:
                lake_prod = pd.Series(0, index=gen.index)
                phs_net = pd.Series(0, index=gen.index)

                for col in gen.columns:
                    col_name = col[0] if isinstance(col, tuple) else col
                    if "B12" in str(col_name):
                        if isinstance(col, tuple) and col[1] == "Actual Aggregated":
                            lake_prod = lake_prod.add(gen[col], fill_value=0)
                        elif not isinstance(col, tuple):
                            lake_prod = lake_prod.add(gen[col], fill_value=0)

                for col in gen.columns:
                    col_name = col[0] if isinstance(col, tuple) else col
                    if "B10" in str(col_name):
                        if isinstance(col, tuple) and col[1] == "Actual Aggregated":
                            phs_net = phs_net.add(gen[col], fill_value=0)
                        elif isinstance(col, tuple) and col[1] == "Actual Consumption":
                            phs_net = phs_net.subtract(gen[col].abs() * eta_phs, fill_value=0)

                total = _to_hourly(lake_prod + phs_net)
                if _entsoe_is_usable(total, start, end):
                    total = _interpolate_gaps(total, variable="lake_inflows", area=area)
                    monthly = total.resample("MS").sum() / 1e6
                    monthly = monthly.clip(lower=0)
                    frames[area] = monthly
                    entsoe_ok = True
        except Exception as e:
            if area != "UK":
                logger.warning(f"Failed to download lake inflows for {area}: {e}")
                continue

        # Elexon fallback for UK
        if not entsoe_ok and area == "UK":
            logger.info(f"  Falling back to Elexon for UK lake inflows")
            monthly = _collect_lake_inflows_elexon(start, end)
            if monthly is not None and len(monthly) > 0:
                frames[area] = monthly

    df = pd.DataFrame(frames)
    df.index.name = "month_dt"
    df = df.reset_index()
    df["month"] = df["month_dt"].dt.strftime("%Y%m")
    df = df.drop(columns=["month_dt"])
    cols = ["month"] + [c for c in df.columns if c != "month"]
    return df[cols]


# ── Hydro max in/out (monthly) ──

def collect_hydro_limits(client, areas, start, end):
    """Collect monthly max hydro charge/discharge power.

    For GB/UK, falls back to Elexon BMRS when ENTSO-E data is unavailable.

    Returns (hMaxIn, hMaxOut) DataFrames with columns ['month', area1, area2, ...].
    Values in GW.
    """
    frames_in = {}
    frames_out = {}
    for area in areas:
        logger.info(f"Downloading hydro limits for {area}")
        entsoe_ok = False

        try:
            gen = client.query_generation(_area_code(area), start=start, end=end, psr_type=None)
            if isinstance(gen, pd.DataFrame) and not gen.empty:
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

                if _entsoe_is_usable(total_out, start, end):
                    monthly_out = total_out.resample("MS").max() / 1000
                    monthly_in = total_in.resample("MS").max() / 1000
                    frames_out[area] = monthly_out
                    frames_in[area] = monthly_in
                    entsoe_ok = True
        except Exception as e:
            if area != "UK":
                logger.warning(f"Failed to download hydro limits for {area}: {e}")
                continue

        # Elexon fallback for UK
        if not entsoe_ok and area == "UK":
            logger.info(f"  Falling back to Elexon for UK hydro limits")
            try:
                gen_df = elexon.fetch_generation(start, end)
                if not gen_df.empty:
                    phs = gen_df["phs"] if "phs" in gen_df.columns else pd.Series(0, index=gen_df.index)
                    river = gen_df["river"] if "river" in gen_df.columns else pd.Series(0, index=gen_df.index)
                    total_out = (river + phs).clip(lower=0)
                    # Elexon does not separate PHS consumption; use PHS production
                    # as an approximation for max charge power.
                    total_in = phs.clip(lower=0)
                    frames_out[area] = total_out.resample("MS").max() / 1000
                    frames_in[area] = total_in.resample("MS").max() / 1000
            except Exception as e:
                logger.warning(f"Elexon fallback also failed for UK hydro limits: {e}")

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

    For GB/UK, falls back to Elexon BMRS when ENTSO-E data is unavailable.

    Returns a DataFrame with columns ['week', area1, area2, ...].
    Values in [0, 1].
    """
    frames = {}
    for area in areas:
        logger.info(f"Downloading nuclear availability for {area}")
        gen = None
        capa_mw = None

        try:
            raw = client.query_generation(
                _area_code(area), start=start, end=end, psr_type=PSR_TYPES["nuclear"]
            )
            if isinstance(raw, pd.DataFrame):
                prod_cols = [c for c in raw.columns
                             if not isinstance(c, tuple) or c[1] == "Actual Aggregated"]
                raw = raw[prod_cols[0]] if prod_cols else raw.iloc[:, 0]
            raw = _to_hourly(raw)

            if _entsoe_is_usable(raw, start, end):
                gen = raw
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
            elif area == "UK":
                logger.info(f"  ENTSO-E nuclear data for UK too sparse, falling back to Elexon")
        except Exception as e:
            if area == "UK":
                logger.info(f"  ENTSO-E unavailable for UK nuclear ({e}), falling back to Elexon")
            else:
                logger.warning(f"Failed to download nuclear data for {area}: {e}")
                continue

        # Elexon fallback for UK
        if gen is None and area == "UK":
            try:
                gen = elexon.fetch_generation_for_fuel(start, end, "nuclear")
                if gen is not None and len(gen) > 0:
                    capa_mw = gen.max()
                else:
                    gen = None
            except Exception as e:
                logger.warning(f"Elexon fallback also failed for UK nuclear: {e}")

        if gen is None:
            logger.warning(f"No nuclear data available for {area}")
            continue

        gen = _interpolate_gaps(gen, variable="nuclear", area=area)

        if capa_mw is None or capa_mw <= 0:
            capa_mw = gen.max() if gen.max() > 0 else 1

        af = (gen / capa_mw).clip(0, 1).fillna(0)

        # Weekly max
        weekly = af.resample("W-MON").max()
        frames[area] = weekly

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
# Renewables.ninja provides two wind fleet variants:
#   - current: technology installed as of ~2020 (current hub heights, rotor diameters)
#   - future:  projected next-generation turbines (taller towers, larger rotors)
NINJA_FILES = {
    "pv": "ninja-pv-country-{iso2}-national-merra2.csv",
    "onshore_current": "ninja-wind-country-{iso2}-current_onshore-merra2.csv",
    "onshore_future": "ninja-wind-country-{iso2}-future_onshore-merra2.csv",
    "offshore_current": "ninja-wind-country-{iso2}-current_offshore-merra2.csv",
    "offshore_future": "ninja-wind-country-{iso2}-future_offshore-merra2.csv",
}

# Countries that have no offshore data (landlocked)
NO_OFFSHORE = {"CH", "LU"}


def _download_ninja_csv(iso2, filename):
    """Download a single CSV from Renewables.ninja and return the NATIONAL column as a Series."""
    import urllib.request

    url = f"{NINJA_BASE_URL.format(iso2=iso2)}/{filename.format(iso2=iso2)}"
    logger.info(f"  Downloading {url}")
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "EOLES-Dispatch/0.1 (energy model; +https://github.com)",
        })
        with urllib.request.urlopen(req, timeout=60) as resp:
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
        # Normalize to UTC tz-naive
        hour_col = pd.to_datetime(df["hour"])
        if hour_col.dt.tz is not None:
            hour_col = hour_col.dt.tz_convert("UTC").dt.tz_localize(None)
        df["hour"] = hour_col

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
    global _gap_report

    if areas is None:
        areas = list(DEFAULT_AREAS)
    if exo_areas is None:
        exo_areas = list(DEFAULT_EXO_AREAS)

    output_dir = Path(output_dir)

    # Initialize gap-fill report for this collection run
    _gap_report = GapFillReport()

    if source in ("all", "entsoe"):
        tv_dir = output_dir / "time_varying_inputs"
        tv_dir.mkdir(parents=True, exist_ok=True)

        start = pd.Timestamp(f"{start_year}-01-01", tz="Europe/Brussels")
        end = pd.Timestamp(f"{end_year}-01-01", tz="Europe/Brussels")

        # Validate API key upfront (fail fast before starting a long run)
        logger.info("=== Validating ENTSO-E API key ===")
        client = _validate_entsoe_key()

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

    # Save gap-fill report
    logger.info("=== Writing gap-fill report ===")
    _gap_report.save(output_dir)

    logger.info("=== Collection complete ===")
    return output_dir
