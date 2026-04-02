"""Timezone helpers and time-series utilities for EOLES-Dispatch.

All data in the pipeline is stored as naive UTC timestamps. The European
electricity calendar follows CET/CEST (Europe/Brussels), so year, month,
and week boundaries are computed in CET then converted to UTC here.

This module is the single source of truth for timezone conventions and
time-series normalization. It is used by every data-collection module
and by format_inputs.py at run creation time.

Convention:
    - CET = Central European Time (UTC+1 winter, UTC+2 summer / CEST).
    - We use "Europe/Brussels" as the IANA timezone.
    - All returned datetimes are naive (no tzinfo), representing UTC.

Used by:
    - datacoll/entsoe.py        (resample_to_hourly)
    - datacoll/elexon.py        (resample_to_hourly)
    - datacoll/main_collect.py  (cet_year_bounds, expected_hours, canonical_index)
    - format_inputs.py          (to_posix_hours, compute_hour_mappings)
    - run.py                    (compute_hour_mappings)

Functions:
    cet_to_utc(dt_naive)                    - Convert naive CET datetime to naive UTC.
    resample_to_hourly(series)              - Resample to hourly naive UTC (mean).
                                              Called from entsoe, elexon.
    canonical_index(year)                   - Hourly DatetimeIndex for a CET year.
                                              Called from _main_collect.
    cet_year_bounds(year)                   - UTC bounds of a CET calendar year.
                                              Called from _main_collect, format_inputs.
    cet_week_bounds(year, week)             - UTC bounds of an ISO week in CET.
    expected_hours(year)                    - Number of hours in a CET year.
                                              Called from _main_collect (collect_production, _validate_year).
    hour_to_cet_month(utc_posix_hours)      - Map POSIX hours to CET month strings.
                                              Called from format_inputs.
    hour_to_cet_week(utc_posix_hours)       - Map POSIX hours to CET week strings.
                                              Called from format_inputs.
    cet_period_bounds(year, months)         - UTC bounds of a year or sub-period in CET.
                                              Called from run._main_run, compute_hour_mappings.
    posix_hours_to_dt(hours_series)         - POSIX hours (int) → UTC-aware Timestamps.
                                              Called from viz/loaders, viz/charts_outputs.
    compute_hour_mappings(simul_year, ...)   - Compute hour-month and hour-week DataFrames.
                                              Called from run.create_run, format_inputs.
"""

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

# CET/CEST timezone used by ENTSO-E and the European electricity calendar.
CET = ZoneInfo("Europe/Brussels")
_UTC = ZoneInfo("UTC")


def cet_to_utc(dt_naive):
    """Convert a naive datetime (interpreted as CET/CEST) to a naive UTC datetime.

    Examples:
        2021-01-01 00:00 CET  → 2020-12-31 23:00 UTC
        2021-07-01 00:00 CEST → 2021-06-30 22:00 UTC
    """
    cet_aware = dt_naive.replace(tzinfo=CET)
    utc_aware = cet_aware.astimezone(tz=_UTC)
    return utc_aware.replace(tzinfo=None)


def resample_to_hourly(series):
    """Resample a time series to hourly frequency (mean) and normalize to UTC tz-naive.

    ENTSO-E returns data in CET/CEST (Europe/Brussels). Elexon returns UTC.
    Sub-hourly data (15min, 30min) is aggregated to hourly means.
    The output is always UTC tz-naive for consistency across all sources.

    Does NOT clip to any range. Use .reindex(canonical_index(year)) after this
    to align to the expected hourly grid.
    """
    # Normalize timezone: convert to UTC then drop tz info
    if series.index.tz is not None:
        series = series.tz_convert("UTC")
        series.index = series.index.tz_localize(None)

    # Resample to hourly if sub-hourly.
    # Detect sub-hourly by checking for timestamps with minutes != 0.
    # The only possible steps are 1h, 30min, 15min.
    if len(series) > 1 and (series.index.minute != 0).any():
        series = series.resample("h").mean()

    return series


# CET bounds


def cet_year_bounds(year):
    """Return (utc_start, utc_end) for a CET calendar year.

    A CET year starts at Jan 1 00:00 CET and ends at Jan 1 00:00 CET of the
    next year. Both bounds are returned as naive UTC datetimes.

    Example:
        cet_year_bounds(2021) → (2020-12-31 23:00, 2021-12-31 23:00)
    """
    start = cet_to_utc(datetime(year, 1, 1))
    end = cet_to_utc(datetime(year + 1, 1, 1))
    return start, end


def cet_week_bounds(year, week):
    """Return (utc_start, utc_end) for an ISO week in CET.

    ISO week starts on Monday 00:00 CET and ends on next Monday 00:00 CET.

    Args:
        year: ISO year.
        week: ISO week number (1-53).
    """
    # Monday of the given ISO week
    monday = datetime.strptime(f"{year}-W{week:02d}-1", "%G-W%V-%u")
    next_monday = monday + pd.Timedelta(days=7)
    start = cet_to_utc(monday)
    end = cet_to_utc(next_monday)
    return start, end


# Conversions


def to_posix_hours(dt_series):
    """Convert a datetime Series to POSIX hours (int, hours since 1970-01-01 UTC)."""
    return ((dt_series - datetime(1970, 1, 1)).dt.total_seconds() / 3600).astype(int)


def posix_hours_to_dt(hours_series):
    """Convert POSIX hours (int, hours since 1970-01-01 UTC) to UTC-aware Timestamps."""
    return pd.to_datetime(hours_series * 3600, unit="s", origin="unix", utc=True)


# Mappings


def expected_hours(year):
    """Number of hours in a CET calendar year.

    This accounts for leap years (8784h) vs normal years (8760h).
    DST transitions cancel out over a full year (spring loses 1h, autumn gains 1h).
    """
    start, end = cet_year_bounds(year)
    return int((end - start).total_seconds() / 3600)


def canonical_index(year):
    """Return a DatetimeIndex of naive UTC hourly timestamps for a CET calendar year.

    The index starts at cet_year_bounds(year)[0] and has exactly
    expected_hours(year) points, spaced 1 hour apart. Half-open: [start, end).

    This is the authoritative hourly grid for a year. Use series.reindex(canonical_index(year))
    to align data: out-of-range timestamps are dropped and missing hours become NaN.
    """
    start, end = cet_year_bounds(year)
    idx = pd.date_range(start, end, freq="h", inclusive="left")
    assert len(idx) == expected_hours(year)
    return idx


def hour_to_cet_month(utc_posix_hours):
    """Map UTC POSIX hours (int) to CET month strings (YYYYMM).

    Args:
        utc_posix_hours: pd.Series of int POSIX hours (hours since 1970-01-01 UTC).

    Returns:
        pd.Series of strings like "202101", "202102", etc.
    """
    ts = pd.to_datetime(utc_posix_hours * 3600, unit="s", origin="1970-01-01", utc=True)
    return ts.dt.tz_convert(CET).dt.strftime("%Y%m")


def hour_to_cet_week(utc_posix_hours):
    """Map UTC POSIX hours (int) to CET week strings (YYYYWW).

    Args:
        utc_posix_hours: pd.Series of int POSIX hours (hours since 1970-01-01 UTC).

    Returns:
        pd.Series of strings like "202101", "202152", etc.
    """
    ts = pd.to_datetime(utc_posix_hours * 3600, unit="s", origin="1970-01-01", utc=True)
    return ts.dt.tz_convert(CET).dt.strftime("%Y%W")


def cet_period_bounds(year, months=None):
    """Return (utc_start, utc_end) for a year or sub-period in CET.

    Args:
        year: Calendar year.
        months: Optional (start_month, end_month) tuple, e.g. (1, 3) for Jan-Mar.
                None = full year (delegates to cet_year_bounds).
    """
    if months is None:
        return cet_year_bounds(year)
    start_m, end_m = months
    start = cet_to_utc(datetime(year, start_m, 1))
    end = (
        cet_to_utc(datetime(year, end_m + 1, 1))
        if end_m < 12
        else cet_to_utc(datetime(year + 1, 1, 1))
    )
    return start, end


def compute_hour_mappings(simul_year, months=None):
    """Compute hour_month and hour_week mapping DataFrames for a simulation period.

    Args:
        simul_year: The simulation year.
        months: Optional (start_month, end_month) tuple, e.g. (1, 3) for Jan-Mar.
                None = full year.

    Returns:
        (hour_month, hour_week) where each is a DataFrame with columns
        ['hour', 'month'] or ['hour', 'week']. 'hour' is POSIX hours (int).
    """
    start, end = cet_period_bounds(simul_year, months)

    # Generate hourly UTC timestamps for the period
    hours_index = pd.date_range(start, end, freq="h", inclusive="left")
    posix_hours = ((hours_index - pd.Timestamp("1970-01-01")).total_seconds() / 3600).astype(int)

    hour_month = pd.DataFrame({"hour": posix_hours})
    hour_month["month"] = hour_to_cet_month(hour_month["hour"])

    hour_week = pd.DataFrame({"hour": posix_hours})
    hour_week["week"] = hour_to_cet_week(hour_week["hour"])

    return hour_month, hour_week
