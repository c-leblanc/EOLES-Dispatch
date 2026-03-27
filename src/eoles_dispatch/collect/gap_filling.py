"""Gap-filling algorithms for hourly time series.

Fills NaN gaps in collected data using a cascade of temporal analogues
(nearby weeks, previous years) with scaling, falling back to linear
interpolation. All operations are logged and recorded in a Report
for audit purposes.

This module is source-agnostic: it operates on any pd.Series with a
DatetimeIndex, regardless of whether the data came from ENTSO-E, Elexon,
or another source.

Called from:
    - main_collect.py   interpolate_gaps is called from collect_demand,
                        collect_production, and collect_exo_prices.
                        Report is instantiated in collect_all, passed down
                        to the collect_* functions, then saved to disk.

Functions:
    interpolate_gaps(series, report, max_gap=3, variable="", area="")
        Fill NaN gaps using cascading strategies:
        1. Linear interpolation for gaps <= max_gap hours.
        2. Same weekday +/-1 week for gaps <= 48h.
        3. Same period +/-1 year for gaps <= 7 days.
        4. Multi-year average for larger gaps.
        5. Linear interpolation as last resort.
        Called from main_collect (collect_demand, collect_production,
        collect_exo_prices).

Classes:
    Report
        Accumulates gap-filling operations and writes a summary
        CSV + human-readable text report.
        Instantiated in main_collect.collect_all, passed to collect_*
        functions, saved via Report.save(output_dir).

Internal helpers:
    _find_gaps(series)                              - Identify contiguous NaN runs.
    _fill_from_analogue(series, gap_start, len, offset) - Fill from a time-shifted period.
"""

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


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


def interpolate_gaps(series, report, max_gap=3, variable="", area=""):
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
            result.iloc[gap_start : gap_start + gap_length] = chunk.iloc[
                offset_in_chunk : offset_in_chunk + gap_length
            ].values
            logger.debug(f"  Gap at {gap_time} ({gap_hours}h): linear interpolation")
            if report:
                report.add(variable, area, gap_time, gap_hours, "linear_interpolation")
            continue

        filled = False
        method = ""

        # Strategy 2: same weekday ±1 week (for gaps up to 48h)
        if gap_hours <= 48:
            for sign in [1, -1]:
                offset = sign * hours_per_week
                fill = _fill_from_analogue(result, gap_start, gap_length, offset)
                if fill is not None:
                    result.iloc[gap_start : gap_start + gap_length] = fill.values
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
                        result.iloc[gap_start : gap_start + gap_length] = fill.values
                        direction = "next" if sign > 0 else "previous"
                        method = f"weekly_analogue_{direction}_±2"
                        logger.info(
                            f"  Gap at {gap_time} ({gap_hours}h): filled from {direction} week (±2)"
                        )
                        filled = True
                        break

        # Strategy 3: same week ±1 year (for gaps 48h-7d, or fallback)
        if not filled and gap_hours <= hours_per_week:
            for sign in [-1, 1]:
                offset = sign * hours_per_year
                fill = _fill_from_analogue(result, gap_start, gap_length, offset)
                if fill is not None:
                    result.iloc[gap_start : gap_start + gap_length] = fill.values
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
                result.iloc[gap_start : gap_start + gap_length] = avg
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
            result.iloc[gap_start : gap_start + gap_length] = interpolated.iloc[
                offset_in_chunk : offset_in_chunk + gap_length
            ].values
            method = "linear_interpolation_fallback"
            logger.warning(
                f"  Gap at {gap_time} ({gap_hours}h): "
                f"no analogue found, used linear interpolation as last resort"
            )

        if report:
            report.add(variable, area, gap_time, gap_hours, method)

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


# ── Gap-fill report ──


class Report:
    """Accumulates gap-filling operations and writes a summary CSV + text report."""

    def __init__(self):
        self.entries = []  # list of dicts

    def add(self, variable, area, gap_start, gap_hours, method, scaling_ratio=None):
        self.entries.append(
            {
                "variable": variable,
                "area": area,
                "gap_start": str(gap_start),
                "gap_end": str(gap_start + pd.Timedelta(hours=gap_hours))
                if isinstance(gap_start, pd.Timestamp)
                else "",
                "gap_hours": gap_hours,
                "method": method,
                "scaling_ratio": round(scaling_ratio, 4) if scaling_ratio is not None else "",
            }
        )

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
            logger.info("  → gap_fill_report.txt (no gaps)")
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
