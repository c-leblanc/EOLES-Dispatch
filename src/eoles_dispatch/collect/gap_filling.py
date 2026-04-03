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
    interpolate_gaps(series, report, variable="", area="", max_interpol=2, max_weeklyAnalog=72)
        Fill NaN gaps using cascading strategies:
        1. Linear interpolation for gaps <= max_interpol hours (if not at start/end).
        2. Same weekday ±1-2 weeks for gaps <= max_weeklyAnalog hours.
        3. No gap filling for gaps > max_weeklyAnalog hours (alert to user).
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

import pandas as pd

logger = logging.getLogger(__name__)


# ── Gap-filling algorithm ──


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


def interpolate_gaps(series, report, variable="", area="", max_interpol=2, max_weeklyAnalog=120):
    """Fill NaN gaps using a cascade of temporal analogues.

    Strategy by gap size:
      - ≤ max_interpol & not at the beginning or end of the series: linear interpolation (signal barely changes)
      - max_interpol-max_weeklyAnalog: same weekday ±1-2 weeks (preserves daily + weekly cycle)
      - > max_weeklyAnalaog: no gap filling and alert to user.

    If a GapFillReport is active, each operation is recorded in it.

    Args:
        series: Time series with potential NaN gaps.
        report: GapFillReport object to which gap filling report is appended.
        variable: Name of the variable (for reporting, e.g. "demand").
        area: Area code (for reporting, e.g. "FR").
        max_interpol: Maximum gap size (hours) for linear interpolation.
        max_weeklyAnalog: Maximum gap size (hours) for using same weekday ±1-2 weeks
    """
        
    total_filled = 0
    total_not_filled = 0
    
    if series.isna().sum() == 0:
        return series, total_filled, total_not_filled

    result = series.copy()
    gaps = _find_gaps(result)
    hours_per_week = 7 * 24

    for gap_start, gap_length in gaps:
        gap_time = series.index[gap_start]
        gap_end = gap_start + gap_length

        filled = False
        method = ""

        # Strategy 1: linear interpolation for small gaps
        if gap_length <= max_interpol and gap_start>0 and gap_end <len(series):
            lo = max(0, gap_start - 1)
            hi = min(len(result), gap_start + gap_length + 1)
            chunk = result.iloc[lo:hi].copy()
            chunk = chunk.interpolate(method="linear")
            offset_in_chunk = gap_start - lo
            result.iloc[gap_start : gap_start + gap_length] = chunk.iloc[
                offset_in_chunk : offset_in_chunk + gap_length
            ].values
            filled = True
            method =  "linear_interpolation"
            total_filled += gap_length

        # Strategy 2: same weekday ±1-2 weeks
        elif gap_length <= max_weeklyAnalog:
            for week_offset in [1, -1, 2, -2]:
                offset = week_offset * hours_per_week
                fill = _fill_from_analogue(result, gap_start, gap_length, offset)
                if fill is not None:
                    result.iloc[gap_start : gap_start + gap_length] = fill.values
                    filled = True
                    direction = "next" if week_offset > 0 else "previous"
                    method = f"weekly_analogue_{direction}_±{abs(week_offset)}"
                    total_filled += gap_length
                    break
            # Alert - Could not fill gap
            if not filled:
                method = "NOT FILLED: NO APPROPRIATE METHOD"
                total_not_filled += gap_length
                logger.warning(
                f"!!  Gap at {gap_time} ({gap_length}h): "
                f"no analogue found, and linear interpolation unsuitable"
                )
        # Alert - Could not fill gap
        else:
            method = "NOT FILLED: GAP TOO LONG"
            total_not_filled += gap_length
            logger.warning(
            f"!!  Gap too long for gap filling ({gap_length}h) at {gap_time}."
            )

        report.add(variable, area, gap_time, gap_length, method)

    return result, total_filled, total_not_filled


# ── Gap-fill report class ──

class Report:
    """Accumulates gap-filling operations and writes CSV + text report to output_dir.

    Each entry is written to CSV immediately by add() for resumable collection.
    The TXT summary is generated by save().
    """

    def __init__(self, output_dir):
        self.entries = []  # list of dicts
        self.output_dir = Path(output_dir)

    @classmethod
    def load(cls, csv_path):
        """Load a previously saved CSV and resume writing to the same directory."""
        csv_path = Path(csv_path)
        output_dir = csv_path.parent
        report = cls(output_dir)
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            report.entries = df.to_dict("records")
        return report

    def add(self, variable, area, gap_start, gap_hours, method, scaling_ratio=None):
        entry = {
            "recorded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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
        self.entries.append(entry)

        # Write to CSV immediately
        csv_path = self.output_dir / "_gap_fill_report.csv"
        df = pd.DataFrame([entry])
        df.to_csv(
            csv_path,
            mode="a",
            header=not csv_path.exists(),
            index=False,
        )

    def save(self):
        """Write the TXT summary (CSV already written incrementally by add())."""
        csv_path = self.output_dir / "_gap_fill_report.csv"

        # Handle empty entries
        if not csv_path.exists():
            txt_path = self.output_dir / "_gap_fill_report.txt"
            txt_path.write_text(
                "Gap-fill report\n"
                "===============\n\n"
                "No missing values were detected. No gap-filling was needed.\n"
            )
            logger.info("  → _gap_fill_report.txt (no gaps)")
            return

        # Load CSV to get the full list of gap filllings
        df = pd.read_csv(csv_path)

        # TXT — human-readable summary
        txt_path = self.output_dir / "_gap_fill_report.txt"
        lines = [
            "Gap-fill report",
            "===============",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]

        # Summary stats
        filled_mask = ~df["method"].str.startswith("NOT FILLED:")
        gaps_filled = filled_mask.sum()
        gaps_not_filled = (~filled_mask).sum()
        hours_filled = df.loc[filled_mask, "gap_hours"].sum()
        hours_not_filled = df.loc[~filled_mask, "gap_hours"].sum()

        lines.append(f"Gaps filled: {gaps_filled} ({int(hours_filled)}h)")
        lines.append(f"Gaps not filled: {gaps_not_filled} ({int(hours_not_filled)}h)")
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

        # Alert 1: Unfilled gaps (critical)
        unfilled = df[~filled_mask]
        if not unfilled.empty:
            lines.append("❌ UNFILLED GAPS — CRITICAL:")
            for _, row in unfilled.iterrows():
                lines.append(
                    f"  {row['variable']:20s} {row['area']:5s}: "
                    f"{row['gap_start']} → {row['gap_end']} "
                    f"({int(row['gap_hours'])}h, {row['method']})"
                )
            lines.append("")

        # Alert 2: Large gaps (>24h) as warnings
        large = df[(df["gap_hours"] > 24) & filled_mask]
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
            f"  → _gap_fill_report.csv ({gaps_filled+gaps_not_filled} entries), "
            f"_gap_fill_report.txt ({int(hours_filled)}h filled, {int(hours_not_filled)}h not filled)"
        )
