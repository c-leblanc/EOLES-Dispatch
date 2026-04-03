"""Tests for data collection utilities and pipeline functions."""

import numpy as np
import pandas as pd
import pytest

from eoles_dispatch.collect._main_collect import _is_production_usable
from eoles_dispatch.collect.entsoe import ENTSOE_COL_NAMES, PRODUCTION_TYPES, col_matches, is_usable
from eoles_dispatch.collect.gap_filling import Report, interpolate_gaps
from eoles_dispatch.utils import (
    canonical_index,
    cet_year_bounds,
    expected_hours,
    resample_to_hourly,
)

# ── col_matches ──


def test_col_matches_human_readable_string():
    assert col_matches("Fossil Gas", "gas") is True


def test_col_matches_human_readable_tuple():
    assert col_matches(("Fossil Gas", "Actual Aggregated"), "gas") is True


def test_col_matches_wrong_prodtype_returns_false():
    assert col_matches("Fossil Gas", "nuclear") is False


def test_col_matches_hydro_phs():
    """Regression: CH data uses 'Hydro Pumped Storage' column name."""
    assert col_matches("Hydro Pumped Storage", "phs") is True


def test_col_matches_all_production_types_have_names():
    """Every key in PRODUCTION_TYPES should also have an entry in ENTSOE_COL_NAMES."""
    for production_type in PRODUCTION_TYPES:
        assert production_type in ENTSOE_COL_NAMES, (
            f"{production_type} is in PRODUCTION_TYPES but missing from ENTSOE_COL_NAMES"
        )


# ── canonical_index ──


def test_canonical_index_length_normal_year():
    idx = canonical_index(2021)
    assert len(idx) == expected_hours(2021)
    assert len(idx) == 8760


def test_canonical_index_length_leap_year():
    idx = canonical_index(2020)
    assert len(idx) == expected_hours(2020)
    assert len(idx) == 8784


def test_canonical_index_bounds():
    idx = canonical_index(2021)
    start, end = cet_year_bounds(2021)
    assert idx[0] == pd.Timestamp(start)
    assert idx[-1] == pd.Timestamp(end) - pd.Timedelta("1h")


def test_canonical_index_is_hourly():
    idx = canonical_index(2021)
    diffs = pd.Series(idx).diff().dropna()
    assert (diffs == pd.Timedelta("1h")).all()


def test_canonical_index_is_naive_utc():
    idx = canonical_index(2021)
    assert idx.tz is None


# ── resample_to_hourly ──


def _make_series(freq, periods, tz=None):
    """Helper: create a constant-value time series at the given frequency."""
    idx = pd.date_range("2023-01-01", periods=periods, freq=freq, tz=tz)
    return pd.Series(np.ones(periods), index=idx)


def test_resample_to_hourly_15min_data():
    s = _make_series("15min", 96)
    result = resample_to_hourly(s)
    assert len(result) == 24


def test_resample_to_hourly_30min_data():
    s = _make_series("30min", 48)
    result = resample_to_hourly(s)
    assert len(result) == 24


def test_resample_to_hourly_already_hourly():
    s = _make_series("h", 24)
    result = resample_to_hourly(s)
    assert len(result) == 24


def test_resample_to_hourly_strips_timezone():
    s = _make_series("h", 24, tz="CET")
    result = resample_to_hourly(s)
    assert result.index.tz is None
    # CET is UTC+1, so first hour should shift back by 1h
    assert result.index[0] == pd.Timestamp("2022-12-31 23:00:00")


def test_resample_to_hourly_utc_input_unchanged():
    s = _make_series("h", 24)
    result = resample_to_hourly(s)
    assert result.index.tz is None
    assert result.index[0] == pd.Timestamp("2023-01-01 00:00:00")


def test_resample_to_hourly_computes_mean():
    """15-min data with values [1,2,3,4] per hour should give mean=2.5."""
    idx = pd.date_range("2023-01-01", periods=8, freq="15min")
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    s = pd.Series(values, index=idx)
    result = resample_to_hourly(s)
    assert len(result) == 2
    assert result.iloc[0] == pytest.approx(2.5)  # mean(1,2,3,4)
    assert result.iloc[1] == pytest.approx(6.5)  # mean(5,6,7,8)


def test_resample_detects_subhourly_by_minutes():
    """Timestamps with minute != 0 trigger resampling."""
    # 30-min data: has :30 timestamps
    idx = pd.date_range("2023-01-01", periods=4, freq="30min")
    s = pd.Series([10.0, 20.0, 30.0, 40.0], index=idx)
    result = resample_to_hourly(s)
    assert len(result) == 2
    assert result.iloc[0] == pytest.approx(15.0)  # mean(10, 20)


# ── reindex onto canonical index ──


def test_reindex_clips_extra_and_exposes_gaps():
    """reindex onto canonical_index clips out-of-range data and shows gaps as NaN."""
    canon_idx = canonical_index(2021)
    start, end = cet_year_bounds(2021)

    # Create a series with extra timestamps and a gap
    extra_before = pd.date_range(start - pd.Timedelta("5h"), periods=5, freq="h")
    extra_after = pd.date_range(end, periods=3, freq="h")
    # Only first 100 hours of the year
    partial = pd.date_range(start, periods=100, freq="h")
    idx = extra_before.append(partial).append(extra_after)
    s = pd.Series(np.ones(len(idx)), index=idx)

    result = s.reindex(canon_idx)

    # Length matches canonical
    assert len(result) == len(canon_idx)
    # First 100 hours have data
    assert result.iloc[:100].notna().all()
    # Rest is NaN
    assert result.iloc[100:].isna().all()


# ── is_usable ──


def test_is_usable_good_coverage():
    idx = pd.date_range("2023-01-01", periods=22, freq="h")
    s = pd.Series(np.ones(22), index=idx)
    assert is_usable(s, 24)  # 22/24 = 91% > 50%


def test_is_usable_sparse():
    idx = pd.date_range("2023-01-01", periods=7, freq="h")
    s = pd.Series(np.ones(7), index=idx)
    assert not is_usable(s, 24)  # 7/24 = 29% < 50%


def test_is_usable_none():
    assert is_usable(None, 24) is False


def test_is_usable_empty_series():
    s = pd.Series(dtype=float)
    assert is_usable(s, 24) is False


def test_is_usable_all_nan():
    idx = pd.date_range("2023-01-01", periods=24, freq="h")
    s = pd.Series([np.nan] * 24, index=idx)
    assert is_usable(s, 24) is False


def test_is_usable_dataframe():
    df = pd.DataFrame({"a": np.ones(20), "b": np.ones(20)})
    assert is_usable(df, 24)  # len=20 > 24*0.5=12


def test_is_usable_zero_expected():
    s = pd.Series([1.0, 2.0])
    assert is_usable(s, 0) is False


# ── interpolate_gaps ──


def _make_gapped_series(gap_start, gap_length, total=48):
    """Create an hourly series with a NaN gap for interpolation testing."""
    idx = pd.date_range("2023-06-01", periods=total, freq="h")
    values = np.arange(1.0, total + 1.0)
    values[gap_start : gap_start + gap_length] = np.nan
    return pd.Series(values, index=idx)


def test_interpolate_small_gap_linear(tmp_path):
    s = _make_gapped_series(gap_start=10, gap_length=2, total=48)
    report = Report(tmp_path)
    result, filled, not_filled = interpolate_gaps(s, report=report, variable="test", area="XX")
    assert result.iloc[10] == pytest.approx(11.0, abs=0.5)
    assert result.iloc[11] == pytest.approx(12.0, abs=0.5)
    assert filled == 2
    assert not_filled == 0


def test_interpolate_no_gaps_passthrough(tmp_path):
    idx = pd.date_range("2023-06-01", periods=24, freq="h")
    s = pd.Series(np.ones(24), index=idx)
    report = Report(tmp_path)
    result, filled, not_filled = interpolate_gaps(s, report=report, variable="test", area="XX")
    pd.testing.assert_series_equal(result, s)
    assert filled == 0
    assert not_filled == 0


def test_interpolate_preserves_index(tmp_path):
    s = _make_gapped_series(gap_start=5, gap_length=2, total=48)
    report = Report(tmp_path)
    result, _, _ = interpolate_gaps(s, report=report, variable="test", area="XX")
    pd.testing.assert_index_equal(result.index, s.index)


def test_interpolate_fills_all_nans(tmp_path):
    s = _make_gapped_series(gap_start=20, gap_length=2, total=48)
    report = Report(tmp_path)
    result, filled, not_filled = interpolate_gaps(s, report=report, variable="test", area="XX")
    assert result.isna().sum() == 0
    assert filled > 0
    assert not_filled == 0


# ── _is_production_usable ──


def test_production_usable_enough_data():
    """DataFrame with 90% valid rows should be usable (> 50% coverage)."""
    n = 100
    df = pd.DataFrame({"hour": range(n), "gas": np.ones(n), "solar": np.ones(n)})
    assert _is_production_usable(df, n)


def test_production_usable_mostly_nan():
    """DataFrame with 90% NaN rows should not be usable."""
    n = 100
    values = np.full(n, np.nan)
    values[:5] = 1.0  # only 5% valid
    df = pd.DataFrame({"hour": range(n), "gas": values, "solar": values})
    assert not _is_production_usable(df, n)


def test_production_usable_none():
    assert not _is_production_usable(None, 100)


# ── Report.load ──


def test_report_load_existing(tmp_path):
    """Report.load should restore entries from a previously saved CSV."""
    report = Report(tmp_path)
    report.add("demand", "FR", pd.Timestamp("2023-01-01"), 3, "linear_interpolation")
    report.add("production", "DE", pd.Timestamp("2023-06-15"), 12, "weekly_analogue_next")
    report.save()

    loaded = Report.load(tmp_path / "_gap_fill_report.csv")
    assert len(loaded.entries) == 2
    assert loaded.entries[0]["variable"] == "demand"
    assert loaded.entries[1]["area"] == "DE"


def test_report_load_nonexistent(tmp_path):
    """Report.load on a missing file should return an empty report."""
    loaded = Report.load(tmp_path / "does_not_exist.csv")
    assert len(loaded.entries) == 0
