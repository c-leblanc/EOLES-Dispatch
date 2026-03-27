"""Tests for eoles_dispatch.format_inputs."""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eoles_dispatch.utils import cet_to_utc, CET, compute_hour_mappings, to_posix_hours, expected_hours
from eoles_dispatch.run.compute import (
    compute_nmd,
    compute_vre_capacity_factors,
    compute_nuclear_max_af,
    compute_lake_inflows,
    compute_hydro_limits,
)
from eoles_dispatch.run.format_inputs import load_ninja_var
from eoles_dispatch.collect._main_collect import _validate_year
from eoles_dispatch.run._main_run import _copy_actual_prices


# ---------------------------------------------------------------------------
# cet_to_utc (now in timezone.py)
# ---------------------------------------------------------------------------


class TestCetToUtc:
    def test_cet_to_utc_winter(self):
        """CET (UTC+1) in winter: 2019-01-01 00:00 CET -> 2018-12-31 23:00 UTC."""
        result = cet_to_utc(datetime(2019, 1, 1))
        assert result == datetime(2018, 12, 31, 23, 0)

    def test_cet_to_utc_summer(self):
        """CEST (UTC+2) in summer: 2019-07-01 00:00 CEST -> 2019-06-30 22:00 UTC."""
        result = cet_to_utc(datetime(2019, 7, 1))
        assert result == datetime(2019, 6, 30, 22, 0)

    def test_cet_to_utc_dst_spring(self):
        """2020-03-29 00:00 CET (last day before spring DST) -> 2020-03-28 23:00 UTC."""
        result = cet_to_utc(datetime(2020, 3, 29))
        assert result == datetime(2020, 3, 28, 23, 0)

    def test_cet_to_utc_dst_autumn(self):
        """2020-10-25 00:00 CEST (still CEST at midnight) -> 2020-10-24 22:00 UTC."""
        result = cet_to_utc(datetime(2020, 10, 25))
        assert result == datetime(2020, 10, 24, 22, 0)

    def test_cet_to_utc_returns_naive(self):
        """The returned datetime must be timezone-naive."""
        result = cet_to_utc(datetime(2020, 6, 15, 12, 0))
        assert result.tzinfo is None


# ---------------------------------------------------------------------------
# compute_nmd
# ---------------------------------------------------------------------------


def _make_production(areas, n_hours=24, fuels=None):
    """Create a mock production dict for testing compute_* functions.

    Returns production with 'hour' as POSIX hours (int), matching the
    pre-filtered format expected by all compute_* functions.
    """
    if fuels is None:
        fuels = ["biomass", "gas", "nuclear", "solar", "onshore", "offshore",
                 "river", "lake", "phs", "phs_in", "geothermal",
                 "marine", "other_renew", "waste", "other"]
    result = {}
    rng = np.random.default_rng(42)
    hours_dt = pd.date_range("2021-01-01", periods=n_hours, freq="h")
    hours_posix = to_posix_hours(hours_dt.to_series())
    for area in areas:
        data = {"hour": hours_posix.values}
        for fuel in fuels:
            data[fuel] = rng.uniform(0, 1000, n_hours)
        # phs_in is negative at all levels (consumption)
        if "phs_in" in data:
            data["phs_in"] = -np.abs(data["phs_in"])
        result[area] = pd.DataFrame(data)
    return result


class TestComputeNmd:
    def test_nmd_sums_correct_fuels(self):
        """NMD should be the sum of biomass, geothermal, marine, other_renew, waste, other."""
        production = _make_production(["FR"], n_hours=4)
        nmd = compute_nmd(production, ["FR"])

        # Compute expected NMD manually
        df = production["FR"]
        nmd_fuels = ["biomass", "geothermal", "marine", "other_renew", "waste", "other"]
        expected = df[nmd_fuels].sum(axis=1).values / 1000  # MW → GW

        fr_nmd = nmd[nmd["area"] == "FR"].sort_values("hour")
        np.testing.assert_array_almost_equal(fr_nmd["value"].values, expected, decimal=6)

    def test_nmd_output_columns(self):
        production = _make_production(["FR", "DE"], n_hours=2)
        nmd = compute_nmd(production, ["FR", "DE"])
        assert list(nmd.columns) == ["area", "hour", "value"]


# ---------------------------------------------------------------------------
# compute_vre_capacity_factors
# ---------------------------------------------------------------------------


class TestComputeVreCapacityFactors:
    def test_cf_bounded_zero_one(self):
        """Capacity factors must be in [0, 1]."""
        production = _make_production(["FR"], n_hours=24)
        capa = pd.DataFrame(
            {"FR": [1000.0, 1000.0, 1000.0]},  # MW
            index=pd.Index(["offshore", "onshore", "solar"], name="tec"),
        )
        cf = compute_vre_capacity_factors(production, capa, ["FR"])
        assert cf["value"].min() >= 0.0
        assert cf["value"].max() <= 1.0

    def test_cf_output_columns(self):
        production = _make_production(["FR"], n_hours=2)
        capa = pd.DataFrame(
            {"FR": [1000.0]},  # MW
            index=pd.Index(["onshore"], name="tec"),
        )
        cf = compute_vre_capacity_factors(production, capa, ["FR"],
                                           technologies=["onshore"])
        assert list(cf.columns) == ["area", "tec", "hour", "value"]


class TestComputeRiverCf:
    def test_river_cf_bounded_zero_one(self):
        """River capacity factors must be in [0, 1]."""
        production = _make_production(["FR"], n_hours=24)
        cf = compute_vre_capacity_factors(production, None, ["FR"],
                                          technologies=["river"])
        assert cf["value"].min() >= 0.0
        assert cf["value"].max() <= 1.0

    def test_river_cf_peaks_at_one(self):
        """River CF should peak at 1.0 (normalized by max production when no capa given)."""
        production = _make_production(["FR"], n_hours=24)
        cf = compute_vre_capacity_factors(production, None, ["FR"],
                                          technologies=["river"])
        assert cf["value"].max() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# compute_nuclear_max_af
# ---------------------------------------------------------------------------


class TestComputeNuclearMaxAf:
    def test_nuclear_af_bounded(self):
        """Nuclear AF must be in [0, 1]."""
        production = _make_production(["FR"], n_hours=168)  # 1 week
        capa = pd.DataFrame(
            {"FR": [60000.0]},  # MW (60 GW)
            index=pd.Index(["nuclear"], name="tec"),
        )
        _, hour_week = compute_hour_mappings(2021)
        nuc_af = compute_nuclear_max_af(production, capa, ["FR"], hour_week)
        assert nuc_af["value"].min() >= 0.0
        assert nuc_af["value"].max() <= 1.0

    def test_nuclear_af_no_nuclear_full_availability(self):
        """Areas without nuclear data should get AF=1.0."""
        production = _make_production(["CH"], n_hours=168,
                                       fuels=["gas", "solar"])  # no nuclear column
        _, hour_week = compute_hour_mappings(2021)
        nuc_af = compute_nuclear_max_af(production, None, ["CH"], hour_week)
        assert all(nuc_af["value"] == 1.0)


# ---------------------------------------------------------------------------
# compute_lake_inflows
# ---------------------------------------------------------------------------


class TestComputeLakeInflows:
    def test_lake_inflows_non_negative(self):
        """Lake inflows must be >= 0."""
        production = _make_production(["FR"], n_hours=730)  # ~1 month
        hour_month, _ = compute_hour_mappings(2021, months=(1, 1))
        inflows = compute_lake_inflows(production, ["FR"], hour_month)
        assert inflows["value"].min() >= 0.0

    def test_lake_inflows_output_columns(self):
        production = _make_production(["FR"], n_hours=730)
        hour_month, _ = compute_hour_mappings(2021, months=(1, 1))
        inflows = compute_lake_inflows(production, ["FR"], hour_month)
        assert list(inflows.columns) == ["area", "month", "value"]


# ---------------------------------------------------------------------------
# compute_hydro_limits
# ---------------------------------------------------------------------------


class TestComputeHydroLimits:
    def test_hydro_limits_non_negative(self):
        """hMaxIn and hMaxOut must be >= 0."""
        production = _make_production(["FR"], n_hours=730)
        hour_month, _ = compute_hour_mappings(2021, months=(1, 1))
        h_in, h_out = compute_hydro_limits(production, ["FR"], hour_month)
        assert h_in["value"].min() >= 0.0
        assert h_out["value"].min() >= 0.0

    def test_hydro_limits_output_columns(self):
        production = _make_production(["FR"], n_hours=730)
        hour_month, _ = compute_hour_mappings(2021, months=(1, 1))
        h_in, h_out = compute_hydro_limits(production, ["FR"], hour_month)
        assert list(h_in.columns) == ["area", "month", "value"]
        assert list(h_out.columns) == ["area", "month", "value"]


# ---------------------------------------------------------------------------
# fuel_timeFactor calendar month expansion (unit logic test)
# ---------------------------------------------------------------------------


class TestFuelTimeFactorExpansion:
    @staticmethod
    def _expand_fuel_timefactor(fuel_tf_raw, sim_months):
        """Reproduce the expansion logic from extract_scenario."""
        sim_months_df = pd.DataFrame({
            "yyyymm": sim_months,
            "month": [int(m[-2:]) for m in sim_months],
        })
        merged = fuel_tf_raw.merge(sim_months_df, on="month", how="inner")
        return merged[["fuel", "yyyymm", "value"]].rename(columns={"yyyymm": "month"})

    def _make_raw(self, months, fuels, value=1.0):
        """Build a melted fuel_timeFactor table with integer month column."""
        rows = []
        for m in months:
            for f in fuels:
                rows.append({"month": m, "fuel": f, "value": value})
        return pd.DataFrame(rows)

    def test_fuel_timefactor_expansion_single_year(self):
        """12 calendar months x N fuels -> 12*N rows with YYYYMM strings."""
        fuels = ["GAS", "COAL"]
        raw = self._make_raw(range(1, 13), fuels)
        sim_months = [f"2020{m:02d}" for m in range(1, 13)]
        result = self._expand_fuel_timefactor(raw, sim_months)
        assert len(result) == 12 * len(fuels)
        assert all(isinstance(v, str) and len(v) == 6 for v in result["month"])

    def test_fuel_timefactor_expansion_partial_year(self):
        """Only months 3-5 in sim_months -> 3*N rows."""
        fuels = ["GAS", "COAL", "OIL"]
        raw = self._make_raw(range(1, 13), fuels)
        sim_months = ["202003", "202004", "202005"]
        result = self._expand_fuel_timefactor(raw, sim_months)
        assert len(result) == 3 * len(fuels)

    def test_fuel_timefactor_mean_is_one(self):
        """Seasonal weights from baseline scenario average to ~1.0 per fuel."""
        baseline_path = (
            Path(__file__).resolve().parents[1]
            / "scenarios"
            / "baseline"
            / "fuel_timeFactor.csv"
        )
        if not baseline_path.exists():
            pytest.skip("baseline scenario not available")
        df = pd.read_csv(baseline_path)
        fuel_cols = [c for c in df.columns if c != "month"]
        for fuel in fuel_cols:
            mean_val = df[fuel].mean()
            assert abs(mean_val - 1.0) < 0.01, (
                f"Mean of {fuel} time factor is {mean_val}, expected ~1.0"
            )


# ---------------------------------------------------------------------------
# Hour-month CET mapping (using timezone helpers)
# ---------------------------------------------------------------------------


class TestHourMonthCetMapping:
    def test_hour_month_march_first_hour_cet(self):
        """The POSIX hour for 2020-02-29T23:00 UTC maps to month '202003' in CET."""
        hour_posix = (
            pd.Timestamp("2020-02-29 23:00") - pd.Timestamp("1970-01-01")
        ).total_seconds() / 3600
        hour_utc = pd.to_datetime(
            hour_posix * 3600, unit="s", origin="1970-01-01", utc=True
        )
        month_cet = hour_utc.tz_convert(CET).strftime("%Y%m")
        assert month_cet == "202003"

    def test_hour_month_full_year_12_months(self):
        """A full year of hourly timestamps yields exactly 12 unique CET months."""
        hour_month, _ = compute_hour_mappings(2020)
        assert len(hour_month["month"].unique()) == 12

    def test_compute_hour_mappings_full_year_length(self):
        """Full year mapping should have expected_hours entries."""
        from eoles_dispatch.utils import expected_hours
        hour_month, hour_week = compute_hour_mappings(2021)
        assert len(hour_month) == expected_hours(2021)
        assert len(hour_week) == expected_hours(2021)

    def test_compute_hour_mappings_partial_year(self):
        """Partial year (Jan-Mar) should have fewer hours than full year."""
        hm_full, _ = compute_hour_mappings(2021)
        hm_partial, _ = compute_hour_mappings(2021, months=(1, 3))
        assert len(hm_partial) < len(hm_full)
        assert set(hm_partial["month"].unique()) == {"202101", "202102", "202103"}


# ---------------------------------------------------------------------------
# _validate_year (collect._main_collect)
# ---------------------------------------------------------------------------


def _make_year_dir(tmp_path, year, areas, exo_areas, include_actual_prices=True):
    """Create a minimal valid year directory for _validate_year tests."""
    n = expected_hours(year)
    hours = list(range(n))

    pd.DataFrame({"hour": hours, **{a: [1000.0] * n for a in areas}}).to_csv(
        tmp_path / "demand.csv", index=False
    )
    pd.DataFrame({"hour": hours, **{a: [50.0] * n for a in exo_areas}}).to_csv(
        tmp_path / "exo_prices.csv", index=False
    )
    fuels = ["biomass", "gas", "nuclear", "solar", "onshore"]
    for area in areas:
        pd.DataFrame({"hour": hours, **{f: [100.0] * n for f in fuels}}).to_csv(
            tmp_path / f"production_{area}.csv", index=False
        )
    if include_actual_prices:
        pd.DataFrame({"hour": hours, **{a: [55.0] * n for a in areas}}).to_csv(
            tmp_path / "actual_prices.csv", index=False
        )


class TestValidateYear:
    def test_valid_year_passes(self, tmp_path):
        """A complete, correctly-sized year directory passes validation."""
        areas, exo_areas = ["FR"], ["CH"]
        _make_year_dir(tmp_path, 2021, areas, exo_areas)
        is_valid, issues = _validate_year(tmp_path, 2021, areas, exo_areas)
        assert is_valid
        assert issues == []

    def test_missing_demand_fails(self, tmp_path):
        areas, exo_areas = ["FR"], ["CH"]
        _make_year_dir(tmp_path, 2021, areas, exo_areas)
        (tmp_path / "demand.csv").unlink()
        is_valid, issues = _validate_year(tmp_path, 2021, areas, exo_areas)
        assert not is_valid
        assert any("demand.csv" in i for i in issues)

    def test_missing_actual_prices_does_not_fail(self, tmp_path):
        """actual_prices.csv is soft validation: missing should not block."""
        areas, exo_areas = ["FR"], ["CH"]
        _make_year_dir(tmp_path, 2021, areas, exo_areas, include_actual_prices=False)
        is_valid, issues = _validate_year(tmp_path, 2021, areas, exo_areas)
        assert is_valid
        assert issues == []

    def test_missing_production_file_fails(self, tmp_path):
        areas, exo_areas = ["FR", "BE"], ["CH"]
        _make_year_dir(tmp_path, 2021, areas, exo_areas)
        (tmp_path / "production_BE.csv").unlink()
        is_valid, issues = _validate_year(tmp_path, 2021, areas, exo_areas)
        assert not is_valid
        assert any("production_BE.csv" in i for i in issues)

    def test_wrong_row_count_fails(self, tmp_path):
        """A demand.csv with incorrect row count is flagged."""
        areas, exo_areas = ["FR"], ["CH"]
        _make_year_dir(tmp_path, 2021, areas, exo_areas)
        n_wrong = expected_hours(2021) - 10
        pd.DataFrame({"hour": list(range(n_wrong)), "FR": [1000.0] * n_wrong}).to_csv(
            tmp_path / "demand.csv", index=False
        )
        is_valid, issues = _validate_year(tmp_path, 2021, areas, exo_areas)
        assert not is_valid
        assert any("demand.csv" in i for i in issues)

    def test_missing_exo_area_in_prices_fails(self, tmp_path):
        """exo_prices.csv missing an exo area column is flagged."""
        areas, exo_areas = ["FR"], ["CH", "NO"]
        _make_year_dir(tmp_path, 2021, areas, exo_areas)
        # Overwrite exo_prices.csv with only one of the two exo areas
        n = expected_hours(2021)
        pd.DataFrame({"hour": list(range(n)), "CH": [50.0] * n}).to_csv(
            tmp_path / "exo_prices.csv", index=False
        )
        is_valid, issues = _validate_year(tmp_path, 2021, areas, exo_areas)
        assert not is_valid
        assert any("exo_prices.csv" in i for i in issues)


# ---------------------------------------------------------------------------
# _copy_actual_prices (run._main_run)
# ---------------------------------------------------------------------------


def _make_actual_prices_csv(year_dir, year, areas, n_hours=None):
    """Write a minimal actual_prices.csv with datetime hour strings."""
    from eoles_dispatch.utils import cet_year_bounds
    start, _ = cet_year_bounds(year)
    if n_hours is None:
        n_hours = expected_hours(year)
    hours = pd.date_range(start, periods=n_hours, freq="h")
    df = pd.DataFrame({"hour": hours, **{a: [55.0] * n_hours for a in areas}})
    year_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(year_dir / "actual_prices.csv", index=False)


class TestCopyActualPrices:
    def test_creates_validation_file(self, tmp_path):
        """Should create validation/actual_prices.csv when source exists."""
        year, areas = 2021, ["FR", "BE"]
        data_dir = tmp_path / "data"
        _make_actual_prices_csv(data_dir / str(year), year, areas)
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _copy_actual_prices(data_dir, run_dir, year, areas, months=None)
        assert (run_dir / "validation" / "actual_prices.csv").exists()

    def test_skips_silently_when_source_missing(self, tmp_path):
        """Should not raise and should not create validation/ if source is absent."""
        data_dir = tmp_path / "data"
        (data_dir / "2021").mkdir(parents=True)
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _copy_actual_prices(data_dir, run_dir, 2021, ["FR"], months=None)
        assert not (run_dir / "validation").exists()

    def test_hour_column_is_posix_int(self, tmp_path):
        """Output hour column must be POSIX hours (int), not datetimes."""
        year, areas = 2021, ["FR"]
        data_dir = tmp_path / "data"
        _make_actual_prices_csv(data_dir / str(year), year, areas)
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _copy_actual_prices(data_dir, run_dir, year, areas, months=None)
        out = pd.read_csv(run_dir / "validation" / "actual_prices.csv")
        assert pd.api.types.is_integer_dtype(out["hour"])

    def test_filters_to_requested_areas(self, tmp_path):
        """Output should only contain requested areas, not all areas in source."""
        year = 2021
        all_areas = ["FR", "BE", "DE"]
        data_dir = tmp_path / "data"
        _make_actual_prices_csv(data_dir / str(year), year, all_areas)
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _copy_actual_prices(data_dir, run_dir, year, ["FR"], months=None)
        out = pd.read_csv(run_dir / "validation" / "actual_prices.csv")
        assert "FR" in out.columns
        assert "BE" not in out.columns
        assert "DE" not in out.columns

    def test_filters_to_months(self, tmp_path):
        """With months=(1, 3), output should cover only Jan-Mar hours."""
        year, areas = 2021, ["FR"]
        data_dir = tmp_path / "data"
        _make_actual_prices_csv(data_dir / str(year), year, areas)
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _copy_actual_prices(data_dir, run_dir, year, areas, months=(1, 3))
        out = pd.read_csv(run_dir / "validation" / "actual_prices.csv")
        assert len(out) < expected_hours(year)
