"""Tests for eoles_dispatch.format_inputs."""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eoles_dispatch.utils import cet_to_utc, CET, compute_hour_mappings
from eoles_dispatch.format_inputs import (
    compute_nmd,
    compute_vre_capacity_factors,
    compute_nuclear_max_af,
    compute_lake_inflows,
    compute_hydro_limits,
    load_ninja_var,
)


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
    """Create a mock production dict for testing compute_* functions."""
    if fuels is None:
        fuels = ["biomass", "gas", "nuclear", "solar", "onshore", "offshore",
                 "river", "lake", "phs_prod", "phs_cons", "geothermal",
                 "marine", "other_renew", "waste", "other"]
    result = {}
    rng = np.random.default_rng(42)
    for area in areas:
        data = {"hour": pd.date_range("2021-01-01", periods=n_hours, freq="h")}
        for fuel in fuels:
            data[fuel] = rng.uniform(0, 1000, n_hours)
        result[area] = pd.DataFrame(data)
    return result


class TestComputeNmd:
    def test_nmd_sums_correct_fuels(self):
        """NMD should be the sum of biomass, geothermal, marine, other_renew, waste, other."""
        production = _make_production(["FR"], n_hours=4)
        start = datetime(2021, 1, 1)
        end = datetime(2021, 1, 1, 4)
        nmd = compute_nmd(production, ["FR"], start, end)

        # Compute expected NMD manually
        df = production["FR"]
        nmd_fuels = ["biomass", "geothermal", "marine", "other_renew", "waste", "other"]
        expected = df[nmd_fuels].sum(axis=1).values / 1000  # MW → GW

        fr_nmd = nmd[nmd["area"] == "FR"].sort_values("hour")
        np.testing.assert_array_almost_equal(fr_nmd["value"].values, expected, decimal=6)

    def test_nmd_output_columns(self):
        production = _make_production(["FR", "DE"], n_hours=2)
        start = datetime(2021, 1, 1)
        end = datetime(2021, 1, 1, 2)
        nmd = compute_nmd(production, ["FR", "DE"], start, end)
        assert list(nmd.columns) == ["area", "hour", "value"]


# ---------------------------------------------------------------------------
# compute_vre_capacity_factors
# ---------------------------------------------------------------------------


class TestComputeVreCapacityFactors:
    def test_cf_bounded_zero_one(self):
        """Capacity factors must be in [0, 1]."""
        production = _make_production(["FR"], n_hours=24)
        capa = pd.DataFrame({
            "area": ["FR", "FR", "FR", "FR"],
            "tec": ["offshore", "onshore", "pv", "river"],
            "value": [1.0, 1.0, 1.0, 0.5],  # GW
        })
        start = datetime(2021, 1, 1)
        end = datetime(2021, 1, 2)
        cf = compute_vre_capacity_factors(production, capa, ["FR"], start, end)
        assert cf["value"].min() >= 0.0
        assert cf["value"].max() <= 1.0

    def test_cf_output_columns(self):
        production = _make_production(["FR"], n_hours=2)
        capa = pd.DataFrame({
            "area": ["FR"], "tec": ["onshore"], "value": [1.0],
        })
        start = datetime(2021, 1, 1)
        end = datetime(2021, 1, 1, 2)
        cf = compute_vre_capacity_factors(production, capa, ["FR"], start, end,
                                           technologies=["onshore"])
        assert list(cf.columns) == ["area", "tec", "hour", "value"]


# ---------------------------------------------------------------------------
# compute_nuclear_max_af
# ---------------------------------------------------------------------------


class TestComputeNuclearMaxAf:
    def test_nuclear_af_bounded(self):
        """Nuclear AF must be in [0, 1]."""
        production = _make_production(["FR"], n_hours=168)  # 1 week
        capa = pd.DataFrame({
            "area": ["FR"], "tec": ["nuclear"], "value": [60.0],  # GW
        })
        _, hour_week = compute_hour_mappings(2021)
        # Filter hour_week to match our small production
        nuc_af = compute_nuclear_max_af(production, capa, ["FR"], hour_week)
        assert nuc_af["value"].min() >= 0.0
        assert nuc_af["value"].max() <= 1.0

    def test_nuclear_af_no_nuclear_full_availability(self):
        """Areas without nuclear data should get AF=1.0."""
        production = _make_production(["CH"], n_hours=168,
                                       fuels=["gas", "solar"])  # no nuclear column
        capa = pd.DataFrame(columns=["area", "tec", "value"])
        _, hour_week = compute_hour_mappings(2021)
        nuc_af = compute_nuclear_max_af(production, capa, ["CH"], hour_week)
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
