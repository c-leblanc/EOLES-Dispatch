"""Tests for eoles_dispatch.format_outputs."""

import time

import numpy as np
import pandas as pd

from eoles_dispatch.run.format_outputs import (
    _safe_gene_values,
    _safe_var_values,
    report_prices,
    report_production,
    write_log,
)

AREAS = ["FR", "DE"]
HOURS = [0, 1, 2]

# ---------------------------------------------------------------------------
# _safe_gene_values
# ---------------------------------------------------------------------------


class TestSafeGeneValues:
    def test_safe_gene_values_existing_tec(self):
        """Keys matching the technology are extracted in order."""
        gene_vals = {
            ("FR", "nuclear", 0): 10.0,
            ("FR", "nuclear", 1): 20.0,
        }
        result = _safe_gene_values(gene_vals, "nuclear", 2, areas=["FR"], hours=[0, 1])
        np.testing.assert_array_equal(result, [10.0, 20.0])

    def test_safe_gene_values_missing_tec(self):
        """Missing technology returns an array of zeros."""
        gene_vals = {("FR", "nuclear", 0): 10.0}
        result = _safe_gene_values(gene_vals, "solar", 3, areas=["FR"], hours=[0, 1, 2])
        np.testing.assert_array_equal(result, [0.0, 0.0, 0.0])

    def test_safe_gene_values_none_replaced_by_zero(self):
        """None values in the dict are replaced by 0.0."""
        gene_vals = {
            ("FR", "wind", 0): None,
            ("FR", "wind", 1): 5.0,
        }
        result = _safe_gene_values(gene_vals, "wind", 2, areas=["FR"], hours=[0, 1])
        np.testing.assert_array_equal(result, [0.0, 5.0])

    def test_safe_gene_values_sorted_order(self):
        """Values follow the order of areas × hours."""
        gene_vals = {
            ("DE", "nuc", 0): 30.0,
            ("BE", "nuc", 0): 10.0,
        }
        result = _safe_gene_values(gene_vals, "nuc", 2, areas=["BE", "DE"], hours=[0])
        np.testing.assert_array_equal(result, [10.0, 30.0])

    def test_safe_gene_values_length(self):
        """Output length matches n_rows when technology exists."""
        gene_vals = {("FR", "gas", i): float(i) for i in range(5)}
        result = _safe_gene_values(gene_vals, "gas", 5, areas=["FR"], hours=list(range(5)))
        assert len(result) == 5


# ---------------------------------------------------------------------------
# _safe_var_values
# ---------------------------------------------------------------------------


class TestSafeVarValues:
    def test_safe_var_values_existing(self):
        """Keys matching the technology are extracted correctly."""
        var_vals = {
            ("FR", "battery", 0): 100.0,
            ("FR", "battery", 1): 200.0,
        }
        result = _safe_var_values(var_vals, "battery", 2, areas=["FR"], hours=[0, 1])
        np.testing.assert_array_equal(result, [100.0, 200.0])

    def test_safe_var_values_missing(self):
        """Missing technology returns zeros."""
        var_vals = {("FR", "battery", 0): 100.0}
        result = _safe_var_values(var_vals, "phs", 4, areas=["FR"], hours=[0, 1, 2, 3])
        np.testing.assert_array_equal(result, [0.0, 0.0, 0.0, 0.0])

    def test_safe_var_values_none_to_zero(self):
        """None values are replaced by 0.0."""
        var_vals = {
            ("DE", "storage", 0): None,
            ("DE", "storage", 1): 50.0,
        }
        result = _safe_var_values(var_vals, "storage", 2, areas=["DE"], hours=[0, 1])
        np.testing.assert_array_equal(result, [0.0, 50.0])


# ---------------------------------------------------------------------------
# report_prices (requires solved model)
# ---------------------------------------------------------------------------


class TestReportPrices:
    def test_prices_csv_created(self, solved_run):
        model, run_dir = solved_run
        report_prices(model, run_dir)
        assert (run_dir / "outputs" / "prices.csv").exists()

    def test_prices_index_is_hours(self, solved_run):
        model, run_dir = solved_run
        report_prices(model, run_dir)
        df = pd.read_csv(run_dir / "outputs" / "prices.csv", index_col="hour")
        assert len(df) == len(HOURS)

    def test_prices_columns_are_areas(self, solved_run):
        model, run_dir = solved_run
        report_prices(model, run_dir)
        df = pd.read_csv(run_dir / "outputs" / "prices.csv", index_col="hour")
        model_areas = list(model.a)
        assert set(df.columns) == set(model_areas)

    def test_prices_values_are_numeric(self, solved_run):
        model, run_dir = solved_run
        report_prices(model, run_dir)
        df = pd.read_csv(run_dir / "outputs" / "prices.csv", index_col="hour")
        assert df.apply(pd.to_numeric, errors="coerce").notna().all().all()

    def test_prices_creates_output_dir_if_missing(self, solved_run, tmp_path):
        """report_prices must mkdir outputs/ if absent."""
        from conftest import _build_input_dir
        from pyomo.opt import SolverFactory

        from eoles_dispatch.models import build_static_thermal_model

        fresh = tmp_path / "fresh_run"
        _build_input_dir(fresh)
        m = build_static_thermal_model(fresh)
        opt = SolverFactory("appsi_highs")
        opt.highs_options["solver"] = "ipm"
        opt.highs_options["run_crossover"] = "on"
        opt.solve(m)
        # Ensure outputs/ does not exist
        assert not (fresh / "outputs").exists()
        report_prices(m, fresh)
        assert (fresh / "outputs" / "prices.csv").exists()


# ---------------------------------------------------------------------------
# report_production (requires solved model)
# ---------------------------------------------------------------------------


class TestReportProduction:
    def test_production_csv_created(self, solved_run):
        model, run_dir = solved_run
        report_production(model, run_dir)
        assert (run_dir / "outputs" / "production.csv").exists()

    def test_production_has_area_hour_index(self, solved_run):
        model, run_dir = solved_run
        report_production(model, run_dir)
        df = pd.read_csv(
            run_dir / "outputs" / "production.csv", index_col=["area", "hour"]
        )
        assert len(df) == len(AREAS) * len(HOURS)

    def test_production_has_demand_column(self, solved_run):
        model, run_dir = solved_run
        report_production(model, run_dir)
        df = pd.read_csv(run_dir / "outputs" / "production.csv")
        assert "demand" in df.columns

    def test_production_values_non_negative_for_gen_cols(self, solved_run):
        model, run_dir = solved_run
        report_production(model, run_dir)
        df = pd.read_csv(
            run_dir / "outputs" / "production.csv", index_col=["area", "hour"]
        )
        gen_cols = [c for c in df.columns if c not in ("demand", "net_imports", "net_exports")]
        assert (df[gen_cols] >= -1e-6).all().all()


# ---------------------------------------------------------------------------
# write_log (requires solved model)
# ---------------------------------------------------------------------------


class TestWriteLog:
    def test_write_log_creates_file(self, solved_run):
        model, run_dir = solved_run
        write_log(run_dir, model, "test_run", "test_scenario", 2020, time.localtime(), "00:00:01")
        assert (run_dir / "_log_test_run.txt").exists()

    def test_write_log_contains_run_name(self, solved_run):
        model, run_dir = solved_run
        write_log(run_dir, model, "test_run", "test_scenario", 2020, time.localtime(), "00:00:01")
        text = (run_dir / "_log_test_run.txt").read_text()
        assert "test_run" in text

    def test_write_log_contains_year(self, solved_run):
        model, run_dir = solved_run
        write_log(run_dir, model, "test_run", "test_scenario", 2020, time.localtime(), "00:00:01")
        text = (run_dir / "_log_test_run.txt").read_text()
        assert "2020" in text

    def test_write_log_contains_cost(self, solved_run):
        model, run_dir = solved_run
        write_log(run_dir, model, "test_run", "test_scenario", 2020, time.localtime(), "00:00:01")
        text = (run_dir / "_log_test_run.txt").read_text()
        assert "bEUR" in text
