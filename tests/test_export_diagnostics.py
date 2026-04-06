"""Tests for eoles_dispatch.run.export_diagnostics."""

import json

import pandas as pd
import pyomo.environ as pyo
import pytest

from eoles_dispatch.run.export_diagnostics import (
    _dual_to_df,
    _numeric_stats,
    _var_to_df,
    export_all_diagnostics,
)

AREAS = ["FR", "DE"]
HOURS = [0, 1, 2]


# ---------------------------------------------------------------------------
# _var_to_df
# ---------------------------------------------------------------------------


class TestVarToDf:
    @pytest.fixture
    def simple_var(self):
        m = pyo.ConcreteModel()
        m.idx = pyo.Set(initialize=[0, 1, 2])
        m.x = pyo.Var(m.idx, initialize=lambda m, i: float(i))
        return m.x

    @pytest.fixture
    def multi_idx_var(self):
        m = pyo.ConcreteModel()
        m.a = pyo.Set(initialize=["FR", "DE"])
        m.h = pyo.Set(initialize=[0, 1])
        m.y = pyo.Var(m.a, m.h, initialize=1.0)
        return m.y

    def test_var_to_df_columns(self, simple_var):
        df = _var_to_df(simple_var)
        assert list(df.columns) == ["idx0", "value"]

    def test_var_to_df_row_count(self, simple_var):
        df = _var_to_df(simple_var)
        assert len(df) == 3

    def test_var_to_df_values(self, simple_var):
        df = _var_to_df(simple_var)
        assert sorted(df["value"].tolist()) == [0.0, 1.0, 2.0]

    def test_var_to_df_multi_index_columns(self, multi_idx_var):
        df = _var_to_df(multi_idx_var)
        assert list(df.columns) == ["idx0", "idx1", "value"]
        assert len(df) == 4

    def test_var_to_df_empty_var_returns_empty_df(self):
        m = pyo.ConcreteModel()
        m.empty_set = pyo.Set(initialize=[])
        m.z = pyo.Var(m.empty_set)
        df = _var_to_df(m.z)
        assert df.empty


# ---------------------------------------------------------------------------
# _numeric_stats
# ---------------------------------------------------------------------------


class TestNumericStats:
    def test_stats_all_fields_present(self):
        result = _numeric_stats(pd.Series([1.0, 2.0, 3.0]))
        for key in ("count", "nonzero_count", "min", "max", "mean", "sum"):
            assert key in result, f"Missing key: {key}"

    def test_stats_count_correct(self):
        assert _numeric_stats(pd.Series([1.0, 2.0, 3.0]))["count"] == 3

    def test_stats_nonzero_count(self):
        assert _numeric_stats(pd.Series([0.0, 1.0, 2.0]))["nonzero_count"] == 2

    def test_stats_empty_series_returns_empty_dict(self):
        assert _numeric_stats(pd.Series([], dtype=float)) == {}

    def test_stats_all_zeros_nonzero_count_zero(self):
        assert _numeric_stats(pd.Series([0.0, 0.0]))["nonzero_count"] == 0

    def test_stats_sum_correct(self):
        assert _numeric_stats(pd.Series([2.0, 3.0, 5.0]))["sum"] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# _dual_to_df (requires solved model with duals)
# ---------------------------------------------------------------------------


class TestDualToDf:
    def test_dual_to_df_columns(self, solved_run):
        model, _ = solved_run
        df = _dual_to_df(model, model.adequacy_constraint)
        assert "dual" in df.columns

    def test_dual_to_df_row_count_matches_constraint(self, solved_run):
        model, _ = solved_run
        df = _dual_to_df(model, model.adequacy_constraint)
        assert len(df) == len(AREAS) * len(HOURS)

    def test_dual_to_df_dual_values_numeric(self, solved_run):
        model, _ = solved_run
        df = _dual_to_df(model, model.adequacy_constraint)
        assert df["dual"].dtype.kind in ("f", "i")
        assert df["dual"].notna().all()


# ---------------------------------------------------------------------------
# export_all_diagnostics (integration)
# ---------------------------------------------------------------------------


class TestExportAllDiagnostics:
    def test_creates_vars_dir(self, solved_run):
        model, run_dir = solved_run
        export_all_diagnostics(model, run_dir)
        assert (run_dir / "diagnostics" / "vars").is_dir()

    def test_creates_duals_dir(self, solved_run):
        model, run_dir = solved_run
        export_all_diagnostics(model, run_dir)
        assert (run_dir / "diagnostics" / "duals").is_dir()

    def test_creates_summary_json(self, solved_run):
        model, run_dir = solved_run
        export_all_diagnostics(model, run_dir)
        assert (run_dir / "diagnostics" / "_summary.json").exists()

    def test_gene_csv_created(self, solved_run):
        model, run_dir = solved_run
        export_all_diagnostics(model, run_dir)
        assert (run_dir / "diagnostics" / "vars" / "gene.csv").exists()

    def test_adequacy_dual_csv_created(self, solved_run):
        model, run_dir = solved_run
        export_all_diagnostics(model, run_dir)
        assert (run_dir / "diagnostics" / "duals" / "adequacy.csv").exists()

    def test_static_model_skips_on_var(self, solved_run):
        """static_thermal has no 'on' variable → vars/on.csv must not be created."""
        model, run_dir = solved_run
        export_all_diagnostics(model, run_dir)
        assert not (run_dir / "diagnostics" / "vars" / "on.csv").exists()

    def test_summary_json_structure(self, solved_run):
        model, run_dir = solved_run
        export_all_diagnostics(model, run_dir)
        with open(run_dir / "diagnostics" / "_summary.json") as f:
            summary = json.load(f)
        for key in ("objective_kEUR", "objective_bEUR", "sets", "variable_stats", "dual_stats"):
            assert key in summary, f"Missing key in summary: {key}"
        assert isinstance(summary["objective_kEUR"], float)
        import math

        assert math.isfinite(summary["objective_kEUR"])

    def test_summary_json_sets_size(self, solved_run):
        model, run_dir = solved_run
        export_all_diagnostics(model, run_dir)
        with open(run_dir / "diagnostics" / "_summary.json") as f:
            summary = json.load(f)
        assert summary["sets"]["a"]["size"] == len(AREAS)

    def test_summary_json_co2(self, solved_run):
        model, run_dir = solved_run
        export_all_diagnostics(model, run_dir)
        with open(run_dir / "diagnostics" / "_summary.json") as f:
            summary = json.load(f)
        assert "total_co2_tCO2" in summary
        # Value may be 0 (no thermal dispatch) but must be a number
        assert summary["total_co2_tCO2"] is None or isinstance(
            summary["total_co2_tCO2"], (int, float)
        )
