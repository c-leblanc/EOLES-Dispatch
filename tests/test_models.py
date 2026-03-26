"""Tests for Pyomo model construction (without solving).

These tests build tiny models (2 areas, 3 hours, few technologies) to verify
that model construction works correctly: sets, variables, constraints, and
objective are created with the right dimensions. No solver is invoked.
"""

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import pyomo.environ as pyo
import pytest

from eoles_dispatch.models import build_default_model, build_static_thermal_model


# ---------------------------------------------------------------------------
# Fixture: generate a minimal set of input CSVs
# ---------------------------------------------------------------------------

AREAS = ["FR", "DE"]
EXO_AREAS = ["NL"]
HOURS = [0, 1, 2]
MONTHS = ["202003"]
WEEKS = ["202013"]
VRE = ["onshore", "solar"]
THR = ["nuclear", "gas_ccgt1G"]
STO = ["lake_phs", "battery"]
TEC = ["nmd"] + VRE + THR + STO
FRR = THR + STO
NO_FRR = VRE + ["nmd"]


def _write_1col(path, values):
    """Write a single-column headerless CSV."""
    pd.DataFrame(values).to_csv(path, index=False, header=False)


def _write_2col(path, col1, col2):
    pd.DataFrame({"a": col1, "b": col2}).to_csv(path, index=False, header=False)


def _write_3col(path, c1, c2, c3):
    pd.DataFrame({"a": c1, "b": c2, "c": c3}).to_csv(path, index=False, header=False)


@pytest.fixture
def input_dir(tmp_path):
    """Create a minimal inputs/ directory with all required CSVs for model construction."""
    d = tmp_path / "inputs"
    d.mkdir()

    n_h = len(HOURS)
    n_a = len(AREAS)

    # ── Sets ──
    _write_1col(d / "areas.csv", AREAS)
    _write_1col(d / "exo_areas.csv", EXO_AREAS)
    _write_1col(d / "hours.csv", HOURS)
    _write_1col(d / "months.csv", MONTHS)
    _write_1col(d / "weeks.csv", WEEKS)
    _write_1col(d / "tec.csv", TEC)
    _write_1col(d / "vre.csv", VRE)
    _write_1col(d / "thr.csv", THR)
    _write_1col(d / "str_tec.csv", STO)
    _write_1col(d / "frr.csv", FRR)
    _write_1col(d / "no_frr.csv", NO_FRR)

    # ── Time-series ──
    # demand: (area, hour, value)
    rows_ah = [(a, h) for a in AREAS for h in HOURS]
    _write_3col(d / "demand.csv",
                [r[0] for r in rows_ah],
                [r[1] for r in rows_ah],
                [50.0] * len(rows_ah))

    # nmd: (area, hour, value)
    _write_3col(d / "nmd.csv",
                [r[0] for r in rows_ah],
                [r[1] for r in rows_ah],
                [5.0] * len(rows_ah))

    # exoPrices: (exo_area, hour, value)
    rows_exo_h = [(ea, h) for ea in EXO_AREAS for h in HOURS]
    _write_3col(d / "exoPrices.csv",
                [r[0] for r in rows_exo_h],
                [r[1] for r in rows_exo_h],
                [40.0] * len(rows_exo_h))

    # vre_profiles: (area, vre, hour, load_factor) — 4 columns
    rows_vre = [(a, v, h) for a in AREAS for v in VRE for h in HOURS]
    pd.DataFrame({
        "a": [r[0] for r in rows_vre],
        "vre": [r[1] for r in rows_vre],
        "h": [r[2] for r in rows_vre],
        "lf": [0.3] * len(rows_vre),
    }).to_csv(d / "vre_profiles.csv", index=False, header=False)

    # lake_inflows: (area, month, value)
    rows_am = [(a, m) for a in AREAS for m in MONTHS]
    _write_3col(d / "lake_inflows.csv",
                [r[0] for r in rows_am],
                [r[1] for r in rows_am],
                [1.0] * len(rows_am))

    # ── Capacity ──
    # capa: (area, tec, value)
    rows_at = [(a, t) for a in AREAS for t in TEC]
    _write_3col(d / "capa.csv",
                [r[0] for r in rows_at],
                [r[1] for r in rows_at],
                [10.0] * len(rows_at))

    # capa_in: (area, sto, value)
    rows_as = [(a, s) for a in AREAS for s in STO]
    _write_3col(d / "capa_in.csv",
                [r[0] for r in rows_as],
                [r[1] for r in rows_as],
                [5.0] * len(rows_as))

    # stockMax: (area, sto, value)
    _write_3col(d / "stockMax.csv",
                [r[0] for r in rows_as],
                [r[1] for r in rows_as],
                [100.0] * len(rows_as))

    # yEAF, maxAF: (area, thr, value)
    rows_athr = [(a, t) for a in AREAS for t in THR]
    _write_3col(d / "yEAF.csv",
                [r[0] for r in rows_athr],
                [r[1] for r in rows_athr],
                [0.85] * len(rows_athr))
    _write_3col(d / "maxAF.csv",
                [r[0] for r in rows_athr],
                [r[1] for r in rows_athr],
                [0.95] * len(rows_athr))

    # nucMaxAF: (area, week, value)
    rows_aw = [(a, w) for a in AREAS for w in WEEKS]
    _write_3col(d / "nucMaxAF.csv",
                [r[0] for r in rows_aw],
                [r[1] for r in rows_aw],
                [0.9] * len(rows_aw))

    # hMaxOut, hMaxIn: (area, month, value)
    _write_3col(d / "hMaxOut.csv",
                [r[0] for r in rows_am],
                [r[1] for r in rows_am],
                [0.8] * len(rows_am))
    _write_3col(d / "hMaxIn.csv",
                [r[0] for r in rows_am],
                [r[1] for r in rows_am],
                [0.8] * len(rows_am))

    # ── Trade ──
    trade_pairs = [(a1, a2) for a1 in AREAS for a2 in AREAS if a1 != a2]
    _write_3col(d / "links.csv",
                [r[0] for r in trade_pairs],
                [r[1] for r in trade_pairs],
                [5.0] * len(trade_pairs))

    rows_exo_trade = [(a, ea) for a in AREAS for ea in EXO_AREAS]
    _write_3col(d / "exo_IM.csv",
                [r[0] for r in rows_exo_trade],
                [r[1] for r in rows_exo_trade],
                [3.0] * len(rows_exo_trade))
    _write_3col(d / "exo_EX.csv",
                [r[0] for r in rows_exo_trade],
                [r[1] for r in rows_exo_trade],
                [3.0] * len(rows_exo_trade))

    # rsv_req: (vre, value)
    _write_2col(d / "rsv_req.csv", VRE, [0.05] * len(VRE))

    # ── Thermal specs ──
    _write_2col(d / "efficiency.csv", THR, [0.45, 0.55])
    _write_2col(d / "eff50.csv", THR, [0.40, 0.50])
    _write_2col(d / "co2_factor.csv", THR, [0.0, 56.1])
    _write_2col(d / "co2_price.csv", THR, [25.0, 25.0])
    _write_2col(d / "nonFuel_vOM.csv", THR, [2.0, 1.5])
    _write_2col(d / "su_fixedCost.csv", THR, [50.0, 30.0])
    _write_2col(d / "su_fuelCons.csv", THR, [3.0, 2.0])
    _write_2col(d / "ramp_fuelCons.csv", THR, [0.5, 0.3])
    _write_2col(d / "minSG.csv", THR, [0.4, 0.3])
    _write_2col(d / "minTimeON.csv", THR, [1, 1])
    _write_2col(d / "minTimeOFF.csv", THR, [1, 1])

    # ── Fuel prices ──
    fuels = ["NUCLEAR", "GAS"]
    _write_2col(d / "thr_fuel.csv", THR, fuels)
    _write_2col(d / "fuel_price.csv", THR, [5.0, 20.0])

    # fuel_timeFactor: (fuel, month, value)
    rows_fm = [(f, m) for f in fuels for m in MONTHS]
    _write_3col(d / "fuel_timeFactor.csv",
                [r[0] for r in rows_fm],
                [r[1] for r in rows_fm],
                [1.0] * len(rows_fm))

    # fuel_areaFactor: (fuel, area, value)
    rows_fa = [(f, a) for f in fuels for a in AREAS]
    _write_3col(d / "fuel_areaFactor.csv",
                [r[0] for r in rows_fa],
                [r[1] for r in rows_fa],
                [0.0] * len(rows_fa))

    # ── Storage ──
    _write_2col(d / "str_vOM.csv", STO, [0.5, 0.3])

    # ── Hour mappings ──
    _write_2col(d / "hour_month.csv", HOURS, [MONTHS[0]] * n_h)
    _write_2col(d / "hour_week.csv", HOURS, [WEEKS[0]] * n_h)

    return tmp_path


# ---------------------------------------------------------------------------
# Tests: Default model (with thermal dynamics)
# ---------------------------------------------------------------------------


class TestDefaultModel:
    def test_builds_without_error(self, input_dir):
        """Model construction should complete without raising."""
        model = build_default_model(input_dir)
        assert model is not None

    def test_sets_sizes(self, input_dir):
        model = build_default_model(input_dir)
        assert len(model.a) == len(AREAS)
        assert len(model.exo_a) == len(EXO_AREAS)
        assert len(model.h) == len(HOURS)
        assert len(model.tec) == len(TEC)
        assert len(model.vre) == len(VRE)
        assert len(model.thr) == len(THR)
        assert len(model.sto) == len(STO)
        assert len(model.month) == len(MONTHS)
        assert len(model.week) == len(WEEKS)

    def test_trade_pairs_exclude_self(self, input_dir):
        model = build_default_model(input_dir)
        for a1, a2 in model.trade_pairs:
            assert a1 != a2
        expected = len(AREAS) * (len(AREAS) - 1)
        assert len(model.trade_pairs) == expected

    def test_has_key_variables(self, input_dir):
        model = build_default_model(input_dir)
        assert hasattr(model, "gene")
        assert hasattr(model, "on")
        assert hasattr(model, "startup")
        assert hasattr(model, "turnoff")
        assert hasattr(model, "ramp_up")
        assert hasattr(model, "storage")
        assert hasattr(model, "stored")
        assert hasattr(model, "rsv")
        assert hasattr(model, "hll")
        assert hasattr(model, "im")
        assert hasattr(model, "ex")
        assert hasattr(model, "exo_im")
        assert hasattr(model, "exo_ex")
        assert hasattr(model, "hcost")
        assert hasattr(model, "hcarb")

    def test_gene_var_size(self, input_dir):
        """gene variable has |areas| × |tec| × |hours| entries."""
        model = build_default_model(input_dir)
        expected = len(AREAS) * len(TEC) * len(HOURS)
        assert len(model.gene) == expected

    def test_has_key_constraints(self, input_dir):
        model = build_default_model(input_dir)
        expected_constraints = [
            "gene_vre_constraint", "gene_nmd_constraint",
            "on_capa_constraint", "gene_on_hmax_constraint", "gene_on_hmin_constraint",
            "yearly_maxON_constraint", "nuc_maxON_constraint",
            "on_off_constraint", "cons_startup_constraint", "cons_turnoff_constraint",
            "ramping_up_constraint",
            "stored_cap_constraint", "stor_in_constraint", "stor_out_constraint",
            "storing_constraint", "lake_res_constraint",
            "reserves_constraint", "no_FRR_contrib_constraint",
            "trade_bal_constraint", "icIM_constraint",
            "exoIM_constraint", "exoEX_constraint",
            "adequacy_constraint",
            "hcost_constraint", "hcarb_constraint",
        ]
        for name in expected_constraints:
            assert hasattr(model, name), f"Missing constraint: {name}"

    def test_adequacy_constraint_size(self, input_dir):
        """One adequacy constraint per (area, hour) pair."""
        model = build_default_model(input_dir)
        assert len(model.adequacy_constraint) == len(AREAS) * len(HOURS)

    def test_objective_exists(self, input_dir):
        model = build_default_model(input_dir)
        assert hasattr(model, "objective")
        assert model.objective.sense == pyo.minimize

    def test_dual_suffix(self, input_dir):
        model = build_default_model(input_dir)
        assert hasattr(model, "dual")

    def test_on_variable_initialized(self, input_dir):
        """on[a, thr, h] should be initialized to capa * maxaf (not zero)."""
        model = build_default_model(input_dir)
        # capa=10, maxaf=0.95 → initial value = 9.5
        for a in AREAS:
            for thr in THR:
                val = model.on[a, thr, HOURS[0]].value
                assert val == pytest.approx(9.5), (
                    f"on[{a}, {thr}, {HOURS[0]}] = {val}, expected 9.5"
                )


# ---------------------------------------------------------------------------
# Tests: Static thermal model
# ---------------------------------------------------------------------------


class TestStaticThermalModel:
    def test_builds_without_error(self, input_dir):
        model = build_static_thermal_model(input_dir)
        assert model is not None

    def test_no_on_variable(self, input_dir):
        """Static model should NOT have 'on', 'startup', 'turnoff', 'ramp_up'."""
        model = build_static_thermal_model(input_dir)
        assert not hasattr(model, "on")
        assert not hasattr(model, "startup")
        assert not hasattr(model, "turnoff")
        assert not hasattr(model, "ramp_up")

    def test_has_gene_capa_constraint(self, input_dir):
        """Static model uses gene_capa_constraint instead of on_capa_constraint."""
        model = build_static_thermal_model(input_dir)
        assert hasattr(model, "gene_capa_constraint")
        assert not hasattr(model, "on_capa_constraint")

    def test_sets_match_default(self, input_dir):
        """Both models should have the same sets."""
        model = build_static_thermal_model(input_dir)
        assert len(model.a) == len(AREAS)
        assert len(model.tec) == len(TEC)
        assert len(model.h) == len(HOURS)

    def test_objective_exists(self, input_dir):
        model = build_static_thermal_model(input_dir)
        assert hasattr(model, "objective")
        assert model.objective.sense == pyo.minimize


# ---------------------------------------------------------------------------
# Tests: Edge cases
# ---------------------------------------------------------------------------


class TestModelEdgeCases:
    def test_missing_input_file_raises(self, tmp_path):
        """Model build should fail if a required CSV is missing."""
        d = tmp_path / "inputs"
        d.mkdir()
        # Only create areas.csv — everything else is missing
        _write_1col(d / "areas.csv", AREAS)
        with pytest.raises(FileNotFoundError):
            build_default_model(tmp_path)

    def test_single_area_single_hour(self, tmp_path):
        """Model should build even with 1 area and 1 hour (minimal case)."""
        d = tmp_path / "inputs"
        d.mkdir()

        areas = ["FR"]
        exo_areas = ["NL"]
        hours = [0]
        months = ["202003"]
        weeks = ["202013"]
        vre = ["onshore"]
        thr = ["nuclear"]
        sto = ["lake_phs", "battery"]
        tec = ["nmd"] + vre + thr + sto
        frr = thr + sto
        no_frr = vre + ["nmd"]
        fuels = ["NUCLEAR"]

        _write_1col(d / "areas.csv", areas)
        _write_1col(d / "exo_areas.csv", exo_areas)
        _write_1col(d / "hours.csv", hours)
        _write_1col(d / "months.csv", months)
        _write_1col(d / "weeks.csv", weeks)
        _write_1col(d / "tec.csv", tec)
        _write_1col(d / "vre.csv", vre)
        _write_1col(d / "thr.csv", thr)
        _write_1col(d / "str_tec.csv", sto)
        _write_1col(d / "frr.csv", frr)
        _write_1col(d / "no_frr.csv", no_frr)

        # For cross-product CSVs, expand manually
        rows_ah = [(a, h) for a in areas for h in hours]
        _write_3col(d / "demand.csv",
                    [r[0] for r in rows_ah], [r[1] for r in rows_ah],
                    [50.0] * len(rows_ah))
        _write_3col(d / "nmd.csv",
                    [r[0] for r in rows_ah], [r[1] for r in rows_ah],
                    [5.0] * len(rows_ah))

        rows_exo_h = [(ea, h) for ea in exo_areas for h in hours]
        _write_3col(d / "exoPrices.csv",
                    [r[0] for r in rows_exo_h], [r[1] for r in rows_exo_h],
                    [40.0] * len(rows_exo_h))

        rows_vre = [(a, v, h) for a in areas for v in vre for h in hours]
        pd.DataFrame({
            "a": [r[0] for r in rows_vre], "vre": [r[1] for r in rows_vre],
            "h": [r[2] for r in rows_vre], "lf": [0.3] * len(rows_vre),
        }).to_csv(d / "vre_profiles.csv", index=False, header=False)

        rows_am = [(a, m) for a in areas for m in months]
        _write_3col(d / "lake_inflows.csv",
                    [r[0] for r in rows_am], [r[1] for r in rows_am],
                    [1.0] * len(rows_am))

        rows_at = [(a, t) for a in areas for t in tec]
        _write_3col(d / "capa.csv",
                    [r[0] for r in rows_at], [r[1] for r in rows_at],
                    [10.0] * len(rows_at))

        rows_as = [(a, s) for a in areas for s in sto]
        _write_3col(d / "capa_in.csv",
                    [r[0] for r in rows_as], [r[1] for r in rows_as],
                    [5.0] * len(rows_as))
        _write_3col(d / "stockMax.csv",
                    [r[0] for r in rows_as], [r[1] for r in rows_as],
                    [100.0] * len(rows_as))

        rows_athr = [(a, t) for a in areas for t in thr]
        _write_3col(d / "yEAF.csv",
                    [r[0] for r in rows_athr], [r[1] for r in rows_athr],
                    [0.85] * len(rows_athr))
        _write_3col(d / "maxAF.csv",
                    [r[0] for r in rows_athr], [r[1] for r in rows_athr],
                    [0.95] * len(rows_athr))

        rows_aw = [(a, w) for a in areas for w in weeks]
        _write_3col(d / "nucMaxAF.csv",
                    [r[0] for r in rows_aw], [r[1] for r in rows_aw],
                    [0.9] * len(rows_aw))

        _write_3col(d / "hMaxOut.csv",
                    [r[0] for r in rows_am], [r[1] for r in rows_am],
                    [0.8] * len(rows_am))
        _write_3col(d / "hMaxIn.csv",
                    [r[0] for r in rows_am], [r[1] for r in rows_am],
                    [0.8] * len(rows_am))

        # No trade pairs for single area
        pd.DataFrame(columns=["a", "b", "c"]).to_csv(
            d / "links.csv", index=False, header=False)

        rows_exo_trade = [(a, ea) for a in areas for ea in exo_areas]
        _write_3col(d / "exo_IM.csv",
                    [r[0] for r in rows_exo_trade], [r[1] for r in rows_exo_trade],
                    [3.0] * len(rows_exo_trade))
        _write_3col(d / "exo_EX.csv",
                    [r[0] for r in rows_exo_trade], [r[1] for r in rows_exo_trade],
                    [3.0] * len(rows_exo_trade))

        _write_2col(d / "rsv_req.csv", vre, [0.05] * len(vre))
        _write_2col(d / "efficiency.csv", thr, [0.45] * len(thr))
        _write_2col(d / "eff50.csv", thr, [0.40] * len(thr))
        _write_2col(d / "co2_factor.csv", thr, [0.0] * len(thr))
        _write_2col(d / "co2_price.csv", thr, [25.0] * len(thr))
        _write_2col(d / "nonFuel_vOM.csv", thr, [2.0] * len(thr))
        _write_2col(d / "su_fixedCost.csv", thr, [50.0] * len(thr))
        _write_2col(d / "su_fuelCons.csv", thr, [3.0] * len(thr))
        _write_2col(d / "ramp_fuelCons.csv", thr, [0.5] * len(thr))
        _write_2col(d / "minSG.csv", thr, [0.4] * len(thr))
        _write_2col(d / "minTimeON.csv", thr, [1] * len(thr))
        _write_2col(d / "minTimeOFF.csv", thr, [1] * len(thr))
        _write_2col(d / "thr_fuel.csv", thr, fuels)
        _write_2col(d / "fuel_price.csv", thr, [5.0] * len(thr))

        rows_fm = [(f, m) for f in fuels for m in months]
        _write_3col(d / "fuel_timeFactor.csv",
                    [r[0] for r in rows_fm], [r[1] for r in rows_fm],
                    [1.0] * len(rows_fm))
        rows_fa = [(f, a) for f in fuels for a in areas]
        _write_3col(d / "fuel_areaFactor.csv",
                    [r[0] for r in rows_fa], [r[1] for r in rows_fa],
                    [0.0] * len(rows_fa))

        _write_2col(d / "str_vOM.csv", sto, [0.3] * len(sto))
        _write_2col(d / "hour_month.csv", hours, [months[0]] * len(hours))
        _write_2col(d / "hour_week.csv", hours, [weeks[0]] * len(hours))

        model = build_default_model(tmp_path)
        assert len(model.a) == 1
        assert len(model.h) == 1
        assert len(model.trade_pairs) == 0
