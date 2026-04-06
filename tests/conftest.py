from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── Session-scoped fixtures ──


@pytest.fixture(scope="session")
def project_root():
    """Return the project root directory."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def areas():
    """Minimal 2-area list for fast tests."""
    return ["FR", "DE"]


@pytest.fixture(scope="session")
def exo_areas():
    """Minimal exogenous areas."""
    return ["NL", "AT"]


@pytest.fixture
def sample_hourly_series():
    """72-hour UTC-naive hourly series with some NaNs for gap-fill tests."""
    idx = pd.date_range("2020-01-01", periods=72, freq="h")
    vals = np.random.default_rng(42).uniform(10, 50, size=72)
    return pd.Series(vals, index=idx)


# ── Shared constants for minimal model tests ──

_AREAS = ["FR", "DE"]
_EXO_AREAS = ["NL"]
_HOURS = [0, 1, 2]
_MONTHS = ["202003"]
_WEEKS = ["202013"]
_VRE = ["onshore", "solar"]
_THR = ["nuclear", "gas_ccgt1G"]
_STO = ["lake_phs", "battery"]
_TEC = ["nmd"] + _VRE + _THR + _STO
_FRR = _THR + _STO
_NO_FRR = _VRE + ["nmd"]
_FUELS = ["NUCLEAR", "GAS"]


# ── CSV write helpers ──


def _write_1col(path, values):
    """Write a single-column headerless CSV."""
    pd.DataFrame(values).to_csv(path, index=False, header=False)


def _write_2col(path, col1, col2):
    pd.DataFrame({"a": col1, "b": col2}).to_csv(path, index=False, header=False)


def _write_3col(path, c1, c2, c3):
    pd.DataFrame({"a": c1, "b": c2, "c": c3}).to_csv(path, index=False, header=False)


# ── Input dir builder ──


def _build_input_dir(base_path):
    """Create a minimal inputs/ directory under base_path with all required CSVs.

    Creates base_path/inputs/ and populates it with all CSVs needed by the Pyomo
    model (both standard and static_thermal variants). Returns base_path.
    """
    base_path = Path(base_path)
    d = base_path / "inputs"
    d.mkdir(parents=True, exist_ok=True)

    n_h = len(_HOURS)

    # ── Sets ──
    _write_1col(d / "areas.csv", _AREAS)
    _write_1col(d / "exo_areas.csv", _EXO_AREAS)
    _write_1col(d / "hours.csv", _HOURS)
    _write_1col(d / "months.csv", _MONTHS)
    _write_1col(d / "weeks.csv", _WEEKS)
    _write_1col(d / "tec.csv", _TEC)
    _write_1col(d / "vre.csv", _VRE)
    _write_1col(d / "thr.csv", _THR)
    _write_1col(d / "str_tec.csv", _STO)
    _write_1col(d / "frr.csv", _FRR)
    _write_1col(d / "no_frr.csv", _NO_FRR)

    # ── Time-series ──
    rows_ah = [(a, h) for a in _AREAS for h in _HOURS]
    _write_3col(
        d / "demand.csv",
        [r[0] for r in rows_ah],
        [r[1] for r in rows_ah],
        [50.0] * len(rows_ah),
    )
    _write_3col(
        d / "nmd.csv",
        [r[0] for r in rows_ah],
        [r[1] for r in rows_ah],
        [5.0] * len(rows_ah),
    )

    rows_exo_h = [(ea, h) for ea in _EXO_AREAS for h in _HOURS]
    _write_3col(
        d / "exoPrices.csv",
        [r[0] for r in rows_exo_h],
        [r[1] for r in rows_exo_h],
        [40.0] * len(rows_exo_h),
    )

    rows_vre = [(a, v, h) for a in _AREAS for v in _VRE for h in _HOURS]
    pd.DataFrame(
        {
            "a": [r[0] for r in rows_vre],
            "vre": [r[1] for r in rows_vre],
            "h": [r[2] for r in rows_vre],
            "lf": [0.3] * len(rows_vre),
        }
    ).to_csv(d / "vre_profiles.csv", index=False, header=False)

    rows_am = [(a, m) for a in _AREAS for m in _MONTHS]
    _write_3col(
        d / "lake_inflows.csv",
        [r[0] for r in rows_am],
        [r[1] for r in rows_am],
        [0.001] * len(rows_am),
    )

    # ── Capacity ──
    rows_at = [(a, t) for a in _AREAS for t in _TEC]
    _write_3col(
        d / "capa.csv",
        [r[0] for r in rows_at],
        [r[1] for r in rows_at],
        [10.0] * len(rows_at),
    )

    rows_as = [(a, s) for a in _AREAS for s in _STO]
    _write_3col(
        d / "capa_in.csv",
        [r[0] for r in rows_as],
        [r[1] for r in rows_as],
        [5.0] * len(rows_as),
    )
    _write_3col(
        d / "stockMax.csv",
        [r[0] for r in rows_as],
        [r[1] for r in rows_as],
        [100.0] * len(rows_as),
    )

    rows_athr = [(a, t) for a in _AREAS for t in _THR]
    _write_3col(
        d / "yEAF.csv",
        [r[0] for r in rows_athr],
        [r[1] for r in rows_athr],
        [0.85] * len(rows_athr),
    )
    _write_3col(
        d / "maxAF.csv",
        [r[0] for r in rows_athr],
        [r[1] for r in rows_athr],
        [0.95] * len(rows_athr),
    )

    rows_aw = [(a, w) for a in _AREAS for w in _WEEKS]
    _write_3col(
        d / "nucMaxAF.csv",
        [r[0] for r in rows_aw],
        [r[1] for r in rows_aw],
        [0.9] * len(rows_aw),
    )

    _write_3col(
        d / "hMaxOut.csv",
        [r[0] for r in rows_am],
        [r[1] for r in rows_am],
        [0.8] * len(rows_am),
    )
    _write_3col(
        d / "hMaxIn.csv",
        [r[0] for r in rows_am],
        [r[1] for r in rows_am],
        [0.8] * len(rows_am),
    )

    # ── Trade ──
    trade_pairs = [(a1, a2) for a1 in _AREAS for a2 in _AREAS if a1 != a2]
    _write_3col(
        d / "links.csv",
        [r[0] for r in trade_pairs],
        [r[1] for r in trade_pairs],
        [5.0] * len(trade_pairs),
    )

    rows_exo_trade = [(a, ea) for a in _AREAS for ea in _EXO_AREAS]
    _write_3col(
        d / "exo_IM.csv",
        [r[0] for r in rows_exo_trade],
        [r[1] for r in rows_exo_trade],
        [3.0] * len(rows_exo_trade),
    )
    _write_3col(
        d / "exo_EX.csv",
        [r[0] for r in rows_exo_trade],
        [r[1] for r in rows_exo_trade],
        [3.0] * len(rows_exo_trade),
    )

    # ── Reserves ──
    _write_2col(d / "rsv_req.csv", _VRE, [0.05] * len(_VRE))

    # ── Thermal specs ──
    _write_2col(d / "efficiency.csv", _THR, [0.45, 0.55])
    _write_2col(d / "eff50.csv", _THR, [0.40, 0.50])
    _write_2col(d / "co2_factor.csv", _THR, [0.0, 56.1])
    _write_2col(d / "co2_price.csv", _THR, [25.0, 25.0])
    _write_2col(d / "nonFuel_vOM.csv", _THR, [2.0, 1.5])
    _write_2col(d / "su_fixedCost.csv", _THR, [50.0, 30.0])
    _write_2col(d / "su_fuelCons.csv", _THR, [3.0, 2.0])
    _write_2col(d / "ramp_fuelCons.csv", _THR, [0.5, 0.3])
    _write_2col(d / "minSG.csv", _THR, [0.4, 0.3])
    _write_2col(d / "minTimeON.csv", _THR, [1, 1])
    _write_2col(d / "minTimeOFF.csv", _THR, [1, 1])

    # ── Fuel prices ──
    _write_2col(d / "thr_fuel.csv", _THR, _FUELS)
    _write_2col(d / "fuel_price.csv", _THR, [5.0, 20.0])

    rows_fm = [(f, m) for f in _FUELS for m in _MONTHS]
    _write_3col(
        d / "fuel_timeFactor.csv",
        [r[0] for r in rows_fm],
        [r[1] for r in rows_fm],
        [1.0] * len(rows_fm),
    )

    rows_fa = [(f, a) for f in _FUELS for a in _AREAS]
    _write_3col(
        d / "fuel_areaFactor.csv",
        [r[0] for r in rows_fa],
        [r[1] for r in rows_fa],
        [0.0] * len(rows_fa),
    )

    # ── Storage ──
    _write_2col(d / "str_vOM.csv", _STO, [0.5, 0.3])

    # ── Hour mappings ──
    _write_2col(d / "hour_month.csv", _HOURS, [_MONTHS[0]] * n_h)
    _write_2col(d / "hour_week.csv", _HOURS, [_WEEKS[0]] * n_h)

    return base_path


# ── Scenario dir builder ──


def _make_scenario_dir(path, areas=None, exo_areas=None):
    """Create a minimal scenario CSV directory with all 13 required CSV files.

    Args:
        path: Directory to populate (created if absent).
        areas: List of modeled area codes (default: _AREAS).
        exo_areas: List of exogenous area codes (default: _EXO_AREAS).

    Returns path.
    """
    if areas is None:
        areas = _AREAS
    if exo_areas is None:
        exo_areas = _EXO_AREAS

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    thr = _THR
    vre = _VRE
    sto = _STO

    # thr_specs.csv — one row per thermal tech, all params as columns
    pd.DataFrame(
        {
            "tec": thr,
            "fuel_price": [5.0, 20.0],
            "efficiency": [0.33, 0.55],
            "eff50": [0.30, 0.50],
            "co2_price": [25.0, 25.0],
            "co2_factor": [0.0, 56.1],
            "nonFuel_vOM": [2.0, 1.5],
            "su_fuelCons": [3.0, 2.0],
            "su_fixedCost": [50.0, 30.0],
            "minTimeOFF": [1, 1],
            "minTimeON": [1, 1],
            "minSG": [0.4, 0.3],
            "ramp_fuelCons": [0.5, 0.3],
            "frr": [True, True],
            "thr_fuel": ["URANIUM", "GAS"],
        }
    ).to_csv(path / "thr_specs.csv", index=False)

    pd.DataFrame({"tec": vre, "rsv_req": [0.027, 0.027]}).to_csv(
        path / "rsv_req.csv", index=False
    )

    pd.DataFrame({"tec": sto, "str_vOM": [5.0, 2.1]}).to_csv(
        path / "str_vOM.csv", index=False
    )

    # Wide-format capacity tables: rows = tec, columns = areas
    all_tec = vre + thr + sto
    for fname, tec_list, val in [
        ("capa", all_tec, 10.0),
        ("maxAF", thr, 0.95),
        ("yEAF", thr, 0.85),
        ("capa_in", sto, 5.0),
        ("stockMax", sto, 100.0),
    ]:
        df = pd.DataFrame({"tec": tec_list})
        for a in areas:
            df[a] = val
        df.to_csv(path / f"{fname}.csv", index=False)

    # links.csv: exporter + one column per area importer
    links_df = pd.DataFrame({"exporter": areas})
    for a in areas:
        links_df[a] = 5.0
    links_df.to_csv(path / "links.csv", index=False)

    # exo_EX.csv: exporter (area) + one column per exo area importer
    exo_ex_df = pd.DataFrame({"exporter": areas})
    for ea in exo_areas:
        exo_ex_df[ea] = 3.0
    exo_ex_df.to_csv(path / "exo_EX.csv", index=False)

    # exo_IM.csv: importer (area) + one column per exo area exporter
    exo_im_df = pd.DataFrame({"importer": areas})
    for ea in exo_areas:
        exo_im_df[ea] = 3.0
    exo_im_df.to_csv(path / "exo_IM.csv", index=False)

    # fuel_timeFactor.csv: month (1-12) + fuel columns
    ftf_df = pd.DataFrame({"month": range(1, 13)})
    for fuel in ["URANIUM", "GAS"]:
        ftf_df[fuel] = 1.0
    ftf_df.to_csv(path / "fuel_timeFactor.csv", index=False)

    # fuel_areaFactor.csv: area + fuel columns
    faf_df = pd.DataFrame({"area": areas})
    for fuel in ["URANIUM", "GAS"]:
        faf_df[fuel] = 0.0
    faf_df.to_csv(path / "fuel_areaFactor.csv", index=False)

    return path


# ── Shared fixtures ──


@pytest.fixture
def input_dir(tmp_path):
    """Create a minimal inputs/ directory with all required CSVs for model construction."""
    return _build_input_dir(tmp_path)


@pytest.fixture
def scenario_dir(tmp_path):
    """Create a minimal scenario CSV directory."""
    p = tmp_path / "scenarios" / "test_scenario"
    _make_scenario_dir(p)
    return p


@pytest.fixture
def solved_run(tmp_path):
    """Build and solve a minimal static_thermal model. Returns (model, run_dir).

    run_dir = tmp_path, which contains inputs/ (from _build_input_dir) and outputs/.
    The model is solved with HiGHS (IPM + crossover) so dual variables are available.
    """
    import pyomo.environ  # noqa: F401 — registers solver plugins
    from pyomo.opt import SolverFactory, TerminationCondition

    from eoles_dispatch.models import build_static_thermal_model

    _build_input_dir(tmp_path)
    model = build_static_thermal_model(tmp_path)
    opt = SolverFactory("appsi_highs")
    opt.highs_options["solver"] = "ipm"
    opt.highs_options["run_crossover"] = "on"
    results = opt.solve(model)
    assert results.solver.termination_condition == TerminationCondition.optimal
    (tmp_path / "outputs").mkdir(exist_ok=True)
    return (model, tmp_path)
