"""Export exhaustive model diagnostics for post-solve analysis.

Writes all Pyomo variable values and constraint duals to CSV files under
runs/<name>/diagnostics/, plus a JSON summary. Intended as input to an AI
diagnostic tool, not for human visualization.

Output structure:
    diagnostics/
        vars/
            gene.csv, on.csv, startup.csv, turnoff.csv, ramp_up.csv,
            storage.csv, stored.csv, rsv.csv, hll.csv,
            im.csv, ex.csv, exo_im.csv, exo_ex.csv,
            hcost.csv, hcarb.csv
        duals/
            adequacy.csv, gene_vre.csv, gene_nmd.csv, on_capa.csv,
            gene_on_hmax.csv, gene_on_hmin.csv, yearly_maxON.csv,
            nuc_maxON.csv, on_off.csv, ramping_up.csv,
            stored_cap.csv, stor_in.csv, stor_out.csv, storing.csv,
            lake_res.csv, reserves.csv, trade_bal.csv, icIM.csv,
            exoIM.csv, exoEX.csv
        _summary.json
"""

import json
import logging
from pathlib import Path

import pandas as pd
import pyomo.environ as pyo

logger = logging.getLogger(__name__)


# ── High-level entry point ──


def export_all_diagnostics(model, run_dir):
    """Export all variables and duals from a solved Pyomo model to diagnostics/.

    Args:
        model: Solved Pyomo ConcreteModel (with dual suffix populated).
        run_dir: Path to the run directory.
    """
    diag_dir = Path(run_dir) / "diagnostics"
    vars_dir = diag_dir / "vars"
    duals_dir = diag_dir / "duals"
    vars_dir.mkdir(parents=True, exist_ok=True)
    duals_dir.mkdir(parents=True, exist_ok=True)

    logger.info("  Exporting full diagnostics...")

    # ── Variables ──
    _VARS = [
        "gene",
        "on",
        "startup",
        "turnoff",
        "ramp_up",
        "storage",
        "stored",
        "rsv",
        "hll",
        "im",
        "ex",
        "exo_im",
        "exo_ex",
        "hcost",
        "hcarb",
    ]
    var_stats = {}
    for name in _VARS:
        if not hasattr(model, name):
            logger.debug(f"    Variable '{name}' not in model, skipping.")
            continue
        df = _var_to_df(getattr(model, name))
        if df.empty:
            continue
        df.to_csv(vars_dir / f"{name}.csv", index=False)
        var_stats[name] = _numeric_stats(df["value"])
        logger.info(f"    vars/{name}.csv ({len(df)} rows)")

    # ── Duals ──
    _CONSTRAINTS = [
        "adequacy_constraint",
        "gene_vre_constraint",
        "gene_nmd_constraint",
        "on_capa_constraint",
        "gene_on_hmax_constraint",
        "gene_on_hmin_constraint",
        "yearly_maxON_constraint",
        "nuc_maxON_constraint",
        "on_off_constraint",
        "cons_startup_constraint",
        "cons_turnoff_constraint",
        "ramping_up_constraint",
        "stored_cap_constraint",
        "stor_in_constraint",
        "stor_out_constraint",
        "storing_constraint",
        "lake_res_constraint",
        "reserves_constraint",
        "no_FRR_contrib_constraint",
        "trade_bal_constraint",
        "icIM_constraint",
        "exoIM_constraint",
        "exoEX_constraint",
    ]
    dual_stats = {}
    for name in _CONSTRAINTS:
        if not hasattr(model, name):
            logger.debug(f"    Constraint '{name}' not in model, skipping.")
            continue
        df = _dual_to_df(model, getattr(model, name))
        if df.empty:
            continue
        short = name.replace("_constraint", "")
        df.to_csv(duals_dir / f"{short}.csv", index=False)
        dual_stats[name] = _numeric_stats(df["dual"])
        logger.info(f"    duals/{short}.csv ({len(df)} rows)")

    # ── Summary JSON ──
    summary = _build_summary(model, var_stats, dual_stats)
    with open(diag_dir / "_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("    _summary.json")
    logger.info(f"  Diagnostics written to {diag_dir}")


# ── Helpers ──


def _var_to_df(var) -> pd.DataFrame:
    """Flatten a Pyomo Var to a DataFrame.

    Columns: idx0[, idx1, idx2, ...], value.
    Scalar keys are wrapped in a tuple for uniform handling.
    None values (fixed or inactive variables) are kept as NaN.
    """
    rows = []
    for k, v in var.items():
        key = k if isinstance(k, tuple) else (k,)
        rows.append((*key, v.value))
    if not rows:
        return pd.DataFrame()
    n_idx = len(rows[0]) - 1
    idx_cols = [f"idx{i}" for i in range(n_idx)]
    return pd.DataFrame(rows, columns=idx_cols + ["value"])


def _dual_to_df(model, constraint) -> pd.DataFrame:
    """Flatten constraint duals to a DataFrame.

    Columns: idx0[, idx1, idx2, ...], dual.
    Missing duals (non-active constraints) default to 0.0.
    """
    dual_dict = dict(model.dual)
    rows = []
    for k, c in constraint.items():
        key = k if isinstance(k, tuple) else (k,)
        rows.append((*key, dual_dict.get(c, 0.0)))
    if not rows:
        return pd.DataFrame()
    n_idx = len(rows[0]) - 1
    idx_cols = [f"idx{i}" for i in range(n_idx)]
    return pd.DataFrame(rows, columns=idx_cols + ["dual"])


def _numeric_stats(series: pd.Series) -> dict:
    """Compute basic descriptive statistics for a numeric series."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {}
    nonzero = s[s != 0]
    return {
        "count": int(len(s)),
        "nonzero_count": int(len(nonzero)),
        "min": float(s.min()),
        "max": float(s.max()),
        "mean": float(s.mean()),
        "sum": float(s.sum()),
    }


def _build_summary(model, var_stats: dict, dual_stats: dict) -> dict:
    """Build a JSON-serialisable summary dict for the solved model."""
    total_cost = pyo.value(model.objective)

    # CO2: sum hcarb if present
    total_co2 = None
    if hasattr(model, "hcarb"):
        total_co2 = float(sum(v.value for v in model.hcarb.values() if v.value is not None))

    # Sets info
    sets_info = {}
    for sname in ("a", "exo_a", "h", "tec", "vre", "thr", "sto", "frr", "no_frr", "week", "month"):
        if hasattr(model, sname):
            s = getattr(model, sname)
            members = [m.item() if hasattr(m, "item") else m for m in sorted(s)]
            sets_info[sname] = {"size": len(members), "members": members}

    return {
        "objective_kEUR": float(total_cost),
        "objective_bEUR": float(total_cost) / 1e6,
        "total_co2_tCO2": total_co2,
        "total_co2_MtCO2": total_co2 / 1e6 if total_co2 is not None else None,
        "sets": sets_info,
        "variable_stats": var_stats,
        "dual_stats": dual_stats,
    }
