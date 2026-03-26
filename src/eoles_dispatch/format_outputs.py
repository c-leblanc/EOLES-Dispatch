"""Extract and format model results into CSV files."""

import datetime
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyomo.environ as pyo

from collections import defaultdict

from .config import TRLOSS, MODEL_TO_AGG


def _ensure_output_dir(run_dir):
    output_dir = Path(run_dir) / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _extract_duals_bulk(model, constraint, keys):
    """Extract dual values for a constraint in bulk.

    Uses Pyomo's internal suffix dict to read all duals at once instead of
    looping through Python one-by-one, which is orders of magnitude faster
    on large models.

    Args:
        model: Solved Pyomo model with dual suffix.
        constraint: Indexed constraint component (e.g. model.adequacy_constraint).
        keys: Iterable of index tuples to extract.

    Returns:
        List of dual values in the same order as keys.
    """
    dual_dict = dict(model.dual)
    return [dual_dict.get(constraint[k], 0.0) for k in keys]


def _extract_var_values_bulk(var):
    """Extract all values from a Pyomo Var as a dict, in bulk.

    Much faster than repeated pyo.value() calls.
    """
    return {k: v.value for k, v in var.items()}


def report_prices(model, run_dir):
    """Extract hourly marginal prices (dual of adequacy constraint) for each area."""
    output_dir = _ensure_output_dir(run_dir)

    areas = sorted(model.a)
    hours = sorted(model.h)

    # Build all (area, hour) keys and extract duals in bulk
    keys = [(a, h) for a in areas for h in hours]
    duals = _extract_duals_bulk(model, model.adequacy_constraint, keys)

    # Reshape into DataFrame: rows=hours, columns=areas
    n_hours = len(hours)
    data = {}
    for i, a in enumerate(areas):
        data[a] = duals[i * n_hours:(i + 1) * n_hours]

    prices = pd.DataFrame(data, index=hours)
    prices.index.name = "hour"
    prices.to_csv(output_dir / "prices.csv", index=True)


def _safe_gene_values(gene_vals, tec, n_rows):
    """Extract generation values for a technology, returning zeros if it doesn't exist."""
    keys = [k for k in gene_vals if k[1] == tec]
    if not keys:
        return np.zeros(n_rows)
    return np.array([gene_vals[k] or 0.0 for k in sorted(keys)])


def _safe_var_values(var_vals, tec, n_rows):
    """Extract variable values for a technology (storage, on), returning zeros if missing."""
    keys = [k for k in var_vals if k[1] == tec]
    if not keys:
        return np.zeros(n_rows)
    return np.array([var_vals[k] or 0.0 for k in sorted(keys)])


def report_production(model, run_dir):
    """Extract hourly production by technology and area."""
    output_dir = _ensure_output_dir(run_dir)
    n_rows = len(model.a) * len(model.h)

    gene_vals = _extract_var_values_bulk(model.gene)
    stor_vals = _extract_var_values_bulk(model.storage)
    im_vals = _extract_var_values_bulk(model.im)
    ex_vals = _extract_var_values_bulk(model.ex)
    exo_im_vals = _extract_var_values_bulk(model.exo_im)
    exo_ex_vals = _extract_var_values_bulk(model.exo_ex)

    areas = sorted(model.a)
    hours = sorted(model.h)

    production = pd.DataFrame({"area": np.repeat(areas, len(hours), axis=0),
                                "hour": hours * len(areas)})

    # Aggregate generation using MODEL_TO_AGG: group model techs by agg name and sum
    agg_groups = defaultdict(lambda: np.zeros(n_rows))
    for model_tec, agg_name in MODEL_TO_AGG.items():
        agg_groups[agg_name] += _safe_gene_values(gene_vals, model_tec, n_rows)
    for agg_name, values in agg_groups.items():
        production[agg_name] = values.tolist()

    # Storage charging (negative = consumption)
    production["phs_in"] = (-_safe_var_values(stor_vals, "lake_phs", n_rows)).tolist()
    production["battery_in"] = (-_safe_var_values(stor_vals, "battery", n_rows)).tolist()

    # Net imports per area: sum over all trading partners
    areas = list(model.a)
    exo_areas = list(model.exo_a)
    hours = sorted(model.h)
    n_hours = len(hours)
    net_imports_list = []
    for a in areas:
        area_net = np.zeros(n_hours)
        # Imports from other modeled areas
        for trader in areas:
            if trader == a:
                continue
            area_net += np.array([im_vals.get((a, trader, h), 0) or 0.0 for h in hours])
            area_net -= np.array([ex_vals.get((a, trader, h), 0) or 0.0 for h in hours])
        # Imports from exogenous areas
        for trader in exo_areas:
            area_net += np.array([exo_im_vals.get((a, trader, h), 0) or 0.0 for h in hours])
            area_net -= np.array([exo_ex_vals.get((a, trader, h), 0) or 0.0 for h in hours])
        net_imports_list.append(area_net)
    production.net_imports = np.concatenate(net_imports_list).tolist()

    production = production.set_index(["area", "hour"])

    # Join demand
    demand = pd.read_csv(
        Path(run_dir) / "inputs" / "demand.csv",
        header=None, names=["area", "hour", "demand"],
    ).set_index(["area", "hour"]).squeeze(axis=1)
    production = production.join(demand)
    production.to_csv(output_dir / "production.csv", index=True)


def report_capa_on(model, run_dir):
    """Extract hourly online thermal capacity by technology and area."""
    if not hasattr(model, "on"):
        import logging
        logging.getLogger(__name__).warning(
            "report_capa_on skipped: model has no 'on' variable (static_thermal model)"
        )
        return

    output_dir = _ensure_output_dir(run_dir)
    n_rows = len(model.a) * len(model.h)
    on_vals = _extract_var_values_bulk(model.on)
    thr_tecs = list(model.thr)

    capa_on = pd.DataFrame(index=range(n_rows))
    areas = sorted(model.a)
    hours = sorted(model.h)
    capa_on["area"] = np.repeat(areas, len(hours), axis=0)
    capa_on["hour"] = hours * len(areas)
    for thr in thr_tecs:
        capa_on[thr] = _safe_var_values(on_vals, thr, n_rows).tolist()
    capa_on = capa_on.set_index(["area", "hour"])
    capa_on.to_csv(output_dir / "capa_on.csv", index=True)


def report_FRtrade(model, run_dir):
    """Extract France's hourly net imports from each trading partner."""
    output_dir = _ensure_output_dir(run_dir)
    hours = sorted(model.h)
    partners = sorted(a for a in model.a if a != "FR")
    im_vals = _extract_var_values_bulk(model.im)
    ex_vals = _extract_var_values_bulk(model.ex)

    FRtrade = pd.DataFrame({"hour": hours})
    for a in partners:
        im_arr = np.array([im_vals.get(("FR", a, h), 0) or 0.0 for h in hours])
        ex_arr = np.array([ex_vals.get(("FR", a, h), 0) or 0.0 for h in hours])
        FRtrade[a] = (im_arr * (1 - TRLOSS) - ex_arr).tolist()
    FRtrade = FRtrade.set_index("hour")
    FRtrade.to_csv(output_dir / "FRtrade.csv", index=True)


def write_log(run_dir, model, run_name, scenario, year, start_time, exec_str, **params):
    """Write a log file summarizing the run parameters and results."""
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)

    total_cost = pyo.value(model.objective)

    # Extract CO2 in bulk instead of looping through 61k pyo.value() calls
    hcarb_values = _extract_var_values_bulk(model.hcarb)
    total_co2 = sum(v for v in hcarb_values.values() if v is not None)

    # Scenario file last modification date
    scenario_date = ""
    scenario_file = run_path.parent.parent / "scenarios" / f"{scenario}.xlsx"
    if scenario_file.exists():
        mod_time = os.path.getmtime(scenario_file)
        scenario_date = datetime.datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M:%S")

    # Cost unit: hcost is in kEUR/h (EUR/MWh × GW), summed over all hours → kEUR total.
    # Convert to billion EUR for display.
    total_cost_beur = total_cost / 1e6

    with open(run_path / f"_log_{run_name}.txt", "w") as f:
        f.write(f"RUN NAME = {run_name}\n")
        f.write(f"Scenario file: {scenario}\n")
        if scenario_date:
            f.write(f"\t last modified on {scenario_date}\n")
        f.write(f"Year simulated: {year}\n\n")
        f.write(f"Total dispatch cost: {total_cost_beur:.2f} bEUR\n")
        f.write(f"Total CO2 emissions: {total_co2 / 1e6:.2f} MtCO2\n\n")
        f.write(f"Started running at: {time.asctime(start_time)}\n")
        f.write(f"Execution time: {exec_str}\n")
