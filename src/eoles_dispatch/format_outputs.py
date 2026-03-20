"""Extract and format model results into CSV files."""

import datetime
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyomo.environ as pyo

from .config import TRLOSS


def _ensure_output_dir(run_dir):
    output_dir = Path(run_dir) / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def report_prices(model, run_dir):
    """Extract hourly marginal prices (dual of adequacy constraint) for each area."""
    output_dir = _ensure_output_dir(run_dir)
    rows = []
    for hour in model.h:
        row = {"hour": hour}
        for a in model.a:
            row[a] = 1_000_000 * model.dual[model.adequacy_constraint[a, hour]]
        rows.append(row)
    prices = pd.DataFrame(rows)
    prices = prices.set_index("hour")
    prices.to_csv(output_dir / "prices.csv", index=True)


def report_production(model, run_dir):
    """Extract hourly production by technology and area."""
    output_dir = _ensure_output_dir(run_dir)
    production = pd.DataFrame(
        index=range(len(model.a) * len(model.h)),
        columns=["area", "hour", "nmd", "pv", "river", "nuclear", "lake_phs",
                 "wind", "coal", "gas", "oil", "battery", "phs_in", "battery_in",
                 "net_imports", "net_exo_imports"],
    )
    production.area = np.repeat(list(model.a._values), len(model.h), axis=0)
    production.hour = list(model.h._values) * len(model.a)
    production.nmd = pyo.value(model.gene[:, "nmd", :])
    production.pv = pyo.value(model.gene[:, "pv", :])
    production.river = pyo.value(model.gene[:, "river", :])
    production.nuclear = pyo.value(model.gene[:, "nuclear", :])
    production.lake_phs = pyo.value(model.gene[:, "lake_phs", :])
    production.wind = (
        np.array(pyo.value(model.gene[:, "onshore", :]))
        + np.array(pyo.value(model.gene[:, "offshore", :]))
    ).tolist()
    production.coal = (
        np.array(pyo.value(model.gene[:, "coal_SA", :]))
        + np.array(pyo.value(model.gene[:, "coal_1G", :]))
        + np.array(pyo.value(model.gene[:, "lignite", :]))
    ).tolist()
    production.gas = (
        np.array(pyo.value(model.gene[:, "gas_ccgt1G", :]))
        + np.array(pyo.value(model.gene[:, "gas_ccgt2G", :]))
        + np.array(pyo.value(model.gene[:, "gas_ccgtSA", :]))
        + np.array(pyo.value(model.gene[:, "gas_ocgtSA", :]))
    ).tolist()
    production.oil = pyo.value(model.gene[:, "oil_light", :])
    production.battery = pyo.value(model.gene[:, "battery", :])
    production.phs_in = (-np.array(pyo.value(model.storage[:, "lake_phs", :]))).tolist()
    production.battery_in = (-np.array(pyo.value(model.storage[:, "battery", :]))).tolist()
    production.net_imports = (
        sum(np.array(pyo.value(model.im[:, trader, :])) - np.array(pyo.value(model.ex[:, trader, :])) for trader in model.a)
        + sum(np.array(pyo.value(model.exo_im[:, trader, :])) - np.array(pyo.value(model.exo_ex[:, trader, :])) for trader in model.exo_a)
    ).tolist()
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
    output_dir = _ensure_output_dir(run_dir)
    capa_on = pd.DataFrame(
        index=range(len(model.a) * len(model.h)),
        columns=["area", "hour", "nuclear", "coal_SA", "coal_1G", "coal_LI",
                 "gas_ccgt1G", "gas_ccgt2G", "gas_ccgtSA", "gas_ocgtSA", "oil"],
    )
    capa_on.area = np.repeat(list(model.a._values), len(model.h), axis=0)
    capa_on.hour = list(model.h._values) * len(model.a)
    capa_on.nuclear = pyo.value(model.on[:, "nuclear", :])
    capa_on.coal_SA = pyo.value(model.on[:, "coal_SA", :])
    capa_on.coal_1G = pyo.value(model.on[:, "coal_1G", :])
    capa_on.coal_LI = pyo.value(model.on[:, "lignite", :])
    capa_on.gas_ccgt1G = pyo.value(model.on[:, "gas_ccgt1G", :])
    capa_on.gas_ccgt2G = pyo.value(model.on[:, "gas_ccgt2G", :])
    capa_on.gas_ccgtSA = pyo.value(model.on[:, "gas_ccgtSA", :])
    capa_on.gas_ocgtSA = pyo.value(model.on[:, "gas_ocgtSA", :])
    capa_on.oil = pyo.value(model.on[:, "oil_light", :])
    capa_on = capa_on.set_index(["area", "hour"])
    capa_on.to_csv(output_dir / "capa_on.csv", index=True)


def report_FRtrade(model, run_dir):
    """Extract France's hourly net imports from each trading partner."""
    output_dir = _ensure_output_dir(run_dir)
    FRtrade = pd.DataFrame(index=range(len(model.h)), columns=["hour"] + list(model.a._values))
    FRtrade.hour = list(model.h._values)
    for a in model.a:
        FRtrade[a] = (
            np.array(pyo.value(model.im["FR", a, :])) * (1 - TRLOSS)
            - np.array(pyo.value(model.ex["FR", a, :]))
        ).tolist()
    FRtrade = FRtrade.set_index("hour")
    FRtrade.to_csv(output_dir / "FRtrade.csv", index=True)


def write_log(run_dir, model, run_name, scenario, year, start_time, exec_time, **params):
    """Write a log file summarizing the run parameters and results."""
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)

    total_cost = pyo.value(model.objective)
    total_co2 = sum(pyo.value(model.hcarb[a, h]) for a in model.a for h in model.h)

    # Scenario file last modification date
    scenario_date = ""
    scenario_file = run_path.parent.parent / "scenarios" / f"{scenario}.xlsx"
    if scenario_file.exists():
        mod_time = os.path.getmtime(scenario_file)
        scenario_date = datetime.datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M:%S")

    with open(run_path / f"_log_{run_name}.txt", "w") as f:
        f.write(f"RUN NAME = {run_name}\n")
        f.write(f"Scenario file: {scenario}\n")
        if scenario_date:
            f.write(f"\t last modified on {scenario_date}\n")
        f.write(f"Year simulated: {year}\n\n")
        f.write(f"Total dispatch cost: {total_cost} bEUR\n")
        f.write(f"Total CO2 emissions: {total_co2 / 1e6} MtCO2\n\n")
        f.write(f"Started running at: {time.asctime(start_time)}\n")
        f.write(f"Execution time: {time.strftime('%H:%M:%S', exec_time)}\n")
