"""Simplified EOLES-Dispatch model without thermal dynamics (no startup, ramping, min stable generation).

This variant treats thermal plants as simple dispatch units with variable costs
proportional to generation only. Useful for fast exploratory runs.
"""

from pathlib import Path

import pandas as pd
import pyomo.environ as pyo

from ..config import LOAD_UNCERTAINTY, DELTA, VOLL, ETA_IN, ETA_OUT, TRLOSS, GJ_MWH


def build_model(run_dir):
    """Build and return the Pyomo ConcreteModel for the static thermal dispatch problem.

    Args:
        run_dir: Path to the run directory containing an inputs/ subdirectory.

    Returns:
        A Pyomo ConcreteModel ready to be solved.
    """
    input_dir = Path(run_dir) / "inputs"
    model = pyo.ConcreteModel()
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    # ── Inputs ──────────────────────────────────────────────────────────

    demand = pd.read_csv(input_dir / "demand.csv", header=None, names=["a", "h", "demand"]).set_index(["a", "h"]).squeeze(axis=1)
    nmd = pd.read_csv(input_dir / "nmd.csv", header=None, names=["a", "h", "nmd"]).set_index(["a", "h"]).squeeze(axis=1)
    exoPrices = pd.read_csv(input_dir / "exoPrices.csv", header=None, names=["exo_a", "h", "exoPrice"]).set_index(["exo_a", "h"]).squeeze(axis=1)
    load_factor = pd.read_csv(input_dir / "vre_profiles.csv", header=None, names=["a", "vre", "h", "load_factor"]).set_index(["a", "vre", "h"]).squeeze(axis=1)
    lake_inflows = pd.read_csv(input_dir / "lake_inflows.csv", header=None, names=["a", "month", "lake_inflows"]).set_index(["a", "month"]).squeeze(axis=1)

    capa = pd.read_csv(input_dir / "capa.csv", header=None, names=["a", "tec", "capa"]).set_index(["a", "tec"]).squeeze(axis=1)
    capa_in = pd.read_csv(input_dir / "capa_in.csv", header=None, names=["a", "sto", "capa"]).set_index(["a", "sto"]).squeeze(axis=1)
    stockMax = pd.read_csv(input_dir / "stockMax.csv", header=None, names=["a", "sto", "capa"]).set_index(["a", "sto"]).squeeze(axis=1)
    eaf = pd.read_csv(input_dir / "yEAF.csv", header=None, names=["a", "thr", "capa"]).set_index(["a", "thr"]).squeeze(axis=1)
    maxaf = pd.read_csv(input_dir / "maxAF.csv", header=None, names=["a", "thr", "capa"]).set_index(["a", "thr"]).squeeze(axis=1)
    nucMaxAF = pd.read_csv(input_dir / "nucMaxAF.csv", header=None, names=["a", "week", "nucMaxAF"]).set_index(["a", "week"]).squeeze(axis=1)
    hMaxOut = pd.read_csv(input_dir / "hMaxOut.csv", header=None, names=["a", "month", "hMaxOut"]).set_index(["a", "month"]).squeeze(axis=1)
    hMaxIn = pd.read_csv(input_dir / "hMaxIn.csv", header=None, names=["a", "month", "hMaxIn"]).set_index(["a", "month"]).squeeze(axis=1)
    links = pd.read_csv(input_dir / "links.csv", header=None, names=["importer", "exporter", "links"]).set_index(["importer", "exporter"]).squeeze(axis=1)
    exo_IM = pd.read_csv(input_dir / "exo_IM.csv", header=None, names=["importer", "exo_exporter", "exo_IM"]).set_index(["importer", "exo_exporter"]).squeeze(axis=1)
    exo_EX = pd.read_csv(input_dir / "exo_EX.csv", header=None, names=["exporter", "exo_importer", "exo_EX"]).set_index(["exporter", "exo_importer"]).squeeze(axis=1)
    rsv_req = pd.read_csv(input_dir / "rsv_req.csv", header=None, names=["vre", "rsv_req"]).set_index("vre").squeeze(axis=1)

    # Simplified variable costs (no eff50, no startup/ramp)
    thr_fuel = pd.read_csv(input_dir / "thr_fuel.csv", header=None, names=["thr", "fuel"]).set_index("thr")
    fuel_price = pd.read_csv(input_dir / "fuel_price.csv", header=None, names=["thr", "price"]).set_index("thr")
    fuel_timeFactor = pd.read_csv(input_dir / "fuel_timeFactor.csv", header=None, names=["fuel", "month", "timeFactor"]).set_index("fuel")
    fuel_areaFactor = pd.read_csv(input_dir / "fuel_areaFactor.csv", header=None, names=["fuel", "area", "areaFactor"]).set_index("fuel")

    fuel_price_adj = fuel_price.join(thr_fuel, how="outer").join(fuel_areaFactor, how="outer", on="fuel").join(fuel_timeFactor, how="outer", on="fuel")
    fuel_price_adj.index.name = "thr"
    fuel_price_adj.set_index(["area", "month"], append=True, inplace=True)
    fuel_price_adj["fuel_price_adj"] = fuel_price_adj.price * fuel_price_adj.timeFactor + fuel_price_adj.areaFactor
    fuel_price_adj = fuel_price_adj.drop(["price", "fuel", "timeFactor", "areaFactor"], axis=1).dropna()

    efficiency = pd.read_csv(input_dir / "efficiency.csv", header=None, names=["thr", "efficiency"]).set_index("thr")
    co2_factor = pd.read_csv(input_dir / "co2_factor.csv", header=None, names=["thr", "co2_factor"]).set_index("thr")
    co2_price = pd.read_csv(input_dir / "co2_price.csv", header=None, names=["thr", "co2_price"]).set_index("thr")
    nonFuel_vOM = pd.read_csv(input_dir / "nonFuel_vOM.csv", header=None, names=["thr", "nonFuel_vOM"]).set_index("thr")

    vOM = fuel_price_adj.join(efficiency, how="left").join(co2_factor, how="left").join(co2_price, how="left").join(nonFuel_vOM, how="left")
    vOM["vOM"] = (1 / vOM.efficiency) * GJ_MWH * (vOM.fuel_price_adj + vOM.co2_factor * vOM.co2_price / 1000) + vOM.nonFuel_vOM
    vOM = vOM[["vOM"]].squeeze(axis=1)

    vCarb = efficiency.join(co2_factor, how="left")
    vCarb["vCarb"] = (1 / vCarb.efficiency) * GJ_MWH * vCarb.co2_factor
    vCarb = vCarb[["vCarb"]].squeeze(axis=1)

    str_vOM = pd.read_csv(input_dir / "str_vOM.csv", header=None, names=["str", "str_vOM"]).set_index("str").squeeze(axis=1)

    months_hours = pd.read_csv(input_dir / "hour_month.csv", header=None, names=["hour", "month"]).set_index("month").squeeze(axis=1)
    hours_months = pd.read_csv(input_dir / "hour_month.csv", header=None, names=["hour", "month"]).set_index("hour").squeeze(axis=1)
    hours_weeks = pd.read_csv(input_dir / "hour_week.csv", header=None, names=["hour", "week"]).set_index("hour").squeeze(axis=1)

    # ── Sets ────────────────────────────────────────────────────────────

    model.a = pyo.Set(initialize=pd.read_csv(input_dir / "areas.csv", header=None).squeeze(axis=1).array, ordered=False)
    model.exo_a = pyo.Set(initialize=pd.read_csv(input_dir / "exo_areas.csv", header=None).squeeze(axis=1).array, ordered=False)
    model.h = pyo.Set(initialize=pd.read_csv(input_dir / "hours.csv", header=None).squeeze(axis=1).array)
    model.week = pyo.Set(initialize=pd.read_csv(input_dir / "weeks.csv", header=None).squeeze(axis=1).array)
    model.month = pyo.Set(initialize=pd.read_csv(input_dir / "months.csv", header=None).squeeze(axis=1).array)
    model.tec = pyo.Set(initialize=pd.read_csv(input_dir / "tec.csv", header=None).squeeze(axis=1).array, ordered=False)
    model.vre = pyo.Set(initialize=pd.read_csv(input_dir / "vre.csv", header=None).squeeze(axis=1).array, ordered=False)
    model.thr = pyo.Set(initialize=pd.read_csv(input_dir / "thr.csv", header=None).squeeze(axis=1).array, ordered=False)
    model.sto = pyo.Set(initialize=pd.read_csv(input_dir / "str_tec.csv", header=None).squeeze(axis=1).array, ordered=False)
    model.frr = pyo.Set(initialize=pd.read_csv(input_dir / "frr.csv", header=None).squeeze(axis=1).array, ordered=False)
    model.no_frr = pyo.Set(initialize=pd.read_csv(input_dir / "no_frr.csv", header=None).squeeze(axis=1).array, ordered=False)
    # Trade pairs: all (a1, a2) where a1 != a2
    model.trade_pairs = pyo.Set(initialize=[(a1, a2) for a1 in model.a for a2 in model.a if a1 != a2], dimen=2)

    # ── Variables (no on/startup/turnoff/ramp_up) ──────────────────────

    model.gene = pyo.Var(((a, tec, h) for a in model.a for tec in model.tec for h in model.h), within=pyo.NonNegativeReals, initialize=0)
    model.storage = pyo.Var(((a, sto, h) for a in model.a for sto in model.sto for h in model.h), within=pyo.NonNegativeReals, initialize=0)
    model.stored = pyo.Var(((a, sto, h) for a in model.a for sto in model.sto for h in model.h), within=pyo.NonNegativeReals, initialize=0)
    model.rsv = pyo.Var(((a, tec, h) for a in model.a for tec in model.tec for h in model.h), within=pyo.NonNegativeReals, initialize=0)
    model.hll = pyo.Var(((a, h) for a in model.a for h in model.h), within=pyo.NonNegativeReals, initialize=0)
    model.im = pyo.Var(((a1, a2, h) for a1, a2 in model.trade_pairs for h in model.h), within=pyo.NonNegativeReals, initialize=0)
    model.ex = pyo.Var(((a1, a2, h) for a1, a2 in model.trade_pairs for h in model.h), within=pyo.NonNegativeReals, initialize=0)
    model.exo_im = pyo.Var(((a, exo_a, h) for a in model.a for exo_a in model.exo_a for h in model.h), within=pyo.NonNegativeReals, initialize=0)
    model.exo_ex = pyo.Var(((a, exo_a, h) for a in model.a for exo_a in model.exo_a for h in model.h), within=pyo.NonNegativeReals, initialize=0)
    model.hcost = pyo.Var(((a, h) for a in model.a for h in model.h), within=pyo.NonNegativeReals, initialize=0)
    model.hcarb = pyo.Var(((a, h) for a in model.a for h in model.h), within=pyo.NonNegativeReals, initialize=0)

    # ── Constraints ─────────────────────────────────────────────────────

    # VRE
    def gene_vre_rule(model, a, h, vre):
        return model.gene[a, vre, h] <= capa[a, vre] * load_factor[a, vre, h]
    model.gene_vre_constraint = pyo.Constraint(model.a, model.h, model.vre, rule=gene_vre_rule)

    def gene_nmd_rule(model, a, h):
        return model.gene[a, "nmd", h] == nmd[a, h]
    model.gene_nmd_constraint = pyo.Constraint(model.a, model.h, rule=gene_nmd_rule)

    # Simplified thermal (capacity-based, no on/off dynamics)
    def gene_capa_rule(model, a, thr, h):
        return model.gene[a, thr, h] <= capa[a, thr] * maxaf[a, thr]
    model.gene_capa_constraint = pyo.Constraint(model.a, model.thr, model.h, rule=gene_capa_rule)

    def yearly_maxGENE_rule(model, a, thr):
        return sum(model.gene[a, thr, h] for h in model.h) / len(model.h) <= capa[a, thr] * eaf[a, thr]
    model.yearly_maxGENE_constraint = pyo.Constraint(model.a, model.thr, rule=yearly_maxGENE_rule)

    def nuc_maxGENE_rule(model, a, h):
        return model.gene[a, "nuclear", h] <= capa[a, "nuclear"] * nucMaxAF[a, hours_weeks[h]]
    model.nuc_maxGENE_constraint = pyo.Constraint(model.a, model.h, rule=nuc_maxGENE_rule)

    # Storage
    def stored_cap_rule(model, a, sto, h):
        return model.stored[a, sto, h] <= stockMax[a, sto] * 1000
    model.stored_cap_constraint = pyo.Constraint(model.a, model.sto, model.h, rule=stored_cap_rule)

    def stor_in_rule(model, a, sto, h):
        if sto == "lake_phs":
            return model.storage[a, sto, h] <= capa_in[a, sto] * hMaxIn[a, hours_months[h]]
        return model.storage[a, sto, h] <= capa_in[a, sto]
    model.stor_in_constraint = pyo.Constraint(model.a, model.sto, model.h, rule=stor_in_rule)

    def stor_out_rule(model, a, sto, h):
        if sto == "lake_phs":
            return model.gene[a, sto, h] + model.rsv[a, sto, h] <= capa[a, sto] * hMaxOut[a, hours_months[h]]
        return model.gene[a, sto, h] + model.rsv[a, sto, h] <= capa[a, sto]
    model.stor_out_constraint = pyo.Constraint(model.a, model.sto, model.h, rule=stor_out_rule)

    def storing_rule(model, a, sto, h):
        h_next = h + 1 if h < model.h.last() else model.h.first()
        if sto == "lake_phs":
            return model.stored[a, sto, h_next] == (
                model.stored[a, sto, h]
                + model.storage[a, sto, h] * ETA_IN[sto]
                - model.gene[a, sto, h] / ETA_OUT[sto]
                + (lake_inflows[a, hours_months[h]] * 1000 / len(months_hours[hours_months[h]])) / ETA_OUT[sto]
            )
        return model.stored[a, sto, h_next] == (
            model.stored[a, sto, h]
            + model.storage[a, sto, h] * ETA_IN[sto]
            - model.gene[a, sto, h] / ETA_OUT[sto]
        )
    model.storing_constraint = pyo.Constraint(model.a, model.sto, model.h, rule=storing_rule)

    def lake_res_rule(model, a, month):
        return sum(
            model.gene[a, "lake_phs", h] - model.storage[a, "lake_phs", h] * ETA_IN["lake_phs"] * ETA_OUT["lake_phs"]
            for h in months_hours[month]
        ) == lake_inflows[a, month] * 1000
    model.lake_res_constraint = pyo.Constraint(model.a, model.month, rule=lake_res_rule)

    # Reserves
    def reserves_rule(model, a, h):
        return sum(model.rsv[a, frr, h] for frr in model.frr) == (
            sum(rsv_req[vre] * capa[a, vre] for vre in model.vre)
            + demand[a, h] * LOAD_UNCERTAINTY * (1 + DELTA)
        )
    model.reserves_constraint = pyo.Constraint(model.a, model.h, rule=reserves_rule)

    def no_FRR_rule(model, a, no_frr, h):
        return model.rsv[a, no_frr, h] == 0
    model.no_FRR_contrib_constraint = pyo.Constraint(model.a, model.no_frr, model.h, rule=no_FRR_rule)

    # Trade
    def trade_bal_rule(model, a1, a2, h):
        return model.im[a1, a2, h] == model.ex[a2, a1, h] * (1 - TRLOSS)
    model.trade_bal_constraint = pyo.Constraint(model.trade_pairs, model.h, rule=trade_bal_rule)

    def icIM_rule(model, a1, a2, h):
        return model.im[a1, a2, h] <= links[a1, a2]
    model.icIM_constraint = pyo.Constraint(model.trade_pairs, model.h, rule=icIM_rule)

    def exoIM_rule(model, a, exo_a, h):
        return model.exo_im[a, exo_a, h] <= exo_IM[a, exo_a]
    model.exoIM_constraint = pyo.Constraint(model.a, model.exo_a, model.h, rule=exoIM_rule)

    def exoEX_rule(model, a, exo_a, h):
        return model.exo_ex[a, exo_a, h] <= exo_EX[a, exo_a]
    model.exoEX_constraint = pyo.Constraint(model.a, model.exo_a, model.h, rule=exoEX_rule)

    # Adequacy
    def adequacy_rule(model, a, h):
        return (
            sum(model.gene[a, tec, h] for tec in model.tec)
            + sum(model.im[a, trader, h] for trader in model.a if trader != a)
            + sum(model.exo_im[a, exo_a, h] for exo_a in model.exo_a)
        ) == (
            demand[a, h]
            + sum(model.ex[a, trader, h] for trader in model.a if trader != a)
            + sum(model.exo_ex[a, exo_a, h] for exo_a in model.exo_a)
            + sum(model.storage[a, sto, h] for sto in model.sto)
            - model.hll[a, h]
        )
    model.adequacy_constraint = pyo.Constraint(model.a, model.h, rule=adequacy_rule)

    # Cost and emissions (simplified — no on/startup/ramp components)
    def hcost_rule(model, a, h):
        return model.hcost[a, h] == (
            sum(model.gene[a, thr, h] * vOM[thr, a, hours_months[h]] for thr in model.thr)
            + sum(model.gene[a, sto, h] * str_vOM[sto] for sto in model.sto)
            + sum(((model.exo_im[a, exo_a, h] - model.exo_ex[a, exo_a, h]) / (1 - TRLOSS)) * exoPrices[exo_a, h] for exo_a in model.exo_a)
            + model.hll[a, h] * VOLL
        )
    model.hcost_constraint = pyo.Constraint(model.a, model.h, rule=hcost_rule)

    def hcarb_rule(model, a, h):
        return model.hcarb[a, h] == sum(model.gene[a, thr, h] * vCarb[thr] for thr in model.thr)
    model.hcarb_constraint = pyo.Constraint(model.a, model.h, rule=hcarb_rule)

    # Objective
    def objective_rule(model):
        return sum(model.hcost[a, h] for a in model.a for h in model.h)
    model.objective = pyo.Objective(rule=objective_rule)

    return model
