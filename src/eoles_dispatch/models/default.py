"""Standard EOLES-Dispatch model with full thermal dynamics (startup, ramping, min stable generation)."""

import itertools
from pathlib import Path

import pandas as pd
import pyomo.environ as pyo

from ..config import DELTA, ETA_IN, ETA_OUT, GJ_MWH, LOAD_UNCERTAINTY, TRLOSS, VOLL


def build_model(run_dir):
    """Build and return the Pyomo ConcreteModel for the standard dispatch problem.

    Args:
        run_dir: Path to the run directory containing an inputs/ subdirectory.

    Returns:
        A Pyomo ConcreteModel ready to be solved.
    """
    input_dir = Path(run_dir) / "inputs"
    model = pyo.ConcreteModel()
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    # ── Inputs ──────────────────────────────────────────────────────────

    # Time-series
    demand = (
        pd.read_csv(input_dir / "demand.csv", header=None, names=["a", "h", "demand"])
        .set_index(["a", "h"])
        .squeeze(axis=1)
    )
    nmd = (
        pd.read_csv(input_dir / "nmd.csv", header=None, names=["a", "h", "nmd"])
        .set_index(["a", "h"])
        .squeeze(axis=1)
    )
    exoPrices = (
        pd.read_csv(input_dir / "exoPrices.csv", header=None, names=["exo_a", "h", "exoPrice"])
        .set_index(["exo_a", "h"])
        .squeeze(axis=1)
    )
    load_factor = (
        pd.read_csv(
            input_dir / "vre_profiles.csv", header=None, names=["a", "vre", "h", "load_factor"]
        )
        .set_index(["a", "vre", "h"])
        .squeeze(axis=1)
    )
    lake_inflows = (
        pd.read_csv(
            input_dir / "lake_inflows.csv", header=None, names=["a", "month", "lake_inflows"]
        )
        .set_index(["a", "month"])
        .squeeze(axis=1)
    )

    # Installed system
    capa = (
        pd.read_csv(input_dir / "capa.csv", header=None, names=["a", "tec", "capa"])
        .set_index(["a", "tec"])
        .squeeze(axis=1)
    )
    capa_in = (
        pd.read_csv(input_dir / "capa_in.csv", header=None, names=["a", "sto", "capa"])
        .set_index(["a", "sto"])
        .squeeze(axis=1)
    )
    stockMax = (
        pd.read_csv(input_dir / "stockMax.csv", header=None, names=["a", "sto", "capa"])
        .set_index(["a", "sto"])
        .squeeze(axis=1)
    )
    eaf = (
        pd.read_csv(input_dir / "yEAF.csv", header=None, names=["a", "thr", "capa"])
        .set_index(["a", "thr"])
        .squeeze(axis=1)
    )
    maxaf = (
        pd.read_csv(input_dir / "maxAF.csv", header=None, names=["a", "thr", "capa"])
        .set_index(["a", "thr"])
        .squeeze(axis=1)
    )
    nucMaxAF = (
        pd.read_csv(input_dir / "nucMaxAF.csv", header=None, names=["a", "week", "nucMaxAF"])
        .set_index(["a", "week"])
        .squeeze(axis=1)
    )
    hMaxOut = (
        pd.read_csv(input_dir / "hMaxOut.csv", header=None, names=["a", "month", "hMaxOut"])
        .set_index(["a", "month"])
        .squeeze(axis=1)
    )
    hMaxIn = (
        pd.read_csv(input_dir / "hMaxIn.csv", header=None, names=["a", "month", "hMaxIn"])
        .set_index(["a", "month"])
        .squeeze(axis=1)
    )
    links = (
        pd.read_csv(input_dir / "links.csv", header=None, names=["importer", "exporter", "links"])
        .set_index(["importer", "exporter"])
        .squeeze(axis=1)
    )
    exo_IM = (
        pd.read_csv(
            input_dir / "exo_IM.csv", header=None, names=["importer", "exo_exporter", "exo_IM"]
        )
        .set_index(["importer", "exo_exporter"])
        .squeeze(axis=1)
    )
    exo_EX = (
        pd.read_csv(
            input_dir / "exo_EX.csv", header=None, names=["exporter", "exo_importer", "exo_EX"]
        )
        .set_index(["exporter", "exo_importer"])
        .squeeze(axis=1)
    )
    rsv_req = (
        pd.read_csv(input_dir / "rsv_req.csv", header=None, names=["vre", "rsv_req"])
        .set_index("vre")
        .squeeze(axis=1)
    )

    # Fuel prices and variable costs
    thr_fuel = pd.read_csv(
        input_dir / "thr_fuel.csv", header=None, names=["thr", "fuel"]
    ).set_index("thr")
    fuel_price = pd.read_csv(
        input_dir / "fuel_price.csv", header=None, names=["thr", "price"]
    ).set_index("thr")
    fuel_timeFactor = pd.read_csv(
        input_dir / "fuel_timeFactor.csv", header=None, names=["fuel", "month", "timeFactor"]
    ).set_index("fuel")
    fuel_areaFactor = pd.read_csv(
        input_dir / "fuel_areaFactor.csv", header=None, names=["fuel", "area", "areaFactor"]
    ).set_index("fuel")

    fuel_price_adj = (
        fuel_price.join(thr_fuel, how="outer")
        .join(fuel_areaFactor, how="outer", on="fuel")
        .join(fuel_timeFactor, how="outer", on="fuel")
    )
    fuel_price_adj.index.name = "thr"
    fuel_price_adj.set_index(["area", "month"], append=True, inplace=True)
    fuel_price_adj["fuel_price_adj"] = (
        fuel_price_adj.price * fuel_price_adj.timeFactor + fuel_price_adj.areaFactor
    )
    fuel_price_adj = fuel_price_adj.drop(
        ["price", "fuel", "timeFactor", "areaFactor"], axis=1
    ).dropna()

    efficiency = pd.read_csv(
        input_dir / "efficiency.csv", header=None, names=["thr", "efficiency"]
    ).set_index("thr")
    eff50 = pd.read_csv(input_dir / "eff50.csv", header=None, names=["thr", "eff50"]).set_index(
        "thr"
    )
    co2_factor = pd.read_csv(
        input_dir / "co2_factor.csv", header=None, names=["thr", "co2_factor"]
    ).set_index("thr")
    co2_price = pd.read_csv(
        input_dir / "co2_price.csv", header=None, names=["thr", "co2_price"]
    ).set_index("thr")
    nonFuel_vOM = pd.read_csv(
        input_dir / "nonFuel_vOM.csv", header=None, names=["thr", "nonFuel_vOM"]
    ).set_index("thr")
    su_fixedCost = pd.read_csv(
        input_dir / "su_fixedCost.csv", header=None, names=["thr", "su_fixedCost"]
    ).set_index("thr")
    su_fuelCons = pd.read_csv(
        input_dir / "su_fuelCons.csv", header=None, names=["thr", "su_fuelCons"]
    ).set_index("thr")
    ramp_fuelCons = pd.read_csv(
        input_dir / "ramp_fuelCons.csv", header=None, names=["thr", "ramp_fuelCons"]
    ).set_index("thr")

    # Compute derived cost parameters
    genOM = (
        fuel_price_adj.join(efficiency, how="left")
        .join(eff50, how="left")
        .join(co2_factor, how="left")
        .join(co2_price, how="left")
        .join(nonFuel_vOM, how="left")
    )
    genOM["genOM"] = (
        (2 / genOM.efficiency - 1 / genOM.eff50)
        * GJ_MWH
        * (genOM.fuel_price_adj + genOM.co2_factor * genOM.co2_price / 1000)
    )
    genOM = genOM[["genOM"]].squeeze(axis=1)

    onOM = (
        fuel_price_adj.join(efficiency, how="left")
        .join(eff50, how="left")
        .join(co2_factor, how="left")
        .join(co2_price, how="left")
        .join(nonFuel_vOM, how="left")
    )
    onOM["onOM"] = (1 / onOM.eff50 - 1 / onOM.efficiency) * GJ_MWH * (
        onOM.fuel_price_adj + onOM.co2_factor * onOM.co2_price / 1000
    ) + onOM.nonFuel_vOM
    onOM = onOM[["onOM"]].squeeze(axis=1)

    su_cost = (
        fuel_price_adj.join(su_fuelCons, how="left")
        .join(co2_factor, how="left")
        .join(co2_price, how="left")
        .join(su_fixedCost, how="left")
    )
    su_cost["su_cost"] = (
        su_cost.su_fuelCons
        * (su_cost.fuel_price_adj + su_cost.co2_factor * su_cost.co2_price / 1000)
        + su_cost.su_fixedCost
    )
    su_cost = su_cost[["su_cost"]].squeeze(axis=1)

    ramp_cost = (
        fuel_price_adj.join(ramp_fuelCons, how="left")
        .join(co2_factor, how="left")
        .join(co2_price, how="left")
    )
    ramp_cost["ramp_cost"] = ramp_cost.ramp_fuelCons * (
        ramp_cost.fuel_price_adj + ramp_cost.co2_factor * ramp_cost.co2_price / 1000
    )
    ramp_cost = ramp_cost[["ramp_cost"]].squeeze(axis=1)

    # Carbon emission factors
    genCarb = efficiency.join(eff50, how="left").join(co2_factor, how="left")
    genCarb["genCarb"] = (2 / genCarb.efficiency - 1 / genCarb.eff50) * GJ_MWH * genCarb.co2_factor
    genCarb = genCarb[["genCarb"]].squeeze(axis=1)

    onCarb = efficiency.join(eff50, how="left").join(co2_factor, how="left")
    onCarb["onCarb"] = (1 / onCarb.eff50 - 1 / onCarb.efficiency) * GJ_MWH * onCarb.co2_factor
    onCarb = onCarb[["onCarb"]].squeeze(axis=1)

    suCarb = su_fuelCons.join(co2_factor, how="left")
    suCarb["suCarb"] = suCarb.su_fuelCons * suCarb.co2_factor
    suCarb = suCarb[["suCarb"]].squeeze(axis=1)

    rampCarb = ramp_fuelCons.join(co2_factor, how="left")
    rampCarb["rampCarb"] = rampCarb.ramp_fuelCons * rampCarb.co2_factor
    rampCarb = rampCarb[["rampCarb"]].squeeze(axis=1)

    # Other thermal parameters
    minSG = (
        pd.read_csv(input_dir / "minSG.csv", header=None, names=["thr", "minSG"])
        .set_index("thr")
        .squeeze(axis=1)
    )
    minTimeOFF = (
        pd.read_csv(input_dir / "minTimeOFF.csv", header=None, names=["thr", "minTimeOFF"])
        .set_index("thr")
        .squeeze(axis=1)
    )
    minTimeON = (
        pd.read_csv(input_dir / "minTimeON.csv", header=None, names=["thr", "minTimeON"])
        .set_index("thr")
        .squeeze(axis=1)
    )
    str_vOM = (
        pd.read_csv(input_dir / "str_vOM.csv", header=None, names=["str", "str_vOM"])
        .set_index("str")
        .squeeze(axis=1)
    )

    # Hour ↔ month/week mappings
    _hm_df = pd.read_csv(input_dir / "hour_month.csv", header=None, names=["hour", "month"])
    months_hours = _hm_df.groupby("month")["hour"].apply(list).to_dict()  # {month: [h1, h2, ...]}
    hours_months = _hm_df.set_index("hour")["month"].to_dict()  # {hour: month}
    hours_weeks = (
        pd.read_csv(input_dir / "hour_week.csv", header=None, names=["hour", "week"])
        .set_index("hour")["week"]
        .to_dict()
    )

    # ── Sets ────────────────────────────────────────────────────────────

    model.a = pyo.Set(
        initialize=pd.read_csv(input_dir / "areas.csv", header=None).squeeze(axis=1).array,
        ordered=False,
    )
    model.exo_a = pyo.Set(
        initialize=pd.read_csv(input_dir / "exo_areas.csv", header=None).squeeze(axis=1).array,
        ordered=False,
    )
    model.h = pyo.Set(
        initialize=pd.read_csv(input_dir / "hours.csv", header=None).squeeze(axis=1).array
    )
    model.week = pyo.Set(
        initialize=pd.read_csv(input_dir / "weeks.csv", header=None).squeeze(axis=1).array
    )
    model.month = pyo.Set(
        initialize=pd.read_csv(input_dir / "months.csv", header=None).squeeze(axis=1).array
    )
    model.tec = pyo.Set(
        initialize=pd.read_csv(input_dir / "tec.csv", header=None).squeeze(axis=1).array,
        ordered=False,
    )
    model.vre = pyo.Set(
        initialize=pd.read_csv(input_dir / "vre.csv", header=None).squeeze(axis=1).array,
        ordered=False,
    )
    model.thr = pyo.Set(
        initialize=pd.read_csv(input_dir / "thr.csv", header=None).squeeze(axis=1).array,
        ordered=False,
    )
    model.sto = pyo.Set(
        initialize=pd.read_csv(input_dir / "str_tec.csv", header=None).squeeze(axis=1).array,
        ordered=False,
    )
    model.frr = pyo.Set(
        initialize=pd.read_csv(input_dir / "frr.csv", header=None).squeeze(axis=1).array,
        ordered=False,
    )
    model.no_frr = pyo.Set(
        initialize=pd.read_csv(input_dir / "no_frr.csv", header=None).squeeze(axis=1).array,
        ordered=False,
    )
    # Trade pairs: all (a1, a2) where a1 != a2
    model.trade_pairs = pyo.Set(
        initialize=[(a1, a2) for a1 in model.a for a2 in model.a if a1 != a2], dimen=2
    )

    # ── Variables ───────────────────────────────────────────────────────

    model.gene = pyo.Var(
        ((a, tec, h) for a in model.a for tec in model.tec for h in model.h),
        within=pyo.NonNegativeReals,
        initialize=0,
    )
    model.on = pyo.Var(
        ((a, thr, h) for a in model.a for thr in model.thr for h in model.h),
        within=pyo.NonNegativeReals,
        initialize=0,
    )
    for a in model.a:
        for thr in model.thr:
            for h in model.h:
                model.on[a, thr, h].value = capa[a, thr] * maxaf[a, thr]
    model.startup = pyo.Var(
        ((a, thr, h) for a in model.a for thr in model.thr for h in model.h),
        within=pyo.NonNegativeReals,
        initialize=0,
    )
    model.turnoff = pyo.Var(
        ((a, thr, h) for a in model.a for thr in model.thr for h in model.h),
        within=pyo.NonNegativeReals,
        initialize=0,
    )
    model.ramp_up = pyo.Var(
        ((a, thr, h) for a in model.a for thr in model.thr for h in model.h),
        within=pyo.NonNegativeReals,
        initialize=0,
    )
    model.storage = pyo.Var(
        ((a, sto, h) for a in model.a for sto in model.sto for h in model.h),
        within=pyo.NonNegativeReals,
        initialize=0,
    )
    model.stored = pyo.Var(
        ((a, sto, h) for a in model.a for sto in model.sto for h in model.h),
        within=pyo.NonNegativeReals,
        initialize=0,
    )
    model.rsv = pyo.Var(
        ((a, tec, h) for a in model.a for tec in model.tec for h in model.h),
        within=pyo.NonNegativeReals,
        initialize=0,
    )
    model.hll = pyo.Var(
        ((a, h) for a in model.a for h in model.h), within=pyo.NonNegativeReals, initialize=0
    )
    model.im = pyo.Var(
        ((a1, a2, h) for a1, a2 in model.trade_pairs for h in model.h),
        within=pyo.NonNegativeReals,
        initialize=0,
    )
    model.ex = pyo.Var(
        ((a1, a2, h) for a1, a2 in model.trade_pairs for h in model.h),
        within=pyo.NonNegativeReals,
        initialize=0,
    )
    model.exo_im = pyo.Var(
        ((a, exo_a, h) for a in model.a for exo_a in model.exo_a for h in model.h),
        within=pyo.NonNegativeReals,
        initialize=0,
    )
    model.exo_ex = pyo.Var(
        ((a, exo_a, h) for a in model.a for exo_a in model.exo_a for h in model.h),
        within=pyo.NonNegativeReals,
        initialize=0,
    )
    model.hcost = pyo.Var(
        ((a, h) for a in model.a for h in model.h), within=pyo.NonNegativeReals, initialize=0
    )
    model.hcarb = pyo.Var(
        ((a, h) for a in model.a for h in model.h), within=pyo.NonNegativeReals, initialize=0
    )

    # ── Constraints ─────────────────────────────────────────────────────

    # VRE generation
    def gene_vre_rule(model, a, h, vre):
        return model.gene[a, vre, h] <= capa[a, vre] * load_factor[a, vre, h]

    model.gene_vre_constraint = pyo.Constraint(model.a, model.h, model.vre, rule=gene_vre_rule)

    def gene_nmd_rule(model, a, h):
        return model.gene[a, "nmd", h] == nmd[a, h]

    model.gene_nmd_constraint = pyo.Constraint(model.a, model.h, rule=gene_nmd_rule)

    # Thermal generation
    def on_capa_rule(model, a, thr, h):
        return model.on[a, thr, h] <= capa[a, thr] * maxaf[a, thr]

    model.on_capa_constraint = pyo.Constraint(model.a, model.thr, model.h, rule=on_capa_rule)

    def gene_on_hmax_rule(model, a, thr, h):
        return model.gene[a, thr, h] + model.rsv[a, thr, h] <= model.on[a, thr, h]

    model.gene_on_hmax_constraint = pyo.Constraint(
        model.a, model.thr, model.h, rule=gene_on_hmax_rule
    )

    def gene_on_hmin_rule(model, a, thr, h):
        return model.on[a, thr, h] * minSG[thr] <= model.gene[a, thr, h]

    model.gene_on_hmin_constraint = pyo.Constraint(
        model.a, model.thr, model.h, rule=gene_on_hmin_rule
    )

    def yearly_maxON_rule(model, a, thr):
        return (
            sum(model.on[a, thr, h] for h in model.h) / len(model.h) <= capa[a, thr] * eaf[a, thr]
        )

    model.yearly_maxON_constraint = pyo.Constraint(model.a, model.thr, rule=yearly_maxON_rule)

    def nuc_maxON_rule(model, a, h):
        return model.on[a, "nuclear", h] <= capa[a, "nuclear"] * nucMaxAF[a, hours_weeks[h]]

    model.nuc_maxON_constraint = pyo.Constraint(model.a, model.h, rule=nuc_maxON_rule)

    # Startup / turnoff dynamics
    def on_off_rule(model, a, thr, h):
        h_next = h + 1 if h < model.h.last() else model.h.first()
        return (
            model.on[a, thr, h_next]
            == model.on[a, thr, h] + model.startup[a, thr, h] - model.turnoff[a, thr, h]
        )

    model.on_off_constraint = pyo.Constraint(model.a, model.thr, model.h, rule=on_off_rule)

    def cons_startup_rule(model, a, thr, h):
        if h - minTimeOFF[thr] >= model.h.first():
            recently_off = range(h - minTimeOFF[thr], h)
        else:
            recently_off = itertools.chain(
                range(model.h.first(), h),
                range(
                    model.h.last() - minTimeOFF[thr] + (h - model.h.first()) + 1, model.h.last() + 1
                ),
            )
        return model.startup[a, thr, h] <= capa[a, thr] * maxaf[a, thr] - model.on[a, thr, h] - sum(
            model.turnoff[a, thr, h_bis] for h_bis in recently_off
        )

    model.cons_startup_constraint = pyo.Constraint(
        model.a, model.thr, model.h, rule=cons_startup_rule
    )

    def cons_turnoff_rule(model, a, thr, h):
        if h - minTimeON[thr] >= model.h.first():
            recently_on = range(h - minTimeON[thr], h)
        else:
            recently_on = itertools.chain(
                range(model.h.first(), h),
                range(
                    model.h.last() - minTimeON[thr] + (h - model.h.first()) + 1, model.h.last() + 1
                ),
            )
        return model.turnoff[a, thr, h] <= model.on[a, thr, h] - sum(
            model.startup[a, thr, h_bis] for h_bis in recently_on
        )

    model.cons_turnoff_constraint = pyo.Constraint(
        model.a, model.thr, model.h, rule=cons_turnoff_rule
    )

    def ramping_up_rule(model, a, thr, h):
        h_next = h + 1 if h < model.h.last() else model.h.first()
        return model.ramp_up[a, thr, h_next] >= model.gene[a, thr, h_next] - model.gene[a, thr, h]

    model.ramping_up_constraint = pyo.Constraint(model.a, model.thr, model.h, rule=ramping_up_rule)

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
            return (
                model.gene[a, sto, h] + model.rsv[a, sto, h]
                <= capa[a, sto] * hMaxOut[a, hours_months[h]]
            )
        return model.gene[a, sto, h] + model.rsv[a, sto, h] <= capa[a, sto]

    model.stor_out_constraint = pyo.Constraint(model.a, model.sto, model.h, rule=stor_out_rule)

    def storing_rule(model, a, sto, h):
        h_next = h + 1 if h < model.h.last() else model.h.first()
        if sto == "lake_phs":
            return model.stored[a, sto, h_next] == (
                model.stored[a, sto, h]
                + model.storage[a, sto, h] * ETA_IN[sto]
                - model.gene[a, sto, h] / ETA_OUT[sto]
                + (lake_inflows[a, hours_months[h]] * 1000 / len(months_hours[hours_months[h]]))
                / ETA_OUT[sto]
            )
        return model.stored[a, sto, h_next] == (
            model.stored[a, sto, h]
            + model.storage[a, sto, h] * ETA_IN[sto]
            - model.gene[a, sto, h] / ETA_OUT[sto]
        )

    model.storing_constraint = pyo.Constraint(model.a, model.sto, model.h, rule=storing_rule)

    def lake_res_rule(model, a, month):
        return (
            sum(
                model.gene[a, "lake_phs", h]
                - model.storage[a, "lake_phs", h] * ETA_IN["lake_phs"] * ETA_OUT["lake_phs"]
                for h in months_hours[month]
            )
            == lake_inflows[a, month] * 1000
        )

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

    model.no_FRR_contrib_constraint = pyo.Constraint(
        model.a, model.no_frr, model.h, rule=no_FRR_rule
    )

    # Trade balance
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

    # Adequacy (supply = demand)
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

    # Cost and emissions definitions
    def hcost_rule(model, a, h):
        return model.hcost[a, h] == (
            sum(model.gene[a, thr, h] * genOM[thr, a, hours_months[h]] for thr in model.thr)
            + sum(model.on[a, thr, h] * onOM[thr, a, hours_months[h]] for thr in model.thr)
            + sum(model.startup[a, thr, h] * su_cost[thr, a, hours_months[h]] for thr in model.thr)
            + sum(
                model.ramp_up[a, thr, h] * ramp_cost[thr, a, hours_months[h]] for thr in model.thr
            )
            + sum(model.gene[a, sto, h] * str_vOM[sto] for sto in model.sto)
            + sum(
                ((model.exo_im[a, exo_a, h] - model.exo_ex[a, exo_a, h]) / (1 - TRLOSS))
                * exoPrices[exo_a, h]
                for exo_a in model.exo_a
            )
            + model.hll[a, h] * VOLL
        )

    model.hcost_constraint = pyo.Constraint(model.a, model.h, rule=hcost_rule)

    def hcarb_rule(model, a, h):
        return model.hcarb[a, h] == (
            sum(model.gene[a, thr, h] * genCarb[thr] for thr in model.thr)
            + sum(model.on[a, thr, h] * onCarb[thr] for thr in model.thr)
            + sum(model.startup[a, thr, h] * suCarb[thr] for thr in model.thr)
            + sum(model.ramp_up[a, thr, h] * rampCarb[thr] for thr in model.thr)
        )

    model.hcarb_constraint = pyo.Constraint(model.a, model.h, rule=hcarb_rule)

    # ── Objective ───────────────────────────────────────────────────────

    def objective_rule(model):
        return sum(model.hcost[a, h] for a in model.a for h in model.h)

    model.objective = pyo.Objective(rule=objective_rule)

    return model
