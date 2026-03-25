"""Load and format input data for the EOLES-Dispatch model.

Reads year-based intermediate data from data/<year>/, computes derived
variables (capacity factors, nuclear availability, lake inflows, hydro limits)
from raw production history, and formats everything for the Pyomo model.

The key design principle: collected data is *raw* (harmonized but not
transformed). All derived computations happen here at run creation time,
using scenario parameters (e.g. installed capacity) when needed.
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import NMD_TYPES
from .utils import (
    CET,
    cet_month_bounds,
    cet_to_utc,
    cet_year_bounds,
    compute_hour_mappings,
    hour_to_cet_month,
    hour_to_cet_week,
)


# ── Data loading from year-based storage ──


def load_year_production(data_dir, year, areas):
    """Load raw production data for all areas from data/<year>/production_<area>.csv.

    Args:
        data_dir: Path to the data/ directory.
        year: The simulation year.
        areas: List of area codes.

    Returns:
        dict {area: pd.DataFrame} with columns ['hour', fuel1, fuel2, ...].
        Values in MW. 'hour' is a UTC tz-naive datetime.
    """
    year_dir = Path(data_dir) / str(year)
    if not year_dir.exists():
        raise FileNotFoundError(
            f"Data for year {year} not found at {year_dir}. "
            f"Run 'eoles-dispatch collect --start {year} --end {year + 1}' first."
        )

    result = {}
    for area in areas:
        prod_path = year_dir / f"production_{area}.csv"
        if not prod_path.exists():
            raise FileNotFoundError(
                f"Production data for {area} not found at {prod_path}. "
                f"Re-run 'eoles-dispatch collect --start {year} --end {year + 1}'."
            )
        df = pd.read_csv(prod_path)
        df["hour"] = pd.to_datetime(df["hour"])
        result[area] = df

    return result


def _load_year_csv(data_dir, year, filename, areas_or_exo):
    """Load a year-based CSV file (demand.csv or exo_prices.csv).

    Args:
        data_dir: Path to the data/ directory.
        year: The simulation year.
        filename: CSV filename (e.g. "demand.csv").
        areas_or_exo: List of area columns to extract.

    Returns:
        pd.DataFrame with columns ['hour', area1, area2, ...].
        'hour' is a UTC tz-naive datetime.
    """
    year_dir = Path(data_dir) / str(year)
    csv_path = year_dir / filename
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{filename} not found at {csv_path}. "
            f"Run 'eoles-dispatch collect --start {year} --end {year + 1}' first."
        )
    df = pd.read_csv(csv_path)
    df["hour"] = pd.to_datetime(df["hour"])
    return df


def _to_posix_hours(dt_series):
    """Convert a datetime Series to POSIX hours (int, hours since 1970-01-01 UTC)."""
    return ((dt_series - datetime(1970, 1, 1)).dt.total_seconds() / 3600).astype(int)


def _melt_hourly(df, areas, start, end):
    """Filter a wide DataFrame to [start, end), convert to POSIX hours, melt to long format.

    Args:
        df: DataFrame with 'hour' (datetime) and area columns.
        areas: List of area codes to include.
        start, end: Naive UTC datetimes for filtering.

    Returns:
        DataFrame with columns ['area', 'hour', 'value'].
    """
    df = df[(df["hour"] >= start) & (df["hour"] < end)].copy()
    df["hour"] = _to_posix_hours(df["hour"])
    available_areas = [a for a in areas if a in df.columns]
    melted = pd.melt(df, id_vars=["hour"], value_vars=available_areas,
                     var_name="area", value_name="value")
    return melted[["area", "hour", "value"]]


def load_ninja_var(data_dir, variable, areas, start, end):
    """Load a Renewable Ninja capacity factor variable.

    Reads from data/renewable_ninja/<variable>.csv and filters to the
    requested period and areas.

    Returns:
        DataFrame with columns ['area', 'hour', 'value'] (POSIX hours).
    """
    csv_path = Path(data_dir) / "renewable_ninja" / f"{variable}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Renewable Ninja data not found at {csv_path}. "
            f"Run 'eoles-dispatch collect --source ninja' to download."
        )
    df = pd.read_csv(csv_path)
    df["hour"] = pd.to_datetime(df["hour"])
    df = df[(df["hour"] >= start) & (df["hour"] < end)]
    if df.empty:
        available_min = pd.read_csv(csv_path, usecols=["hour"], nrows=1)["hour"].iloc[0]
        available_max = pd.read_csv(csv_path, usecols=["hour"]).iloc[-1]["hour"]
        raise ValueError(
            f"No data for '{variable}' in the requested period "
            f"{start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}. "
            f"Available data covers {available_min} to {available_max}. "
            f"Run 'eoles-dispatch collect --source ninja' to download data."
        )
    df["hour"] = _to_posix_hours(df["hour"])
    melted = pd.melt(df, id_vars=["hour"], value_vars=areas,
                     var_name="area", value_name="value")
    return melted[["area", "hour", "value"]]


# ── Derived variable computations (from raw production data) ──


def compute_nmd(production, areas, start, end):
    """Compute NMD (non-market-dependent) production from raw generation data.

    NMD = sum of biomass, geothermal, marine, other_renew, waste, other.

    Args:
        production: dict {area: DataFrame} from load_year_production.
        areas: List of area codes.
        start, end: Naive UTC datetimes.

    Returns:
        DataFrame with columns ['area', 'hour', 'value'] in GW.
    """
    frames = {}
    for area in areas:
        if area not in production:
            continue
        df = production[area]
        df_filtered = df[(df["hour"] >= start) & (df["hour"] < end)].copy()
        nmd_cols = [c for c in NMD_TYPES if c in df_filtered.columns]
        if nmd_cols:
            nmd_series = df_filtered[nmd_cols].sum(axis=1) / 1000  # MW → GW
        else:
            nmd_series = pd.Series(0, index=df_filtered.index)
        frames[area] = pd.DataFrame({
            "hour": _to_posix_hours(df_filtered["hour"]),
            "value": nmd_series.values,
        })
        frames[area]["area"] = area

    if not frames:
        return pd.DataFrame(columns=["area", "hour", "value"])

    result = pd.concat(frames.values(), ignore_index=True)
    return result[["area", "hour", "value"]]


def compute_vre_capacity_factors(production, scenario_capa, areas, start, end,
                                  technologies=None):
    """Compute VRE capacity factors from raw production and scenario capacity.

    CF = hourly_production (MW) / installed_capacity (GW * 1000), clipped to [0, 1].

    Args:
        production: dict {area: DataFrame} from load_year_production.
        scenario_capa: DataFrame with columns ['area', 'tec', 'value'] (GW).
        areas: List of area codes.
        start, end: Naive UTC datetimes.
        technologies: List of VRE tech names (default: offshore, onshore, pv, river).

    Returns:
        DataFrame with columns ['area', 'tec', 'hour', 'value'].
    """
    if technologies is None:
        technologies = ["offshore", "onshore", "pv", "river"]

    frames = []
    for tec in technologies:
        for area in areas:
            if area not in production:
                continue
            df = production[area]
            if tec not in df.columns:
                continue

            df_filtered = df[(df["hour"] >= start) & (df["hour"] < end)].copy()
            prod_mw = df_filtered[tec].values

            # Get installed capacity from scenario (GW → MW)
            capa_row = scenario_capa[
                (scenario_capa["area"] == area) & (scenario_capa["tec"] == tec)
            ]
            if not capa_row.empty:
                capa_mw = float(capa_row["value"].iloc[0]) * 1000  # GW → MW
            else:
                # Fallback: use max observed production
                capa_mw = prod_mw.max() if prod_mw.max() > 0 else 1

            cf = np.clip(prod_mw / max(capa_mw, 1), 0, 1)

            frame = pd.DataFrame({
                "area": area,
                "tec": tec,
                "hour": _to_posix_hours(df_filtered["hour"]),
                "value": cf,
            })
            frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=["area", "tec", "hour", "value"])

    return pd.concat(frames, ignore_index=True)[["area", "tec", "hour", "value"]]


def compute_nuclear_max_af(production, scenario_capa, areas, hour_week):
    """Compute weekly max nuclear availability factor from raw production.

    AF = hourly_production / installed_capacity, clipped to [0, 1].
    Weekly max = max AF within each week.

    Args:
        production: dict {area: DataFrame} from load_year_production.
        scenario_capa: DataFrame with columns ['area', 'tec', 'value'] (GW).
        areas: List of area codes.
        hour_week: DataFrame with columns ['hour', 'week'] (POSIX hours → YYWW).

    Returns:
        DataFrame with columns ['area', 'week', 'value'].
    """
    frames = []
    weeks = hour_week["week"].unique().tolist()

    for area in areas:
        if area not in production:
            # No nuclear data → full availability
            for w in weeks:
                frames.append({"area": area, "week": w, "value": 1.0})
            continue

        df = production[area]
        if "nuclear" not in df.columns:
            for w in weeks:
                frames.append({"area": area, "week": w, "value": 1.0})
            continue

        # Merge production with week mapping
        prod_hours = pd.DataFrame({
            "hour": _to_posix_hours(df["hour"]),
            "nuclear": df["nuclear"].values,
        })
        merged = prod_hours.merge(hour_week, on="hour", how="inner")

        # Get installed nuclear capacity (GW → MW)
        capa_row = scenario_capa[
            (scenario_capa["area"] == area) & (scenario_capa["tec"] == "nuclear")
        ]
        if not capa_row.empty:
            capa_mw = float(capa_row["value"].iloc[0]) * 1000
        else:
            capa_mw = merged["nuclear"].max() if merged["nuclear"].max() > 0 else 1

        merged["af"] = np.clip(merged["nuclear"] / max(capa_mw, 1), 0, 1)

        # Weekly max
        weekly = merged.groupby("week")["af"].max().reset_index()
        weekly["area"] = area
        weekly = weekly.rename(columns={"af": "value"})
        frames.append(weekly[["area", "week", "value"]])

    if not frames:
        return pd.DataFrame(columns=["area", "week", "value"])

    # frames can be list of dicts or DataFrames
    result_parts = []
    for f in frames:
        if isinstance(f, dict):
            result_parts.append(pd.DataFrame([f]))
        else:
            result_parts.append(f)
    return pd.concat(result_parts, ignore_index=True)[["area", "week", "value"]]


def compute_lake_inflows(production, areas, hour_month, eta_phs=0.855):
    """Compute monthly lake inflows from raw production data.

    lake_inflows = lake_prod + phs_prod - η * phs_cons, summed monthly, in TWh.

    Args:
        production: dict {area: DataFrame} from load_year_production.
        areas: List of area codes.
        hour_month: DataFrame with columns ['hour', 'month'] (POSIX hours → YYYYMM).
        eta_phs: PHS round-trip efficiency (default: 0.9 * 0.95 = 0.855).

    Returns:
        DataFrame with columns ['area', 'month', 'value'] in TWh.
    """
    frames = []

    for area in areas:
        if area not in production:
            continue
        df = production[area]
        prod_hours = pd.DataFrame({"hour": _to_posix_hours(df["hour"])})

        # Lake production
        if "lake" in df.columns:
            prod_hours["lake"] = df["lake"].values
        else:
            prod_hours["lake"] = 0.0

        # PHS prod/cons
        if "phs_prod" in df.columns:
            prod_hours["phs_prod"] = df["phs_prod"].values
        else:
            prod_hours["phs_prod"] = 0.0

        if "phs_cons" in df.columns:
            prod_hours["phs_cons"] = df["phs_cons"].values
        else:
            prod_hours["phs_cons"] = 0.0

        # Net inflow = lake_prod + phs_prod - η * phs_cons
        prod_hours["inflow_mw"] = (
            prod_hours["lake"] + prod_hours["phs_prod"] - eta_phs * prod_hours["phs_cons"]
        )

        # Merge with month mapping and aggregate
        merged = prod_hours.merge(hour_month, on="hour", how="inner")
        monthly = merged.groupby("month")["inflow_mw"].sum().reset_index()
        monthly["value"] = monthly["inflow_mw"].clip(lower=0) / 1e6  # MWh → TWh
        monthly["area"] = area
        frames.append(monthly[["area", "month", "value"]])

    if not frames:
        return pd.DataFrame(columns=["area", "month", "value"])

    return pd.concat(frames, ignore_index=True)[["area", "month", "value"]]


def compute_hydro_limits(production, areas, hour_month):
    """Compute monthly max hydro charge/discharge from raw production.

    hMaxOut = monthly max of (lake_prod + phs_prod), in GW.
    hMaxIn  = monthly max of phs_cons, in GW.

    Args:
        production: dict {area: DataFrame} from load_year_production.
        areas: List of area codes.
        hour_month: DataFrame with columns ['hour', 'month'].

    Returns:
        (hMaxIn, hMaxOut) DataFrames with columns ['area', 'month', 'value'] in GW.
    """
    frames_in = []
    frames_out = []

    for area in areas:
        if area not in production:
            continue
        df = production[area]
        prod_hours = pd.DataFrame({"hour": _to_posix_hours(df["hour"])})

        # Discharge: lake + PHS production
        lake_prod = df["lake"].values if "lake" in df.columns else np.zeros(len(df))
        phs_prod = df["phs_prod"].values if "phs_prod" in df.columns else np.zeros(len(df))
        prod_hours["out_mw"] = np.clip(lake_prod + phs_prod, 0, None)

        # Charge: PHS consumption
        phs_cons = df["phs_cons"].values if "phs_cons" in df.columns else np.zeros(len(df))
        prod_hours["in_mw"] = np.clip(phs_cons, 0, None)

        # Merge with month mapping and get monthly max
        merged = prod_hours.merge(hour_month, on="hour", how="inner")

        monthly_out = merged.groupby("month")["out_mw"].max().reset_index()
        monthly_out["value"] = monthly_out["out_mw"] / 1000  # MW → GW
        monthly_out["area"] = area
        frames_out.append(monthly_out[["area", "month", "value"]])

        monthly_in = merged.groupby("month")["in_mw"].max().reset_index()
        monthly_in["value"] = monthly_in["in_mw"] / 1000  # MW → GW
        monthly_in["area"] = area
        frames_in.append(monthly_in[["area", "month", "value"]])

    def _combine(frames):
        if not frames:
            return pd.DataFrame(columns=["area", "month", "value"])
        return pd.concat(frames, ignore_index=True)[["area", "month", "value"]]

    return _combine(frames_in), _combine(frames_out)


# ── Main entry point: load all time-varying inputs ──


def load_tv_inputs(data_dir, simul_year, areas, exo_areas,
                   actCF=False, rn_horizon="current", months=None,
                   scenario_capa=None):
    """Load all time-varying inputs from year-based data and compute derived variables.

    This is the main data loading function called at run creation time.

    Flow:
        1. Compute UTC boundaries from CET calendar
        2. Load raw production, demand, exo_prices from data/<year>/
        3. Load Ninja profiles from data/renewable_ninja/ (if not actCF)
        4. Compute hour_month and hour_week mappings
        5. Compute NMD from raw production
        6. Compute VRE capacity factors (from production if actCF, else from Ninja)
        7. Compute nucMaxAF, lake_inflows, hMaxIn, hMaxOut from raw production
        8. Return the same dict format as before (compatible with save_inputs + models)

    Args:
        data_dir: Path to the data/ directory.
        simul_year: Simulation year.
        areas: List of modeled country codes.
        exo_areas: List of non-modeled country codes.
        actCF: Use actual historical capacity factors instead of Renewable Ninja.
        rn_horizon: Renewables.ninja wind fleet ("current" or "future").
        months: Optional (start_month, end_month) tuple.
        scenario_capa: DataFrame with columns ['area', 'tec', 'value'] (GW).
            Required for computing CFs and nuclear AF from production data.

    Returns:
        Dict with keys: demand, nmd, exoPrices, vre_profiles, hour_month,
        hour_week, lake_inflows, hMaxIn, hMaxOut, nucMaxAF, hours, weeks, months.
    """
    data_dir = Path(data_dir)

    # 1. Compute UTC boundaries
    if months:
        start_m, end_m = months
        start = cet_to_utc(datetime(simul_year, start_m, 1))
        if end_m < 12:
            end = cet_to_utc(datetime(simul_year, end_m + 1, 1))
        else:
            end = cet_to_utc(datetime(simul_year + 1, 1, 1))
    else:
        start, end = cet_year_bounds(simul_year)

    # 2. Load raw data from year-based storage
    production = load_year_production(data_dir, simul_year, areas)
    demand_raw = _load_year_csv(data_dir, simul_year, "demand.csv", areas)
    exo_prices_raw = _load_year_csv(data_dir, simul_year, "exo_prices.csv", exo_areas)

    # 3. Format demand and exo prices (MW → GW for demand, prices stay EUR/MWh)
    demand_filtered = demand_raw[(demand_raw["hour"] >= start) & (demand_raw["hour"] < end)].copy()
    demand_filtered["hour"] = _to_posix_hours(demand_filtered["hour"])
    area_cols_demand = [c for c in areas if c in demand_filtered.columns]
    demand_filtered[area_cols_demand] = demand_filtered[area_cols_demand] / 1000  # MW → GW
    demand = pd.melt(demand_filtered, id_vars=["hour"], value_vars=area_cols_demand,
                     var_name="area", value_name="value")[["area", "hour", "value"]]

    exo_filtered = exo_prices_raw[(exo_prices_raw["hour"] >= start) & (exo_prices_raw["hour"] < end)].copy()
    exo_filtered["hour"] = _to_posix_hours(exo_filtered["hour"])
    exo_cols = [c for c in exo_areas if c in exo_filtered.columns]
    exoPrices = pd.melt(exo_filtered, id_vars=["hour"], value_vars=exo_cols,
                        var_name="area", value_name="value")[["area", "hour", "value"]]

    # 4. Hour-month and hour-week mappings
    hour_month, hour_week = compute_hour_mappings(simul_year, months)

    # 5. Compute NMD from production
    nmd = compute_nmd(production, areas, start, end)

    # 6. VRE capacity factors
    vre_profiles = pd.DataFrame()
    if actCF:
        if scenario_capa is None:
            raise ValueError(
                "scenario_capa is required when actCF=True "
                "(needed to compute capacity factors from production data)"
            )
        vre_cf = compute_vre_capacity_factors(
            production, scenario_capa, areas, start, end,
            technologies=["offshore", "onshore", "pv"],
        )
        vre_profiles = vre_cf
    else:
        offshore = load_ninja_var(data_dir, f"offshore_{rn_horizon}", areas, start, end)
        onshore = load_ninja_var(data_dir, f"onshore_{rn_horizon}", areas, start, end)
        pv = load_ninja_var(data_dir, "pv", areas, start, end)
        offshore["tec"] = "offshore"
        onshore["tec"] = "onshore"
        pv["tec"] = "pv"
        vre_profiles = pd.concat([
            offshore[["area", "tec", "hour", "value"]],
            onshore[["area", "tec", "hour", "value"]],
            pv[["area", "tec", "hour", "value"]],
        ])

    # River CF (always from production data, since Ninja doesn't have it)
    if scenario_capa is not None:
        river_cf = compute_vre_capacity_factors(
            production, scenario_capa, areas, start, end,
            technologies=["river"],
        )
    else:
        # Fallback: use production/max as CF proxy
        river_cf = compute_vre_capacity_factors(
            production, pd.DataFrame(columns=["area", "tec", "value"]),
            areas, start, end, technologies=["river"],
        )
    vre_profiles = pd.concat([vre_profiles, river_cf])

    # 7. Derived monthly/weekly variables from production
    if scenario_capa is not None:
        nucMaxAF = compute_nuclear_max_af(production, scenario_capa, areas, hour_week)
    else:
        # No scenario capacity → assume full availability
        weeks_list = hour_week["week"].unique().tolist()
        nuc_rows = [{"area": a, "week": w, "value": 1.0}
                    for a in areas for w in weeks_list]
        nucMaxAF = pd.DataFrame(nuc_rows)

    lake_inflows = compute_lake_inflows(production, areas, hour_month)
    hMaxIn, hMaxOut = compute_hydro_limits(production, areas, hour_month)

    # Ensure all areas are present in monthly/weekly data (fill missing with defaults)
    all_months = sorted(hour_month["month"].unique().tolist())
    all_weeks = sorted(hour_week["week"].unique().tolist())

    for a in areas:
        if a not in lake_inflows["area"].values:
            filler = pd.DataFrame({"area": a, "month": all_months, "value": 0.0})
            lake_inflows = pd.concat([lake_inflows, filler], ignore_index=True)
        if a not in hMaxIn["area"].values:
            filler = pd.DataFrame({"area": a, "month": all_months, "value": 0.0})
            hMaxIn = pd.concat([hMaxIn, filler], ignore_index=True)
            hMaxOut = pd.concat([hMaxOut, filler], ignore_index=True)
        if a not in nucMaxAF["area"].values:
            filler = pd.DataFrame({"area": a, "week": all_weeks, "value": 1.0})
            nucMaxAF = pd.concat([nucMaxAF, filler], ignore_index=True)

    hours = hour_month["hour"].unique().tolist()
    weeks = all_weeks
    months_list = all_months

    return {
        "demand": demand,
        "nmd": nmd,
        "exoPrices": exoPrices,
        "vre_profiles": vre_profiles,
        "hour_month": hour_month,
        "hour_week": hour_week,
        "lake_inflows": lake_inflows,
        "hMaxIn": hMaxIn,
        "hMaxOut": hMaxOut,
        "nucMaxAF": nucMaxAF,
        "hours": hours,
        "weeks": weeks,
        "months": months_list,
    }


# ── Scenario loading ──


def _read_scenario_table(scenario_path, name, **kwargs):
    """Read a scenario table from either a CSV directory or an Excel file."""
    scenario_path = Path(scenario_path)
    if scenario_path.is_dir():
        return pd.read_csv(scenario_path / f"{name}.csv", **kwargs)
    else:
        return pd.read_excel(scenario_path, sheet_name=name, **kwargs)


def extract_scenario(scenario_path, areas, exo_areas, hour_month):
    """Extract and format scenario parameters.

    Args:
        scenario_path: Path to a scenario directory (containing CSVs) or an Excel file.
        areas: List of modeled country codes.
        exo_areas: List of non-modeled country codes.
        hour_month: DataFrame with hour-month mapping.

    Returns a dict with all scenario DataFrames and technology sets.
    """
    scenario_path = Path(scenario_path)

    thr_specs = _read_scenario_table(scenario_path, "thr_specs")

    # Extract individual columns from thr_specs as separate DataFrames
    thr_params = {}
    for col in thr_specs.columns[1:]:
        thr_params[col] = thr_specs[["tec", col]]

    rsv_req = _read_scenario_table(scenario_path, "rsv_req")
    str_vOM = _read_scenario_table(scenario_path, "str_vOM")

    # Technology sets
    thr = thr_specs["tec"].tolist()
    vre = rsv_req["tec"].tolist()
    str_tec = str_vOM["tec"].tolist()
    tec = ["nmd"] + vre + thr + str_tec

    frr_df = thr_params["frr"]
    frr = frr_df[frr_df["frr"]].tec.tolist() + str_tec
    no_frr = list(set(tec) - set(frr))

    # Installed capacity data
    def _read_melt(name, id_col, filter_areas):
        df = pd.melt(_read_scenario_table(scenario_path, name), id_vars=id_col, var_name="area")
        return df[df["area"].isin(filter_areas)][["area", id_col, "value"]] if id_col != "area" else df[df["area"].isin(filter_areas)]

    capa = _read_melt("capa", "tec", areas)
    maxAF = _read_melt("maxAF", "tec", areas)
    yEAF = _read_melt("yEAF", "tec", areas)
    capa_in = _read_melt("capa_in", "tec", areas)
    stockMax = _read_melt("stockMax", "tec", areas)

    # Interconnections
    links = pd.melt(_read_scenario_table(scenario_path, "links"), id_vars="exporter", var_name="importer")
    links = links[(links["importer"].isin(areas)) & (links["exporter"].isin(areas))][["importer", "exporter", "value"]]

    exo_EX = pd.melt(_read_scenario_table(scenario_path, "exo_EX"), id_vars="exporter", var_name="importer")
    exo_EX = exo_EX[(exo_EX["exporter"].isin(areas)) & (exo_EX["importer"].isin(exo_areas))]

    exo_IM = pd.melt(_read_scenario_table(scenario_path, "exo_IM"), id_vars="importer", var_name="exporter")
    exo_IM = exo_IM[(exo_IM["importer"].isin(areas)) & (exo_IM["exporter"].isin(exo_areas))]

    # Fuel price seasonal weights (calendar months 1-12, mean=1 per fuel).
    # Expand to YYYYMM strings matching the simulation period.
    fuel_timeFactor_raw = pd.melt(
        _read_scenario_table(scenario_path, "fuel_timeFactor"),
        id_vars="month", var_name="fuel",
    )
    sim_months = hour_month["month"].unique()
    sim_months_df = pd.DataFrame({
        "yyyymm": sim_months,
        "month": [int(m[-2:]) for m in sim_months],
    })
    fuel_timeFactor = fuel_timeFactor_raw.merge(sim_months_df, on="month", how="inner")
    fuel_timeFactor = fuel_timeFactor[["fuel", "yyyymm", "value"]].rename(columns={"yyyymm": "month"})

    fuel_areaFactor = pd.melt(
        _read_scenario_table(scenario_path, "fuel_areaFactor"),
        id_vars="area", var_name="fuel",
    )
    fuel_areaFactor = fuel_areaFactor[fuel_areaFactor["area"].isin(areas)][["fuel", "area", "value"]]

    return {
        "thr_params": thr_params,
        "rsv_req": rsv_req,
        "str_vOM": str_vOM,
        "thr": thr,
        "vre": vre,
        "str_tec": str_tec,
        "tec": tec,
        "frr": frr,
        "no_frr": no_frr,
        "capa": capa,
        "maxAF": maxAF,
        "yEAF": yEAF,
        "capa_in": capa_in,
        "stockMax": stockMax,
        "links": links,
        "exo_EX": exo_EX,
        "exo_IM": exo_IM,
        "fuel_timeFactor": fuel_timeFactor,
        "fuel_areaFactor": fuel_areaFactor,
    }


def xlsx_to_scenario(xlsx_path, output_dir=None):
    """Convert a scenario Excel file to a directory of CSVs.

    Args:
        xlsx_path: Path to the .xlsx scenario file.
        output_dir: Output directory. Defaults to the xlsx stem name in the same parent dir.

    Returns:
        Path to the created scenario directory.
    """
    xlsx_path = Path(xlsx_path)
    if output_dir is None:
        output_dir = xlsx_path.parent / xlsx_path.stem.lower().replace("scenario_", "")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sheets = [
        "thr_specs", "rsv_req", "str_vOM", "capa", "maxAF", "yEAF",
        "capa_in", "stockMax", "links", "exo_IM", "exo_EX",
        "fuel_timeFactor", "fuel_areaFactor",
    ]
    for sheet in sheets:
        df = pd.read_excel(xlsx_path, sheet_name=sheet)
        df.to_csv(output_dir / f"{sheet}.csv", index=False)

    print(f"Scenario exported to {output_dir}/")
    return output_dir


def save_inputs(run_dir, tv_data, scenario_data, areas, exo_areas):
    """Save all formatted inputs as CSVs to the run's input directory."""
    input_dir = Path(run_dir) / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)

    # Save time-varying data
    for name, data in tv_data.items():
        if isinstance(data, pd.DataFrame):
            data.to_csv(input_dir / f"{name}.csv", index=False, header=False)
        elif isinstance(data, list):
            pd.DataFrame(data).to_csv(input_dir / f"{name}.csv", index=False, header=False)

    # Save scenario data
    for name, data in scenario_data.items():
        if name == "thr_params":
            for param_name, param_df in data.items():
                param_df.to_csv(input_dir / f"{param_name}.csv", index=False, header=False)
        elif isinstance(data, pd.DataFrame):
            data.to_csv(input_dir / f"{name}.csv", index=False, header=False)
        elif isinstance(data, list):
            pd.DataFrame(data).to_csv(input_dir / f"{name}.csv", index=False, header=False)

    # Save area lists
    pd.DataFrame(areas).to_csv(input_dir / "areas.csv", index=False, header=False)
    pd.DataFrame(exo_areas).to_csv(input_dir / "exo_areas.csv", index=False, header=False)
