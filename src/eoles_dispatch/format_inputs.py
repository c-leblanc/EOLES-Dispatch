"""Load and format input data for the EOLES-Dispatch model.

Reads time-varying data (demand, VRE profiles, hydro, nuclear) and scenario
parameters from an Excel file, then saves formatted CSVs to a run directory.
"""

import os
from datetime import datetime
from pathlib import Path

import pandas as pd


def load_time_varying_var(data_dir, variable, areas, start, end):
    """Load a single time-varying variable from CSV, filter by date and areas."""
    csv_path = data_dir / "time_varying_inputs" / f"{variable}.csv"
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
            f"Run 'eoles-dispatch collect' to download data for this period."
        )
    df["hour"] = (df["hour"] - datetime(1970, 1, 1)).dt.total_seconds() / 3600
    df["hour"] = df["hour"].astype(int)
    melted = pd.melt(df, id_vars=["hour"], value_vars=areas, var_name="area", value_name="value")
    return melted[["area", "hour", "value"]]


def load_ninja_var(data_dir, variable, areas, start, end):
    """Load a Renewable Ninja capacity factor variable."""
    csv_path = data_dir / "renewable_ninja" / f"{variable}.csv"
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
            f"Run 'eoles-dispatch collect' to download data for this period."
        )
    df["hour"] = (df["hour"] - datetime(1970, 1, 1)).dt.total_seconds() / 3600
    df["hour"] = df["hour"].astype(int)
    melted = pd.melt(df, id_vars=["hour"], value_vars=areas, var_name="area", value_name="value")
    return melted[["area", "hour", "value"]]


def load_tv_inputs(data_dir, simul_year, areas, exo_areas, actCF=False, rn_horizon="current", months=None):
    """Load all time-varying inputs and return them as a dict of DataFrames/lists.

    Args:
        months: Tuple (start_month, end_month) e.g. (1,3) for Jan-Mar,
                (8,8) for August only. None = full year.
    """
    if months:
        start_m, end_m = months
        start = datetime(simul_year, start_m, 1)
        if end_m < 12:
            end = datetime(simul_year, end_m + 1, 1)
        else:
            end = datetime(simul_year + 1, 1, 1)
    else:
        start = datetime(simul_year, 1, 1)
        end = datetime(simul_year + 1, 1, 1)

    demand = load_time_varying_var(data_dir, "demand", areas, start, end)
    nmd = load_time_varying_var(data_dir, "nmd", areas, start, end)
    exoPrices = load_time_varying_var(data_dir, "exoPrices", exo_areas, start, end)

    # VRE production profiles
    vre_profiles = pd.DataFrame()
    if actCF:
        for tec in ["offshore", "onshore", "pv"]:
            temp = load_time_varying_var(data_dir, tec, areas, start, end)
            temp["tec"] = tec
            vre_profiles = pd.concat([vre_profiles, temp[["area", "tec", "hour", "value"]]])
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

    river = load_time_varying_var(data_dir, "river", areas, start, end)
    river["tec"] = "river"
    vre_profiles = pd.concat([vre_profiles, river[["area", "tec", "hour", "value"]]])

    # Hour-month and hour-week mappings
    hour_month = pd.DataFrame({"hour": demand["hour"].unique()})
    hour_month["hour_POSIX"] = pd.to_datetime(hour_month["hour"] * 3600, unit="s", origin="1970-01-01", utc=True)
    hour_month["month"] = hour_month["hour_POSIX"].dt.strftime("%Y%m").astype(str)
    hour_month = hour_month[["hour", "month"]]

    hour_week = pd.DataFrame({"hour": demand["hour"].unique()})
    hour_week["hour_POSIX"] = pd.to_datetime(hour_week["hour"] * 3600, unit="s", origin="1970-01-01", utc=True)
    hour_week["week"] = hour_week["hour_POSIX"].dt.strftime("%Y%W").astype(str)
    hour_week = hour_week[["hour", "week"]]

    # Monthly data
    tv_dir = data_dir / "time_varying_inputs"

    lake_inflows = pd.read_csv(tv_dir / "lake_inflows.csv", dtype={"month": str})
    lake_inflows = lake_inflows[["month"] + areas]
    lake_inflows = pd.melt(lake_inflows, id_vars="month", var_name="area", value_name="value")
    lake_inflows = lake_inflows[lake_inflows["month"].isin(hour_month["month"])]
    lake_inflows = lake_inflows[["area", "month", "value"]]

    hMaxIn = pd.read_csv(tv_dir / "hMaxIn.csv", dtype={"month": str})
    hMaxIn = hMaxIn[["month"] + areas]
    hMaxIn = pd.melt(hMaxIn, id_vars="month", var_name="area", value_name="value")
    hMaxIn = hMaxIn[hMaxIn["month"].isin(hour_month["month"])]
    hMaxIn = hMaxIn[["area", "month", "value"]]

    hMaxOut = pd.read_csv(tv_dir / "hMaxOut.csv", dtype={"month": str})
    hMaxOut = hMaxOut[["month"] + areas]
    hMaxOut = pd.melt(hMaxOut, id_vars="month", var_name="area", value_name="value")
    hMaxOut = hMaxOut[hMaxOut["month"].isin(hour_month["month"])]
    hMaxOut = hMaxOut[["area", "month", "value"]]

    nucMaxAF = pd.read_csv(tv_dir / "nucMaxAF.csv", dtype={"week": str})
    nucMaxAF = nucMaxAF[["week"] + areas]
    nucMaxAF = pd.melt(nucMaxAF, id_vars="week", var_name="area", value_name="value")
    nucMaxAF = nucMaxAF[nucMaxAF["week"].isin(hour_week["week"])]
    nucMaxAF = nucMaxAF[["area", "week", "value"]]

    hours = hour_month["hour"].unique().tolist()
    weeks = hour_week["week"].unique().tolist()
    months = hour_month["month"].unique().tolist()

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
        "months": months,
    }


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

    # Fuel price correction factors
    fuel_timeFactor = pd.melt(
        _read_scenario_table(scenario_path, "fuel_timeFactor", dtype={"month": str}),
        id_vars="month", var_name="fuel",
    )
    fuel_timeFactor = fuel_timeFactor[fuel_timeFactor.month.isin(hour_month["month"])][["fuel", "month", "value"]]

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


def prepare_run(run_dir, data_dir, scenario_path, year, areas, exo_areas,
                actCF=False, rn_horizon="current"):
    """Full pipeline: load data, extract scenario, save formatted inputs.

    Args:
        run_dir: Path to the run directory (e.g. runs/my_run)
        data_dir: Path to the data directory containing time_varying_inputs/ and renewable_ninja/
        scenario_path: Path to a scenario directory (CSVs) or Excel file (.xlsx)
        year: Simulation year (2016-2019)
        areas: List of modeled country codes
        exo_areas: List of non-modeled country codes
        actCF: Use actual historical capacity factors instead of Renewable Ninja
        rn_horizon: Renewables.ninja wind fleet ("current" or "future")
    """
    data_dir = Path(data_dir)
    tv_data = load_tv_inputs(data_dir, year, areas, exo_areas, actCF, rn_horizon)
    scenario_data = extract_scenario(scenario_path, areas, exo_areas, tv_data["hour_month"])
    save_inputs(run_dir, tv_data, scenario_data, areas, exo_areas)
    return tv_data, scenario_data
