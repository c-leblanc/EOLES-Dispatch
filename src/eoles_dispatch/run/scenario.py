"""Scenario loading, extraction, and conversion for EOLES-Dispatch.

Reads scenario parameters from CSV directories or Excel files and formats
them for the Pyomo model. Also provides xlsx_to_scenario for converting
Excel scenario files to CSV directories.
"""

from pathlib import Path

import pandas as pd


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
