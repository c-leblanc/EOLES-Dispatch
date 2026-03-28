"""Load and format input data for the EOLES-Dispatch model.

Reads year-based intermediate data from data/<year>/, delegates derived
variable computations to compute.py, scenario handling to scenario.py,
and formats everything for the Pyomo model.
"""

from pathlib import Path

import pandas as pd

from ..utils import to_posix_hours
from .compute import (
    compute_hydro_limits,
    compute_lake_inflows,
    compute_nmd,
    compute_nuclear_max_af,
    compute_vre_capacity_factors,
)

# ── High-level entry points ──


def load_tv_inputs(
    data_dir, simul_year, areas, exo_areas, hour_month, hour_week, actCF=False, rn_horizon="current"
):
    """Load all time-varying inputs from year-based data and compute derived variables.

    This is the main data loading function called at run creation time.
    hour_month and hour_week (computed once by the caller) define the simulation
    period — all data is filtered to match their hours.

    Args:
        data_dir: Path to the data/ directory.
        simul_year: Simulation year.
        areas: List of modeled country codes.
        exo_areas: List of non-modeled country codes.
        hour_month: DataFrame with columns ['hour', 'month'] (POSIX hours).
        hour_week: DataFrame with columns ['hour', 'week'] (POSIX hours).
        actCF: Use actual historical capacity factors instead of Renewable Ninja.
        rn_horizon: Renewables.ninja wind fleet ("current" or "future").

    Returns:
        Dict with keys: demand, nmd, exoPrices, vre_profiles, hour_month,
        hour_week, lake_inflows, hMaxIn, hMaxOut, nucMaxAF, hours, weeks, months.
    """
    data_dir = Path(data_dir)
    valid_hours = set(hour_month["hour"].unique())

    # 1. Load raw data and filter to the simulation period
    production_raw = load_year_production(data_dir, simul_year, areas)
    production = {area: _filter_to_posix(df, valid_hours) for area, df in production_raw.items()}

    demand_raw = _load_year_csv(data_dir, simul_year, "demand.csv", areas)
    demand_filtered = _filter_to_posix(demand_raw, valid_hours)
    area_cols_demand = [c for c in areas if c in demand_filtered.columns]
    demand_filtered[area_cols_demand] = demand_filtered[area_cols_demand] / 1000  # MW → GW
    demand = pd.melt(
        demand_filtered,
        id_vars=["hour"],
        value_vars=area_cols_demand,
        var_name="area",
        value_name="value",
    )[["area", "hour", "value"]]

    exo_prices_raw = _load_year_csv(data_dir, simul_year, "exo_prices.csv", exo_areas)
    exo_filtered = _filter_to_posix(exo_prices_raw, valid_hours)
    exo_cols = [c for c in exo_areas if c in exo_filtered.columns]
    exoPrices = pd.melt(
        exo_filtered, id_vars=["hour"], value_vars=exo_cols, var_name="area", value_name="value"
    )[["area", "hour", "value"]]

    # Load installed capacity (wide format: tec index, area columns, MW)
    icapa_path = data_dir / str(simul_year) / "installed_capacity.csv"
    if icapa_path.exists():
        installed_capa = pd.read_csv(icapa_path, index_col="tec")
    else:
        installed_capa = None

    # 2. NMD
    nmd = compute_nmd(production, areas)

    # 3. VRE capacity factors
    if actCF:
        vre_profiles = compute_vre_capacity_factors(
            production,
            installed_capa,
            areas,
            technologies=["offshore", "onshore", "solar"],
        )
    else:
        offshore = load_ninja_var(data_dir, f"offshore_{rn_horizon}", areas, valid_hours)
        onshore = load_ninja_var(data_dir, f"onshore_{rn_horizon}", areas, valid_hours)
        solar = load_ninja_var(data_dir, "solar", areas, valid_hours)
        offshore["tec"] = "offshore"
        onshore["tec"] = "onshore"
        solar["tec"] = "solar"
        vre_profiles = pd.concat(
            [
                offshore[["area", "tec", "hour", "value"]],
                onshore[["area", "tec", "hour", "value"]],
                solar[["area", "tec", "hour", "value"]],
            ]
        )

    # River CF: always from production, installed_capa used if available
    river_cf = compute_vre_capacity_factors(
        production, installed_capa, areas, technologies=["river"]
    )
    vre_profiles = pd.concat([vre_profiles, river_cf])

    # 4. Derived monthly/weekly variables
    nucMaxAF = compute_nuclear_max_af(production, installed_capa, areas, hour_week)

    lake_inflows = compute_lake_inflows(production, areas, hour_month)
    hMaxIn, hMaxOut = compute_hydro_limits(production, areas, hour_month)

    # Fill missing areas with defaults
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
        "hours": sorted(valid_hours),
        "weeks": all_weeks,
        "months": all_months,
    }


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


# ── Data loaders and helpers ──


def load_year_production(data_dir, year, areas):
    """Load raw production data for all areas from data/<year>/production_<area>.csv.

    Args:
        data_dir: Path to the data/ directory.
        year: The simulation year.
        areas: List of area codes.

    Returns:
        dict {area: pd.DataFrame} with columns ['hour', prodtype1, prodtype2, ...].
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


def load_ninja_var(data_dir, variable, areas, valid_hours):
    """Load a Renewable Ninja capacity factor variable.

    Reads from data/renewable_ninja/<variable>.csv and filters to the
    requested hours.

    Args:
        data_dir: Path to the data/ directory.
        variable: Ninja variable name (e.g. "offshore_current").
        areas: List of area codes.
        valid_hours: Set or array of POSIX hours to keep.

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
    df["hour"] = to_posix_hours(pd.to_datetime(df["hour"]))
    df = df[df["hour"].isin(valid_hours)]
    if df.empty:
        raise ValueError(
            f"No data for '{variable}' in the requested period. "
            f"Run 'eoles-dispatch collect --source ninja' to download data."
        )
    melted = pd.melt(df, id_vars=["hour"], value_vars=areas, var_name="area", value_name="value")
    return melted[["area", "hour", "value"]]


# ── Filtering helper ──


def _filter_to_posix(df, valid_hours):
    """Filter a raw CSV DataFrame to valid_hours and convert 'hour' to POSIX hours.

    Args:
        df: DataFrame with a datetime 'hour' column.
        valid_hours: Set of POSIX hours to keep.

    Returns:
        Filtered DataFrame with 'hour' as POSIX int.
    """
    df = df.copy()
    df["hour"] = to_posix_hours(df["hour"])
    return df[df["hour"].isin(valid_hours)]
