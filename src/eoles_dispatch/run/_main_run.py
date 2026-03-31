"""Run management for EOLES-Dispatch.

A run is a self-contained directory with inputs, outputs, and metadata.
It captures everything needed to reproduce and understand a simulation result.

Directory structure:
    runs/<name>/
        run.yaml          - metadata (scenario, year, parameters, timestamps)
        inputs/           - formatted model inputs (CSVs)
        outputs/          - model results (CSVs)
        scenario/         - copy of the scenario used
"""

import logging
import shutil
import time
from datetime import datetime
from pathlib import Path

import yaml

from ..collect._main_collect import collect_all
from ..config import DEFAULT_AREAS, DEFAULT_EXO_AREAS

logger = logging.getLogger(__name__)


# ── High-level entry points ──


def create_run(
    name,
    scenario,
    year,
    project_dir=None,
    areas=None,
    exo_areas=None,
    actCF=False,
    rn_horizon="current",
    auto_download=True,
    months=None,
):
    """Create a new run: fetch data if needed, format inputs, copy scenario.

    Args:
        name: Run name (determines directory name).
        scenario: Scenario name (looks for scenarios/<name>/ dir, then .xlsx).
        year: Simulation year.
        project_dir: Root project directory. Defaults to cwd.
        areas: List of modeled country codes.
        exo_areas: List of non-modeled country codes.
        actCF: Use actual historical capacity factors.
        rn_horizon: Renewables.ninja wind fleet ("current" or "future").
        auto_download: Automatically download missing data.

    Returns:
        Path to the created run directory.
    """
    if areas is None:
        areas = list(DEFAULT_AREAS)
    if exo_areas is None:
        exo_areas = list(DEFAULT_EXO_AREAS)
    if project_dir is None:
        project_dir = Path.cwd()
    else:
        project_dir = Path(project_dir)

    run_dir = project_dir / "runs" / name
    if run_dir.exists():
        raise FileExistsError(f"Run '{name}' already exists at {run_dir}")

    data_dir = project_dir / "data"

    # Resolve scenario
    scenario_dir = project_dir / "scenarios" / scenario
    scenario_xlsx = project_dir / "scenarios" / f"{scenario}.xlsx"
    if scenario_dir.is_dir():
        scenario_path = scenario_dir
    elif scenario_xlsx.exists():
        scenario_path = scenario_xlsx
    else:
        raise FileNotFoundError(
            f"Scenario '{scenario}' not found. Expected either "
            f"{scenario_dir}/ (CSV directory) or {scenario_xlsx}"
        )

    # Auto-download data if needed
    if auto_download:
        _ensure_data_available(data_dir, year, areas, exo_areas)

    # Create run directory
    run_dir.mkdir(parents=True)
    logger.info(f"Creating run '{name}'...")

    # Format and save inputs
    # New flow: scenario first (needs hour_month), then tv_data (needs scenario_capa)
    from ..utils import compute_hour_mappings
    from .format_inputs import load_tv_inputs, save_inputs
    from .scenario import extract_scenario

    logger.info("  Computing time mappings...")
    hour_month, hour_week = compute_hour_mappings(year, months=months)

    logger.info("  Loading scenario parameters...")
    scenario_data = extract_scenario(scenario_path, areas, exo_areas, hour_month)

    logger.info("  Loading time-varying data and computing derived variables...")
    tv_data = load_tv_inputs(
        data_dir,
        year,
        areas,
        exo_areas,
        hour_month,
        hour_week,
        actCF=actCF,
        rn_horizon=rn_horizon,
    )

    logger.info("  Saving formatted inputs...")
    save_inputs(run_dir, tv_data, scenario_data, areas, exo_areas)

    # TODO : Load validation data upon viz --validation, not upon create
    # Copy historical prices and production for validation (if available)
    _copy_actual_prices(data_dir, run_dir, year, areas, months)
    _copy_actual_production(data_dir, run_dir, year, areas, months)

    # Copy scenario into run directory for reproducibility
    scenario_copy_dir = run_dir / "scenario"
    scenario_copy_dir.mkdir()
    if scenario_path.is_dir():
        for csv_file in scenario_path.glob("*.csv"):
            shutil.copy2(csv_file, scenario_copy_dir / csv_file.name)
    else:
        shutil.copy2(scenario_path, scenario_copy_dir / scenario_path.name)

    # Write metadata
    metadata = {
        "name": name,
        "scenario": scenario,
        "year": year,
        "areas": areas,
        "exo_areas": exo_areas,
        "actCF": actCF,
        "rn_horizon": rn_horizon,
        "months": f"{months[0]}-{months[1]}"
        if months and months[0] != months[1]
        else str(months[0])
        if months
        else None,
        "created": datetime.now().isoformat(timespec="seconds"),
        "status": "created",
    }
    with open(run_dir / "run.yaml", "w") as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)

    logger.info(f"  Run created at {run_dir}")
    return run_dir


def solve_run(
    name,
    project_dir=None,
    solver="highs",
    version="standard",
    reports=None,
    full_diag=False,
):
    """Solve an existing run.

    Args:
        name: Run name.
        project_dir: Root project directory.
        solver: Solver name (e.g. "highs", "cbc", "gurobi").
        version: Model version ("standard" or "static_thermal").
        reports: List of reports to generate.
        full_diag: If True, export exhaustive diagnostics (all variables and
            duals) to runs/<name>/diagnostics/.

    Returns:
        Solver results object.
    """
    import gc

    import pyomo.environ  # noqa: F401 — registers solver plugins
    from pyomo.opt import SolverFactory

    from ..models import MODEL_REGISTRY
    from .format_outputs import (
        report_capa_on,
        report_FRtrade,
        report_prices,
        report_production,
        write_log,
    )

    if reports is None:
        reports = ["prices", "production"]

    if project_dir is None:
        project_dir = Path.cwd()
    else:
        project_dir = Path(project_dir)

    run_dir = project_dir / "runs" / name
    meta_path = run_dir / "run.yaml"

    if not meta_path.exists():
        raise FileNotFoundError(f"Run '{name}' not found. Create it first with 'create'.")

    with open(meta_path) as f:
        metadata = yaml.safe_load(f)

    logger.info(f"Solving run '{name}' (scenario={metadata['scenario']}, year={metadata['year']})")
    start_time = time.localtime()
    start_monotonic = time.monotonic()

    # Build model
    build_model = MODEL_REGISTRY.get(version)
    if build_model is None:
        raise ValueError(
            f"Unknown model version '{version}'. Choose from: {list(MODEL_REGISTRY.keys())}"
        )
    logger.info(f"  Building model [{version}]...")
    model = build_model(run_dir)

    # Solve
    # Pyomo uses "appsi_highs" as the SolverFactory name for HiGHS
    solver_name = "appsi_highs" if solver == "highs" else solver
    logger.info(f"  Solving with {solver}...")
    opt = SolverFactory(solver_name)

    # HiGHS-specific tuning for large LP models
    if solver == "highs":
        opt.highs_options["solver"] = "ipm"  # Interior point method (faster on large LPs)
        opt.highs_options["run_crossover"] = "on"  # Get a basic feasible solution for duals

    results = opt.solve(model, tee=True)

    # Check solver status
    from pyomo.opt import TerminationCondition

    tc = results.solver.termination_condition
    if tc not in (TerminationCondition.optimal, TerminationCondition.feasible):
        raise RuntimeError(
            f"Solver did not find an optimal solution. "
            f"Termination condition: {tc}. Check model feasibility."
        )
    if tc == TerminationCondition.feasible:
        logger.warning("  Solver returned a feasible (but not proven optimal) solution.")

    # Create outputs directory
    (run_dir / "outputs").mkdir(exist_ok=True)

    # Generate reports
    logger.info("  Generating reports...")
    report_map = {
        "prices": report_prices,
        "production": report_production,
        "capa_on": report_capa_on,
        "FRtrade": report_FRtrade,
    }
    for report_name in reports:
        if report_name in report_map:
            report_map[report_name](model, run_dir)

    if full_diag:
        from .export_diagnostics import export_all_diagnostics

        export_all_diagnostics(model, run_dir)

    elapsed_seconds = int(time.monotonic() - start_monotonic)
    hours, remainder = divmod(elapsed_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    exec_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    logger.info(f"  Done in {exec_str}")

    # Update metadata
    metadata["status"] = "solved"
    metadata["solved"] = datetime.now().isoformat(timespec="seconds")
    metadata["solver"] = solver
    metadata["model_version"] = version
    metadata["reports"] = reports
    metadata["exec_time"] = exec_str
    with open(meta_path, "w") as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)

    write_log(run_dir, model, name, metadata["scenario"], metadata["year"], start_time, exec_str)

    del model, opt
    gc.collect()

    return results


def load_run_metadata(name, project_dir=None):
    """Load metadata for a run."""
    if project_dir is None:
        project_dir = Path.cwd()
    else:
        project_dir = Path(project_dir)

    meta_path = project_dir / "runs" / name / "run.yaml"
    if not meta_path.exists():
        raise FileNotFoundError(f"Run '{name}' not found at {meta_path}")

    with open(meta_path) as f:
        return yaml.safe_load(f)


def list_runs(project_dir=None):
    """List all runs and their status.

    Returns a list of dicts with run metadata.
    """
    if project_dir is None:
        project_dir = Path.cwd()
    else:
        project_dir = Path(project_dir)

    runs_dir = project_dir / "runs"
    if not runs_dir.exists():
        return []

    runs = []
    for run_dir in sorted(runs_dir.iterdir()):
        meta_path = run_dir / "run.yaml"
        if meta_path.exists():
            with open(meta_path) as f:
                runs.append(yaml.safe_load(f))
    return runs


# ── Helpers ──


def _copy_actual_prices(data_dir, run_dir, year, areas, months):
    """Copy historical day-ahead prices into runs/<name>/validation/.

    Reads actual_prices.csv from the year data directory, filters to the
    requested areas and time period, converts timestamps to POSIX hours
    (matching outputs/prices.csv format), and saves into validation/.

    Silently skips if actual_prices.csv does not exist.
    """
    import pandas as pd

    from ..utils import cet_period_bounds, to_posix_hours

    start, end = cet_period_bounds(year, months)
    area_frames = []
    for area in areas:
        src = data_dir / str(year) / f"prices_{area}.csv"
        if not src.exists():
            continue
    
        df = pd.read_csv(src, parse_dates=["hour"])
        df = df[(df["hour"] >= start) & (df["hour"] < end)].copy()
        if df.empty:
            continue
        # Add "area" variable
        result = pd.DataFrame({"area": area, "hour": df["hour"], "price": df["prices"]})
        # Convert hour to POSIX hours (int)
        result["hour"] = to_posix_hours(result["hour"])

        area_frames.append(result)

    if not area_frames:
        return
    combined_long = pd.concat(area_frames, ignore_index=True)
    combined_wide = combined_long.pivot_table(index="hour", columns="area", values="price")

    validation_dir = run_dir / "validation"
    validation_dir.mkdir(exist_ok=True)
    combined_wide.to_csv(validation_dir / "actual_prices.csv", index=True)


def _copy_actual_production(data_dir, run_dir, year, areas, months):
    """Copy historical generation data into runs/<name>/validation/.

    Reads production_<area>.csv files from the year data directory, filters to
    the requested areas and time period, aggregates raw technology columns to
    the agg level (via RAW_TO_AGG), converts timestamps to
    POSIX hours.  Then joins demand from the run inputs and derives
    net_imports / net_exports as demand minus total production.

    The resulting CSV mirrors the column layout of outputs/production.csv:
    area, hour, <tec_cols>, net_imports, net_exports, demand.

    Silently skips areas whose production file does not exist.
    """
    import numpy as np
    import pandas as pd

    from ..config import RAW_TO_AGG
    from ..utils import cet_period_bounds, to_posix_hours

    start, end = cet_period_bounds(year, months)

    area_frames = []
    for area in areas:
        src = data_dir / str(year) / f"production_{area}.csv"
        if not src.exists():
            continue

        df = pd.read_csv(src, parse_dates=["hour"])
        df = df[(df["hour"] >= start) & (df["hour"] < end)].copy()
        if df.empty:
            continue

        # TODO: this RAW_TO_AGG aggregation loop (raw cols → agg categories, MW→GW)
        # is the canonical implementation. Extract to a shared helper in collect/ or utils.
        agg_cols = {}
        for raw_col, agg_name in RAW_TO_AGG.items():
            if raw_col not in df.columns:
                continue
            if agg_name not in agg_cols:
                agg_cols[agg_name] = df[raw_col].values.copy()
            else:
                agg_cols[agg_name] = agg_cols[agg_name] + df[raw_col].values

        result = pd.DataFrame({"hour": df["hour"], "area": area})
        for agg_name, vals in agg_cols.items():
            result[agg_name] = vals

        # Convert hour to POSIX hours (int)
        result["hour"] = to_posix_hours(result["hour"])

        area_frames.append(result)

    if not area_frames:
        return

    combined = pd.concat(area_frames, ignore_index=True)

    # Join demand from run inputs (already in GW / POSIX hours)
    demand_path = run_dir / "inputs" / "demand.csv"
    demand = pd.read_csv(demand_path, header=None, names=["area", "hour", "demand"])
    combined = combined.merge(demand, on=["area", "hour"], how="left")

    # Derive net imports / exports = demand − total production
    tec_cols = [c for c in combined.columns if c not in ("area", "hour", "demand")]
    total_prod = combined[tec_cols].sum(axis=1)
    net = combined["demand"] - total_prod
    combined["net_imports"] = np.maximum(net, 0)
    combined["net_exports"] = np.minimum(net, 0)

    # Reorder to match outputs/production.csv layout
    combined = combined[["area", "hour"] + tec_cols + ["net_imports", "net_exports", "demand"]]

    validation_dir = run_dir / "validation"
    validation_dir.mkdir(exist_ok=True)
    combined.to_csv(validation_dir / "actual_production.csv", index=False)

# TODO : Detail and tailor to the list of areas necessary for simulation
def _ensure_data_available(data_dir, year, areas, exo_areas):
    """Check if data for the given year is available, download if not.

    Checks for year-based directory structure: data/<year>/.
    If the year directory is missing, triggers a collect.
    If marked as corrupt (data/<year>_corrupt), raises an error.
    """

    # Check history data
    year_dir = data_dir / str(year)
    history_missing = not year_dir.exists()

    # Check Renewables.ninja data
    ninja_dir = data_dir / "renewable_ninja"
    ninja_files = [
        "solar.csv",
        "onshore_current.csv",
        "onshore_future.csv",
        "offshore_current.csv",
        "offshore_future.csv",
    ]
    ninja_missing = not ninja_dir.exists() or not all((ninja_dir / f).exists() for f in ninja_files)

    if history_missing or ninja_missing:
        logger.info("Some necessary data is missing, launching data collection...")
        collect_all(data_dir, year, year + 1, areas=areas, exo_areas=exo_areas, source="all")
        # Verify history data download succeeded
        if not year_dir.exists():
            if (data_dir / f"{year}_corrupt").exists():
                raise RuntimeError(
                    f"Data collection for {year} failed validation. "
                    f"Check {data_dir / f'{year}_corrupt'} for details."
                )
            raise RuntimeError(f"Data collection for {year} did not produce {year_dir}.")
        # Verify ninja data download succeeded
        still_missing = [f for f in ninja_files if not (ninja_dir / f).exists()]
        if still_missing:
            raise RuntimeError(
                f"Failed to download Renewables.ninja data. "
                f"Missing files: {still_missing}. "
                f"Check your internet connection, or provide the data manually in {ninja_dir}/"
            )
