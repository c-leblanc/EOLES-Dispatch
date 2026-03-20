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

import shutil
import time
from datetime import datetime
from pathlib import Path

import yaml

from .config import DEFAULT_AREAS, DEFAULT_EXO_AREAS


def _ensure_data_available(data_dir, year, areas, exo_areas):
    """Check if data for the given year is available, download if not."""
    tv_dir = data_dir / "time_varying_inputs"
    required_files = ["demand.csv", "nmd.csv", "exoPrices.csv", "lake_inflows.csv",
                      "nucMaxAF.csv", "hMaxIn.csv", "hMaxOut.csv"]

    # Check if any required file is missing
    missing = not tv_dir.exists() or not all((tv_dir / f).exists() for f in required_files)

    if missing:
        print(f"Data not found in {tv_dir}, downloading from ENTSO-E...")
        from .collect import collect_all
        collect_all(data_dir, year, year + 1, areas=areas, exo_areas=exo_areas, source="entsoe")

    # Check if year is covered in existing data
    if tv_dir.exists() and (tv_dir / "demand.csv").exists():
        import pandas as pd
        demand = pd.read_csv(tv_dir / "demand.csv", nrows=5)
        first_date = pd.to_datetime(demand["hour"].iloc[0])
        if first_date.year > year:
            print(f"Data starts at {first_date.year}, need {year}. Downloading...")
            from .collect import collect_all
            collect_all(data_dir, year, year + 1, areas=areas, exo_areas=exo_areas, source="entsoe")

    # Check Renewables.ninja data
    ninja_dir = data_dir / "renewable_ninja"
    ninja_files = ["pv.csv", "onshore_CU.csv", "offshore_CU.csv"]
    ninja_missing = not ninja_dir.exists() or not all((ninja_dir / f).exists() for f in ninja_files)

    if ninja_missing:
        print(f"Renewable Ninja data not found in {ninja_dir}, downloading...")
        from .collect import collect_ninja
        collect_ninja(ninja_dir, areas=areas)


def create_run(
    name,
    scenario,
    year,
    project_dir=None,
    areas=None,
    exo_areas=None,
    actCF=False,
    rn_horizon="CU",
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
        rn_horizon: Renewable Ninja horizon ("CU", "NT", "LT").
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
    print(f"Creating run '{name}'...")

    # Format and save inputs
    from .format_inputs import load_tv_inputs, extract_scenario, save_inputs

    print("  Loading time-varying data...")
    tv_data = load_tv_inputs(data_dir, year, areas, exo_areas, actCF, rn_horizon, months=months)

    print("  Loading scenario parameters...")
    scenario_data = extract_scenario(scenario_path, areas, exo_areas, tv_data["hour_month"])

    print("  Saving formatted inputs...")
    save_inputs(run_dir, tv_data, scenario_data, areas, exo_areas)

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
        "months": f"{months[0]}-{months[1]}" if months and months[0] != months[1]
                  else str(months[0]) if months else None,
        "created": datetime.now().isoformat(timespec="seconds"),
        "status": "created",
    }
    with open(run_dir / "run.yaml", "w") as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)

    print(f"  Run created at {run_dir}")
    return run_dir


def solve_run(
    name,
    project_dir=None,
    solver="highs",
    version="standard",
    reports=None,
):
    """Solve an existing run.

    Args:
        name: Run name.
        project_dir: Root project directory.
        solver: Solver name (e.g. "highs", "cbc", "gurobi").
        version: Model version ("standard" or "static_thermal").
        reports: List of reports to generate.

    Returns:
        Solver results object.
    """
    import gc

    import pyomo.environ  # noqa: F401 — registers solver plugins
    from pyomo.opt import SolverFactory

    from .format_outputs import report_prices, report_production, report_capa_on, report_FRtrade, write_log
    from .models import MODEL_REGISTRY

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

    print(f"Solving run '{name}' (scenario={metadata['scenario']}, year={metadata['year']})")
    start_time = time.localtime()

    # Build model
    build_model = MODEL_REGISTRY.get(version)
    if build_model is None:
        raise ValueError(f"Unknown model version '{version}'. Choose from: {list(MODEL_REGISTRY.keys())}")
    print(f"  Building model [{version}]...")
    model = build_model(run_dir)

    # Solve
    # Pyomo uses "appsi_highs" as the SolverFactory name for HiGHS
    solver_name = "appsi_highs" if solver == "highs" else solver
    print(f"  Solving with {solver}...")
    opt = SolverFactory(solver_name)

    # HiGHS-specific tuning for large LP models
    if solver == "highs":
        opt.highs_options["solver"] = "ipm"          # Interior point method (faster on large LPs)
        opt.highs_options["run_crossover"] = "on"     # Get a basic feasible solution for duals

    results = opt.solve(model, tee=True)

    # Create outputs directory
    (run_dir / "outputs").mkdir(exist_ok=True)

    # Generate reports
    print("  Generating reports...")
    report_map = {
        "prices": report_prices,
        "production": report_production,
        "capa_on": report_capa_on,
        "FRtrade": report_FRtrade,
    }
    for report_name in reports:
        if report_name in report_map:
            report_map[report_name](model, run_dir)

    exec_time = time.gmtime(time.time() - time.mktime(start_time))
    exec_str = time.strftime("%H:%M:%S", exec_time)
    print(f"  Done in {exec_str}")

    # Update metadata
    metadata["status"] = "solved"
    metadata["solved"] = datetime.now().isoformat(timespec="seconds")
    metadata["solver"] = solver
    metadata["model_version"] = version
    metadata["reports"] = reports
    metadata["exec_time"] = exec_str
    with open(meta_path, "w") as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)

    write_log(run_dir, model, name, metadata["scenario"], metadata["year"], start_time, exec_time)

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
