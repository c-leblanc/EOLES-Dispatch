"""CLI entry point for EOLES-Dispatch.

Usage:
    eoles-dispatch create my_run --scenario baseline --year 2021
    eoles-dispatch solve my_run
    eoles-dispatch solve my_run --solver gurobi
    eoles-dispatch list
    eoles-dispatch collect --start 2020 --end 2025
    eoles-dispatch convert-scenario Scenario_BASELINE.xlsx
"""

import argparse
import logging
from pathlib import Path

from . import __version__


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(
        prog="eoles-dispatch",
        description="EOLES-Dispatch: Cost-minimization dispatch model for wholesale electricity prices",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    # Shared arguments
    def _add_project_dir(p):
        p.add_argument(
            "--project-dir", type=Path, default=None, help="Project root directory (default: cwd)"
        )

    # --- create command ---
    create_parser = subparsers.add_parser(
        "create", help="Create a new run (fetch data, format inputs)"
    )
    create_parser.add_argument("name", help="Run name")
    create_parser.add_argument(
        "--scenario",
        required=True,
        help="Scenario name (looks for scenarios/<name>/ dir, then .xlsx)",
    )
    create_parser.add_argument("--year", type=int, required=True, help="Simulation year")
    create_parser.add_argument(
        "--rn-horizon",
        default="current",
        choices=["current", "future"],
        help="Renewables.ninja wind fleet: current (installed ~2020) or future (larger turbines, taller towers)",
    )
    create_parser.add_argument(
        "--actual-cf",
        action="store_true",
        help="Use actual historical capacity factors instead of Renewable Ninja",
    )
    create_parser.add_argument(
        "--months",
        type=str,
        default=None,
        metavar="M or M1-M2",
        help="Restrict to specific months (e.g. '1' for Jan, '8' for Aug, '1-3' for Jan-Mar)",
    )
    create_parser.add_argument(
        "--no-download", action="store_true", help="Don't auto-download missing data"
    )
    _add_project_dir(create_parser)

    # --- solve command ---
    solve_parser = subparsers.add_parser("solve", help="Solve an existing run")
    solve_parser.add_argument("name", help="Run name")
    solve_parser.add_argument("--solver", default="highs", help="Solver to use (default: highs)")
    solve_parser.add_argument(
        "--model-version", default="standard", help="Model version (default: standard)"
    )
    solve_parser.add_argument(
        "--reports",
        nargs="+",
        default=["prices", "production"],
        choices=["prices", "production", "capa_on", "FRtrade"],
        help="Reports to generate (default: prices production)",
    )
    solve_parser.add_argument(
        "--fulldiag",
        action="store_true",
        help="Export exhaustive diagnostics (all variables and duals) to diagnostics/",
    )
    _add_project_dir(solve_parser)

    # --- list command ---
    list_parser = subparsers.add_parser("list", help="List all runs")
    _add_project_dir(list_parser)

    # --- collect command ---
    collect_parser = subparsers.add_parser(
        "collect", help="Download data from ENTSO-E and/or Renewables.ninja"
    )
    collect_parser.add_argument(
        "--start", type=int, required=True, help="First year to download (e.g. 2020)"
    )
    collect_parser.add_argument(
        "--end", type=int, required=True, help="Last year (exclusive, e.g. 2025 for 2020-2024)"
    )
    collect_parser.add_argument(
        "--source",
        default="all",
        choices=["all", "entsoe", "ninja"],
        help="Data source to collect (default: all)",
    )
    collect_parser.add_argument(
        "--output-dir", type=Path, default=None, help="Output directory (default: data/)"
    )
    collect_parser.add_argument(
        "--force", action="store_true", help="Re-download even if year data already exists"
    )
    _add_project_dir(collect_parser)

    # --- viz command ---
    viz_parser = subparsers.add_parser("viz", help="Generate an interactive HTML report for a run")
    viz_parser.add_argument("name", nargs="+", help="Run name(s)")
    viz_parser.add_argument(
        "--no-open", action="store_true", help="Don't open the report in the browser"
    )
    viz_parser.add_argument(
        "--validate",
        action="store_true",
        help="Compare simulated prices against historical day-ahead prices",
    )
    _add_project_dir(viz_parser)

    # --- convert-scenario command ---
    conv_parser = subparsers.add_parser(
        "convert-scenario", help="Convert an Excel scenario to a CSV directory"
    )
    conv_parser.add_argument("xlsx_path", type=Path, help="Path to the .xlsx scenario file")
    conv_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: derived from xlsx name)",
    )

    args = parser.parse_args()

    if args.command == "create":
        from .run._main_run import create_run

        # Parse --months: "3" → (3,3), "1-3" → (1,3), None → None
        month_range = None
        if args.months:
            if "-" in args.months:
                parts = args.months.split("-", 1)
                month_range = (int(parts[0]), int(parts[1]))
            else:
                m = int(args.months)
                month_range = (m, m)
            if not (
                1 <= month_range[0] <= 12
                and 1 <= month_range[1] <= 12
                and month_range[0] <= month_range[1]
            ):
                parser.error(f"Invalid --months: {args.months} (expected 1-12 range)")

        create_run(
            name=args.name,
            scenario=args.scenario,
            year=args.year,
            project_dir=args.project_dir,
            rn_horizon=args.rn_horizon,
            actCF=args.actual_cf,
            auto_download=not args.no_download,
            months=month_range,
        )

    elif args.command == "solve":
        from .run._main_run import solve_run

        solve_run(
            name=args.name,
            project_dir=args.project_dir,
            solver=args.solver,
            version=args.model_version,
            reports=args.reports,
            full_diag=args.fulldiag,
        )

    elif args.command == "list":
        from .run._main_run import list_runs

        runs = list_runs(project_dir=args.project_dir)
        if not runs:
            print("No runs found.")
        else:
            print(f"{'NAME':<30} {'SCENARIO':<15} {'YEAR':<6} {'STATUS':<10} {'CREATED':<20}")
            print("-" * 85)
            for r in runs:
                print(
                    f"{r.get('name', '?'):<30} {r.get('scenario', '?'):<15} "
                    f"{r.get('year', '?'):<6} {r.get('status', '?'):<10} "
                    f"{r.get('created', '?'):<20}"
                )

    elif args.command == "collect":
        if args.end <= args.start:
            parser.error(
                f"--end ({args.end}) must be greater than --start ({args.start}). "
                f"Note: --end is exclusive, so use --end {args.start + 1} to collect year {args.start}."
            )
        from .collect._main_collect import collect_all

        project_dir = args.project_dir or Path.cwd()
        output_dir = args.output_dir or project_dir / "data"
        collect_all(output_dir, args.start, args.end, source=args.source, force=args.force)

    elif args.command == "viz":
        from .viz import generate_report

        project_dir = args.project_dir or Path.cwd()
        for run_name in args.name:
            run_dir = project_dir / "runs" / run_name
            if not run_dir.exists():
                print(f"Run '{run_name}' not found at {run_dir}")
                continue
            out = generate_report(run_dir, open_browser=not args.no_open, validate=args.validate)
            print(f"Report: {out}")

    elif args.command == "convert-scenario":
        from .run.scenario import xlsx_to_scenario

        xlsx_to_scenario(args.xlsx_path, args.output_dir)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
