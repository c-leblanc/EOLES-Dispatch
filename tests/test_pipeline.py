"""Integration tests for the solve / create pipeline.

TestSolveRun exercises the full solve_run() path using a pre-built
minimal inputs/ directory (3 areas, 3 hours, static_thermal variant).

TestCreateRun checks error-path behaviour of create_run() without
touching the data-loading pipeline (those paths are exercised by
test_format_inputs.py and test_scenario.py).
"""

import pytest
import yaml
from conftest import _build_input_dir, _make_scenario_dir

AREAS = ["FR", "DE"]
EXO_AREAS = ["NL"]
HOURS = [0, 1, 2]
RUN_NAME = "test_run"


# ---------------------------------------------------------------------------
# Fixture: minimal project directory with a pre-built run
# ---------------------------------------------------------------------------


@pytest.fixture
def run_project_dir(tmp_path):
    """Create a project with a ready-to-solve run (inputs/ + run.yaml)."""
    run_d = tmp_path / "runs" / RUN_NAME
    _build_input_dir(run_d)  # creates run_d/inputs/

    metadata = {
        "name": RUN_NAME,
        "scenario": "test_scenario",
        "year": 2020,
        "areas": AREAS,
        "exo_areas": EXO_AREAS,
        "actCF": False,
        "rn_horizon": "current",
        "months": None,
        "created": "2024-01-01T00:00:00",
        "status": "created",
    }
    with open(run_d / "run.yaml", "w") as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)

    return tmp_path


# ---------------------------------------------------------------------------
# TestSolveRun
# ---------------------------------------------------------------------------


class TestSolveRun:
    def test_solve_run_completes_without_error(self, run_project_dir):
        from eoles_dispatch.run._main_run import solve_run

        results = solve_run(RUN_NAME, project_dir=run_project_dir, version="static_thermal")
        assert results is not None

    def test_solve_run_creates_outputs_dir(self, run_project_dir):
        from eoles_dispatch.run._main_run import solve_run

        solve_run(RUN_NAME, project_dir=run_project_dir, version="static_thermal")
        assert (run_project_dir / "runs" / RUN_NAME / "outputs").is_dir()

    def test_solve_run_prices_csv_created(self, run_project_dir):
        from eoles_dispatch.run._main_run import solve_run

        solve_run(
            RUN_NAME,
            project_dir=run_project_dir,
            version="static_thermal",
            reports=["prices"],
        )
        assert (run_project_dir / "runs" / RUN_NAME / "outputs" / "prices.csv").exists()

    def test_solve_run_production_csv_created(self, run_project_dir):
        from eoles_dispatch.run._main_run import solve_run

        solve_run(
            RUN_NAME,
            project_dir=run_project_dir,
            version="static_thermal",
            reports=["production"],
        )
        assert (run_project_dir / "runs" / RUN_NAME / "outputs" / "production.csv").exists()

    def test_solve_run_prices_shape(self, run_project_dir):
        """prices.csv must have one row per hour and one column per area."""
        import pandas as pd

        from eoles_dispatch.run._main_run import solve_run

        solve_run(
            RUN_NAME,
            project_dir=run_project_dir,
            version="static_thermal",
            reports=["prices"],
        )
        df = pd.read_csv(
            run_project_dir / "runs" / RUN_NAME / "outputs" / "prices.csv", index_col="hour"
        )
        assert len(df) == len(HOURS)
        assert set(df.columns) == set(AREAS)

    def test_solve_run_production_has_demand_column(self, run_project_dir):
        import pandas as pd

        from eoles_dispatch.run._main_run import solve_run

        solve_run(
            RUN_NAME,
            project_dir=run_project_dir,
            version="static_thermal",
            reports=["production"],
        )
        df = pd.read_csv(run_project_dir / "runs" / RUN_NAME / "outputs" / "production.csv")
        assert "demand" in df.columns

    def test_solve_run_updates_yaml_status(self, run_project_dir):
        from eoles_dispatch.run._main_run import solve_run

        solve_run(RUN_NAME, project_dir=run_project_dir, version="static_thermal")
        with open(run_project_dir / "runs" / RUN_NAME / "run.yaml") as f:
            meta = yaml.safe_load(f)
        assert meta["status"] == "solved"

    def test_solve_run_writes_log_file(self, run_project_dir):
        from eoles_dispatch.run._main_run import solve_run

        solve_run(RUN_NAME, project_dir=run_project_dir, version="static_thermal")
        assert (run_project_dir / "runs" / RUN_NAME / f"_log_{RUN_NAME}.txt").exists()

    def test_solve_run_full_diag_creates_diagnostics(self, run_project_dir):
        from eoles_dispatch.run._main_run import solve_run

        solve_run(
            RUN_NAME,
            project_dir=run_project_dir,
            version="static_thermal",
            full_diag=True,
        )
        assert (
            run_project_dir / "runs" / RUN_NAME / "diagnostics" / "_summary.json"
        ).exists()

    def test_solve_run_invalid_version_raises(self, run_project_dir):
        from eoles_dispatch.run._main_run import solve_run

        with pytest.raises((KeyError, ValueError)):
            solve_run(RUN_NAME, project_dir=run_project_dir, version="nonexistent_version")

    def test_solve_run_missing_run_raises(self, tmp_path):
        from eoles_dispatch.run._main_run import solve_run

        with pytest.raises(FileNotFoundError):
            solve_run("no_such_run", project_dir=tmp_path, version="static_thermal")


# ---------------------------------------------------------------------------
# TestCreateRun  (error paths only — data loading tested elsewhere)
# ---------------------------------------------------------------------------


class TestCreateRun:
    @pytest.fixture
    def project_with_scenario(self, tmp_path):
        """Project dir with a valid scenario but no data."""
        _make_scenario_dir(tmp_path / "scenarios" / "test_scenario", areas=AREAS, exo_areas=EXO_AREAS)
        return tmp_path

    def test_create_run_duplicate_name_raises(self, project_with_scenario):
        """A second create_run with the same name must raise FileExistsError."""
        from eoles_dispatch.run._main_run import create_run

        # Pre-create the run directory so it already exists
        (project_with_scenario / "runs" / "dup_run").mkdir(parents=True)

        with pytest.raises(FileExistsError):
            create_run(
                "dup_run",
                scenario="test_scenario",
                year=2020,
                project_dir=project_with_scenario,
                auto_download=False,
            )

    def test_create_run_missing_scenario_raises(self, tmp_path):
        """Referencing a non-existent scenario must raise FileNotFoundError."""
        from eoles_dispatch.run._main_run import create_run

        with pytest.raises(FileNotFoundError):
            create_run(
                "new_run",
                scenario="nonexistent_scenario",
                year=2020,
                project_dir=tmp_path,
                auto_download=False,
            )
