"""Tests for eoles_dispatch.viz (loaders and report generation)."""

from unittest.mock import patch

import pandas as pd
import pytest
import yaml

from eoles_dispatch.viz.loaders import (
    _parse_months,
    load_actual_prices,
    load_actual_production,
    load_metadata,
    prepare_validation_data,
)
from eoles_dispatch.viz.report import generate_report

AREAS = ["FR", "DE"]
RUN_NAME = "test_run"


# ---------------------------------------------------------------------------
# Fixture: solved run with outputs/ and run.yaml in place
# ---------------------------------------------------------------------------


@pytest.fixture
def viz_run_dir(solved_run):
    """Extend solved_run with outputs/ CSVs and run.yaml for viz tests."""
    from eoles_dispatch.run.format_outputs import report_prices, report_production

    model, run_dir = solved_run
    report_prices(model, run_dir)
    report_production(model, run_dir)

    metadata = {
        "name": RUN_NAME,
        "scenario": "test_scenario",
        "year": 2020,
        "areas": AREAS,
        "exo_areas": ["NL"],
        "actCF": False,
        "rn_horizon": "current",
        "months": None,
        "created": "2024-01-01T00:00:00",
        "status": "solved",
        "solved": "2024-01-01T00:01:00",
        "solver": "highs",
        "model_version": "static_thermal",
        "exec_time": "00:00:01",
    }
    with open(run_dir / "run.yaml", "w") as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)

    return run_dir


# ---------------------------------------------------------------------------
# TestLoadMetadata
# ---------------------------------------------------------------------------


class TestLoadMetadata:
    def test_load_metadata_returns_dict(self, viz_run_dir):
        result = load_metadata(viz_run_dir)
        assert isinstance(result, dict)
        assert result["name"] == RUN_NAME

    def test_load_metadata_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_metadata(tmp_path)


# ---------------------------------------------------------------------------
# TestLoadActualPrices / TestLoadActualProduction
# ---------------------------------------------------------------------------


class TestLoadActualPrices:
    def test_returns_none_when_missing(self, viz_run_dir):
        assert load_actual_prices(viz_run_dir) is None

    def test_returns_df_when_present(self, viz_run_dir):
        val_dir = viz_run_dir / "validation"
        val_dir.mkdir(exist_ok=True)
        pd.DataFrame({"hour": [0, 1], "FR": [50.0, 55.0]}).to_csv(
            val_dir / "actual_prices.csv", index=False
        )
        result = load_actual_prices(viz_run_dir)
        assert result is not None
        assert len(result) == 2


class TestLoadActualProduction:
    def test_returns_none_when_missing(self, viz_run_dir):
        assert load_actual_production(viz_run_dir) is None

    def test_returns_df_when_present(self, viz_run_dir):
        val_dir = viz_run_dir / "validation"
        val_dir.mkdir(exist_ok=True)
        pd.DataFrame({"area": ["FR"], "hour": [0], "nuclear": [5.0]}).to_csv(
            val_dir / "actual_production.csv", index=False
        )
        result = load_actual_production(viz_run_dir)
        assert result is not None
        assert "nuclear" in result.columns


# ---------------------------------------------------------------------------
# TestParseMonths
# ---------------------------------------------------------------------------


class TestParseMonths:
    def test_parse_months_range(self):
        assert _parse_months("1-3") == (1, 3)

    def test_parse_months_single(self):
        assert _parse_months("5") == (5, 5)

    def test_parse_months_none(self):
        assert _parse_months(None) is None


# ---------------------------------------------------------------------------
# TestPrepareValidationData
# ---------------------------------------------------------------------------


class TestPrepareValidationData:
    def test_skips_when_both_files_present(self, viz_run_dir):
        """If both validation CSV files exist, preparation helpers are not called."""
        val_dir = viz_run_dir / "validation"
        val_dir.mkdir(exist_ok=True)
        pd.DataFrame().to_csv(val_dir / "actual_prices.csv", index=False)
        pd.DataFrame().to_csv(val_dir / "actual_production.csv", index=False)

        with patch("eoles_dispatch.viz.loaders._prepare_actual_prices") as mock_p:
            with patch("eoles_dispatch.viz.loaders._prepare_actual_production") as mock_pp:
                meta = load_metadata(viz_run_dir)
                prepare_validation_data(viz_run_dir, meta)
                mock_p.assert_not_called()
                mock_pp.assert_not_called()

    def test_calls_prepare_when_files_missing(self, viz_run_dir):
        """When validation files are absent, both helpers must be called."""
        meta = load_metadata(viz_run_dir)
        with patch("eoles_dispatch.viz.loaders._ensure_price_data"):
            with patch("eoles_dispatch.viz.loaders._prepare_actual_prices") as mock_p:
                with patch("eoles_dispatch.viz.loaders._prepare_actual_production") as mock_pp:
                    prepare_validation_data(viz_run_dir, meta)
                    mock_p.assert_called_once()
                    mock_pp.assert_called_once()


# ---------------------------------------------------------------------------
# TestGenerateReport
# ---------------------------------------------------------------------------


class TestGenerateReport:
    def test_generate_report_creates_html(self, viz_run_dir):
        generate_report(viz_run_dir, open_browser=False)
        assert (viz_run_dir / "viz.html").exists()

    def test_generate_report_returns_path(self, viz_run_dir):
        result = generate_report(viz_run_dir, open_browser=False)
        assert result == viz_run_dir / "viz.html"

    def test_html_contains_run_name(self, viz_run_dir):
        generate_report(viz_run_dir, open_browser=False)
        text = (viz_run_dir / "viz.html").read_text(encoding="utf-8")
        assert RUN_NAME in text

    def test_html_has_tab_structure(self, viz_run_dir):
        generate_report(viz_run_dir, open_browser=False)
        text = (viz_run_dir / "viz.html").read_text(encoding="utf-8")
        assert "Inputs" in text
        assert "Outputs" in text

    def test_generate_report_no_browser_open(self, viz_run_dir):
        with patch("eoles_dispatch.viz.report.webbrowser.open") as mock_open:
            generate_report(viz_run_dir, open_browser=False)
            mock_open.assert_not_called()

    def test_generate_report_with_validate_calls_prepare(self, viz_run_dir):
        """validate=True must trigger prepare_validation_data."""
        with patch("eoles_dispatch.viz.report.prepare_validation_data") as mock_prep:
            generate_report(viz_run_dir, open_browser=False, validate=True)
            mock_prep.assert_called_once()
