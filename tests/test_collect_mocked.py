"""Tests for eoles_dispatch.collect._main_collect with mocked external APIs."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from eoles_dispatch.utils import canonical_index

YEAR = 2021
AREAS = ["FR"]
EXO_AREAS = ["NL"]


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def canon_idx():
    return canonical_index(YEAR)


@pytest.fixture
def canonical_series(canon_idx):
    """Full-year Series of ones indexed by the canonical UTC hourly index."""
    return pd.Series(np.ones(len(canon_idx)), index=canon_idx)


@pytest.fixture
def canonical_production_df(canon_idx):
    """Full-year production DataFrame (nuclear + gas columns)."""
    return pd.DataFrame(
        {
            "nuclear": np.ones(len(canon_idx)),
            "gas": np.ones(len(canon_idx)) * 0.5,
        },
        index=canon_idx,
    )


# ---------------------------------------------------------------------------
# TestCollectHistoryMocked
# ---------------------------------------------------------------------------


class TestCollectHistoryMocked:
    @pytest.fixture
    def year_dir(self, tmp_path):
        d = tmp_path / str(YEAR)
        d.mkdir()
        return d

    @pytest.fixture
    def mock_client(self):
        return MagicMock()

    @pytest.fixture
    def mock_entsoe(self, canonical_series, canonical_production_df):
        m = MagicMock()
        m.fetch_demand.return_value = canonical_series
        m.fetch_generation.return_value = canonical_production_df
        m.fetch_day_ahead_prices.return_value = canonical_series
        m.is_usable.return_value = True
        m.fetch_installed_capacity.return_value = {"nuclear": 10.0, "gas": 5.0}
        return m

    def _run(self, mock_entsoe, year_dir, mock_client, **kwargs):
        from eoles_dispatch.collect._main_collect import collect_history

        with patch("eoles_dispatch.collect._main_collect.entsoe", mock_entsoe):
            collect_history(
                year_dir,
                mock_client,
                YEAR,
                AREAS,
                EXO_AREAS,
                include_area_prices=False,
                **kwargs,
            )

    def test_demand_file_written(self, mock_entsoe, year_dir, mock_client):
        self._run(mock_entsoe, year_dir, mock_client)
        assert (year_dir / f"demand_{AREAS[0]}.csv").exists()

    def test_production_file_written(self, mock_entsoe, year_dir, mock_client):
        self._run(mock_entsoe, year_dir, mock_client)
        assert (year_dir / f"production_{AREAS[0]}.csv").exists()

    def test_prices_file_written(self, mock_entsoe, year_dir, mock_client):
        self._run(mock_entsoe, year_dir, mock_client)
        assert (year_dir / f"prices_{EXO_AREAS[0]}.csv").exists()

    def test_installed_capacity_file_written(self, mock_entsoe, year_dir, mock_client):
        self._run(mock_entsoe, year_dir, mock_client)
        assert (year_dir / f"installed_capacity_{AREAS[0]}.csv").exists()

    def test_demand_file_has_correct_row_count(self, mock_entsoe, year_dir, mock_client):
        self._run(mock_entsoe, year_dir, mock_client)
        df = pd.read_csv(year_dir / f"demand_{AREAS[0]}.csv")
        assert len(df) == len(canonical_index(YEAR))

    def test_skips_existing_demand_file(self, mock_entsoe, year_dir, mock_client):
        """Pre-existing demand file prevents re-fetching from ENTSO-E."""
        (year_dir / f"demand_{AREAS[0]}.csv").write_text("hour,demand\n")
        self._run(mock_entsoe, year_dir, mock_client)
        mock_entsoe.fetch_demand.assert_not_called()

    def test_unusable_data_not_written(self, mock_entsoe, year_dir, mock_client):
        """When is_usable returns False the demand file must not be created."""
        mock_entsoe.is_usable.return_value = False
        self._run(mock_entsoe, year_dir, mock_client)
        assert not (year_dir / f"demand_{AREAS[0]}.csv").exists()


# ---------------------------------------------------------------------------
# TestCollectAllMocked
# ---------------------------------------------------------------------------


class TestCollectAllMocked:
    @pytest.fixture
    def mock_entsoe(self, canonical_series, canonical_production_df):
        m = MagicMock()
        m.set_client.return_value = MagicMock()
        m.fetch_demand.return_value = canonical_series
        m.fetch_generation.return_value = canonical_production_df
        m.fetch_day_ahead_prices.return_value = canonical_series
        m.is_usable.return_value = True
        m.fetch_installed_capacity.return_value = {}
        return m

    def test_collect_all_creates_year_dir(self, tmp_path, mock_entsoe):
        from eoles_dispatch.collect._main_collect import collect_all

        with patch("eoles_dispatch.collect._main_collect.entsoe", mock_entsoe):
            collect_all(
                tmp_path,
                YEAR,
                YEAR + 1,
                areas=AREAS,
                exo_areas=EXO_AREAS,
                source="entsoe",
            )
        assert (tmp_path / str(YEAR)).is_dir()

    def test_missing_api_key_raises_systemexit(self, tmp_path):
        mock_entsoe = MagicMock()
        mock_entsoe.set_client.side_effect = EnvironmentError("No API key set")

        from eoles_dispatch.collect._main_collect import collect_all

        with patch("eoles_dispatch.collect._main_collect.entsoe", mock_entsoe):
            with pytest.raises(SystemExit):
                collect_all(
                    tmp_path,
                    YEAR,
                    YEAR + 1,
                    areas=AREAS,
                    exo_areas=EXO_AREAS,
                    source="entsoe",
                )


# ---------------------------------------------------------------------------
# TestCollectNinjaMocked
# ---------------------------------------------------------------------------


class TestCollectNinjaMocked:
    def test_collect_ninja_creates_solar_file(self, tmp_path):
        """Mocked _download_ninja_csv → solar.csv is created in output dir."""
        idx = pd.date_range("2020-01-01", periods=8760, freq="h")
        mock_series = pd.Series([0.3] * 8760, index=idx)

        with patch("eoles_dispatch.collect.rninja._download_ninja_csv", return_value=mock_series):
            from eoles_dispatch.collect.rninja import collect_ninja

            collect_ninja(tmp_path / "ninja", areas=["FR"])

        assert (tmp_path / "ninja" / "solar.csv").exists()

    def test_collect_ninja_failed_download_skips_gracefully(self, tmp_path):
        """_download_ninja_csv returning None must not crash or create files."""
        with patch("eoles_dispatch.collect.rninja._download_ninja_csv", return_value=None):
            from eoles_dispatch.collect.rninja import collect_ninja

            collect_ninja(tmp_path / "ninja", areas=["FR"])

        # No file if no area succeeded
        assert not (tmp_path / "ninja" / "solar.csv").exists()
