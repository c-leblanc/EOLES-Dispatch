from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def project_root():
    """Return the project root directory."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def areas():
    """Minimal 2-area list for fast tests."""
    return ["FR", "DE"]


@pytest.fixture(scope="session")
def exo_areas():
    """Minimal exogenous areas."""
    return ["NL", "AT"]


@pytest.fixture
def sample_hourly_series():
    """72-hour UTC-naive hourly series with some NaNs for gap-fill tests."""
    idx = pd.date_range("2020-01-01", periods=72, freq="h")
    vals = np.random.default_rng(42).uniform(10, 50, size=72)
    return pd.Series(vals, index=idx)
