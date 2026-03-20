"""Default parameters and constants for the EOLES-Dispatch model."""

import pandas as pd

# Uncertainty coefficient for hourly demand
LOAD_UNCERTAINTY = 0.01

# Load variation factor
DELTA = 0.1

# Value of lost load in EUR/MWh (virtual cost of unserved demand)
VOLL = 15000

# Charging efficiency of storage technologies
ETA_IN = pd.Series([0.95, 0.9], index=["lake_phs", "battery"])

# Discharging efficiency of storage technologies
ETA_OUT = pd.Series([0.9, 0.95], index=["lake_phs", "battery"])

# Transportation loss applied to power trade
TRLOSS = 0.02

# Conversion factor from GJ to MWh
GJ_MWH = 3.6

# Default modeled areas
DEFAULT_AREAS = ["FR", "BE", "DE", "CH", "IT", "ES", "UK"]

# Default exogenous (non-modeled) areas
DEFAULT_EXO_AREAS = ["NL", "DK1", "DK2", "SE4", "PL", "CZ", "AT", "GR", "SI", "PT", "IE"]
