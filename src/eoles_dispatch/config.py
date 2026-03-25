"""Project-wide configuration constants and environment loading.

Centralizes all parameters, area code mappings, and environment variables
used across the pipeline. Loaded once at import time.

Used by:
    - datacoll/entsoe.py    (ENTSOE_API_KEY, AREA_CODES, AREA_CODES_PRICE, ENTSOE_MIN_COVERAGE)
    - datacoll/main_collect.py (DEFAULT_AREAS, DEFAULT_EXO_AREAS, ENTSOE_MIN_COVERAGE)
    - datacoll/rninja.py    (DEFAULT_AREAS)
    - format_inputs.py      (NMD_TYPES)
    - run.py                (DEFAULT_AREAS, DEFAULT_EXO_AREAS)

Constants:
    _load_dotenv()          - Load .env file at module import time.

    LOAD_UNCERTAINTY        - Uncertainty coefficient for hourly demand.
    DELTA                   - Load variation factor.
    VOLL                    - Value of lost load (EUR/MWh).
    ETA_IN / ETA_OUT        - Storage charging/discharging efficiencies.
    TRLOSS                  - Transportation loss on power trade.
    GJ_MWH                  - GJ to MWh conversion factor.

    ENTSOE_API_KEY          - API key read from environment.
    ENTSOE_MIN_COVERAGE     - Minimum data coverage ratio for ENTSO-E series.

    DEFAULT_AREAS           - Default modeled country codes.
    DEFAULT_EXO_AREAS       - Default exogenous (non-modeled) country codes.
    NMD_TYPES               - Non-market-dependent fuel types.
    AREA_CODES              - Mapping our area codes -> ENTSO-E bidding zone codes.
    AREA_CODES_PRICE        - Override mapping for day-ahead price queries.
"""

import pandas as pd
import os
from dotenv import load_dotenv, find_dotenv


#---------------------
## Loading Environment
#---------------------

def _load_dotenv():
    """Load environment variables from .env file.

    Searches for .env in the current directory and up to 3 parent directories.
    """
    env_path = find_dotenv(usecwd=True)
    if env_path:
        load_dotenv(env_path)

# Load environment variables at module import time
_load_dotenv()

#------------------
## Model Parameters
#------------------

LOAD_UNCERTAINTY = 0.01 # Uncertainty coefficient for hourly demand
DELTA = 0.1 # Load variation factor
VOLL = 15000 # Value of lost load in EUR/MWh (virtual cost of unserved demand)
ETA_IN = pd.Series([0.95, 0.9], index=["lake_phs", "battery"]) # Charging efficiency of storage technologies
ETA_OUT = pd.Series([0.9, 0.95], index=["lake_phs", "battery"]) # Discharging efficiency of storage technologies
TRLOSS = 0.02 # Transportation loss applied to power trade

#------------------
## Unit conversions
#------------------

GJ_MWH = 3.6 # Conversion factor from GJ to MWh

#-----------------
## Data Collection
#-----------------

ENTSOE_API_KEY = os.environ.get("ENTSOE_API_KEY")

# Minimum valid-data ratio to accept an ENTSO-E series before falling back to
# an alternative source. Below this threshold, the series is considered too
# sparse and the Elexon fallback is triggered for GB.
ENTSOE_MIN_COVERAGE = 0.5


#--------------------
## Perimeter Settings
#--------------------

DEFAULT_AREAS = ["FR", "BE", "DE", "CH", "IT", "ES", "UK"] # Default modeled areas
DEFAULT_EXO_AREAS = ["NL", "DK1", "DK2", "SE4", "PL", "CZ", "AT", "GR", "SI", "PT", "IE"] # Default exogenous (non-modeled) areas

NMD_TYPES = ["biomass", "geothermal", "marine", "other_renew", "waste", "other"] # NMD (non-market-dependent) fuel types treated as exogenous


AREA_CODES = { # Matching to ENTSOE area codes
    "FR": "FR",
    "BE": "BE",
    "DE": "DE_LU",
    "CH": "CH",
    "IT": "IT",
    "ES": "ES",
    "UK": "GB",
    "NL": "NL",
    "DK1": "DK_1",
    "DK2": "DK_2",
    "SE4": "SE_4",
    "PL": "PL",
    "CZ": "CZ",
    "AT": "AT",
    "GR": "GR",
    "SI": "SI",
    "PT": "PT",
    "IE": "IE_SEM",
}
#   DE → "DE_LU" (bidding zone, includes Luxembourg since Oct 2018).
#        LU is ~0.6 GW peak vs DE ~80 GW, so the impact is negligible.
#        We use the bidding zone rather than control area because ENTSO-E
#        data availability is better at bidding-zone level, and prices are
#        only published for DE_LU (not DE alone).
#        Renewables.ninja uses DE-only data, which is consistent since LU
#        wind/solar capacity is negligible relative to DE.
#   IT → "IT" (whole country, control area). NOT IT_NORD or other sub-zones.
#        The model treats Italy as a single node, so we use the national
#        aggregate. ENTSO-E publishes load/generation at this level.
#   UK → "GB" (Great Britain = England + Scotland + Wales).
#        Excludes Northern Ireland which is part of IE_SEM (all-island market).
#        This matches the R scripts which used EIC 10YGB----------A.
#        Renewables.ninja also uses GB.


AREA_CODES_PRICE = {
    "IT": "IT_NORD",
}
# For day-ahead prices, some areas need a different code than for load/generation.
# DE prices were published under DE-AT-LU until Oct 2018, then DE-LU.
# entsoe-py handles this automatically when using DE_LU.
# IT prices use IT_NORD (the reference price zone), not IT (which has no price).




