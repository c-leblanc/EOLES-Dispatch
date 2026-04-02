"""Project-wide configuration constants and environment loading.

Centralizes all parameters, area code mappings, technology nomenclature
mappings, and environment variables used across the pipeline.
Loaded once at import time.

Three technology nomenclature levels exist in the project:
    - **raw**:   production types as collected from external sources (ENTSO-E, Elexon).
                 Used in data/<year>/production_<area>.csv.
    - **model**: technologies as defined in the LP model scenarios.
                 Used in scenario CSVs and Pyomo sets.
    - **agg**:   aggregated categories for outputs and visualizations.
                 Used in runs/<name>/outputs/production.csv and charts.

RAW_TO_AGG and MODEL_TO_AGG define the canonical mappings between levels.
"""

import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------
## Loading Environment
# ---------------------


def _load_dotenv():
    """Load environment variables from .env file if python-dotenv is installed.

    python-dotenv is only required for the collect extra (ENTSO-E API key).
    Silently skipped when running the model without the collect dependencies.
    """
    try:
        from dotenv import find_dotenv, load_dotenv

        env_path = find_dotenv(usecwd=True)
        if env_path:
            load_dotenv(env_path)
    except ImportError:
        pass


# Load environment variables at module import time
_load_dotenv()

# ------------------
## Model Parameters
# ------------------

LOAD_UNCERTAINTY = 0.05  # Uncertainty coefficient for hourly demand
DELTA = 0.1  # Load variation factor
VOLL = 15000  # Value of lost load in EUR/MWh (virtual cost of unserved demand)
ETA_IN = pd.Series(
    [0.95, 0.9], index=["lake_phs", "battery"]
)  # Charging efficiency of storage technologies
ETA_OUT = pd.Series(
    [0.9, 0.95], index=["lake_phs", "battery"]
)  # Discharging efficiency of storage technologies
TRLOSS = 0.02  # Transportation loss applied to power trade

# ------------------
## Unit conversions
# ------------------

GJ_MWH = 3.6  # Conversion factor from GJ to MWh

# -----------------
## Data Collection
# -----------------

ENTSOE_API_KEY = os.environ.get("ENTSOE_API_KEY")

# Minimum valid-data ratio to accept an ENTSO-E series before falling back to
# an alternative source. Below this threshold, the series is considered too
# sparse and the Elexon fallback is triggered for GB.
ENTSOE_MIN_COVERAGE = 0.5


# --------------------
## Perimeter Settings
# --------------------

DEFAULT_AREAS = ["FR", "BE", "DE", "CH", "IT", "ES", "UK"]  # Default modeled areas
DEFAULT_EXO_AREAS = [
    "NL",
    "DK1",
    "DK2",
    "SE4",
    "PL",
    "CZ",
    "AT",
    "GR",
    "SI",
    "PT",
    "IE",
]  # Default exogenous (non-modeled) areas

AREA_CODES = {  # Matching to ENTSOE area codes
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
#        Before Oct 2018 the zone was DE_AT_LU (incl. Austria + Luxembourg).
#        Time-dependent resolution (DE_AT_LU / DE_LU) is handled by
#        entsoe.py:_resolve_area(), not here — this dict stores the
#        *current* default.
#        LU is ~0.6 GW peak vs DE ~80 GW, so the impact is negligible.
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
# DE prices followed the same zone transition as load/generation
# (DE_AT_LU → DE_LU in Oct 2018); this is handled by
# entsoe.py:_resolve_area_price().
# IT prices use IT_NORD (the reference price zone), not IT (which has no price).


# ----------------------------------
## Technology nomenclature mappings
# ----------------------------------

# RAW_TO_AGG: raw (collected data) → agg (output/viz).
#   Keys = column names in data/<year>/production_<area>.csv.
#   Values = column names in runs/<name>/outputs/production.csv.
#   Also used to derive NMD_TYPES for compute_nmd().
RAW_TO_AGG = {
    # Renewables
    "solar": "solar",
    "onshore": "wind",
    "offshore": "wind",
    "river": "river",
    "lake": "lake_phs",
    # Nuclear
    "nuclear": "nuclear",
    # Thermal
    "gas": "gas",
    "coal_gas": "gas",
    "hard_coal": "coal",
    "lignite": "coal",
    "oil": "oil",
    "oil_shale": "oil",
    "peat": "nmd",
    # NMD (non-market-dependent)
    "biomass": "nmd",
    "geothermal": "nmd",
    "marine": "nmd",
    "other_renew": "nmd",
    "waste": "nmd",
    "other": "nmd",
    # Storage (hydro pumped storage)
    "phs": "lake_phs",
    "phs_in": "phs_in",  # negative at all levels
}

# MODEL_TO_AGG: model (LP technologies) → agg (output/viz).
#   Keys = technology names from scenario CSVs (capa.csv, thr_specs.csv).
#   Values = same agg namespace as RAW_TO_AGG.
#   Used by format_outputs.py and viz/_theme.py.
MODEL_TO_AGG = {
    # Thermal sub-types
    "gas_ccgt1G": "gas",
    "gas_ccgt2G": "gas",
    "gas_ccgtSA": "gas",
    "gas_ocgtSA": "gas",
    "coal_1G": "coal",
    "coal_SA": "coal",
    "lignite": "coal",
    "oil_light": "oil",
    # VRE
    "onshore": "wind",
    "offshore": "wind",
    # Identity mappings (no sub-types)
    "nuclear": "nuclear",
    "solar": "solar",
    "river": "river",
    "lake_phs": "lake_phs",
    "battery": "battery",
    "nmd": "nmd",
}

# NMD production types, derived from RAW_TO_AGG (single source of truth).
NMD_TYPES = [k for k, v in RAW_TO_AGG.items() if v == "nmd"]
