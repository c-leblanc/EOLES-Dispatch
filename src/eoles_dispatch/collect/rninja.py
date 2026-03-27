"""Renewables.ninja capacity-factor profile downloads.

Downloads solar and wind (onshore/offshore, current/future fleet) capacity
factor time series from the public Renewables.ninja country downloads.
These profiles are year-independent (long-term averages from MERRA-2
reanalysis) and are saved once in data/renewable_ninja/.

Unlike ENTSO-E/Elexon data, Ninja profiles do not need gap-filling or
yearly updates — they are downloaded once and reused across all years.

Delegates to:
    - config.py     DEFAULT_AREAS (default country list).

Called from:
    - main_collect.py   collect_ninja is called from collect_all
                        (when source="all" or source="ninja").
    - run.py            collect_ninja is called from _ensure_data_available
                        when Ninja files are missing at run creation time.

Functions:
    collect_ninja(output_dir, areas=None)
        Download all Ninja profiles for the given areas, save as CSVs
        with columns ['hour', area1, area2, ...].
        Called from main_collect.collect_all, run._ensure_data_available.

Internal helpers:
    _download_ninja_csv(iso2, filename)
        Download a single CSV and extract the NATIONAL column.

Constants:
    NINJA_BASE_URL      URL template for country downloads.
    NINJA_ISO2          Our area codes -> Ninja ISO2 codes.
    NINJA_FILES         File definitions (name -> URL template).
    NO_OFFSHORE         Landlocked countries (no offshore data).
"""

import io
import logging
from pathlib import Path

import pandas as pd

from ..config import DEFAULT_AREAS

logger = logging.getLogger(__name__)


# URL template for Renewables.ninja public country downloads.
# Format: ninja-{type}-country-{ISO2}-{variant}-merra2.csv
NINJA_BASE_URL = "https://www.renewables.ninja/country_downloads/{iso2}"

# Maps our area codes to Renewables.ninja ISO2 codes.
# Perimeter consistency with ENTSO-E:
#   DE → "DE" (Germany only). Ninja has no DE_LU aggregate, but LU capacity
#        is negligible so DE-only profiles are representative of the DE_LU zone.
#   IT → "IT" (whole country). Matches ENTSO-E IT control area.
#   UK → "GB" (Great Britain). Matches ENTSO-E GB bidding zone.
NINJA_ISO2 = {
    "FR": "FR",
    "BE": "BE",
    "DE": "DE",
    "CH": "CH",
    "IT": "IT",
    "ES": "ES",
    "UK": "GB",
    "LU": "LU",
}

# File definitions: (our_name, url_filename_template)
# {iso2} is replaced with the country code.
# Renewables.ninja provides two wind fleet variants:
#   - current: technology installed as of ~2020 (current hub heights, rotor diameters)
#   - future:  projected next-generation turbines (taller towers, larger rotors)
NINJA_FILES = {
    "solar": "ninja-pv-country-{iso2}-national-merra2.csv",
    "onshore_current": "ninja-wind-country-{iso2}-current_onshore-merra2.csv",
    "onshore_future": "ninja-wind-country-{iso2}-future_onshore-merra2.csv",
    "offshore_current": "ninja-wind-country-{iso2}-current_offshore-merra2.csv",
    "offshore_future": "ninja-wind-country-{iso2}-future_offshore-merra2.csv",
}

# Countries that have no offshore data (landlocked)
NO_OFFSHORE = {"CH", "LU"}


def _download_ninja_csv(iso2, filename):
    """Download a single CSV from Renewables.ninja and return the NATIONAL column as a Series."""
    import urllib.request

    url = f"{NINJA_BASE_URL.format(iso2=iso2)}/{filename.format(iso2=iso2)}"
    logger.info(f"  Downloading {url}")
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "EOLES-Dispatch/0.1 (energy model; +https://github.com)",
            },
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8")
    except Exception as e:
        logger.warning(f"  Failed to download {url}: {e}")
        return None

    # Skip comment lines (start with #)
    lines = raw.split("\n")
    data_lines = [line for line in lines if not line.startswith('"#') and not line.startswith("#")]
    csv_text = "\n".join(data_lines)

    df = pd.read_csv(io.StringIO(csv_text), parse_dates=["time"])
    df = df.set_index("time")

    if "NATIONAL" not in df.columns:
        logger.warning(f"  No NATIONAL column in {filename} for {iso2}")
        return None

    return df["NATIONAL"]


def collect_ninja(output_dir, areas=None):
    """Download Renewables.ninja capacity factor profiles for all areas.

    Downloads solar, onshore (current/future), and offshore (current/future)
    from the public country downloads. Produces CSVs in the same format as
    the existing renewable_ninja/ directory: columns ['hour', area1, area2, ...].

    Args:
        output_dir: Path to data/renewable_ninja/ directory.
        areas: List of area codes (default: DEFAULT_AREAS).
    """
    if areas is None:
        areas = list(DEFAULT_AREAS)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for file_key, filename_template in NINJA_FILES.items():
        logger.info(f"=== Collecting {file_key} ===")
        is_offshore = "offshore" in file_key
        series_dict = {}

        for area in areas:
            iso2 = NINJA_ISO2.get(area)
            if iso2 is None:
                logger.warning(f"  No Renewables.ninja ISO2 mapping for {area}, skipping")
                continue
            if is_offshore and area in NO_OFFSHORE:
                logger.info(f"  Skipping offshore for {area} (landlocked)")
                continue

            series = _download_ninja_csv(iso2, filename_template)
            if series is not None:
                series_dict[area] = series

        if not series_dict:
            logger.warning(f"  No data collected for {file_key}")
            continue

        # Align all countries on the same time index (intersection)
        df = pd.DataFrame(series_dict)
        df.index.name = "hour"

        # Fill missing offshore countries with 0
        if is_offshore:
            for area in areas:
                if area not in df.columns:
                    df[area] = 0.0
            df = df[areas]  # reorder columns

        df = df.reset_index()
        # Normalize to UTC tz-naive
        hour_col = pd.to_datetime(df["hour"])
        if hour_col.dt.tz is not None:
            hour_col = hour_col.dt.tz_convert("UTC").dt.tz_localize(None)
        df["hour"] = hour_col

        out_path = output_dir / f"{file_key}.csv"
        df.to_csv(out_path, index=False)
        logger.info(f"  → {file_key}.csv ({len(df)} rows, {len(df.columns) - 1} areas)")

    logger.info("=== Ninja collection complete ===")
    return output_dir
