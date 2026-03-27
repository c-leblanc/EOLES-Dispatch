# EOLES-Dispatch

A cost-minimization dispatch model for simulating realistic wholesale electricity prices at an hourly time-step, with a focus on the French power system.

## Overview

EOLES-Dispatch is a **linear relaxation of a unit commitment model** that strikes a balance between price realism and computational tractability. It takes as input a detailed description of installed generation capacity and time series of power dispatch determinants (hourly demand, renewable availability, hydro resources, nuclear maintenance planning, fossil fuel prices) and minimizes total dispatch cost subject to physical and operational constraints.

The model is **centered on France**: six neighboring countries (Belgium, Germany, Switzerland, Italy, Spain, UK) are modeled explicitly to capture cross-border trade dynamics, while 11 additional neighbors are represented as exogenous price zones. This multi-country setup ensures that French wholesale prices reflect realistic import/export patterns rather than assuming an isolated system.

**Key features:**
- France-focused, with 6 coupled neighbors and 11 exogenous price zones
- Hourly resolution, up to a full year (8760 hours)
- LP relaxation of unit commitment: captures startup costs, min stable generation, and ramping without integer variables
- Two model variants: **Standard** (full thermal dynamics) and **Static Thermal** (simplified, faster)
- HiGHS solver by default (open-source, no license required)
- Integrated data collection from ENTSO-E, Elexon BMRS (GB), and Renewables.ninja
- Interactive HTML visualization of inputs and outputs (Plotly)
- Scenario editor in the browser

## Quick start

```bash
# 1. Clone and install
git clone https://github.com/c-leblanc/EOLES-Dispatch.git
cd EOLES-Dispatch
python -m venv .venv
source .venv/bin/activate
pip install ".[collect,viz]"

# 2. Set up your ENTSO-E API key
cp .env.example .env
# Edit .env and add your key (register at https://transparency.entsoe.eu/)

# 3. Create a run (downloads data automatically on first use)
eoles-dispatch create my_run --scenario baseline --year 2019

# 4. Solve it
eoles-dispatch solve my_run

# 5. Visualize results
eoles-dispatch viz my_run
```

## Installation

**Requirements:** Python >= 3.9

```bash
pip install ".[collect]"    # Core + ENTSO-E/Elexon data collection
pip install ".[viz]"        # Core + Plotly visualization
pip install ".[collect,viz]"# Both (recommended)
```

For development (editable install + test tools):

```bash
pip install -e ".[dev]"
```

The default solver is [HiGHS](https://highs.dev/) (bundled via `highspy`). Other solvers (Gurobi, CBC, GLPK) are supported via `--solver`.

## Usage

EOLES-Dispatch follows a **create → solve → visualize** workflow. Each run is a self-contained directory under `runs/`.

### Creating a run

```bash
eoles-dispatch create <name> --scenario <scenario> --year <year> [options]
```

| Option | Description |
|--------|-------------|
| `--scenario baseline` | Scenario name (looks for `scenarios/<name>/` directory) |
| `--year 2019` | Simulation year (data must cover this year) |
| `--months 1` | Restrict to a single month (e.g. January) for fast testing |
| `--months 6-8` | Restrict to a range of months (e.g. June to August) |
| `--rn-horizon current\|future` | Renewables.ninja wind fleet: `current` (installed ~2020) or `future` (next-gen turbines) |
| `--actual-cf` | Use historical capacity factors instead of Renewables.ninja |
| `--no-download` | Don't auto-download missing data |

The `--months` option is useful for fast testing: 1 month solves in ~4 minutes on a laptop.

### Solving a run

```bash
eoles-dispatch solve <name> [options]
```

| Option | Description |
|--------|-------------|
| `--solver highs` | LP solver (default: highs). Also: gurobi, cbc, glpk |
| `--model-version standard` | Model variant: `standard` (full thermal dynamics) or `static_thermal` |
| `--reports prices production` | Output reports to generate (also: `capa_on`, `FRtrade`) |

### Visualizing results

```bash
eoles-dispatch viz <name>           # opens interactive HTML in browser
eoles-dispatch viz run1 run2        # generate reports for multiple runs
eoles-dispatch viz <name> --no-open # generate without opening browser
```

The report is a self-contained HTML file at `runs/<name>/viz.html` with four tabs:
- **France — Inputs**: demand, VRE profiles, nuclear availability, capacity mix, etc.
- **France — Outputs**: spot price statistics, price duration curve, production mix
- **Other countries — Inputs/Outputs**: same charts for the 6 other countries

### Other commands

```bash
# List all runs and their status
eoles-dispatch list

# Download data from ENTSO-E and Renewables.ninja
eoles-dispatch collect --start 2020 --end 2025

# Download only from one source
eoles-dispatch collect --start 2020 --end 2025 --source entsoe
eoles-dispatch collect --start 2020 --end 2025 --source ninja

# Convert an old Excel scenario to CSV directory
eoles-dispatch convert-scenario scenarios/Scenario_BASELINE.xlsx
```

## Project structure

```
EOLES-Dispatch/
├── pyproject.toml                  # Package config & dependencies
├── src/eoles_dispatch/
│   ├── __main__.py                 # CLI entry point
│   ├── config.py                   # Model constants & default parameters
│   ├── format_outputs.py           # Result extraction & export
│   ├── utils.py                    # Utility functions
│   ├── run/                        # Run orchestration module
│   │   ├── __init__.py
│   │   ├── _main_run.py            # Run lifecycle (create, solve, list)
│   │   ├── format_inputs.py        # Data loading & preprocessing
│   │   ├── compute.py              # Model building & solving
│   │   └── scenario.py             # Scenario loading & management
│   ├── collect/                    # Data collection module
│   │   ├── __init__.py
│   │   ├── _main_collect.py        # Data collection orchestrator
│   │   ├── entsoe.py               # ENTSO-E API client
│   │   ├── elexon.py               # Elexon BMRS API client (GB post-Brexit fallback)
│   │   ├── rninja.py               # Renewables.ninja data fetching
│   │   └── gap_filling.py          # Missing data gap-filling logic
│   ├── viz/                        # Visualization module
│   │   ├── __init__.py
│   │   ├── report.py               # Interactive HTML report generation
│   │   ├── charts_inputs.py        # Input visualization charts
│   │   ├── charts_outputs.py       # Output visualization charts
│   │   ├── loaders.py              # Data loading for visualization
│   │   └── theme.py                # Plotly theme configuration
│   └── models/
│       ├── __init__.py
│       ├── default.py              # Standard model (startup, ramping, part-load)
│       └── static_thermal.py       # Simplified model (no thermal dynamics)
├── tests/                          # Test suite (pytest)
├── .github/                        # CI/CD configuration
│   └── workflows/
│       └── test.yml                # GitHub Actions test workflow
├── docs/                           # Documentation
├── scenarios/
│   ├── baseline/                   # Default scenario (12 CSV files)
│   └── scenario_editor.html        # Browser-based scenario editor
├── .env.example                    # Template for environment variables (API keys)
├── data/                           # Historical data (gitignored, regenerable)
│   ├── time_varying_inputs/        # Demand, prices, hydro, nuclear (ENTSO-E / Elexon)
│   ├── renewable_ninja/            # Wind & solar capacity factors (Ninja)
│   ├── gap_fill_report.csv/.txt    # Log of gap-filled missing values
│   └── DATA_COLLECTION.md          # Detailed data pipeline documentation
└── runs/                           # Run directories (gitignored)
    └── <run_name>/
        ├── run.yaml                # Metadata (scenario, year, status, timestamps)
        ├── inputs/                 # Formatted model inputs
        ├── outputs/                # Model results (prices, production, ...)
        ├── scenario/               # Copy of the scenario used
        └── viz.html                # Interactive report
```

## Scenarios

A scenario is a directory of 12 CSV files describing the power system configuration:

| File | Description |
|------|-------------|
| `thr_specs.csv` | Thermal technology specs (fuel type, efficiency, variable costs, min stable generation, ramp rates, startup costs) |
| `capa.csv` | Installed capacity by area and technology (GW) |
| `links.csv` | Interconnection capacity between modeled countries (GW) |
| `exo_IM.csv` / `exo_EX.csv` | Import/export capacity with exogenous neighbors (GW) |
| `rsv_req.csv` | Reserve requirements by VRE technology |
| `str_vOM.csv` | Storage variable O&M costs |
| `maxAF.csv` | Maximum hourly availability factors by technology |
| `yEAF.csv` | Yearly energy availability factors |
| `capa_in.csv` | Storage charging capacity (GW) |
| `stockMax.csv` | Storage reservoir volume (GWh) |
| `fuel_timeFactor.csv` | Seasonal fuel price weights (calendar months 1–12, mean = 1 per fuel) |
| `fuel_areaFactor.csv` | Country-specific fuel price correction factors |

### Editing scenarios

Open `scenarios/scenario_editor.html` in a browser for a tabbed spreadsheet-like interface. You can:
- Load an existing scenario folder or individual CSVs
- Edit values directly in the browser
- Download the modified scenario as a ZIP (Unzip before using for simulation)

Excel scenarios (`.xlsx`) are also supported as fallback — convert them with `eoles-dispatch convert-scenario`.

## Data

Historical data is **not included in the repository** (too large) but is downloaded automatically on first run creation, or manually via:

```bash
eoles-dispatch collect --start 2019 --end 2020
```

### Sources

| Source | Data | API |
|--------|------|-----|
| [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/) | Demand, generation by fuel, day-ahead prices, hydro inflows, nuclear availability | via [`entsoe-py`](https://github.com/EnergieID/entsoe-py) (requires `ENTSOE_API_KEY`) |
| [Elexon BMRS](https://bmrs.elexon.co.uk/) | Same as ENTSO-E, for GB only (post-Brexit fallback) | [Insights API](https://data.elexon.co.uk/bmrs/api/v1) (free, no key required) |
| [Renewables.ninja](https://www.renewables.ninja/) | Wind (onshore/offshore) and solar PV capacity factors | Public country-level downloads (no key required) |

**API key setup:**

```bash
cp .env.example .env
# Edit .env and add your ENTSO-E API key
# (register for free at https://transparency.entsoe.eu/)
```

The key is validated at startup with a quick test query — you'll get a clear error immediately if it's missing or invalid. See [`data/DATA_COLLECTION.md`](data/DATA_COLLECTION.md) for full details on the data pipeline.

### Country zone mapping

| Model area | ENTSO-E bidding zone | Renewables.ninja | Notes |
|------------|---------------------|-------------------|-------|
| FR | FR | FR | |
| BE | BE | BE | |
| DE | DE_LU | DE + LU (weighted) | Germany-Luxembourg bidding zone |
| CH | CH | CH | |
| IT | IT (load), IT_NORD (prices) | IT | Whole country for volumes, North for prices |
| ES | ES | ES | |
| UK | GB (→ Elexon post-2021) | GB | Great Britain (excl. Northern Ireland). ENTSO-E data unavailable post-Brexit; automatic Elexon fallback. |

## Model description

EOLES-Dispatch is a linear programming model that minimizes total system dispatch cost over all hours and areas, subject to:

- **Adequacy**: supply meets demand + reserves in every hour and area
- **Thermal constraints** (standard model): minimum stable generation, startup/shutdown, ramp rates
- **VRE curtailment**: renewable generation bounded by capacity × availability factor
- **Storage**: energy balance, charge/discharge limits, reservoir capacity
- **Hydro**: monthly inflow limits, max turbining/pumping rates
- **Nuclear**: weekly availability factor caps
- **Trade**: bilateral flows bounded by interconnection capacity, with 2% transmission losses

The dual variable of the adequacy constraint gives the **marginal price** in each area and hour (EUR/MWh).

### Key parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `VOLL` | 15,000 EUR/MWh | Value of lost load (demand shedding penalty) |
| `LOAD_UNCERTAINTY` | 1% | Hourly demand reserve margin |
| `TRLOSS` | 2% | Transmission losses on cross-border flows |
| `ETA_IN` | 95% / 90% | Charging efficiency (PHS / battery) |
| `ETA_OUT` | 90% / 95% | Discharging efficiency (PHS / battery) |

## Tests

```bash
# Run the full test suite (~74 tests, <2s)
pytest tests/ -v

# With timing details
pytest tests/ -v --durations=0
```

Tests cover configuration constants, ENTSO-E column matching, time series processing, timezone handling, fuel price seasonality expansion, model construction (both standard and static thermal variants), CLI smoke tests, and more. No solver or API calls are needed — everything runs offline with synthetic fixtures.

CI runs automatically on push via GitHub Actions (Python 3.9 / 3.11 / 3.12).

## Performance notes

Solve times depend heavily on the time horizon and available RAM:

| Horizon | Variables (approx.) | Solve time (MacBook Air M2, 8GB) |
|---------|-------------------|----------------------------------|
| 1 month (744h) | ~600K | ~5 min |
| 3 months (2160h) | ~1.8M | ~45 min |
| Full year (8760h) | ~7M | ~9 hours |

## Acknowledgements

- Based on the [EOLES family of models](https://www.centre-cired.fr/the-eoles-family-of-models/) by Behrang Shirizadeh, Quentin Perrier & Philippe Quirion (CIRED)
- Python implementation originally by Nilam De Oliveira-Gill
- Renewable generation data by Iain Staffell and Stefan Pfenninger ([Renewables.ninja](https://www.renewables.ninja/))
- Power system data from [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/)

## License

MIT
