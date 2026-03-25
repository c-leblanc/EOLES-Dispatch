# Data collection and preparation — EOLES-Dispatch

This document describes the full pipeline for collecting, cleaning, and
formatting the input data required by the EOLES-Dispatch model. The
corresponding source code lives in
[`src/eoles_dispatch/datacoll/`](../src/eoles_dispatch/datacoll/) (main orchestrator in
[`main_collect.py`](../src/eoles_dispatch/datacoll/main_collect.py)).

---

## 1. Overview

EOLES-Dispatch requires hourly time series for every modeled country. Two data
sources are used:

- **ENTSO-E Transparency Platform** — historical actuals for electricity demand,
  generation by fuel type, day-ahead prices, and installed capacities.
- **Renewables.ninja** — simulated capacity factor profiles for variable
  renewable energy (wind, solar), based on MERRA-2 meteorological reanalysis.

Data collection is launched via:

```bash
eoles-dispatch collect --start 2020 --end 2025 [--source entsoe|ninja|all]
```

or programmatically through `collect_all()`.

---

## 2. Geographic scope

### 2.1 Modeled areas (`DEFAULT_AREAS`)

| Code | Country | Notes |
|------|---------|-------|
| `FR` | France | |
| `BE` | Belgium | |
| `DE` | Germany | ENTSO-E bidding zone DE_LU (includes Luxembourg) |
| `CH` | Switzerland | |
| `IT` | Italy | National aggregate |
| `ES` | Spain | |
| `UK` | Great Britain | England + Scotland + Wales; excludes Northern Ireland |

### 2.2 Exogenous areas (`DEFAULT_EXO_AREAS`)

`NL`, `DK1`, `DK2`, `SE4`, `PL`, `CZ`, `AT`, `GR`, `SI`, `PT`, `IE`

These areas are not modeled explicitly. Only their day-ahead prices are
collected, serving as boundary conditions for cross-border trade.

### 2.3 Notable perimeter choices

- **DE → DE_LU**: The ENTSO-E bidding zone has been DE_LU since October 2018.
  Luxembourg represents ~0.6 GW peak vs ~80 GW for Germany, so the impact is
  negligible (<1%). Renewables.ninja uses DE alone, which is consistent since
  Luxembourg's renewable capacity is marginal.

- **IT**: Load and generation use the IT zone (national aggregate). Prices use
  IT_NORD (the reference price zone), because ENTSO-E does not publish prices
  at the IT level.

- **UK → GB**: Corresponds to Great Britain only. Northern Ireland belongs to
  the IE_SEM all-island market and is treated as an exogenous area via IE.
  Since Brexit (~mid-2021), GB no longer transmits data to ENTSO-E. For recent
  years, data is automatically sourced from the Elexon BMRS API instead (see
  section 3.8).

---

## 3. Data collected from ENTSO-E

Access to ENTSO-E uses the `entsoe-py` library and requires an API key
(environment variable `ENTSOE_API_KEY`).

### 3.1 Electricity demand (`demand.csv`)

| | |
|---|---|
| **Source** | `query_load` (Actual Total Load) |
| **Resolution** | Hourly |
| **Unit** | GW (converted from MW, ÷1000) |
| **Processing** | Hourly resampling → gap-filling |

Demand represents the actual system load as published by TSOs through ENTSO-E.

### 3.2 Non-market-dependent generation — NMD (`nmd.csv`)

| | |
|---|---|
| **Source** | `query_generation`, aggregating PSR types: B01 (biomass), B09 (geothermal), B13 (marine), B15 (other renewable), B17 (waste), B20 (other) |
| **Resolution** | Hourly |
| **Unit** | GW (converted from MW, ÷1000) |
| **Processing** | Hourly resampling → gap-filling |
| **Fallback** | If the aggregate query fails, each PSR type is queried individually and summed |

NMD represents non-dispatchable must-run generation. It is subtracted from gross
demand to obtain the residual demand that the model must satisfy.

### 3.3 VRE capacity factors (`offshore.csv`, `onshore.csv`, `pv.csv`, `river.csv`)

| | |
|---|---|
| **Source** | `query_generation` (production) + `query_installed_generation_capacity` |
| **PSR types** | B18 (wind offshore), B19 (wind onshore), B16 (solar PV), B11 (run-of-river hydro) |
| **Calculation** | CF = production / installed capacity, clipped to [0, 1] |
| **Resolution** | Hourly |
| **Unit** | Dimensionless (capacity factor, 0 to 1) |
| **Fallback** | If installed capacity is unavailable, the observed production maximum is used as proxy |

These capacity factors reflect the actual historical performance of installed
fleets. They differ from Renewables.ninja profiles, which are weather-based and
use technology assumptions independent of the existing fleet.

### 3.4 Exogenous prices (`exoPrices.csv`)

| | |
|---|---|
| **Source** | `query_day_ahead_prices` |
| **Areas** | NL, DK1, DK2, SE4, PL, CZ, AT, GR, SI, PT, IE |
| **Resolution** | Hourly |
| **Unit** | EUR/MWh |
| **Processing** | Hourly resampling → gap-filling (linear interpolation threshold = 24h) |

Exogenous prices serve as a price signal for cross-border exchanges with
non-modeled areas. The linear interpolation threshold is set higher (24h instead
of 3h) because prices exhibit more frequent gaps and less structured
intra-day variability.

### 3.5 Lake inflows (`lake_inflows.csv`)

| | |
|---|---|
| **Source** | `query_generation` (PSR types B12 lake + B10 PHS) |
| **Calculation** | inflows = lake_production + PHS_production − η × PHS_consumption, with η = 0.9 × 0.95 = 0.855 |
| **Resolution** | Monthly (sum of hourly values) |
| **Unit** | TWh (converted from MWh, ÷10⁶) |
| **Processing** | Hourly gap-filling → monthly aggregation → clip ≥ 0 |

Inflows are estimated from observed net hydro production. This is not a direct
measurement of natural inflows but a reasonable approximation of the hydraulic
energy available each month.

### 3.6 Hydro power limits (`hMaxIn.csv`, `hMaxOut.csv`)

| | |
|---|---|
| **Source** | `query_generation` (PSR types B10 PHS + B12 lake) |
| **Calculation** | Monthly maximum of hourly power output |
| **Resolution** | Monthly |
| **Unit** | GW (converted from MW, ÷1000) |

- `hMaxOut` = monthly max of (lake production + PHS production)
- `hMaxIn` = monthly max of PHS consumption

### 3.7 Nuclear availability (`nucMaxAF.csv`)

| | |
|---|---|
| **Source** | `query_generation` (PSR type B14 nuclear) + `query_installed_generation_capacity` |
| **Calculation** | Availability factor = production / installed capacity, weekly max, clipped to [0, 1] |
| **Resolution** | Weekly |
| **Unit** | Dimensionless (0 to 1) |

The weekly maximum (rather than the mean) reflects the peak achievable output
during each week, capturing the nuclear fleet's maintenance schedule.

### 3.8 Elexon BMRS fallback for GB (post-Brexit)

Since ~mid-2021, Great Britain no longer reports data to ENTSO-E. When the
collection pipeline detects that ENTSO-E data for GB is missing or too sparse
(< 50% coverage of the requested period), it automatically falls back to the
**Elexon BMRS Insights API**, which is the official data platform for the GB
electricity market.

| | |
|---|---|
| **API** | Elexon BMRS Insights API |
| **Base URL** | `https://data.elexon.co.uk/bmrs/api/v1` |
| **Authentication** | None required (free, public, no registration) |
| **Resolution** | Half-hourly (resampled to hourly to match ENTSO-E) |
| **Source code** | [`src/eoles_dispatch/datacoll/elexon.py`](../src/eoles_dispatch/datacoll/elexon.py) |

**Endpoint mapping:**

| EOLES variable | Elexon endpoint | Notes |
|---|---|---|
| Demand | `/demand/actual/total` | Actual Total Load (ATL dataset) |
| NMD generation | `/generation/actual/per-type` | Biomass + Other from AGPT dataset |
| VRE capacity factors | `/generation/actual/per-type` | Wind Onshore/Offshore, Solar, Hydro Run-of-river |
| Nuclear availability | `/generation/actual/per-type` | Nuclear generation from AGPT dataset |
| Hydro limits | `/generation/actual/per-type` | PHS + Hydro from AGPT dataset |
| Lake inflows | `/generation/actual/per-type` | PHS + Hydro net production |
| Day-ahead prices | `/balancing/pricing/market-index` | N2EX or APX market index (GBP/MWh) |

The fallback is transparent: the gap-filling report logs which data came from
Elexon, and the output format is identical regardless of the source. The
pipeline queries ENTSO-E first and only contacts Elexon when needed, so
historical data from before Brexit still comes from ENTSO-E.

**Note on prices**: Elexon reports prices in GBP/MWh. Since UK is a modeled
area (not exogenous), its prices are determined by the model rather than
downloaded. The Elexon price endpoint is available in the module for reference
but is not used in the standard collection pipeline.

---

## 4. Data collected from Renewables.ninja

Renewables.ninja provides simulated capacity factor profiles derived from
MERRA-2 meteorological reanalysis. Unlike ENTSO-E data, these profiles are
independent of the historically installed fleet and can be used for prospective
scenarios.

### 4.1 Downloaded files

| File | Content |
|------|---------|
| `pv.csv` | Solar PV (national aggregate) |
| `onshore_current.csv` | Onshore wind — current fleet (installed ~2020) |
| `onshore_future.csv` | Onshore wind — future fleet (next-gen turbines, taller towers) |
| `offshore_current.csv` | Offshore wind — current fleet |
| `offshore_future.csv` | Offshore wind — future fleet |

### 4.2 Country code mapping

`FR→FR`, `BE→BE`, `DE→DE`, `CH→CH`, `IT→IT`, `ES→ES`, `UK→GB`, `LU→LU`

Landlocked countries (CH, LU) have no offshore data; their offshore columns are
filled with zeros.

### 4.3 Download format

- **URL**: `https://www.renewables.ninja/country_downloads/{ISO2}/{filename}`
- **Method**: Direct HTTP download (no API key required)
- **Column extracted**: `NATIONAL` (nationally weighted aggregate)
- **Resolution**: Hourly
- **Unit**: Dimensionless (capacity factor, 0 to 1)

### 4.4 Wind fleet variants

Renewables.ninja provides two wind turbine technology assumptions:

- **current**: reflects the fleet installed as of ~2020. Current-generation
  turbines with typical hub heights and rotor diameters of the existing fleet.
- **future**: projected next-generation turbines. Taller towers, larger rotors,
  improved power curves — representative of turbines being installed in the
  mid-2020s and beyond.

The variant is selectable via `--rn-horizon current|future` when creating a run.
PV profiles have a single variant (no fleet distinction).

---

## 5. Missing value treatment (gap-filling)

ENTSO-E data regularly contains gaps due to sensor failures, publication delays,
or API errors. EOLES-Dispatch requires complete time series with no missing
values. An intelligent gap-filling process is applied to all hourly series.

### 5.1 Approach: cascading temporal analogy

Energy time series exhibit strong multi-periodic structure:

- Intra-day cycle (day/night)
- Weekly cycle (weekday/weekend)
- Seasonal cycle (summer/winter)
- Inter-annual repetition

Gap-filling exploits this structure by searching for the **closest analogous
period** that has valid data. Four strategies are applied in cascade, from most
precise to most coarse:

| Gap size | Method | Rationale |
|---|---|---|
| ≤ 3h (24h for prices) | Linear interpolation | Signal changes little over a few hours |
| 3h – 48h | Same weekday, ±1 week (then ±2 weeks) | Preserves daily + weekly cycle |
| 48h – 7 days | Same week, ±1 year | Preserves seasonality |
| > 7 days | Multi-year average (±1 year, ±2 years) | Last resort, with level normalization |

### 5.2 Level scaling

When data from another week or year is used as an analogue, a scaling ratio is
computed from observed data around the gap (±24h):

```
ratio = mean(observed data around gap)
      / mean(analogue data around source period)

filled value = analogue data × ratio
```

This captures the **temporal profile** of the analogue (shape of the daily
cycle, peaks, troughs) while respecting the **current absolute level** (if
demand is higher this year, the copied profile is scaled up proportionally).

Scaling is only applied when sufficient context is available (> 6 valid hours)
on both sides. Otherwise the ratio defaults to 1 (direct copy).

### 5.3 Analogue quality criterion

An analogue is accepted only if **at least 80%** of its values are valid
(non-NaN). This prevents filling a gap with another gap. If the analogue
contains a few residual NaN values, they are linearly interpolated within the
copied period.

### 5.4 Safety net

After all cascade strategies, if any NaN values remain (which should only happen
in extreme cases), forward-fill then back-fill is applied. No value is ever
silently replaced with zero.

### 5.5 Special case: exogenous prices

For day-ahead prices (`exoPrices`), the linear interpolation threshold is raised
to 24h instead of 3h. This accounts for the fact that prices are published on a
daily basis (full-day gaps are common) and that intra-day price variability is
less structured than demand or renewable generation.

---

## 6. Gap-filling report

Each collection run automatically generates a detailed report of all gap-filling
operations:

```
data/gap_fill_report.csv   — detailed log of every filled gap
data/gap_fill_report.txt   — human-readable summary
```

### 6.1 CSV report contents

Each row corresponds to one filled gap:

| Column | Description |
|--------|-------------|
| `variable` | Variable name (`demand`, `nmd`, `cf_onshore`, `exo_price`, ...) |
| `area` | Area code (`FR`, `DE`, ...) |
| `gap_start` | Timestamp of gap start |
| `gap_end` | Timestamp of gap end |
| `gap_hours` | Gap duration in hours |
| `method` | Method used (see below) |
| `scaling_ratio` | Scaling ratio applied (empty if not applicable) |

**Available methods**:
- `linear_interpolation` — short gap, direct interpolation
- `weekly_analogue_next` / `weekly_analogue_previous` — same weekday ±1 week
- `weekly_analogue_next_±2` / `weekly_analogue_previous_±2` — same weekday ±2 weeks
- `yearly_analogue_next` / `yearly_analogue_previous` — same week ±1 year
- `multi_year_average_Ny` — average over N years
- `linear_interpolation_fallback` — last resort
- `ffill_bfill_safety_net` — final safety net

### 6.2 Text report contents

- Total number of gaps and hours filled
- Breakdown by method (gap count, total hours)
- Breakdown by variable × area (gap count, total hours, largest gap)
- **Warning section** for gaps longer than 24h (recommended for manual review)

---

## 7. Unit conversions

| Raw ENTSO-E data | Stored unit |
|---|---|
| Power in MW | **GW** (÷ 1,000) |
| Energy in MWh | **TWh** (÷ 1,000,000) |
| Capacity factors | Dimensionless, clipped to [0, 1] |
| Prices | EUR/MWh (unchanged) |

---

## 8. Output file structure

```
data/
├── time_varying_inputs/
│   ├── demand.csv              Hourly demand by area (GW)
│   ├── nmd.csv                 Hourly NMD generation by area (GW)
│   ├── offshore.csv            Hourly offshore wind CF by area
│   ├── onshore.csv             Hourly onshore wind CF by area
│   ├── pv.csv                  Hourly solar PV CF by area
│   ├── river.csv               Hourly run-of-river CF by area
│   ├── exoPrices.csv           Hourly day-ahead prices, exogenous areas (EUR/MWh)
│   ├── lake_inflows.csv        Monthly lake inflows by area (TWh)
│   ├── hMaxIn.csv              Monthly max hydro charging power (GW)
│   ├── hMaxOut.csv             Monthly max hydro discharging power (GW)
│   └── nucMaxAF.csv            Weekly nuclear availability factor (0–1)
│
├── renewable_ninja/
│   ├── pv.csv                  Solar PV CF (Ninja)
│   ├── onshore_current.csv     Onshore wind CF, current fleet
│   ├── onshore_future.csv      Onshore wind CF, future fleet
│   ├── offshore_current.csv    Offshore wind CF, current fleet
│   └── offshore_future.csv     Offshore wind CF, future fleet
│
├── gap_fill_report.csv         Detailed gap-filling log
├── gap_fill_report.txt         Human-readable gap-filling summary
└── DATA_COLLECTION.md          This document
```

**Hourly file format**:
```csv
hour,FR,BE,DE,CH,IT,ES,UK
2020-01-01 00:00:00,62.3,10.1,55.8,...
```

**Monthly format**: `month,FR,BE,DE,...` with `month` as `YYYYMM`.

**Weekly format**: `week,FR,BE,DE,...` with `week` as `YYYYWW`.

---

## 9. Error handling and fallbacks

The collection pipeline is designed to be robust against partial ENTSO-E API
unavailability:

- **Entire area unavailable**: a warning is logged and the area is omitted from
  the output file. Other areas are collected normally.
- **Aggregate NMD query fails**: each NMD fuel type is queried individually as a
  fallback.
- **Installed capacity unavailable**: the observed production maximum is used as
  a proxy for capacity factor calculation. If the maximum is zero, a capacity of
  1 MW is used to avoid division by zero.
- **Ninja download fails**: a warning is logged and the area is omitted.
  Landlocked countries without offshore data have their offshore columns filled
  with zeros.

---

## 10. Technical requirements

**Python dependencies**:
- `pandas`, `numpy`
- `entsoe-py` (`pip install entsoe-py`)

**Environment variables**:
- `ENTSOE_API_KEY`: ENTSO-E Transparency Platform API key
  (free registration at https://transparency.entsoe.eu/)
- Can be set via shell (`export ENTSOE_API_KEY=...`) or in a `.env` file at
  the project root (see `.env.example`). The `.env` file is loaded automatically
  via `python-dotenv` if installed.
- The API key is validated at the start of each collection run with a lightweight
  test query. Invalid or missing keys are caught immediately rather than after
  partial downloads.

**Renewables.ninja**:
- No API key required (public download)
- HTTP access to `https://www.renewables.ninja/country_downloads/`

**Elexon BMRS** (GB fallback):
- No API key or registration required
- HTTP access to `https://data.elexon.co.uk/bmrs/api/v1`

---

## 11. Known limitations

- **ENTSO-E / Ninja temporal mismatch**: ENTSO-E data may cover more recent
  years than those available on Renewables.ninja (~2015–2024). If a run requests
  a year not covered by Ninja, the corresponding profiles will be absent.

- **Germany DE vs DE_LU**: the perimeter inconsistency between ENTSO-E (DE_LU)
  and Ninja (DE only) introduces an error of ~0.75% on renewable profiles. This
  is negligible for modeling purposes.

- **Hydro inflows**: lake inflows are estimated from observed net production, not
  measured directly. This approximation may underestimate actual inflows during
  periods of spillage.

- **No automatic physical plausibility checks**: values are not validated against
  physical bounds (demand always positive, prices within reasonable range, etc.).
  The gap-filling report does, however, help identify problematic series.

- **GB hydro data from Elexon**: the Elexon AGPT dataset does not separate PHS
  production from consumption in the same way as ENTSO-E. Lake inflow and hydro
  limit estimates for GB post-Brexit are therefore rougher approximations than
  for other countries.
