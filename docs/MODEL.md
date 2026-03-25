# EOLES-Dispatch — Model description

## Overview

EOLES-Dispatch is a **linear relaxation of a unit commitment model** designed to simulate realistic wholesale electricity prices at an hourly time-step, with a focus on the French power system. It minimizes total dispatch cost over a set of interconnected areas, subject to supply-demand adequacy, reserve requirements, thermal plant operational constraints, and storage/hydro management.

**Type**: Linear program (LP), no integer variables.
**Solver**: HiGHS (interior point method + crossover).
**Time resolution**: hourly (up to 8,760 hours/year).

### Geographic scope

The model is **centered on France**. Six neighboring countries (BE, DE, CH, IT, ES, UK) are modeled explicitly to capture cross-border trade dynamics and their impact on French prices. Without this multi-country coupling, simulated prices would reflect an isolated system.

- **France**: primary zone, focus of the analysis
- **6 coupled neighbors** (endogenous): BE, DE, CH, IT, ES, UK — co-optimized with France
- **11 exogenous price zones**: NL, DK1, DK2, SE4, PL, CZ, AT, GR, SI, PT, IE

Exogenous zones are not optimized: their day-ahead price is an input, and trade with them is bounded by interconnection capacities.

### Model variants

1. **`standard`** (default): Full thermal dynamics with startup, shutdown, ramping, and minimum stable generation.
2. **`static_thermal`**: Simplified model without thermal dynamics (proportional dispatch only).

---

## Notation and sets

| Symbol | Description |
|--------|-------------|
| $a \in \mathcal{A}$ | Modeled areas |
| $a^{exo} \in \mathcal{A}^{exo}$ | Exogenous areas |
| $h \in \mathcal{H}$ | Hours (POSIX timestamps) |
| $m \in \mathcal{M}$ | Months (YYYYMM) |
| $w \in \mathcal{W}$ | Weeks (YYWW) |
| $tec \in \mathcal{T}$ | All technologies |
| $thr \in \mathcal{T}_{th} \subset \mathcal{T}$ | Thermal technologies (nuclear, coal_SA, coal_1G, lignite, gas_ccgt1G, gas_ccgt2G, gas_ccgtSA, gas_ocgtSA, oil_light) |
| $vre \in \mathcal{T}_{vre} \subset \mathcal{T}$ | Variable renewables (offshore, onshore, pv, river) |
| $sto \in \mathcal{T}_{sto} \subset \mathcal{T}$ | Storage (lake_phs, battery) |
| $nmd$ | Non-market-dependent (biomass, geothermal, marine, waste, etc.) |
| $frr \subset \mathcal{T}$ | Technologies eligible for reserve provision |
| $m(h)$ | Month corresponding to hour $h$ |
| $w(h)$ | Week corresponding to hour $h$ |

---

## Decision variables (endogenous)

All variables are non-negative reals ($\geq 0$).

### Generation and dispatch

| Variable | Indexing | Unit | Description |
|----------|----------|------|-------------|
| $gene_{a,tec,h}$ | area, technology, hour | GW | Hourly generation |
| $on_{a,thr,h}$ | area, thermal, hour | GW | Online capacity (`standard` model only) |
| $startup_{a,thr,h}$ | area, thermal, hour | GW | Capacity started up |
| $turnoff_{a,thr,h}$ | area, thermal, hour | GW | Capacity shut down |
| $ramp\_up_{a,thr,h}$ | area, thermal, hour | GW | Upward ramp |

### Storage

| Variable | Indexing | Unit | Description |
|----------|----------|------|-------------|
| $storage_{a,sto,h}$ | area, storage, hour | GW | Charging power |
| $stored_{a,sto,h}$ | area, storage, hour | GWh | Stored energy at end of hour |

### Reserves

| Variable | Indexing | Unit | Description |
|----------|----------|------|-------------|
| $rsv_{a,tec,h}$ | area, technology, hour | GW | Spinning reserve (FRR) |

### Trade

| Variable | Indexing | Unit | Description |
|----------|----------|------|-------------|
| $im_{a_1,a_2,h}$ | importer, exporter, hour | GW | Import between modeled areas |
| $ex_{a_1,a_2,h}$ | exporter, importer, hour | GW | Export between modeled areas |
| $exo\_im_{a,a^{exo},h}$ | modeled area, exogenous area, hour | GW | Import from exogenous zone |
| $exo\_ex_{a,a^{exo},h}$ | modeled area, exogenous area, hour | GW | Export to exogenous zone |

### Shortfall and accounting

| Variable | Indexing | Unit | Description |
|----------|----------|------|-------------|
| $hll_{a,h}$ | area, hour | GW | Unserved energy (load loss) |
| $hcost_{a,h}$ | area, hour | kEUR | Hourly total cost |
| $hcarb_{a,h}$ | area, hour | kg CO2 | Hourly emissions |

---

## Exogenous parameters (inputs)

### Time series

| Parameter | Indexing | Unit | Source |
|-----------|----------|------|--------|
| $demand_{a,h}$ | area, hour | GW | ENTSO-E |
| $nmd_{a,h}$ | area, hour | GW | ENTSO-E (sum of biomass, geothermal, marine, waste, other) |
| $exoPrice_{a^{exo},h}$ | exogenous area, hour | EUR/MWh | ENTSO-E (day-ahead prices) |
| $lf_{a,vre,h}$ | area, VRE, hour | [0,1] | Renewables.ninja (capacity factors) |
| $lakeInflows_{a,m}$ | area, month | TWh | Derived from ENTSO-E data |

### Installed capacities

| Parameter | Indexing | Unit | Description |
|-----------|----------|------|-------------|
| $capa_{a,tec}$ | area, technology | GW | Installed capacity |
| $capa\_in_{a,sto}$ | area, storage | GW | Charging capacity |
| $stockMax_{a,sto}$ | area, storage | TWh | Maximum stored energy |

### Thermal availability

| Parameter | Indexing | Unit | Description |
|-----------|----------|------|-------------|
| $maxaf_{a,thr}$ | area, thermal | [0,1] | Maximum availability factor |
| $eaf_{a,thr}$ | area, thermal | [0,1] | Yearly average availability factor |
| $nucMaxAF_{a,w}$ | area, week | [0,1] | Weekly nuclear availability factor |

### Hydro constraints

| Parameter | Indexing | Unit | Description |
|-----------|----------|------|-------------|
| $hMaxOut_{a,m}$ | area, month | [0,1] | Monthly max turbining factor |
| $hMaxIn_{a,m}$ | area, month | [0,1] | Monthly max pumping factor |

### Interconnections

| Parameter | Indexing | Unit | Description |
|-----------|----------|------|-------------|
| $links_{a_1,a_2}$ | importer, exporter | GW | Interconnection capacity |
| $exo\_IM_{a,a^{exo}}$ | area, exogenous area | GW | Import capacity from exogenous zone |
| $exo\_EX_{a,a^{exo}}$ | area, exogenous area | GW | Export capacity to exogenous zone |

### Thermal parameters

| Parameter | Indexing | Description |
|-----------|----------|-------------|
| $\eta_{thr}$ (`efficiency`) | thermal | Full-load efficiency |
| $\eta^{50}_{thr}$ (`eff50`) | thermal | Efficiency at 50% load |
| $cf_{thr}$ (`co2_factor`) | thermal | CO2 emission factor (tCO2/GJ) |
| $p^{CO2}_{thr}$ (`co2_price`) | thermal | CO2 price (EUR/tCO2) |
| $vOM^{nf}_{thr}$ (`nonFuel_vOM`) | thermal | Non-fuel variable O&M (EUR/MWh) |
| $su^{fuel}_{thr}$ (`su_fuelCons`) | thermal | Startup fuel consumption (GJ) |
| $su^{fix}_{thr}$ (`su_fixedCost`) | thermal | Fixed startup cost (EUR) |
| $ramp^{fuel}_{thr}$ (`ramp_fuelCons`) | thermal | Ramping fuel consumption (GJ/MW) |
| $minSG_{thr}$ | thermal | Minimum stable generation (fraction of $on$) |
| $minON_{thr}$ | thermal | Minimum on-time (hours) |
| $minOFF_{thr}$ | thermal | Minimum off-time (hours) |
| $fp_{thr,a,m}$ (`fuel_price_adj`) | thermal, area, month | Adjusted fuel price (EUR/GJ) |

The adjusted fuel price is computed as:

$$fp_{thr,a,m} = price_{thr} \times timeFactor_{fuel(thr),m} + areaFactor_{fuel(thr),a}$$

### Reserves

| Parameter | Description |
|-----------|-------------|
| $rsv\_req_{vre}$ | Reserve requirement per VRE technology (fraction of installed capacity) |

### Constants

| Constant | Value | Description |
|----------|-------|-------------|
| $\sigma$ (`LOAD_UNCERTAINTY`) | 0.01 | Demand uncertainty (1%) |
| $\delta$ (`DELTA`) | 0.1 | Load variation factor (10%) |
| $VOLL$ | 15,000 EUR/MWh | Value of lost load |
| $\eta^{in}_{lake\_phs}$ | 0.95 | Pumping efficiency (PHS) |
| $\eta^{in}_{battery}$ | 0.90 | Charging efficiency (battery) |
| $\eta^{out}_{lake\_phs}$ | 0.90 | Turbining efficiency (PHS) |
| $\eta^{out}_{battery}$ | 0.95 | Discharging efficiency (battery) |
| $\lambda$ (`TRLOSS`) | 0.02 | Transmission losses (2%) |
| $GJ\_MWH$ | 3.6 | GJ to MWh conversion factor |

---

## Objective function

The model minimizes total dispatch cost over all areas and hours:

$$\min \sum_{a \in \mathcal{A}} \sum_{h \in \mathcal{H}} hcost_{a,h}$$

---

## Constraints

### 1. Supply-demand balance (adequacy)

For each area $a$ and hour $h$:

$$\sum_{tec} gene_{a,tec,h} + \sum_{a_2 \neq a} im_{a,a_2,h} + \sum_{a^{exo}} exo\_im_{a,a^{exo},h} = demand_{a,h} + \sum_{a_2 \neq a} ex_{a,a_2,h} + \sum_{a^{exo}} exo\_ex_{a,a^{exo},h} + \sum_{sto} storage_{a,sto,h} - hll_{a,h}$$

This is the central constraint. Its **dual value** (shadow price) gives the **electricity price** in each area at each hour.

### 2. Variable renewables (VRE)

Generation is bounded by installed capacity times the capacity factor:

$$gene_{a,vre,h} \leq capa_{a,vre} \times lf_{a,vre,h} \quad \forall a, vre, h$$

### 3. Non-market-dependent generation (NMD)

NMD generation is fixed exogenously:

$$gene_{a,nmd,h} = nmd_{a,h} \quad \forall a, h$$

### 4. Thermal plants

#### 4.1 Online capacity (`standard` model)

$$on_{a,thr,h} \leq capa_{a,thr} \times maxaf_{a,thr}$$

#### 4.2 Upper generation bound

**`standard` model**: generation + reserve is bounded by online capacity:

$$gene_{a,thr,h} + rsv_{a,thr,h} \leq on_{a,thr,h}$$

**`static_thermal` model**: generation is directly bounded by capacity:

$$gene_{a,thr,h} \leq capa_{a,thr} \times maxaf_{a,thr}$$

#### 4.3 Minimum stable generation (`standard` model)

$$minSG_{thr} \times on_{a,thr,h} \leq gene_{a,thr,h}$$

#### 4.4 Yearly availability

**`standard` model** (bound on online capacity):

$$\frac{1}{|\mathcal{H}|} \sum_{h} on_{a,thr,h} \leq capa_{a,thr} \times eaf_{a,thr}$$

**`static_thermal` model** (bound on generation):

$$\frac{1}{|\mathcal{H}|} \sum_{h} gene_{a,thr,h} \leq capa_{a,thr} \times eaf_{a,thr}$$

#### 4.5 Weekly nuclear availability

$$on_{a,nuclear,h} \leq capa_{a,nuclear} \times nucMaxAF_{a,w(h)}$$

(In the `static_thermal` model, $gene$ replaces $on$.)

#### 4.6 Startup/shutdown dynamics (`standard` model only)

**State continuity** (with cyclic wrap-around):

$$on_{a,thr,h+1} = on_{a,thr,h} + startup_{a,thr,h} - turnoff_{a,thr,h}$$

**Startup constraint** (respects minimum off-time):

$$startup_{a,thr,h} \leq capa_{a,thr} \times maxaf_{a,thr} - on_{a,thr,h} - \sum_{h' \in [h-minOFF_{thr},\, h)} turnoff_{a,thr,h'}$$

**Shutdown constraint** (respects minimum on-time):

$$turnoff_{a,thr,h} \leq on_{a,thr,h} - \sum_{h' \in [h-minON_{thr},\, h)} startup_{a,thr,h'}$$

#### 4.7 Ramping (`standard` model only)

Upward ramps are tracked for cost accounting (no upper bound is enforced):

$$ramp\_up_{a,thr,h+1} \geq gene_{a,thr,h+1} - gene_{a,thr,h}$$

### 5. Storage

#### 5.1 Storage capacity

$$stored_{a,sto,h} \leq stockMax_{a,sto} \times 1000$$

(stockMax in TWh, stored in GWh)

#### 5.2 Power bounds

**Charging**:

$$storage_{a,sto,h} \leq \begin{cases} capa\_in_{a,sto} \times hMaxIn_{a,m(h)} & \text{if } sto = lake\_phs \\ capa\_in_{a,sto} & \text{otherwise} \end{cases}$$

**Discharging**:

$$gene_{a,sto,h} + rsv_{a,sto,h} \leq \begin{cases} capa_{a,sto} \times hMaxOut_{a,m(h)} & \text{if } sto = lake\_phs \\ capa_{a,sto} & \text{otherwise} \end{cases}$$

#### 5.3 State of charge evolution

For batteries:

$$stored_{a,sto,h+1} = stored_{a,sto,h} + storage_{a,sto,h} \times \eta^{in}_{sto} - \frac{gene_{a,sto,h}}{\eta^{out}_{sto}}$$

For lakes/PHS, with natural inflows:

$$stored_{a,lake,h+1} = stored_{a,lake,h} + storage_{a,lake,h} \times \eta^{in}_{lake} - \frac{gene_{a,lake,h}}{\eta^{out}_{lake}} + \frac{lakeInflows_{a,m(h)} \times 1000 / |h \in m(h)|}{\eta^{out}_{lake}}$$

#### 5.4 Monthly water balance

For lakes/PHS, net monthly production must equal inflows:

$$\sum_{h \in m} \left( gene_{a,lake,h} - storage_{a,lake,h} \times \eta^{in}_{lake} \times \eta^{out}_{lake} \right) = lakeInflows_{a,m} \times 1000$$

### 6. Reserves (FRR)

The reserve requirement combines a VRE-related component and a demand-related component:

$$\sum_{frr} rsv_{a,frr,h} = \sum_{vre} rsv\_req_{vre} \times capa_{a,vre} + demand_{a,h} \times \sigma \times (1 + \delta)$$

Technologies not eligible for FRR do not contribute:

$$rsv_{a,tec,h} = 0 \quad \forall tec \notin frr$$

### 7. Trade

**Physical balance** (with transmission losses):

$$im_{a_1,a_2,h} = ex_{a_2,a_1,h} \times (1 - \lambda)$$

**Interconnection capacities**:

$$im_{a_1,a_2,h} \leq links_{a_1,a_2}$$

$$exo\_im_{a,a^{exo},h} \leq exo\_IM_{a,a^{exo}}$$

$$exo\_ex_{a,a^{exo},h} \leq exo\_EX_{a,a^{exo}}$$

---

## Cost definition

### `standard` model: 4-component decomposition

The hourly cost is defined by the constraint:

$$hcost_{a,h} = \underbrace{\sum_{thr} gene_{a,thr,h} \times genOM_{thr,a,m(h)}}_{\text{generation cost}} + \underbrace{\sum_{thr} on_{a,thr,h} \times onOM_{thr,a,m(h)}}_{\text{no-load cost}} + \underbrace{\sum_{thr} startup_{a,thr,h} \times su\_cost_{thr,a,m(h)}}_{\text{startup cost}} + \underbrace{\sum_{thr} ramp\_up_{a,thr,h} \times ramp\_cost_{thr,a,m(h)}}_{\text{ramping cost}}$$

$$+ \sum_{sto} gene_{a,sto,h} \times str\_vOM_{sto} + \sum_{a^{exo}} \frac{exo\_im_{a,a^{exo},h} - exo\_ex_{a,a^{exo},h}}{1 - \lambda} \times exoPrice_{a^{exo},h} + hll_{a,h} \times VOLL$$

The cost coefficients are derived from a two-efficiency model ($\eta$ at full load, $\eta^{50}$ at 50% load):

$$genOM_{thr,a,m} = \left(\frac{2}{\eta_{thr}} - \frac{1}{\eta^{50}_{thr}}\right) \times GJ\_MWH \times \left(fp_{thr,a,m} + \frac{cf_{thr} \times p^{CO2}_{thr}}{1000}\right)$$

$$onOM_{thr,a,m} = \left(\frac{1}{\eta^{50}_{thr}} - \frac{1}{\eta_{thr}}\right) \times GJ\_MWH \times \left(fp_{thr,a,m} + \frac{cf_{thr} \times p^{CO2}_{thr}}{1000}\right) + vOM^{nf}_{thr}$$

$$su\_cost_{thr,a,m} = su^{fuel}_{thr} \times \left(fp_{thr,a,m} + \frac{cf_{thr} \times p^{CO2}_{thr}}{1000}\right) + su^{fix}_{thr}$$

$$ramp\_cost_{thr,a,m} = ramp^{fuel}_{thr} \times \left(fp_{thr,a,m} + \frac{cf_{thr} \times p^{CO2}_{thr}}{1000}\right)$$

**Interpretation**: the total marginal cost of a thermal plant is decomposed into a cost proportional to output ($genOM$) and a no-load cost ($onOM$, independent of output level but dependent on being online). This decomposition uses a linear interpolation between the efficiency at 50% and 100% load.

### `static_thermal` model: simplified cost

$$hcost_{a,h} = \sum_{thr} gene_{a,thr,h} \times vOM_{thr,a,m(h)} + \sum_{sto} gene_{a,sto,h} \times str\_vOM_{sto} + \sum_{a^{exo}} \frac{exo\_im - exo\_ex}{1 - \lambda} \times exoPrice_{a^{exo},h} + hll_{a,h} \times VOLL$$

where:

$$vOM_{thr,a,m} = \frac{1}{\eta_{thr}} \times GJ\_MWH \times \left(fp_{thr,a,m} + \frac{cf_{thr} \times p^{CO2}_{thr}}{1000}\right) + vOM^{nf}_{thr}$$

---

## CO2 emissions

### `standard` model

$$hcarb_{a,h} = \sum_{thr} \left( gene_{a,thr,h} \times genCarb_{thr} + on_{a,thr,h} \times onCarb_{thr} + startup_{a,thr,h} \times suCarb_{thr} + ramp\_up_{a,thr,h} \times rampCarb_{thr} \right)$$

where:

$$genCarb_{thr} = \left(\frac{2}{\eta_{thr}} - \frac{1}{\eta^{50}_{thr}}\right) \times GJ\_MWH \times cf_{thr}$$

$$onCarb_{thr} = \left(\frac{1}{\eta^{50}_{thr}} - \frac{1}{\eta_{thr}}\right) \times GJ\_MWH \times cf_{thr}$$

$$suCarb_{thr} = su^{fuel}_{thr} \times cf_{thr}$$

$$rampCarb_{thr} = ramp^{fuel}_{thr} \times cf_{thr}$$

### `static_thermal` model

$$hcarb_{a,h} = \sum_{thr} gene_{a,thr,h} \times \frac{GJ\_MWH \times cf_{thr}}{\eta_{thr}}$$

---

## Output computation

### Electricity prices

Prices are the **dual values (shadow prices)** of the adequacy constraint. The dual represents the marginal cost of serving one additional MW of demand in a given area at a given hour.

$$price_{a,h} = \frac{\partial \text{Objective}}{\partial demand_{a,h}} = dual(adequacy\_constraint_{a,h})$$

**Economic interpretation**:
- The price reflects the **marginal generator cost** (the most expensive plant needed to balance supply and demand).
- If $hll_{a,h} > 0$ (load shedding), then $price_{a,h} = VOLL = 15{,}000$ EUR/MWh.
- If an interconnection is congested, prices differ between neighboring areas.
- In case of surplus (abundant wind/solar), the price is driven down by the cheapest online technology.
- Prices equalize between connected areas as long as transmission lines are not saturated.

Extraction is done via the Pyomo `model.dual` suffix on `model.adequacy_constraint[a, h]`.

### Hourly production

Hourly production by technology and area is extracted directly from the $gene_{a,tec,h}$ variable values. Detailed technologies are aggregated into categories for reporting:
- **wind** = onshore + offshore
- **coal** = coal_SA + coal_1G + lignite
- **gas** = gas_ccgt1G + gas_ccgt2G + gas_ccgtSA + gas_ocgtSA
- Storage charging appears with a negative sign (phs_in, battery_in).
- Net imports per area are computed as $\sum im - \sum ex$ over all trading partners.

### Online capacity (`standard` model)

The values of $on_{a,thr,h}$ are exported for each thermal technology and area.

### France trade

France's net imports from each partner are computed as:

$$FRtrade_{partner,h} = im_{FR,partner,h} \times (1 - \lambda) - ex_{FR,partner,h}$$

### Summary log

- **Total dispatch cost**: $\sum_{a,h} hcost_{a,h}$ converted to billion EUR.
- **Total emissions**: $\sum_{a,h} hcarb_{a,h}$ converted to MtCO2.

---

## Modeling assumptions

1. **Linear relaxation**: No binary variables for unit commitment. The $on$ variable is continuous, allowing partial startups/shutdowns.
2. **Perfect foresight**: The model has complete knowledge of future demand, VRE profiles, hydro inflows, and exogenous prices over the entire horizon.
3. **Single node per area**: No intra-area transmission congestion.
4. **VRE and NMD not optimized**: Variable renewables are bounded by profiles; NMD generation is fixed.
5. **Cyclic wrap-around**: The last hour wraps to the first for dynamic constraints (storage, startup/shutdown).
6. **Two-efficiency cost model**: The $genOM$/$onOM$ decomposition captures the effect of part-load operation on thermal efficiency via a linear interpolation between 50% and 100% load.
7. **Unconstrained ramping**: Upward ramps are tracked for cost accounting, but no upper bound is enforced.
8. **Simplified reserves**: Reserves are sized deterministically (1% of demand + margin + VRE contribution), without distinguishing aFRR from mFRR.
