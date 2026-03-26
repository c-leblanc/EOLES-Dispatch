"""Derived variable computations for EOLES-Dispatch.

Computes capacity factors, nuclear availability, lake inflows, and hydro
limits from raw production history. All functions expect pre-filtered
production dicts with POSIX hours.
"""

import numpy as np
import pandas as pd

from ..config import NMD_TYPES, ETA_IN, ETA_OUT


def compute_nmd(production, areas, nmd_tecs = NMD_TYPES):
    """Compute NMD (non-market-dependent) production from raw generation data.

    NMD = sum of biomass, geothermal, marine, other_renew, waste, other.

    Args:
        production: dict {area: DataFrame} with columns ['hour', fuel1, ...].
            Already filtered to the simulation period. 'hour' is POSIX hours.
        areas: List of area codes.

    Returns:
        DataFrame with columns ['area', 'hour', 'value'] in GW.
    """
    frames = {}
    for area in areas:
        df = production[area]
        nmd_cols = [c for c in nmd_tecs if c in df.columns]
        if nmd_cols:
            nmd_series = df[nmd_cols].sum(axis=1) / 1000  # MW → GW
        else:
            nmd_series = pd.Series(0, index=df.index)
        frames[area] = pd.DataFrame({
            "hour": df["hour"].values,
            "value": nmd_series.values,
        })
        frames[area]["area"] = area

    if not frames:
        return pd.DataFrame(columns=["area", "hour", "value"])

    result = pd.concat(frames.values(), ignore_index=True)
    return result[["area", "hour", "value"]]


def compute_vre_capacity_factors(
        production,
        installed_capa,
        areas,
        technologies=["offshore", "onshore", "solar", "river"]
        ):
    """Compute VRE capacity factors from raw production and installed capacity.

    CF = hourly_production / installed_capacity.
    Falls back to max(hourly_production) when installed capacity is unavailable.

    Args:
        production: dict {area: DataFrame} with columns ['hour', fuel1, ...].
            Already filtered to the simulation period. 'hour' is POSIX hours.
        installed_capa: DataFrame with tec as index and area codes as columns (MW).
            Loaded from data/<year>/installed_capacity.csv.
        areas: List of area codes.
        technologies: List of VRE tech names (default: offshore, onshore, solar, river).

    Returns:
        DataFrame with columns ['area', 'tec', 'hour', 'value'].
    """
    frames = []
    for tec in technologies:
        for area in areas:
            df = production[area]
            if tec not in df.columns:
                continue

            prod_mw = df[tec].values

            # Use installed capacity if available, otherwise approximate from max production
            if (installed_capa is not None
                    and tec in installed_capa.index
                    and area in installed_capa.columns):
                capa_mw = installed_capa.loc[tec, area]
            else:
                capa_mw = prod_mw.max()
            cf = prod_mw / capa_mw if capa_mw > 0 else np.zeros_like(prod_mw)

            frame = pd.DataFrame({
                "area": area,
                "tec": tec,
                "hour": df["hour"].values,
                "value": cf,
            })
            frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=["area", "tec", "hour", "value"])

    return pd.concat(frames, ignore_index=True)[["area", "tec", "hour", "value"]]


def compute_nuclear_max_af(production, installed_capa, areas, hour_week):
    """Compute weekly max nuclear availability factor from raw production.

    AF = hourly_production / installed_capacity, clipped to [0, 1].
    Weekly max = max AF within each week.

    Args:
        production: dict {area: DataFrame} with columns ['hour', ..., 'nuclear'].
            Already filtered to the simulation period. 'hour' is POSIX hours.
        installed_capa: DataFrame with tec as index and area codes as columns (MW).
            Loaded from data/<year>/installed_capacity.csv.
        areas: List of area codes.
        hour_week: DataFrame with columns ['hour', 'week'] (POSIX hours → YYWW).

    Returns:
        DataFrame with columns ['area', 'week', 'value'].
    """
    frames = []
    weeks = hour_week["week"].unique().tolist()

    for area in areas:
        df = production[area]
        if "nuclear" not in df.columns:
            for w in weeks:
                frames.append({"area": area, "week": w, "value": 1.0})
            continue

        merged = df[["hour", "nuclear"]].merge(hour_week, on="hour", how="inner")

        # Get installed nuclear capacity (MW)
        if (installed_capa is not None
                and "nuclear" in installed_capa.index
                and area in installed_capa.columns):
            capa_mw = installed_capa.loc["nuclear", area]
        else:
            capa_mw = merged["nuclear"].max() if merged["nuclear"].max() > 0 else 1

        merged["af"] = np.clip(merged["nuclear"] / max(capa_mw, 1), 0, 1)

        # Weekly max
        weekly = merged.groupby("week")["af"].max().reset_index()
        weekly["area"] = area
        weekly = weekly.rename(columns={"af": "value"})
        frames.append(weekly[["area", "week", "value"]])

    if not frames:
        return pd.DataFrame(columns=["area", "week", "value"])

    # frames can be list of dicts or DataFrames
    result_parts = []
    for f in frames:
        if isinstance(f, dict):
            result_parts.append(pd.DataFrame([f]))
        else:
            result_parts.append(f)
    return pd.concat(result_parts, ignore_index=True)[["area", "week", "value"]]


def compute_lake_inflows(production, areas, hour_month,
                         eta_phs=ETA_IN["lake_phs"]*ETA_OUT["lake_phs"]):
    """Compute monthly lake inflows from raw production data.

    lake_inflows = lake + phs + η * phs_in, summed monthly, in TWh.
    (phs_in is negative, so η * phs_in subtracts the pumping losses.)

    Args:
        production: dict {area: DataFrame} with columns ['hour', ...].
            Already filtered to the simulation period. 'hour' is POSIX hours.
        areas: List of area codes.
        hour_month: DataFrame with columns ['hour', 'month'] (POSIX hours → YYYYMM).
        eta_phs: PHS round-trip efficiency (default: 0.9 * 0.95 = 0.855).

    Returns:
        DataFrame with columns ['area', 'month', 'value'] in TWh.
    """
    frames = []

    for area in areas:
        if area not in production:
            continue
        df = production[area]
        prod_hours = pd.DataFrame({"hour": df["hour"].values})

        # Lake production
        if "lake" in df.columns:
            prod_hours["lake"] = df["lake"].values
        else:
            prod_hours["lake"] = 0.0

        # PHS production (positive) and consumption (negative)
        if "phs" in df.columns:
            prod_hours["phs"] = df["phs"].values
        else:
            prod_hours["phs"] = 0.0

        if "phs_in" in df.columns:
            prod_hours["phs_in"] = df["phs_in"].values
        else:
            prod_hours["phs_in"] = 0.0

        # Net inflow = lake + phs + η * phs_in  (phs_in is negative)
        prod_hours["inflow_mw"] = (
            prod_hours["lake"] + prod_hours["phs"] + eta_phs * prod_hours["phs_in"]
        )

        # Merge with month mapping and aggregate
        merged = prod_hours.merge(hour_month, on="hour", how="inner")
        monthly = merged.groupby("month")["inflow_mw"].sum().reset_index()
        monthly["value"] = monthly["inflow_mw"].clip(lower=0) / 1e6  # MWh → TWh
        monthly["area"] = area
        frames.append(monthly[["area", "month", "value"]])

    if not frames:
        return pd.DataFrame(columns=["area", "month", "value"])

    return pd.concat(frames, ignore_index=True)[["area", "month", "value"]]


def compute_hydro_limits(production, areas, hour_month):
    """Compute monthly max hydro charge/discharge from raw production.

    hMaxOut = monthly max of (lake + phs), in GW.
    hMaxIn  = monthly max of abs(phs_in), in GW.

    Args:
        production: dict {area: DataFrame} with columns ['hour', ...].
            Already filtered to the simulation period. 'hour' is POSIX hours.
        areas: List of area codes.
        hour_month: DataFrame with columns ['hour', 'month'].

    Returns:
        (hMaxIn, hMaxOut) DataFrames with columns ['area', 'month', 'value'] in GW.
    """
    frames_in = []
    frames_out = []

    for area in areas:
        if area not in production:
            continue
        df = production[area]
        prod_hours = pd.DataFrame({"hour": df["hour"].values})

        # Discharge: lake + PHS production
        lake_prod = df["lake"].values if "lake" in df.columns else np.zeros(len(df))
        phs_gen = df["phs"].values if "phs" in df.columns else np.zeros(len(df))
        prod_hours["out_mw"] = np.clip(lake_prod + phs_gen, 0, None)

        # Charge: abs(phs_in) — phs_in is negative
        phs_in = df["phs_in"].values if "phs_in" in df.columns else np.zeros(len(df))
        prod_hours["in_mw"] = np.abs(phs_in)

        # Merge with month mapping and get monthly max
        merged = prod_hours.merge(hour_month, on="hour", how="inner")

        monthly_out = merged.groupby("month")["out_mw"].max().reset_index()
        monthly_out["value"] = monthly_out["out_mw"] / 1000  # MW → GW
        monthly_out["area"] = area
        frames_out.append(monthly_out[["area", "month", "value"]])

        monthly_in = merged.groupby("month")["in_mw"].max().reset_index()
        monthly_in["value"] = monthly_in["in_mw"] / 1000  # MW → GW
        monthly_in["area"] = area
        frames_in.append(monthly_in[["area", "month", "value"]])

    def _combine(frames):
        if not frames:
            return pd.DataFrame(columns=["area", "month", "value"])
        return pd.concat(frames, ignore_index=True)[["area", "month", "value"]]

    return _combine(frames_in), _combine(frames_out)
