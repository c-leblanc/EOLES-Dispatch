"""Visualization module for EOLES-Dispatch runs.

Generates an interactive HTML report with Plotly charts for run inputs
(and outputs when available). Open the resulting file in any browser.

Usage:
    eoles-dispatch viz my_run
    eoles-dispatch viz run1 run2  # compare two runs
"""

import webbrowser
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yaml
import numpy as np


# ── Color palettes (from pyREADING.R) ──

COUNTRY_COLORS = {
    "FR": "#2E86C1",
    "BE": "#D4AC0D",
    "DE": "#2C3E50",
    "ES": "#F39C12",
    "UK": "#E74C3C",
    "IT": "#27AE60",
    "CH": "#9B59B6",
}

TEC_COLORS = {
    "nuclear":      "#ffd966",
    "gas":          "#9d0208",
    "coal":         "#6c757d",
    "oil":          "#262626",
    "wind":         "#005900",
    "pv":           "#ffff00",
    "river":        "#011f4b",
    "lake_phs":     "#6497b1",
    "battery":      "#ffa500",
    "biomass":      "#262626",
    "nmd":          "#262626",
    "net_imports":  "#adb5bd",
    "net_exports":  "#adb5bd",
    "phs_in":       "#6497b1",
    "battery_in":   "#ffa500",
}

TEC_AGGREGATION = {
    "nuclear":      "Nuclear",
    "gas_ccgt1G":   "Gas",
    "gas_ccgt2G":   "Gas",
    "gas_ccgtSA":   "Gas",
    "gas_ocgtSA":   "Gas",
    "gas":          "Gas",
    "coal_1G":      "Coal",
    "coal_SA":      "Coal",
    "lignite":      "Coal",
    "coal":         "Coal",
    "oil_light":    "Oil",
    "oil":          "Oil",
    "onshore":      "Wind",
    "offshore":     "Wind",
    "wind":         "Wind",
    "pv":           "PV",
    "river":        "River",
    "lake_phs":     "Lake/PHS",
    "phs_in":       "Lake/PHS (charge)",
    "battery":      "Battery",
    "battery_in":   "Battery (charge)",
    "biomass":      "Biomass",
    "nmd":          "NMD",
    "net_imports":  "Net imports",
    "net_exo_imports": "Net imports (exo)",
}

AGG_COLORS = {
    "Nuclear":           "#ffd966",
    "Gas":               "#9d0208",
    "Coal":              "#6c757d",
    "Oil":               "#262626",
    "Wind":              "#005900",
    "PV":                "#ffff00",
    "River":             "#011f4b",
    "Lake/PHS":          "#6497b1",
    "Lake/PHS (charge)": "#b5d4e8",
    "Battery":           "#ffa500",
    "Battery (charge)":  "#ffd699",
    "Biomass":           "#556b2f",
    "NMD":               "#262626",
    "Net imports":       "#adb5bd",
    "Net imports (exo)": "#dee2e6",
}

# Order for stacked area (bottom to top) — positive generation
AGG_ORDER_POS = [
    "Nuclear", "Gas", "Coal", "Oil", "Wind", "PV",
    "River", "Lake/PHS", "Battery", "Biomass", "NMD",
    "Net imports", "Net imports (exo)",
]
# Negative (consumption/charge)
AGG_ORDER_NEG = ["Lake/PHS (charge)", "Battery (charge)"]

AGG_ORDER = AGG_ORDER_POS + AGG_ORDER_NEG

# ── Theme ──

_FONT = "'Nunito', -apple-system, BlinkMacSystemFont, sans-serif"
_FONT_MONO = "'DM Mono', monospace"

_CLEAN_LAYOUT = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family=_FONT, size=12, color="#3b2a0e"),
    title_font=dict(size=14, color="#3b2a0e", family=_FONT),
    xaxis=dict(gridcolor="rgba(180,120,20,0.06)", linecolor="rgba(180,120,20,0.15)",
               linewidth=1, zeroline=False),
    yaxis=dict(gridcolor="rgba(180,120,20,0.06)", linecolor="rgba(180,120,20,0.15)",
               linewidth=1, zeroline=False),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, bgcolor="rgba(0,0,0,0)",
                font=dict(size=11, color="#a08050")),
    margin=dict(l=55, r=20, t=55, b=40),
)


def _apply_theme(fig, extra_top_margin=0, keep_legend=False):
    """Apply clean minimal theme to a figure."""
    layout = dict(_CLEAN_LAYOUT)
    if extra_top_margin:
        layout["margin"] = dict(l=55, r=20, t=65 + extra_top_margin, b=40)
    if keep_legend:
        del layout["legend"]
    fig.update_layout(**layout)
    fig.update_xaxes(gridcolor="#e8e8e8", linecolor="#bbb", linewidth=1, zeroline=False)
    fig.update_yaxes(gridcolor="#e8e8e8", linecolor="#bbb", linewidth=1, zeroline=False)
    return fig


# ── Data loading helpers ──

def _posix_hours_to_dt(hours_series):
    return pd.to_datetime(hours_series * 3600, unit="s", origin="unix", utc=True)


def _load_hourly(run_dir, filename, col_names):
    path = run_dir / "inputs" / filename
    if not path.exists():
        return None
    df = pd.read_csv(path, header=None, names=col_names)
    if "hour" in df.columns:
        df["datetime"] = _posix_hours_to_dt(df["hour"])
    return df


def _load_metadata(run_dir):
    meta_path = run_dir / "run.yaml"
    if meta_path.exists():
        with open(meta_path) as f:
            return yaml.safe_load(f)
    return {}


def _country_color(area, idx=0):
    return COUNTRY_COLORS.get(area, f"hsl({(idx * 51) % 360}, 70%, 50%)")


def _tec_color(tec):
    return TEC_COLORS.get(tec, AGG_COLORS.get(tec, "#888888"))


# ── Input chart builders ──

def chart_demand(run_dir, areas):
    df = _load_hourly(run_dir, "demand.csv", ["area", "hour", "value"])
    if df is None:
        return None
    df = df[df["area"].isin(areas)]
    if df.empty:
        return None
    fig = go.Figure()
    for i, area in enumerate(sorted(df["area"].unique())):
        sub = df[df["area"] == area]
        fig.add_trace(go.Scatter(
            x=sub["datetime"], y=sub["value"], name=area, mode="lines",
            line=dict(color=_country_color(area, i), width=1),
        ))
    fig.update_layout(title="Hourly demand", yaxis_title="GW", height=380)
    return _apply_theme(fig)


def chart_vre_profiles(run_dir, areas):
    df = _load_hourly(run_dir, "vre_profiles.csv", ["area", "tec", "hour", "value"])
    if df is None:
        return None
    df = df[df["area"].isin(areas)]
    if df.empty:
        return None
    tecs = [t for t in ["onshore", "offshore", "pv", "river"] if t in df["tec"].unique()]
    labels = {"onshore": "Onshore wind", "offshore": "Offshore wind", "pv": "Solar PV", "river": "Run-of-river"}
    fig = make_subplots(rows=len(tecs), cols=1, shared_xaxes=True,
                        subplot_titles=[labels.get(t, t.capitalize()) for t in tecs],
                        vertical_spacing=0.06)
    for row, tec in enumerate(tecs, 1):
        sub_tec = df[df["tec"] == tec]
        for i, area in enumerate(sorted(sub_tec["area"].unique())):
            sub = sub_tec[sub_tec["area"] == area]
            fig.add_trace(go.Scatter(
                x=sub["datetime"], y=sub["value"], name=area, mode="lines",
                line=dict(color=_country_color(area, i), width=1),
                legendgroup=area, showlegend=(row == 1),
            ), row=row, col=1)
        fig.update_yaxes(title_text="CF", row=row, col=1)
    fig.update_layout(title="VRE capacity factors", height=220 * len(tecs),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
    return _apply_theme(fig, extra_top_margin=20)


def chart_nmd(run_dir, areas):
    df = _load_hourly(run_dir, "nmd.csv", ["area", "hour", "value"])
    if df is None:
        return None
    df = df[df["area"].isin(areas)]
    if df.empty:
        return None
    fig = go.Figure()
    for i, area in enumerate(sorted(df["area"].unique())):
        sub = df[df["area"] == area]
        fig.add_trace(go.Scatter(
            x=sub["datetime"], y=sub["value"], name=area, mode="lines",
            line=dict(color=_country_color(area, i), width=1),
        ))
    fig.update_layout(title="Non-market-dependent production", yaxis_title="GW", height=350)
    return _apply_theme(fig)


def chart_exo_prices(run_dir, areas):
    df = _load_hourly(run_dir, "exoPrices.csv", ["area", "hour", "value"])
    if df is None:
        return None
    df = df[df["area"].isin(areas)]
    if df.empty:
        return None
    fig = go.Figure()
    for i, area in enumerate(sorted(df["area"].unique())):
        sub = df[df["area"] == area]
        fig.add_trace(go.Scatter(
            x=sub["datetime"], y=sub["value"], name=area, mode="lines",
            line=dict(width=1),
        ))
    fig.update_layout(title="Exogenous day-ahead prices", yaxis_title="EUR/MWh", height=380)
    return _apply_theme(fig)


def chart_nuclear_availability(run_dir, areas):
    path = run_dir / "inputs" / "nucMaxAF.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, header=None, names=["area", "week", "value"])
    df = df[df["area"].isin(areas)]
    if df.empty:
        return None
    week_str = df["week"].astype(str).str.zfill(6)
    df["date"] = pd.to_datetime(week_str.str[:4] + "-W" + week_str.str[4:] + "-1",
                                 format="%Y-W%W-%w", errors="coerce")
    fig = go.Figure()
    for i, area in enumerate(sorted(df["area"].unique())):
        sub = df[df["area"] == area]
        fig.add_trace(go.Scatter(
            x=sub["date"], y=sub["value"], name=area, mode="lines+markers",
            line=dict(color=_country_color(area, i), width=2),
            marker=dict(size=4),
        ))
    fig.update_layout(
        title="Nuclear weekly max availability factor",
        yaxis_title="Availability factor", yaxis_range=[0, 1.05], height=350,
    )
    return _apply_theme(fig)


def chart_lake_inflows(run_dir, areas):
    path = run_dir / "inputs" / "lake_inflows.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, header=None, names=["area", "month", "value"])
    df = df[df["area"].isin(areas)]
    if df.empty:
        return None
    df["date"] = pd.to_datetime(df["month"], format="%Y%m", errors="coerce")
    fig = go.Figure()
    for i, area in enumerate(sorted(df["area"].unique())):
        sub = df[df["area"] == area]
        fig.add_trace(go.Bar(
            x=sub["date"], y=sub["value"], name=area,
            marker_color=_country_color(area, i),
        ))
    fig.update_layout(
        title="Monthly lake/PHS inflows", yaxis_title="TWh",
        barmode="group", height=350,
    )
    return _apply_theme(fig)


def chart_capacity_mix(run_dir, areas):
    path = run_dir / "inputs" / "capa.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, header=None, names=["area", "tec", "value"])
    df = df[(df["value"] > 0) & (df["area"].isin(areas))]
    if df.empty:
        return None

    df["group"] = df["tec"].map(TEC_AGGREGATION).fillna(df["tec"])
    agg = df.groupby(["area", "group"])["value"].sum().reset_index()

    area_list = [a for a in areas if a in agg["area"].values]
    n = len(area_list)
    if n == 0:
        return None

    pie_legend = dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.02,
                      bgcolor="rgba(0,0,0,0)", font=dict(size=12))

    if n == 1:
        sub = agg[agg["area"] == area_list[0]]
        order_map = {name: i for i, name in enumerate(AGG_ORDER)}
        sub = sub.copy()
        sub["order"] = sub["group"].map(order_map).fillna(99)
        sub = sub.sort_values("order")
        colors = [AGG_COLORS.get(g, "#888888") for g in sub["group"]]
        fig = go.Figure(go.Pie(
            labels=sub["group"], values=sub["value"],
            marker=dict(colors=colors),
            textinfo="percent",
            texttemplate="%{percent:.0%}",
            hovertemplate="%{label}: %{value:.2f} GW (%{percent})<extra></extra>",
        ))
        fig.update_layout(
            title=f"Installed capacity — {area_list[0]} (GW)",
            height=420, legend=pie_legend, margin=dict(l=20, r=160, t=60, b=20),
        )
    else:
        cols = min(n, 4)
        rows = (n + cols - 1) // cols
        fig = make_subplots(
            rows=rows, cols=cols,
            specs=[[{"type": "pie"}] * cols for _ in range(rows)],
            subplot_titles=area_list,
        )
        for idx, area in enumerate(area_list):
            r = idx // cols + 1
            c = idx % cols + 1
            sub = agg[agg["area"] == area].copy()
            order_map = {name: i for i, name in enumerate(AGG_ORDER)}
            sub["order"] = sub["group"].map(order_map).fillna(99)
            sub = sub.sort_values("order")
            colors = [AGG_COLORS.get(g, "#888888") for g in sub["group"]]
            fig.add_trace(go.Pie(
                labels=sub["group"], values=sub["value"],
                marker=dict(colors=colors),
                textinfo="percent",
                texttemplate="%{percent:.0%}",
                hovertemplate="%{label}: %{value:.2f} GW (%{percent})<extra></extra>",
                showlegend=(idx == 0),
            ), row=r, col=c)
        fig.update_layout(
            title="Installed capacity by area (GW)",
            height=350 * rows, legend=pie_legend, margin=dict(l=20, r=160, t=60, b=20),
        )
    return _apply_theme(fig, keep_legend=True)


def chart_interconnections(run_dir, areas):
    path = run_dir / "inputs" / "links.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, header=None, names=["exporter", "importer", "value"])
    df = df[df["exporter"].isin(areas) | df["importer"].isin(areas)]
    if df.empty:
        return None
    all_areas = sorted(set(df["exporter"].tolist() + df["importer"].tolist()))
    matrix = pd.DataFrame(0.0, index=all_areas, columns=all_areas)
    for _, row in df.iterrows():
        matrix.loc[row["exporter"], row["importer"]] = row["value"]

    fig = go.Figure(data=go.Heatmap(
        z=matrix.values, x=matrix.columns.tolist(), y=matrix.index.tolist(),
        colorscale="Blues", text=matrix.values.round(1),
        texttemplate="%{text}", hovertemplate="From %{y} to %{x}: %{z:.1f} GW",
    ))
    fig.update_layout(
        title="Interconnection capacity (GW) — exporter → importer",
        height=max(300, 55 * len(all_areas)), width=max(400, 55 * len(all_areas)),
    )
    return _apply_theme(fig)


# ── Output chart builders ──

def chart_prices(run_dir, areas):
    """Hourly price time series."""
    path = run_dir / "outputs" / "prices.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "hour" not in df.columns:
        return None
    df["datetime"] = _posix_hours_to_dt(df["hour"])
    cols = [c for c in areas if c in df.columns]
    if not cols:
        return None
    fig = go.Figure()
    for i, area in enumerate(cols):
        fig.add_trace(go.Scatter(
            x=df["datetime"], y=df[area], name=area, mode="lines",
            line=dict(color=_country_color(area, i), width=1),
        ))
    # y-axis starts at 0 unless there are negative prices
    all_vals = pd.concat([df[c] for c in cols])
    y_min = all_vals.min()
    fig.update_layout(title="Simulated spot price", yaxis_title="EUR/MWh", height=380,
                      yaxis_rangemode="tozero" if y_min >= 0 else "normal")
    return _apply_theme(fig)






def html_price_overview(run_dir, areas):
    """Side-by-side: HTML stats table (left) + Plotly price duration curve (right).

    Returns raw HTML string with embedded Plotly chart.
    """
    path = run_dir / "outputs" / "prices.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    cols = [c for c in areas if c in df.columns]
    if not cols:
        return None

    # ── Stats table ──
    stat_fns = [
        ("Mean",   lambda v: f"{v.mean():.1f}"),
        ("Median", lambda v: f"{v.median():.1f}"),
        ("Std",    lambda v: f"{v.std():.1f}"),
        ("Min",    lambda v: f"{v.min():.1f}"),
        ("Max",    lambda v: f"{v.max():.1f}"),
        ("P5",     lambda v: f"{v.quantile(0.05):.1f}"),
        ("P95",    lambda v: f"{v.quantile(0.95):.1f}"),
    ]

    header = "".join(f"<th>{a}</th>" for a in cols)
    rows_html = ""
    for stat_name, fn in stat_fns:
        cells = "".join(f"<td>{fn(df[a].dropna())}</td>" for a in cols)
        rows_html += f"<tr><td class='row-label'>{stat_name}</td>{cells}</tr>\n"

    table_html = f"""<table class="stats-table">
<thead><tr><th></th>{header}</tr></thead>
<tbody>{rows_html}</tbody>
</table>"""

    # ── Duration curve ──
    fig = go.Figure()
    pct = np.linspace(0.005, 0.995, 200)
    y_min = 0
    for i, area in enumerate(cols):
        values = df[area].dropna().values
        quantiles = np.quantile(values, 1 - pct)
        y_min = min(y_min, quantiles.min())
        fig.add_trace(go.Scatter(
            x=pct * 100, y=quantiles, name=area, mode="lines",
            line=dict(color=_country_color(area, i), width=2),
        ))
    fig.update_layout(
        title="Price duration curve",
        xaxis_title="Duration (%)", yaxis_title="EUR/MWh",
        height=400, margin=dict(l=50, r=15, t=50, b=40),
        yaxis_rangemode="tozero" if y_min >= 0 else "normal",
    )
    _apply_theme(fig)
    chart_html = fig.to_html(full_html=False, include_plotlyjs=False)

    return f"""<div class="price-overview">
  <div class="price-overview-left">
    <h3 class="section-title">Spot price statistics (EUR/MWh)</h3>
    {table_html}
  </div>
  <div class="price-overview-right">
    {chart_html}
  </div>
</div>"""


def chart_production(run_dir, areas):
    """Stacked area production mix with aggregated technologies."""
    path = run_dir / "outputs" / "production.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "hour" not in df.columns or "area" not in df.columns:
        return None
    df = df[df["area"].isin(areas)]
    if df.empty:
        return None

    df["datetime"] = _posix_hours_to_dt(df["hour"])
    tec_cols = [c for c in df.columns if c not in ("hour", "area", "datetime", "demand")]

    # Aggregate technologies
    agg_data = pd.DataFrame()
    agg_data["datetime"] = df["datetime"]
    agg_data["area"] = df["area"]
    if "demand" in df.columns:
        agg_data["demand"] = df["demand"]

    agg_groups = {}
    for tec in tec_cols:
        group = TEC_AGGREGATION.get(tec, tec)
        if group not in agg_groups:
            agg_groups[group] = df[tec].values.copy()
        else:
            agg_groups[group] = agg_groups[group] + df[tec].values
    for group, vals in agg_groups.items():
        agg_data[group] = vals

    area_list = [a for a in areas if a in df["area"].unique()]
    n = len(area_list)
    if n == 0:
        return None

    fig = make_subplots(rows=n, cols=1, shared_xaxes=True,
                        subplot_titles=area_list if n > 1 else None,
                        vertical_spacing=0.03)

    # Determine which groups are present
    present_pos = [g for g in AGG_ORDER_POS if g in agg_groups]
    present_neg = [g for g in AGG_ORDER_NEG if g in agg_groups]

    for row, area in enumerate(area_list, 1):
        sub = agg_data[agg_data["area"] == area]
        # Positive stack (generation)
        for group in present_pos:
            if sub[group].abs().sum() < 0.001:
                continue
            fig.add_trace(go.Scatter(
                x=sub["datetime"], y=sub[group], name=group,
                mode="lines", stackgroup="pos",
                line=dict(width=0),
                fillcolor=AGG_COLORS.get(group, "#888"),
                legendgroup=group, showlegend=(row == 1),
                hovertemplate=f"{group}: %{{y:.2f}} GW<extra></extra>",
            ), row=row, col=1)
        # Negative stack (charge)
        for group in present_neg:
            if sub[group].abs().sum() < 0.001:
                continue
            fig.add_trace(go.Scatter(
                x=sub["datetime"], y=sub[group], name=group,
                mode="lines", stackgroup="neg",
                line=dict(width=0),
                fillcolor=AGG_COLORS.get(group, "#888"),
                legendgroup=group, showlegend=(row == 1),
                hovertemplate=f"{group}: %{{y:.2f}} GW<extra></extra>",
            ), row=row, col=1)
        # Demand line
        if "demand" in sub.columns:
            fig.add_trace(go.Scatter(
                x=sub["datetime"], y=sub["demand"], name="Demand",
                mode="lines", line=dict(color="#e00000", width=1.5),
                legendgroup="demand", showlegend=(row == 1),
                hovertemplate="Demand: %{y:.2f} GW<extra></extra>",
            ), row=row, col=1)
        fig.update_yaxes(title_text="GW", row=row, col=1)
    fig.update_layout(title="Production mix", height=500 * n,
                      legend=dict(orientation="h", yanchor="bottom", y=1.01,
                                  font=dict(size=11)))
    return _apply_theme(fig, extra_top_margin=20)


# ── Chart lists ──

_INPUT_CHARTS_FR = [
    ("demand", chart_demand),
    ("vre_profiles", chart_vre_profiles),
    ("nmd", chart_nmd),
    ("nuclear", chart_nuclear_availability),
    ("lake_inflows", chart_lake_inflows),
    ("capacity", chart_capacity_mix),
]

_INPUT_CHARTS_OTHER = [
    ("demand", chart_demand),
    ("vre_profiles", chart_vre_profiles),
    ("nmd", chart_nmd),
    ("exo_prices", chart_exo_prices),
    ("nuclear", chart_nuclear_availability),
    ("lake_inflows", chart_lake_inflows),
    ("capacity", chart_capacity_mix),
    ("interconnections", chart_interconnections),
]

_OUTPUT_CHARTS = [
    ("price_overview", html_price_overview),
    ("prices", chart_prices),
    ("production", chart_production),
]


def _render_charts(run_dir, chart_list, areas):
    """Render a list of charts to HTML divs.

    Chart functions can return a Plotly figure or a raw HTML string.
    Plotly CDN is loaded once in the page <head>.
    """
    parts = []
    for name, chart_fn in chart_list:
        result = chart_fn(run_dir, areas)
        if result is None:
            continue
        if isinstance(result, str):
            parts.append(f'<div class="chart">{result}</div>')
        else:
            parts.append(f'<div class="chart">{result.to_html(full_html=False, include_plotlyjs=False)}</div>')
    return parts


def generate_report(run_dir, open_browser=True):
    """Generate an interactive HTML report for a run.

    Tabs: France Inputs | France Outputs | Neighbors Inputs | Neighbors Outputs
    """
    run_dir = Path(run_dir)
    meta = _load_metadata(run_dir)
    all_areas = meta.get("areas", ["FR"])
    focus = "FR"
    other_areas = [a for a in all_areas if a != focus]

    has_outputs = (run_dir / "outputs").exists() and any((run_dir / "outputs").glob("*.csv"))

    # Build all chart HTML
    fr_input_parts = _render_charts(run_dir, _INPUT_CHARTS_FR, [focus])
    fr_output_parts = []
    if has_outputs:
        fr_output_parts = _render_charts(run_dir, _OUTPUT_CHARTS, [focus])

    other_input_parts = _render_charts(run_dir, _INPUT_CHARTS_OTHER, other_areas)
    other_output_parts = []
    if has_outputs:
        other_output_parts = _render_charts(run_dir, _OUTPUT_CHARTS, other_areas)

    fr_input_html = "\n".join(fr_input_parts) if fr_input_parts else '<div class="no-data">No input data found.</div>'
    fr_output_html = "\n".join(fr_output_parts) if fr_output_parts else '<div class="no-data">No output data — run not solved yet.</div>'
    other_input_html = "\n".join(other_input_parts) if other_input_parts else '<div class="no-data">No input data found.</div>'
    other_output_html = "\n".join(other_output_parts) if other_output_parts else '<div class="no-data">No output data — run not solved yet.</div>'

    _MONTH_NAMES = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                    7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    months_raw = meta.get("months")
    if months_raw:
        ms = str(months_raw)
        if "-" in ms:
            a, b = ms.split("-", 1)
            months_label = f" · {_MONTH_NAMES.get(int(a),a)}–{_MONTH_NAMES.get(int(b),b)}"
        else:
            months_label = f" · {_MONTH_NAMES.get(int(ms), ms)} only"
    else:
        months_label = ""
    exec_time = meta.get("exec_time", "")
    status = meta.get("status", "?")
    solver_name = meta.get("solver", "")
    if solver_name.lower() == "highs":
        solver_name = "HiGHS"
    else:
        solver_name = solver_name.upper()
    solver_info = solver_name if solver_name else ""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<title>EOLES-Dispatch — {meta.get('name', run_dir.name)}</title>
<style>
  :root {{
    --bg:       #fef6e4;
    --surface:  #fffdf7;
    --border:   rgba(214, 158, 46, 0.20);
    --shadow:   rgba(180, 120, 20, 0.10);
    --text:     #3b2a0e;
    --muted:    #a08050;
    --accent:   #2E86C1;
    --accent-bg: rgba(46, 134, 193, 0.08);
    --accent-border: rgba(46, 134, 193, 0.25);
    --radius:   16px;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'Nunito', -apple-system, BlinkMacSystemFont, sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    -webkit-font-smoothing: antialiased;
    background-image:
      radial-gradient(ellipse 80% 50% at 50% 0%, rgba(245,166,35,0.08) 0%, transparent 70%);
  }}

  /* ── Header ── */
  .header {{
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 22px 32px 18px;
    box-shadow: 0 1px 8px var(--shadow);
  }}
  .header-top {{
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 14px;
  }}
  .header-top h1 {{
    font-size: 1.35rem;
    font-weight: 800;
    color: var(--text);
    letter-spacing: -0.02em;
  }}
  .header-top .subtitle {{
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: var(--muted);
    letter-spacing: 0.06em;
    text-transform: uppercase;
  }}
  .badge {{
    display: inline-flex; align-items: center; gap: 5px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.68rem;
    font-family: 'DM Mono', monospace;
    font-weight: 500;
    letter-spacing: 0.04em;
    text-transform: uppercase;
  }}
  .badge-solved {{
    background: rgba(52, 199, 89, 0.12);
    color: #1b7a3d;
    border: 1px solid rgba(52, 199, 89, 0.25);
  }}
  .badge-created {{
    background: rgba(245, 166, 35, 0.12);
    color: #92400e;
    border: 1px solid rgba(245, 166, 35, 0.3);
  }}
  .meta-pills {{
    display: flex; flex-wrap: wrap; gap: 8px;
  }}
  .meta-pill {{
    display: inline-flex; align-items: center; gap: 6px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.75rem;
    color: var(--muted);
    box-shadow: 0 1px 3px var(--shadow);
  }}
  .meta-pill b {{
    color: var(--text);
    font-weight: 700;
  }}

  /* ── Tabs ── */
  .tabs {{
    display: flex; gap: 0;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 0 32px;
    position: sticky; top: 0; z-index: 10;
    box-shadow: 0 1px 4px var(--shadow);
  }}
  .tab {{
    padding: 11px 22px;
    cursor: pointer;
    font-size: 0.82rem;
    font-weight: 700;
    color: var(--muted);
    border-bottom: 3px solid transparent;
    transition: all 0.15s ease;
    user-select: none;
    white-space: nowrap;
  }}
  .tab:hover {{
    color: var(--text);
    background: var(--accent-bg);
  }}
  .tab.active {{
    color: var(--accent);
    border-bottom-color: var(--accent);
  }}

  /* ── Content area ── */
  .tab-content {{
    display: none;
    padding: 20px 32px 40px;
    max-width: 1400px;
    margin: 0 auto;
  }}
  .tab-content.active {{
    display: block;
  }}

  /* ── Chart cards ── */
  .chart {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    box-shadow: 0 2px 0 rgba(255,255,255,0.7) inset, 0 3px 14px var(--shadow);
    padding: 18px 22px;
    margin: 18px 0;
    transition: box-shadow 0.2s ease, transform 0.2s ease;
  }}
  .chart:hover {{
    box-shadow: 0 2px 0 rgba(255,255,255,0.7) inset, 0 6px 24px rgba(180,120,20,0.14);
    transform: translateY(-1px);
  }}

  /* ── Price overview side-by-side ── */
  .price-overview {{
    display: flex;
    align-items: stretch;
    gap: 18px;
    margin: 18px 0;
  }}
  .price-overview-left {{
    flex: 0 0 33.333%;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    box-shadow: 0 2px 0 rgba(255,255,255,0.7) inset, 0 3px 14px var(--shadow);
    padding: 22px 24px;
    display: flex;
    flex-direction: column;
    justify-content: center;
  }}
  .price-overview-right {{
    flex: 1 1 66.667%;
    min-width: 0;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    box-shadow: 0 2px 0 rgba(255,255,255,0.7) inset, 0 3px 14px var(--shadow);
    padding: 14px 18px;
  }}
  .section-title {{
    font-size: 0.7rem;
    font-family: 'DM Mono', monospace;
    font-weight: 500;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 0 0 14px 0;
  }}

  /* ── Stats table ── */
  .stats-table {{
    border-collapse: collapse;
    font-size: 0.82rem;
    width: 100%;
  }}
  .stats-table th, .stats-table td {{
    padding: 7px 14px;
    text-align: center;
    border-bottom: 1px solid rgba(180,120,20,0.06);
  }}
  .stats-table thead th {{
    background: rgba(180,120,20,0.04);
    font-family: 'DM Mono', monospace;
    font-weight: 500;
    font-size: 0.7rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    border-bottom: 1px solid var(--border);
  }}
  .stats-table .row-label {{
    text-align: left;
    font-weight: 700;
    color: var(--muted);
    font-size: 0.78rem;
  }}
  .stats-table td {{
    font-variant-numeric: tabular-nums;
    font-weight: 600;
    color: var(--text);
  }}
  .stats-table tbody tr:hover {{
    background: rgba(180,120,20,0.04);
  }}

  /* ── No data ── */
  .no-data {{
    text-align: center;
    padding: 56px 20px;
    color: var(--muted);
    font-size: 0.9rem;
    font-weight: 600;
  }}

  /* ── Footer ── */
  .footer {{
    text-align: center;
    padding: 20px;
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    color: var(--muted);
    letter-spacing: 0.06em;
    text-transform: uppercase;
  }}
</style>
</head>
<body>
<div class="header">
  <div class="header-top">
    <div>
      <h1>EOLES-Dispatch</h1>
      <div class="subtitle">{meta.get('name', run_dir.name)}</div>
    </div>
    <span class="badge badge-{status}">{status}{(' · ' + exec_time) if exec_time else ''}</span>
  </div>
  <div class="meta-pills">
    <div class="meta-pill">Scenario <b>{meta.get('scenario', '?')}</b></div>
    <div class="meta-pill">Year <b>{meta.get('year', '?')}</b>{months_label}</div>
    <div class="meta-pill">Areas <b>{', '.join(all_areas)}</b></div>
    <div class="meta-pill">Solver: <b>{solver_info or 'N/A'}</b></div>
    <div class="meta-pill">Created: <b>{meta.get('created', '?')[:16].replace('T', ' ')}</b></div>
    {"<div class='meta-pill'>Solved: <b>" + meta.get('solved', '')[:16].replace('T', ' ') + "</b></div>" if meta.get('solved') else ""}
  </div>
</div>
<div class="tabs">
  <div class="tab active" onclick="switchTab('fr-in')">France — Inputs</div>
  <div class="tab" onclick="switchTab('fr-out')">France — Outputs</div>
  <div class="tab" onclick="switchTab('other-in')">Neighbors — Inputs</div>
  <div class="tab" onclick="switchTab('other-out')">Neighbors — Outputs</div>
</div>
<div id="tab-fr-in" class="tab-content active">
{fr_input_html}
</div>
<div id="tab-fr-out" class="tab-content">
{fr_output_html}
</div>
<div id="tab-other-in" class="tab-content">
{other_input_html}
</div>
<div id="tab-other-out" class="tab-content">
{other_output_html}
</div>
<div class="footer">EOLES-Dispatch &middot; {meta.get('name', run_dir.name)} &middot; {meta.get('created', '')[:10]}</div>
<script>
function switchTab(id) {{
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));
  document.getElementById('tab-' + id).classList.add('active');
  const tabs = document.querySelectorAll('.tab');
  const map = {{'fr-in': 'France — Inputs', 'fr-out': 'France — Outputs',
                'other-in': 'Other countries — Inputs', 'other-out': 'Other countries — Outputs'}};
  tabs.forEach(t => {{ if (t.textContent === map[id]) t.classList.add('active'); }});
  window.dispatchEvent(new Event('resize'));
}}
</script>
</body>
</html>"""

    out_path = run_dir / "viz.html"
    out_path.write_text(html, encoding="utf-8")

    if open_browser:
        webbrowser.open(f"file://{out_path.resolve()}")

    return out_path
