"""Input chart builders (pre-solve data from inputs/ folder)."""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .theme import TEC_AGGREGATION, AGG_COLORS, AGG_ORDER, _apply_theme
from .loaders import _load_hourly, _country_color


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
    tecs = [t for t in ["onshore", "offshore", "solar", "river"] if t in df["tec"].unique()]
    labels = {"onshore": "Onshore wind", "offshore": "Offshore wind", "solar": "Solar solar", "river": "Run-of-river"}
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
