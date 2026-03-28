"""Output chart builders (post-solve data from outputs/ folder)."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..utils import hour_to_cet_month
from .loaders import (
    _country_color,
    _load_actual_prices,
    _load_actual_production,
    _posix_hours_to_dt,
)
from .theme import (
    AGG_COLORS,
    AGG_NEGATIVE,
    AGG_ORDER,
    LEGEND_BELOW,
    TEC_AGGREGATION,
    _apply_theme,
)


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
        fig.add_trace(
            go.Scatter(
                x=df["datetime"],
                y=df[area],
                name=area,
                mode="lines",
                line=dict(color=_country_color(area, i), width=1),
            )
        )
    # y-axis starts at 0 unless there are negative prices
    all_vals = pd.concat([df[c] for c in cols])
    y_min = all_vals.min()
    fig.update_layout(
        yaxis_title="EUR/MWh",
        height=380,
        yaxis_rangemode="tozero" if y_min >= 0 else "normal",
    )
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
        ("Mean", lambda v: f"{v.mean():.1f}"),
        ("Median", lambda v: f"{v.median():.1f}"),
        ("Std", lambda v: f"{v.std():.1f}"),
        ("Min", lambda v: f"{v.min():.1f}"),
        ("Max", lambda v: f"{v.max():.1f}"),
        ("P5", lambda v: f"{v.quantile(0.05):.1f}"),
        ("P95", lambda v: f"{v.quantile(0.95):.1f}"),
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
        fig.add_trace(
            go.Scatter(
                x=pct * 100,
                y=quantiles,
                name=area,
                mode="lines",
                line=dict(color=_country_color(area, i), width=2),
            )
        )
    fig.update_layout(
        xaxis_title="Duration (%)",
        yaxis_title="EUR/MWh",
        height=50,
        yaxis_rangemode="tozero" if y_min >= 0 else "normal",
    )
    _apply_theme(fig)
    fig.update_layout(legend=LEGEND_BELOW, margin=dict(l=45, r=15, t=15, b=40))
    chart_html = fig.to_html(full_html=False, include_plotlyjs=False)

    return f"""<div class="price-overview">
  <div class="price-overview-left">
    <h3 class="section-title">Spot price statistics (EUR/MWh)</h3>
    <div class="price-overview-table">{table_html}</div>
  </div>
  <div class="price-overview-right">
    <h3 class="section-title">Price Duration Curve</h3>
    <div class="price-chart-wrapper">{chart_html}</div>
  </div>
</div>"""


# ── Validation chart builders (--validate mode) ──


def chart_prices_validate(run_dir, areas):
    """Hourly price time series with historical overlay."""
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

    actual_df = _load_actual_prices(run_dir)
    has_actual = actual_df is not None

    fig = go.Figure()
    for i, area in enumerate(cols):
        color = _country_color(area, i)
        label = f"{area} (simulated)" if has_actual and area in actual_df.columns else area
        fig.add_trace(
            go.Scatter(
                x=df["datetime"],
                y=df[area],
                name=label,
                mode="lines",
                line=dict(color=color, width=1),
            )
        )

    # Overlay historical prices (dashed, reduced opacity)
    if has_actual:
        for i, area in enumerate(cols):
            if area not in actual_df.columns:
                continue
            color = _country_color(area, i)
            fig.add_trace(
                go.Scatter(
                    x=actual_df["datetime"],
                    y=actual_df[area],
                    name=f"{area} (actual)",
                    mode="lines",
                    line=dict(color=color, width=1, dash="dot"),
                    opacity=0.5,
                )
            )

    # y-axis starts at 0 unless there are negative prices
    all_vals = pd.concat([df[c] for c in cols])
    if has_actual:
        act_cols = [c for c in cols if c in actual_df.columns]
        if act_cols:
            all_vals = pd.concat([all_vals] + [actual_df[c] for c in act_cols])
    y_min = all_vals.min()
    fig.update_layout(
        yaxis_title="EUR/MWh",
        height=380,
        yaxis_rangemode="tozero" if y_min >= 0 else "normal",
    )
    return _apply_theme(fig)


def html_price_overview_validate(run_dir, areas):
    """Side-by-side: HTML stats table with Sim/Act columns + error metrics + duration curve.

    Returns raw HTML string with embedded Plotly chart.
    """
    path = run_dir / "outputs" / "prices.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    cols = [c for c in areas if c in df.columns]
    if not cols:
        return None

    actual_df = _load_actual_prices(run_dir)
    actual_cols = [c for c in cols if actual_df is not None and c in actual_df.columns]
    has_actual = len(actual_cols) > 0

    # ── Stats table ──
    stat_fns = [
        ("Mean", lambda v: f"{v.mean():.1f}"),
        ("Median", lambda v: f"{v.median():.1f}"),
        ("Std", lambda v: f"{v.std():.1f}"),
        ("Min", lambda v: f"{v.min():.1f}"),
        ("Max", lambda v: f"{v.max():.1f}"),
        ("P5", lambda v: f"{v.quantile(0.05):.1f}"),
        ("P95", lambda v: f"{v.quantile(0.95):.1f}"),
    ]

    if has_actual:
        # Two sub-columns per area: Sim | Act
        header = ""
        for a in cols:
            if a in actual_cols:
                header += f'<th colspan="2">{a}</th>'
            else:
                header += f"<th>{a}</th>"
        sub_header = ""
        for a in cols:
            if a in actual_cols:
                sub_header += "<th>Sim</th><th>Act</th>"
            else:
                sub_header += "<th>Sim</th>"

        rows_html = ""
        for stat_name, fn in stat_fns:
            cells = ""
            for a in cols:
                cells += f"<td>{fn(df[a].dropna())}</td>"
                if a in actual_cols:
                    cells += f"<td>{fn(actual_df[a].dropna())}</td>"
            rows_html += f"<tr><td class='row-label'>{stat_name}</td>{cells}</tr>\n"

        # Error metrics section
        merged = df[["hour"] + actual_cols].merge(
            actual_df[["hour"] + actual_cols], on="hour", suffixes=("_sim", "_act")
        )
        err_metrics = []
        for a in actual_cols:
            sim_v = merged[f"{a}_sim"]
            act_v = merged[f"{a}_act"]
            valid = sim_v.notna() & act_v.notna()
            sim_v, act_v = sim_v[valid], act_v[valid]
            if len(sim_v) == 0:
                err_metrics.append(None)
                continue
            error = sim_v.values - act_v.values
            bias = error.mean()
            rmse = np.sqrt((error**2).mean())
            mae = np.abs(error).mean()
            ss_res = (error**2).sum()
            ss_tot = ((act_v.values - act_v.values.mean()) ** 2).sum()
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            corr = np.corrcoef(sim_v.values, act_v.values)[0, 1]
            err_metrics.append({"bias": bias, "rmse": rmse, "mae": mae, "r2": r2, "corr": corr})

        err_defs = [
            ("Bias", "bias", ".1f"),
            ("RMSE", "rmse", ".1f"),
            ("MAE", "mae", ".1f"),
            ("R\u00b2", "r2", ".3f"),
            ("Corr", "corr", ".3f"),
        ]
        err_rows = ""
        for label, key, fmt in err_defs:
            cells = ""
            for a in cols:
                if a in actual_cols:
                    ai = actual_cols.index(a)
                    m = err_metrics[ai]
                    if m is not None:
                        cells += f'<td colspan="2">{m[key]:{fmt}}</td>'
                    else:
                        cells += '<td colspan="2">\u2014</td>'
                else:
                    cells += "<td>\u2014</td>"
            err_rows += f"<tr><td class='row-label'>{label}</td>{cells}</tr>\n"

        table_html = f"""<table class="stats-table">
<thead>
<tr><th></th>{header}</tr>
<tr><th></th>{sub_header}</tr>
</thead>
<tbody>{rows_html}
<tr><td colspan="100" style="border-top:2px solid var(--border);padding:4px 0 0 0;font-size:0.65rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.05em">Error metrics (sim \u2212 act)</td></tr>
{err_rows}</tbody>
</table>"""
    else:
        # No actual data — fallback to simple table
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
        color = _country_color(area, i)
        values = df[area].dropna().values
        quantiles = np.quantile(values, 1 - pct)
        y_min = min(y_min, quantiles.min())
        label = f"{area} (sim)" if has_actual and area in actual_cols else area
        fig.add_trace(
            go.Scatter(
                x=pct * 100,
                y=quantiles,
                name=label,
                mode="lines",
                line=dict(color=color, width=2),
            )
        )

    # Overlay actual duration curves
    if has_actual:
        for i, area in enumerate(cols):
            if area not in actual_cols:
                continue
            color = _country_color(area, i)
            values = actual_df[area].dropna().values
            quantiles = np.quantile(values, 1 - pct)
            y_min = min(y_min, quantiles.min())
            fig.add_trace(
                go.Scatter(
                    x=pct * 100,
                    y=quantiles,
                    name=f"{area} (actual)",
                    mode="lines",
                    line=dict(color=color, width=2, dash="dot"),
                    opacity=0.5,
                )
            )

    fig.update_layout(
        xaxis_title="Duration (%)",
        yaxis_title="EUR/MWh",
        height=50,
        yaxis_rangemode="tozero" if y_min >= 0 else "normal",
    )
    _apply_theme(fig)
    fig.update_layout(legend=LEGEND_BELOW, margin=dict(l=45, r=15, t=15, b=40))
    chart_html = fig.to_html(full_html=False, include_plotlyjs=False)

    title_label = (
        "Spot price \u2014 simulated vs actual (EUR/MWh)"
        if has_actual
        else "Spot price statistics (EUR/MWh)"
    )

    return f"""<div class="price-overview">
  <div class="price-overview-left">
    <h3 class="section-title">{title_label}</h3>
    <div class="price-overview-table">{table_html}</div>
  </div>
  <div class="price-overview-right">
    <h3 class="section-title">Price Duration Curve</h3>
    <div class="price-chart-wrapper">{chart_html}</div>
  </div>
</div>"""


def chart_price_scatter(run_dir, areas):
    """Scatter plot of simulated vs actual prices (when historical data is available)."""
    path = run_dir / "outputs" / "prices.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    actual_df = _load_actual_prices(run_dir)
    if actual_df is None:
        return None

    cols = [c for c in areas if c in df.columns and c in actual_df.columns]
    if not cols:
        return None

    merged = df[["hour"] + cols].merge(
        actual_df[["hour"] + cols], on="hour", suffixes=("_sim", "_act")
    )
    if merged.empty:
        return None

    fig = go.Figure()
    for i, area in enumerate(cols):
        sim_v = merged[f"{area}_sim"]
        act_v = merged[f"{area}_act"]
        valid = sim_v.notna() & act_v.notna()
        fig.add_trace(
            go.Scattergl(
                x=act_v[valid],
                y=sim_v[valid],
                name=area,
                mode="markers",
                marker=dict(color=_country_color(area, i), size=3, opacity=0.3),
            )
        )

    # 45° reference line
    all_sim = pd.concat([merged[f"{a}_sim"] for a in cols]).dropna()
    all_act = pd.concat([merged[f"{a}_act"] for a in cols]).dropna()
    lo = min(all_sim.min(), all_act.min(), 0)
    hi = max(all_sim.max(), all_act.max())
    fig.add_trace(
        go.Scatter(
            x=[lo, hi],
            y=[lo, hi],
            mode="lines",
            name="y = x",
            line=dict(color="#e00000", width=1, dash="dash"),
            showlegend=False,
        )
    )

    fig.update_layout(
        xaxis_title="Actual price (EUR/MWh)",
        yaxis_title="Simulated price (EUR/MWh)",
        height=450,
    )
    return _apply_theme(fig)


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

    fig = make_subplots(
        rows=n,
        cols=1,
        shared_xaxes=True,
        subplot_titles=area_list if n > 1 else None,
        vertical_spacing=0.02,
    )

    # Negative groups added in reverse AGG_ORDER so the first trace added is closest
    # to 0 (Battery charge), giving: net_exports at bottom, battery_in at top.
    neg_present = [g for g in reversed(AGG_ORDER) if g in agg_groups and g in AGG_NEGATIVE]
    pos_present = [g for g in AGG_ORDER if g in agg_groups and g not in AGG_NEGATIVE]

    for row, area in enumerate(area_list, 1):
        sub = agg_data[agg_data["area"] == area]
        for group in neg_present:
            if sub[group].abs().sum() < 0.001:
                continue
            fig.add_trace(
                go.Scatter(
                    x=sub["datetime"],
                    y=sub[group],
                    name=group,
                    mode="lines",
                    stackgroup="neg",
                    line=dict(width=0),
                    fillcolor=AGG_COLORS.get(group, "#888"),
                    legendgroup=group,
                    showlegend=(row == 1),
                    hovertemplate=f"{group}: %{{y:.2f}} GW<extra></extra>",
                ),
                row=row,
                col=1,
            )
        for group in pos_present:
            if sub[group].abs().sum() < 0.001:
                continue
            fig.add_trace(
                go.Scatter(
                    x=sub["datetime"],
                    y=sub[group],
                    name=group,
                    mode="lines",
                    stackgroup="pos",
                    line=dict(width=0),
                    fillcolor=AGG_COLORS.get(group, "#888"),
                    legendgroup=group,
                    showlegend=(row == 1),
                    hovertemplate=f"{group}: %{{y:.2f}} GW<extra></extra>",
                ),
                row=row,
                col=1,
            )
        # Demand line
        if "demand" in sub.columns:
            fig.add_trace(
                go.Scatter(
                    x=sub["datetime"],
                    y=sub["demand"],
                    name="Demand",
                    mode="lines",
                    line=dict(color="#e00000", width=1.5),
                    legendgroup="demand",
                    showlegend=(row == 1),
                    hovertemplate="Demand: %{y:.2f} GW<extra></extra>",
                ),
                row=row,
                col=1,
            )
        fig.update_yaxes(title_text="GW", row=row, col=1)
    fig.update_layout(height=500 * n)
    return _apply_theme(fig, extra_top_margin=20)


# ── Energy mix chart builders ──
# FIXME : Fix the display of negative values, consider migrating away from plotly
# TODO : Add table format of the energy mix

def chart_energy_mix_annual(run_dir, areas):
    """Annual energy mix: stacked bar chart with one bar per area, stacked by technology."""
    path = run_dir / "outputs" / "production.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "hour" not in df.columns or "area" not in df.columns:
        return None

    area_list, agg_data = _build_energy_agg(df, areas)
    if not area_list:
        return None

    tec_display_cols = [g for g in AGG_ORDER if g in agg_data.columns]
    annual = agg_data.groupby("area")[tec_display_cols].sum() / 1000  # TWh

    fig = go.Figure()
    _add_stacked_bars(fig, area_list, annual)
    fig.update_layout(yaxis_title="TWh", barmode="stack", height=600)
    _apply_theme(fig)
    fig.update_layout(margin_t=20)
    return fig


def chart_energy_mix_monthly(run_dir, areas):
    """Monthly energy mix: stacked bar chart by month, one subplot per area."""
    path = run_dir / "outputs" / "production.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "hour" not in df.columns or "area" not in df.columns:
        return None

    area_list, agg_data = _build_energy_agg(df, areas)
    if not area_list:
        return None

    agg_data["month"] = hour_to_cet_month(agg_data["hour"])  # "YYYYMM" in CET
    tec_display_cols = [g for g in AGG_ORDER if g in agg_data.columns]
    n = len(area_list)

    fig = make_subplots(
        rows=n,
        cols=1,
        shared_xaxes=True,
        subplot_titles=area_list if n > 1 else None,
        vertical_spacing=0.02,
    )
    for row, area in enumerate(area_list, 1):
        sub = agg_data[agg_data["area"] == area]
        monthly = sub.groupby("month")[tec_display_cols].sum() / 1000  # TWh
        months_present = sorted(monthly.index.tolist())
        _add_monthly_stacked_bars(
            fig, monthly, months_present, row=row, col=1, show_legend=(row == 1)
        )
        fig.update_yaxes(title_text="TWh", row=row, col=1)

    fig.update_layout(barmode="stack", height=600 * n)
    _apply_theme(fig)
    fig.update_layout(margin_t=20)
    return fig


def chart_energy_mix_annual_validate(run_dir, areas):
    """Annual energy mix: areas interleaved (sim / act) per area (--validate mode)."""
    path = run_dir / "outputs" / "production.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "hour" not in df.columns or "area" not in df.columns:
        return None

    area_list, agg_data = _build_energy_agg(df, areas)
    if not area_list:
        return None

    tec_display_cols = [g for g in AGG_ORDER if g in agg_data.columns]
    annual_sim = agg_data.groupby("area")[tec_display_cols].sum() / 1000  # TWh

    actual_df = _load_actual_production(run_dir)
    if actual_df is None:
        return chart_energy_mix_annual(run_dir, areas)

    _, agg_actual = _build_energy_agg(actual_df, area_list)
    if agg_actual is None:
        return chart_energy_mix_annual(run_dir, areas)

    tec_display_cols_act = [g for g in AGG_ORDER if g in agg_actual.columns]
    annual_act = agg_actual.groupby("area")[tec_display_cols_act].sum() / 1000  # TWh

    # Interleaved x: [area1 (sim), area1 (act), area2 (sim), area2 (act), ...]
    x_labels = [lbl for a in area_list for lbl in (f"{a} (sim)", f"{a} (act)")]

    fig = go.Figure()
    neg_order = [g for g in reversed(AGG_ORDER) if g in AGG_NEGATIVE]
    pos_order = [g for g in AGG_ORDER if g not in AGG_NEGATIVE]
    for group in neg_order + pos_order:
        y_vals = []
        for area in area_list:
            sim_val = (
                float(annual_sim.at[area, group])
                if area in annual_sim.index and group in annual_sim.columns
                else 0.0
            )
            act_val = (
                float(annual_act.at[area, group])
                if area in annual_act.index and group in annual_act.columns
                else 0.0
            )
            y_vals += [sim_val, act_val]
        if sum(abs(v) for v in y_vals) < 0.001:
            continue
        fig.add_trace(
            go.Bar(
                x=x_labels,
                y=y_vals,
                name=group,
                marker_color=AGG_COLORS.get(group, "#888"),
                legendgroup=group,
                hovertemplate=f"{group}: %{{y:.2f}} TWh<extra></extra>",
            )
        )

    fig.update_layout(yaxis_title="TWh", barmode="stack", height=600)
    _apply_theme(fig)
    fig.update_layout(margin_t=20)
    return fig


def chart_energy_mix_monthly_validate(run_dir, areas):
    """Monthly energy mix: months interleaved (sim / act) per area (--validate mode)."""
    path = run_dir / "outputs" / "production.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "hour" not in df.columns or "area" not in df.columns:
        return None

    area_list, agg_data = _build_energy_agg(df, areas)
    if not area_list:
        return None

    agg_data["month"] = hour_to_cet_month(agg_data["hour"])  # "YYYYMM" in CET
    tec_display_cols = [g for g in AGG_ORDER if g in agg_data.columns]

    actual_df = _load_actual_production(run_dir)
    if actual_df is None:
        return chart_energy_mix_monthly(run_dir, areas)

    _, agg_actual = _build_energy_agg(actual_df, area_list)
    if agg_actual is None:
        return chart_energy_mix_monthly(run_dir, areas)

    agg_actual["month"] = hour_to_cet_month(agg_actual["hour"])  # "YYYYMM" in CET
    tec_display_cols_act = [g for g in AGG_ORDER if g in agg_actual.columns]

    n = len(area_list)
    fig = make_subplots(
        rows=n,
        cols=1,
        shared_xaxes=False,
        subplot_titles=area_list if n > 1 else None,
        vertical_spacing=0.02,
    )
    for row, area in enumerate(area_list, 1):
        sub_sim = agg_data[agg_data["area"] == area]
        monthly_sim = sub_sim.groupby("month")[tec_display_cols].sum() / 1000

        sub_act = (
            agg_actual[agg_actual["area"] == area] if area in agg_actual["area"].values else None
        )
        monthly_act = (
            sub_act.groupby("month")[tec_display_cols_act].sum() / 1000
            if sub_act is not None and not sub_act.empty
            else None
        )

        # Union of all months, sorted, interleaved x
        all_months = sorted(
            set(monthly_sim.index) | (set(monthly_act.index) if monthly_act is not None else set())
        )
        x_labels = [
            lbl
            for m in all_months
            for lbl in (
                f"{_MONTH_LABELS.get(int(m[4:]), m)} (sim)",
                f"{_MONTH_LABELS.get(int(m[4:]), m)} (act)",
            )
        ]

        show_legend = row == 1
        neg_order = [g for g in reversed(AGG_ORDER) if g in AGG_NEGATIVE]
        pos_order = [g for g in AGG_ORDER if g not in AGG_NEGATIVE]
        for group in neg_order + pos_order:
            y_vals = []
            for m in all_months:
                sim_val = (
                    float(monthly_sim.at[m, group])
                    if m in monthly_sim.index and group in monthly_sim.columns
                    else 0.0
                )
                act_val = (
                    float(monthly_act.at[m, group])
                    if monthly_act is not None
                    and m in monthly_act.index
                    and group in monthly_act.columns
                    else 0.0
                )
                y_vals += [sim_val, act_val]
            if sum(abs(v) for v in y_vals) < 0.001:
                continue
            fig.add_trace(
                go.Bar(
                    x=x_labels,
                    y=y_vals,
                    name=group,
                    marker_color=AGG_COLORS.get(group, "#888"),
                    legendgroup=group,
                    showlegend=show_legend,
                    hovertemplate=f"{group}: %{{y:.2f}} TWh<extra></extra>",
                ),
                row=row,
                col=1,
            )
        fig.update_yaxes(title_text="TWh", row=row, col=1)

    fig.update_layout(barmode="stack", height=600 * n)
    _apply_theme(fig)
    fig.update_layout(margin_t=20)
    return fig




# ── Energy mix helpers ──

_MONTH_LABELS = {
    1: "Jan",
    2: "Feb",
    3: "Mar",
    4: "Apr",
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Aug",
    9: "Sep",
    10: "Oct",
    11: "Nov",
    12: "Dec",
}


def _build_energy_agg(df, areas):
    """Return (area_list, agg_data) for energy mix charts.

    agg_data has columns: area, hour, <display_label>...
    Values are in GW (one row per simulation hour).
    """
    df = df[df["area"].isin(areas)].copy()
    if df.empty:
        return [], None

    df["datetime"] = _posix_hours_to_dt(df["hour"])
    tec_cols = [c for c in df.columns if c not in ("hour", "area", "datetime", "demand")]

    agg_groups = {}
    for tec in tec_cols:
        group = TEC_AGGREGATION.get(tec, tec)
        if group not in agg_groups:
            agg_groups[group] = df[tec].values.copy()
        else:
            agg_groups[group] = agg_groups[group] + df[tec].values

    agg_data = df[["area", "hour"]].copy()
    for group, vals in agg_groups.items():
        agg_data[group] = vals

    area_list = [a for a in areas if a in df["area"].unique()]
    return area_list, agg_data


def _add_stacked_bars(fig, area_list, totals_twh, row=None, col=None, show_legend=True):
    """Add positive + negative stacked bar traces to a figure or subplot."""
    # Negative groups added in reverse AGG_ORDER: first trace = closest to 0 (Battery
    # charge), last = farthest (Net exports), giving bottom-to-top: net_exports → battery.
    neg_present = [g for g in reversed(AGG_ORDER) if g in totals_twh.columns and g in AGG_NEGATIVE]
    pos_present = [g for g in AGG_ORDER if g in totals_twh.columns and g not in AGG_NEGATIVE]
    subplot_kwargs = {"row": row, "col": col} if row is not None else {}

    for group in neg_present + pos_present:
        vals = totals_twh.reindex(area_list)[group].fillna(0)
        if vals.abs().sum() < 0.001:
            continue
        fig.add_trace(
            go.Bar(
                x=area_list,
                y=vals.values,
                name=group,
                marker_color=AGG_COLORS.get(group, "#888"),
                legendgroup=group,
                showlegend=show_legend,
                hovertemplate=f"{group}: %{{y:.2f}} TWh<extra></extra>",
            ),
            **subplot_kwargs,
        )


def _add_monthly_stacked_bars(
    fig, monthly_twh, months_present, row=None, col=None, show_legend=True
):
    """Add monthly stacked bar traces (x = month labels) to a figure or subplot.

    months_present are YYYYMM strings (e.g. "201902"), sorted ascending.
    """
    month_labels = [_MONTH_LABELS.get(int(m[4:]), m) for m in months_present]
    neg_present = [g for g in reversed(AGG_ORDER) if g in monthly_twh.columns and g in AGG_NEGATIVE]
    pos_present = [g for g in AGG_ORDER if g in monthly_twh.columns and g not in AGG_NEGATIVE]
    subplot_kwargs = {"row": row, "col": col} if row is not None else {}

    for group in neg_present + pos_present:
        vals = monthly_twh.reindex(months_present)[group].fillna(0)
        if vals.abs().sum() < 0.001:
            continue
        fig.add_trace(
            go.Bar(
                x=month_labels,
                y=vals.values,
                name=group,
                marker_color=AGG_COLORS.get(group, "#888"),
                legendgroup=group,
                showlegend=show_legend,
                hovertemplate=f"{group}: %{{y:.2f}} TWh<extra></extra>",
            ),
            **subplot_kwargs,
        )




