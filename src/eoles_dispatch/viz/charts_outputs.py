"""Output chart builders (post-solve data from outputs/ folder)."""

import io
import re

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

matplotlib.use("Agg")
from plotly.subplots import make_subplots

from ..utils import hour_to_cet_month, posix_hours_to_dt
from .loaders import (
    _country_color,
    _load_actual_prices,
    _load_actual_production,
)
from .theme import (
    _MONTH_LABELS,
    AGG_COLORS,
    AGG_NEGATIVE,
    AGG_ORDER,
    LEGEND_BELOW,
    TEC_AGGREGATION,
    _apply_theme,
)

# ── Price overview ──


def html_price_overview(run_dir, areas, *, validate=False):
    """Side-by-side: HTML stats table (left) + Plotly price duration curve (right).

    When validate=True, loads actual prices and adds Sim/Act columns, error metrics,
    and actual duration curve overlay.

    Returns raw HTML string with embedded Plotly chart.
    """
    path = run_dir / "outputs" / "prices.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    cols = [c for c in areas if c in df.columns]
    if not cols:
        return None

    actual_df = None
    actual_cols = []
    if validate:
        actual_df = _load_actual_prices(run_dir)
        actual_cols = [c for c in cols if actual_df is not None and c in actual_df.columns]

    table_html = _price_stats_table(df, cols, actual_df, actual_cols)
    chart_html = _price_duration_curve(df, cols, actual_df, actual_cols)

    has_actual = len(actual_cols) > 0
    title_label = (
        "Spot price \u2014 simulated vs actual (EUR/MWh)"
        if validate and has_actual
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


def _price_stats_table(df, cols, actual_df=None, actual_cols=()):
    """Return an HTML stats table for price data.

    If actual_df/actual_cols are provided, renders Sim/Act sub-columns and error metrics.
    """
    stat_fns = [
        ("Mean", lambda v: f"{v.mean():.1f}"),
        ("Median", lambda v: f"{v.median():.1f}"),
        ("Std", lambda v: f"{v.std():.1f}"),
        ("Min", lambda v: f"{v.min():.1f}"),
        ("Max", lambda v: f"{v.max():.1f}"),
        ("P5", lambda v: f"{v.quantile(0.05):.1f}"),
        ("P95", lambda v: f"{v.quantile(0.95):.1f}"),
    ]
    has_actual = len(actual_cols) > 0

    if has_actual:
        header = ""
        for a in cols:
            header += f'<th colspan="2">{a}</th>' if a in actual_cols else f"<th>{a}</th>"
        sub_header = ""
        for a in cols:
            sub_header += "<th>Sim</th><th>Act</th>" if a in actual_cols else "<th>Sim</th>"

        rows_html = ""
        for stat_name, fn in stat_fns:
            cells = ""
            for a in cols:
                cells += f"<td>{fn(df[a].dropna())}</td>"
                if a in actual_cols:
                    cells += f"<td>{fn(actual_df[a].dropna())}</td>"
            rows_html += f"<tr><td class='row-label'>{stat_name}</td>{cells}</tr>\n"

        merged = df[["hour"] + list(actual_cols)].merge(
            actual_df[["hour"] + list(actual_cols)], on="hour", suffixes=("_sim", "_act")
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
                    ai = list(actual_cols).index(a)
                    m = err_metrics[ai]
                    cells += (
                        f'<td colspan="2">{m[key]:{fmt}}</td>'
                        if m is not None
                        else '<td colspan="2">\u2014</td>'
                    )
                else:
                    cells += "<td>\u2014</td>"
            err_rows += f"<tr><td class='row-label'>{label}</td>{cells}</tr>\n"

        return f"""<table class="stats-table">
<thead>
<tr><th></th>{header}</tr>
<tr><th></th>{sub_header}</tr>
</thead>
<tbody>{rows_html}
<tr><td colspan="100" style="border-top:2px solid var(--border);padding:4px 0 0 0;font-size:0.65rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.05em">Error metrics (sim \u2212 act)</td></tr>
{err_rows}</tbody>
</table>"""

    # Simple table (no actual data)
    header = "".join(f"<th>{a}</th>" for a in cols)
    rows_html = ""
    for stat_name, fn in stat_fns:
        cells = "".join(f"<td>{fn(df[a].dropna())}</td>" for a in cols)
        rows_html += f"<tr><td class='row-label'>{stat_name}</td>{cells}</tr>\n"
    return f"""<table class="stats-table">
<thead><tr><th></th>{header}</tr></thead>
<tbody>{rows_html}</tbody>
</table>"""


def _price_duration_curve(df, cols, actual_df=None, actual_cols=()):
    """Return embedded Plotly HTML for a price duration curve.

    If actual_df/actual_cols are provided, overlays actual curves as dashed traces.
    """
    has_actual = len(actual_cols) > 0
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
        height=500,
        yaxis_rangemode="tozero" if y_min >= 0 else "normal",
    )
    _apply_theme(fig)
    fig.update_layout(legend=LEGEND_BELOW, margin=dict(l=45, r=15, t=15, b=40))
    return fig.to_html(full_html=False, include_plotlyjs=False)


# ── Price full series ──


def chart_prices(run_dir, areas, *, validate=False):
    """Hourly price time series.

    When validate=True, overlays actual prices as dashed traces.
    """
    path = run_dir / "outputs" / "prices.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "hour" not in df.columns:
        return None
    df["datetime"] = posix_hours_to_dt(df["hour"])
    cols = [c for c in areas if c in df.columns]
    if not cols:
        return None

    actual_df = _load_actual_prices(run_dir) if validate else None
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


# ── Energy mix chart builders ──

# Row groupings for the energy mix table: (section_label, [(group_key, display_label), ...])
_TABLE_SECTIONS = [
    (
        "Fossil",
        [
            ("Gas", "Gas"),
            ("Coal", "Coal"),
            ("Oil", "Oil"),
        ],
    ),
    (
        "Renewables",
        [
            ("Wind", "Wind"),
            ("Solar", "Solar"),
            ("River", "River"),
        ],
    ),
    (
        "Other",
        [
            ("Nuclear", "Nuclear"),
            ("NMD", "NMD"),
        ],
    ),
    (
        "Storage",
        [
            ("Lake/PHS", "Lake/PHS (generation)"),
            ("Lake/PHS (charge)", "PHS (storage)"),
            ("Battery", "Battery (generation)"),
            ("Battery (charge)", "Battery (storage)"),
        ],
    ),
]
_TABLE_EXCHANGES = [
    ("Net imports", "Net imports"),
    ("Net exports", "Net exports"),
]


def _compute_energy_mix(run_dir, areas, validate=False):
    """Load and aggregate energy mix data over the full simulation period.

    Returns (area_list, enermix_sim, enermix_act, demand_sim, demand_act) where:
    - enermix_sim / enermix_act are DataFrames indexed by area, columns = tech groups, values in TWh.
    - demand_sim / demand_act are Series indexed by area, values in TWh.
    - enermix_act / demand_act are None when validate=False or actual data is unavailable.
    Returns ([], None, None, None, None) on missing/invalid input data.
    """
    path = run_dir / "outputs" / "production.csv"
    if not path.exists():
        return [], None, None, None, None
    df = pd.read_csv(path)
    if "hour" not in df.columns or "area" not in df.columns:
        return [], None, None, None, None

    area_list, agg_data = _build_energy_agg(df, areas)
    if not area_list:
        return [], None, None, None, None

    tec_cols_sim = [g for g in AGG_ORDER if g in agg_data.columns]
    enermix_sim = agg_data.groupby("area")[tec_cols_sim].sum() / 1000  # TWh

    if "Net imports" in enermix_sim.columns and "Net exports" in enermix_sim.columns:
        # Net exports are negative; compute net position and assign to the right side
        net_pos = enermix_sim["Net imports"] + enermix_sim["Net exports"]
        enermix_sim["Net imports"] = net_pos.clip(lower=0)
        enermix_sim["Net exports"] = net_pos.clip(upper=0)

    demand_sim = None
    if "demand" in agg_data.columns:
        demand_sim = agg_data.groupby("area")["demand"].sum() / 1000  # TWh

    enermix_act = None
    demand_act = None
    if validate:
        actual_df = _load_actual_production(run_dir)
        if actual_df is not None:
            _, agg_actual = _build_energy_agg(actual_df, area_list)
            if agg_actual is not None:
                tec_cols_act = [g for g in AGG_ORDER if g in agg_actual.columns]
                enermix_act = agg_actual.groupby("area")[tec_cols_act].sum() / 1000  # TWh
                if "Net imports" in enermix_act.columns and "Net exports" in enermix_act.columns:
                    net_pos = enermix_act["Net imports"] + enermix_act["Net exports"]
                    enermix_act["Net imports"] = net_pos.clip(lower=0)
                    enermix_act["Net exports"] = net_pos.clip(upper=0)
                if "demand" in agg_actual.columns:
                    demand_act = agg_actual.groupby("area")["demand"].sum() / 1000

    return area_list, enermix_sim, enermix_act, demand_sim, demand_act


def _energy_mix_fig(area_list, enermix_sim, enermix_act, width_px=620, height_px=320):
    """Build energy mix stacked bar chart as an inline SVG HTML string.

    Positive groups stack above the x-axis from zero; negative groups (Net exports,
    PHS charge, Battery charge) stack below from zero, independent of the positive stack.
    In validate mode, bars are interleaved: [area1 (sim), area1 (act), ...].

    width_px / height_px: target embed dimensions in pixels; the figure is rendered at
    this exact size so fonts and proportions are correct.
    """
    has_actual = enermix_act is not None

    if has_actual:
        bars = [
            item
            for a in area_list
            for item in ((f"{a}\n(sim)", enermix_sim, a), (f"{a}\n(act)", enermix_act, a))
        ]
    else:
        bars = [(a, enermix_sim, a) for a in area_list]

    n_bars = len(bars)
    x = np.arange(n_bars)
    bar_width = 0.65

    def _get(df, area, group):
        if area in df.index and group in df.columns:
            return float(df.at[area, group])
        return 0.0

    # neg_groups: Battery (charge) first (closest to 0), Net exports last (farthest)
    neg_groups = [g for g in reversed(AGG_ORDER) if g in AGG_NEGATIVE]
    pos_groups = [g for g in AGG_ORDER if g not in AGG_NEGATIVE]

    dpi = 100
    fig, ax = plt.subplots(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)

    bottom_pos = np.zeros(n_bars)
    for group in pos_groups:
        vals = np.array([_get(df, area, group) for _, df, area in bars])
        if np.abs(vals).sum() < 0.001:
            continue
        ax.bar(
            x,
            vals,
            bar_width,
            bottom=bottom_pos,
            color=AGG_COLORS.get(group, "#888"),
            label=group,
            zorder=2,
        )
        bottom_pos += vals

    bottom_neg = np.zeros(n_bars)
    for group in neg_groups:
        vals = np.array([_get(df, area, group) for _, df, area in bars])
        if np.abs(vals).sum() < 0.001:
            continue
        ax.bar(
            x,
            vals,
            bar_width,
            bottom=bottom_neg,
            color=AGG_COLORS.get(group, "#888"),
            label=group,
            zorder=2,
        )
        bottom_neg += vals

    ax.set_xticks(x)
    ax.set_xticklabels([lbl for lbl, _, _ in bars], fontsize=9)
    ax.set_ylabel("TWh", fontsize=9)
    ax.axhline(0, color="#333", linewidth=0.8, zorder=3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=9)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles, labels, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8, frameon=False
    )

    fig.tight_layout()

    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    svg = buf.getvalue()
    svg = svg[svg.find("<svg") :]

    # Force embed at exact target dimensions (viewBox handles internal scaling)
    svg = re.sub(r'(<svg[^>]*)\bwidth="[^"]*"', rf'\1width="{width_px}px"', svg, count=1)
    svg = re.sub(r'(<svg[^>]*)\bheight="[^"]*"', rf'\1height="{height_px}px"', svg, count=1)
    return svg


def _energy_mix_table(area_list, enermix_sim, enermix_act, demand_sim=None, demand_act=None):
    """Build and return the energy mix HTML table from pre-computed data.

    Rows are grouped by section (Fossil / Renewables / Other), followed by a Total row,
    then Exchanges (Net imports / Net exports) separated by a border.
    demand_sim / demand_act are Series indexed by area (TWh). When provided, the
    Demand row uses these values directly instead of computing gen + exchanges.
    """
    has_actual = enermix_act is not None
    n_cols = 1 + len(area_list) * (2 if has_actual else 1)

    def _get(df, area, group):
        if df is not None and area in df.index and group in df.columns:
            return float(df.at[area, group])
        return 0.0

    def _data_row(label, group, totals_sim=None, totals_act=None):
        cells = ""
        row_vals = []
        for area in area_list:
            sim_val = _get(enermix_sim, area, group)
            row_vals.append(sim_val)
            if has_actual:
                act_val = _get(enermix_act, area, group)
                row_vals.append(act_val)
                cells += f"<td>{sim_val:.1f}</td><td>{act_val:.1f}</td>"
                if totals_sim is not None:
                    totals_sim[area] += sim_val
                if totals_act is not None:
                    totals_act[area] += act_val
            else:
                cells += f"<td>{sim_val:.1f}</td>"
                if totals_sim is not None:
                    totals_sim[area] += sim_val
        if sum(abs(v) for v in row_vals) < 0.001:
            return ""
        return f"<tr><td class='row-label'>{label}</td>{cells}</tr>\n"

    def _section_header(title):
        return (
            f"<tr style='background:var(--bg-muted,#f5f0e8)'>"
            f"<td class='row-label' style='font-weight:600'>{title}</td>"
            f"<td colspan='{n_cols - 1}'></td></tr>\n"
        )

    if has_actual:
        header = "".join(f'<th colspan="2">{a}</th>' for a in area_list)
        sub_header = "<th>Sim</th><th>Act</th>" * len(area_list)
    else:
        header = "".join(f"<th>{a}</th>" for a in area_list)
        sub_header = ""

    rows_html = ""
    totals_sim = {a: 0.0 for a in area_list}
    totals_act = {a: 0.0 for a in area_list}

    for section_name, items in _TABLE_SECTIONS:
        section_rows = "".join(
            _data_row(label, group, totals_sim=totals_sim, totals_act=totals_act)
            for group, label in items
        )
        if section_rows:
            rows_html += _section_header(section_name)
            rows_html += section_rows

    total_cells = ""
    for area in area_list:
        if has_actual:
            total_cells += f"<td><strong>{totals_sim[area]:.1f}</strong></td><td><strong>{totals_act[area]:.1f}</strong></td>"
        else:
            total_cells += f"<td><strong>{totals_sim[area]:.1f}</strong></td>"
    rows_html += f"<tr style='border-top:2px solid var(--border);background:var(--bg-muted,#f5f0e8)'><td class='row-label'><strong>Total Generation</strong></td>{total_cells}</tr>\n"

    exchange_rows = "".join(_data_row(label, group) for group, label in _TABLE_EXCHANGES)
    rows_html += exchange_rows

    # Use actual demand data when available, otherwise fall back to gen + exchanges
    def _demand_val(series, area, fallback):
        if series is not None and area in series.index:
            return float(series[area])
        return fallback

    demand_cells = ""
    for area in area_list:
        sim_val = _demand_val(demand_sim, area, totals_sim[area])
        if has_actual:
            act_val = _demand_val(demand_act, area, totals_act[area])
            demand_cells += (
                f"<td><strong>{sim_val:.1f}</strong></td><td><strong>{act_val:.1f}</strong></td>"
            )
        else:
            demand_cells += f"<td><strong>{sim_val:.1f}</strong></td>"
    rows_html += f"<tr style='border-top:2px solid var(--border);background:var(--bg-muted,#f5f0e8)'><td class='row-label'><strong>Demand</strong></td>{demand_cells}</tr>\n"

    sub_header_row = f"<tr><th></th>{sub_header}</tr>" if has_actual else ""
    return f"""<table class="stats-table">
<thead>
<tr><th></th>{header}</tr>
{sub_header_row}
</thead>
<tbody>{rows_html}</tbody>
</table>"""


def chart_energy_mix(run_dir, areas, validate=False):
    """Energy mix over the full simulation period: stacked bar per area, as inline SVG HTML.

    Positive groups stack above the x-axis from zero; negative groups below from zero.
    If validate=True, interleaves simulated and actual bars per area (--validate mode).
    Falls back to validate=False if actual data is unavailable.
    """
    area_list, enermix_sim, enermix_act, _, _ = _compute_energy_mix(run_dir, areas, validate)
    if not area_list:
        return None
    return _energy_mix_fig(area_list, enermix_sim, enermix_act)


def html_energy_mix(run_dir, areas, validate=False):
    """Energy mix over the full simulation period as an HTML table (TWh, one row per technology group).

    In validate mode, each area gets Sim / Act sub-columns.
    Returns an HTML string, or None if data is unavailable.
    """
    area_list, enermix_sim, enermix_act, demand_sim, demand_act = _compute_energy_mix(
        run_dir, areas, validate
    )
    if not area_list:
        return None
    return _energy_mix_table(area_list, enermix_sim, enermix_act, demand_sim, demand_act)


def html_energy_mix_overview(run_dir, areas, validate=False):
    """Side-by-side: static SVG energy mix chart (left) + HTML summary table (right).

    When validate=True, the chart interleaves Sim/Act bars and the table adds Sim/Act columns.
    Falls back to non-validate display if actual data is unavailable.

    Returns a raw HTML string with embedded SVG chart.
    """
    area_list, enermix_sim, enermix_act, demand_sim, demand_act = _compute_energy_mix(
        run_dir, areas, validate
    )
    if not area_list:
        return None

    has_actual = enermix_act is not None
    table_html = _energy_mix_table(area_list, enermix_sim, enermix_act, demand_sim, demand_act)

    # Match chart height to the table: count <tr> rows × 28px + thead (~40px)
    n_tr = table_html.count("<tr")
    height_px = max(280, n_tr * 28 + 40)
    # Panel is ~50% of 1400px max-width minus padding → ~620px wide
    chart_html = _energy_mix_fig(
        area_list, enermix_sim, enermix_act, width_px=620, height_px=height_px
    )

    title_label = (
        "Energy mix \u2014 simulated vs actual (TWh)"
        if validate and has_actual
        else "Energy mix (TWh)"
    )

    return f"""<div class="price-overview">
  <div class="price-overview-right">
    <h3 class="section-title">{title_label}</h3>
    <div class="price-chart-wrapper">{chart_html}</div>
  </div>
  <div class="price-overview-left">
    <h3 class="section-title">Annual breakdown (TWh)</h3>
    <div class="price-overview-table">{table_html}</div>
  </div>
</div>"""


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

    df["datetime"] = posix_hours_to_dt(df["hour"])
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


# ── Energy mix helpers ──

def _build_energy_agg(df, areas):
    """Return (area_list, agg_data) for energy mix charts.

    agg_data has columns: area, hour, <display_label>...
    Values are in GW (one row per simulation hour).
    """
    df = df[df["area"].isin(areas)].copy()
    if df.empty:
        return [], None

    df["datetime"] = posix_hours_to_dt(df["hour"])
    tec_cols = [c for c in df.columns if c not in ("hour", "area", "datetime", "demand")]

    agg_groups = {}
    for tec in tec_cols:
        group = TEC_AGGREGATION.get(tec, tec)
        if group not in agg_groups:
            agg_groups[group] = df[tec].values.copy()
        else:
            agg_groups[group] = agg_groups[group] + df[tec].values

    agg_data = df[["area", "hour"]].copy()
    if "demand" in df.columns:
        agg_data["demand"] = df["demand"].values
    for group, vals in agg_groups.items():
        agg_data[group] = vals

    area_list = [a for a in areas if a in df["area"].unique()]
    return area_list, agg_data


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
