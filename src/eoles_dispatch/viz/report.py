"""Chart registries, renderer, and main HTML report generator."""

import webbrowser
from pathlib import Path

from .charts_inputs import (
    chart_capacity_mix,
    chart_demand,
    chart_exo_prices,
    chart_interconnections,
    chart_lake_inflows,
    chart_nmd,
    chart_nuclear_availability,
    chart_vre_profiles,
)
from .charts_outputs import (
    chart_energy_mix_annual,
    chart_energy_mix_annual_validate,
    chart_energy_mix_monthly,
    chart_energy_mix_monthly_validate,
    chart_price_scatter,
    chart_prices,
    chart_prices_validate,
    chart_production,
    html_price_overview,
    html_price_overview_validate,
)
from .loaders import _load_metadata

# ── Main orchestrator ──


def generate_report(run_dir, open_browser=True, validate=False):
    """Generate an interactive HTML report for a run.

    Args:
        run_dir: Path to the run directory.
        open_browser: Open the report in the default browser.
        validate: If True, overlay historical day-ahead prices on price charts
            and show error metrics (Bias, RMSE, MAE, R², Corr).

    Tabs: France Inputs | France Outputs | Neighbors Inputs | Neighbors Outputs
    """
    run_dir = Path(run_dir)
    meta = _load_metadata(run_dir)
    all_areas = meta.get("areas", ["FR"])
    focus = "FR"
    other_areas = [a for a in all_areas if a != focus]

    has_outputs = (run_dir / "outputs").exists() and any((run_dir / "outputs").glob("*.csv"))
    output_charts = _OUTPUT_CHARTS_VALIDATE if validate else _OUTPUT_CHARTS

    # Build all chart HTML
    fr_input_parts = _render_charts(run_dir, _INPUT_CHARTS_FR, [focus])
    fr_output_parts = []
    if has_outputs:
        fr_output_parts = _render_charts(run_dir, output_charts, [focus])

    other_input_parts = _render_charts(run_dir, _INPUT_CHARTS_OTHER, other_areas)
    other_output_parts = []
    if has_outputs:
        other_output_parts = _render_charts(run_dir, output_charts, other_areas)

    fr_input_html = (
        "\n".join(fr_input_parts)
        if fr_input_parts
        else '<div class="no-data">No input data found.</div>'
    )
    fr_output_html = (
        "\n".join(fr_output_parts)
        if fr_output_parts
        else '<div class="no-data">No output data — run not solved yet.</div>'
    )
    other_input_html = (
        "\n".join(other_input_parts)
        if other_input_parts
        else '<div class="no-data">No input data found.</div>'
    )
    other_output_html = (
        "\n".join(other_output_parts)
        if other_output_parts
        else '<div class="no-data">No output data — run not solved yet.</div>'
    )

    _MONTH_NAMES = {
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
    months_raw = meta.get("months")
    if months_raw:
        ms = str(months_raw)
        if "-" in ms:
            a, b = ms.split("-", 1)
            months_label = f" · {_MONTH_NAMES.get(int(a), a)}–{_MONTH_NAMES.get(int(b), b)}"
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
<title>EOLES-Dispatch — {meta.get("name", run_dir.name)}</title>
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

  .chart-raw {{
    margin: 18px 0;
  }}

  /* ── Price overview side-by-side ── */
  .price-overview {{
    display: flex;
    gap: 16px;
    margin: 18px 0;
    min-height: 400px;
    align-items: stretch;
  }}
  .price-overview-left {{
    flex: 0 0 50%;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    box-shadow: 0 2px 0 rgba(255,255,255,0.7) inset, 0 3px 14px var(--shadow);
    padding: 16px 18px;
    display: flex;
    flex-direction: column;
    overflow-y: auto;
  }}
  .price-overview-table {{
    flex: 1;
    overflow: auto;
  }}
  .price-overview-right {{
    flex: 0 0 50%;
    min-width: 0;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    box-shadow: 0 2px 0 rgba(255,255,255,0.7) inset, 0 3px 14px var(--shadow);
    padding: 16px 18px;
    display: flex;
    flex-direction: column;
  }}
  .price-overview-left > .section-title,
  .price-overview-right > .section-title {{
    margin: 0 0 8px 0;
    flex-shrink: 0;
  }}
  .price-chart-wrapper {{
    flex: 1;
    min-height: 0;
    width: 100%;
  }}
  .price-chart-wrapper > div {{
    width: 100%;
    height: 100%;
  }}
  .price-chart-wrapper .plotly-graph-div {{
    width: 100% !important;
    height: 100% !important;
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
    font-size: 0.75rem;
    width: 100%;
  }}
  .stats-table th, .stats-table td {{
    padding: 4px 8px;
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

  /* ── Output sections ── */
  .output-section {{
    margin: 32px 0 0;
    padding-bottom: 8px;
    border-bottom: 2px solid var(--border);
  }}
  .output-section h2 {{
    margin: 0;
    font-size: 0.72rem;
    font-family: 'DM Mono', monospace;
    font-weight: 500;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.12em;
  }}
  .chart-label {{
    font-size: 0.92rem;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 6px;
    letter-spacing: -0.01em;
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
      <div class="subtitle">{meta.get("name", run_dir.name)}</div>
    </div>
    <span class="badge badge-{status}">{status}{(" · " + exec_time) if exec_time else ""}</span>
  </div>
  <div class="meta-pills">
    <div class="meta-pill">Scenario <b>{meta.get("scenario", "?")}</b></div>
    <div class="meta-pill">Year <b>{meta.get("year", "?")}</b>{months_label}</div>
    <div class="meta-pill">Areas <b>{", ".join(all_areas)}</b></div>
    <div class="meta-pill">Solver: <b>{solver_info or "N/A"}</b></div>
    <div class="meta-pill">Created: <b>{meta.get("created", "?")[:16].replace("T", " ")}</b></div>
    {"<div class='meta-pill'>Solved: <b>" + meta.get("solved", "")[:16].replace("T", " ") + "</b></div>" if meta.get("solved") else ""}
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
<div class="footer">EOLES-Dispatch &middot; {meta.get("name", run_dir.name)} &middot; {meta.get("created", "")[:10]}</div>
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


# ── Chart configuration and rendering ──

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
    ("Prices", None),
    ("", html_price_overview),
    ("Energy Mix", None),
    ("", chart_energy_mix_annual),
    ("Prices — Details", None),
    ("Spot price", chart_prices),
    ("Energy Mix — Details", None),
    ("Monthly breakdown", chart_energy_mix_monthly),
    ("Dispatch", chart_production),
]

_OUTPUT_CHARTS_VALIDATE = [
    ("Prices", None),
    ("", html_price_overview_validate),
    ("Energy Mix", None),
    ("", chart_energy_mix_annual_validate),
    ("Prices — Details", None),
    ("Spot price", chart_prices_validate),
    ("Simulated vs actual", chart_price_scatter),
    ("Energy Mix — Details", None),
    ("Monthly breakdown", chart_energy_mix_monthly_validate),
    ("Dispatch", chart_production),
]


def _render_charts(run_dir, chart_list, areas):
    """Render a list of charts to HTML divs.

    Registry entries are (label, chart_fn) pairs.
    - If chart_fn is None: label is a section header → renders an <h2> divider.
    - Otherwise: label is an optional HTML chart title (shown if non-empty and
      starts with an uppercase letter); chart_fn returns a Plotly figure or raw HTML.
    Plotly CDN is loaded once in the page <head>.
    """
    parts = []
    for label, chart_fn in chart_list:
        if chart_fn is None:
            parts.append(f'<div class="output-section"><h2>{label}</h2></div>')
            continue
        result = chart_fn(run_dir, areas)
        if result is None:
            continue
        title_html = (
            f'<div class="chart-label">{label}</div>'
            if label and label[0].isupper()
            else ""
        )
        if isinstance(result, str):
            parts.append(f'<div class="chart-raw">{title_html}{result}</div>')
        else:
            parts.append(
                f'<div class="chart">{title_html}{result.to_html(full_html=False, include_plotlyjs=False)}</div>'
            )
    return parts
