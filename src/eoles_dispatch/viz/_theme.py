"""Color palettes, typography, and Plotly theme helpers."""

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
