"""Shared visual themes and styling configurations for Plotly figures."""

import plotly.graph_objects as go

# Palette tailored for high-contrast dark mode visualization
PALETTE = [
    "#38bdf8",  # Sky 400 (Primary accent)
    "#10b981",  # Emerald 500 (Success/Positive)
    "#f59e0b",  # Amber 500 (Warning/Moderate)
    "#f43f5e",  # Rose 500 (Alert/Negative)
    "#8b5cf6",  # Violet 500
    "#ec4899",  # Pink 500
    "#06b6d4",  # Cyan 500
    "#a855f7",  # Purple 500
]

COLOR_ACCENT = "#38bdf8"
COLOR_MUTED = "#94a3b8"
COLOR_GRID = "rgba(51, 65, 85, 0.4)"
BG_PLOT = "rgba(15, 23, 42, 0.6)"
BG_PAPER = "rgba(0, 0, 0, 0)"


def apply_dark_theme(
    fig: go.Figure,
    title: str | None = None,
    height: int = 420,
    show_legend: bool = True,
    x_title: str | None = None,
    y_title: str | None = None,
) -> go.Figure:
    """Apply consistent styling and dark theme to any Plotly figure."""
    layout_update = {
        "template": "plotly_dark",
        "paper_bgcolor": BG_PAPER,
        "plot_bgcolor": BG_PLOT,
        "height": height,
        "margin": {"l": 30, "r": 20, "t": 50 if title else 25, "b": 35},
        "font": {
            "family": "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
            "color": "#f1f5f9",
            "size": 12,
        },
        "showlegend": show_legend,
        "legend": {
            "bgcolor": "rgba(15, 23, 42, 0.8)",
            "bordercolor": "rgba(51, 65, 85, 0.6)",
            "borderwidth": 1,
            "font": {"size": 11, "color": "#cbd5e1"},
        },
        "hoverlabel": {
            "bgcolor": "#0f172a",
            "bordercolor": "#38bdf8",
            "font": {"color": "#f8fafc", "family": "Inter, sans-serif", "size": 12},
        },
    }

    if title:
        layout_update["title"] = {
            "text": f"<b>{title}</b>",
            "font": {"size": 14, "color": "#f8fafc"},
            "x": 0.02,
            "xanchor": "left",
        }

    fig.update_layout(**layout_update)

    fig.update_xaxes(
        gridcolor=COLOR_GRID,
        zerolinecolor=COLOR_GRID,
        showline=True,
        linecolor="rgba(100, 116, 139, 0.3)",
        tickfont={"color": COLOR_MUTED, "size": 11},
        title={"text": x_title, "font": {"color": "#e2e8f0", "size": 12}} if x_title else None,
    )
    fig.update_yaxes(
        gridcolor=COLOR_GRID,
        zerolinecolor=COLOR_GRID,
        showline=True,
        linecolor="rgba(100, 116, 139, 0.3)",
        tickfont={"color": COLOR_MUTED, "size": 11},
        title={"text": y_title, "font": {"color": "#e2e8f0", "size": 12}} if y_title else None,
    )
    return fig
