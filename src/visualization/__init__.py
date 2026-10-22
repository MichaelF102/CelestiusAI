"""Visualization package exporting styling themes and high-level Plotly chart builders."""

from src.visualization.charts import (
    build_bivariate_scatter,
    build_boxplot,
    build_categorical_bar,
    build_correlation_heatmap,
    build_crosstab_heatmap,
    build_grouped_bar,
    build_histogram,
    build_qq_plot,
    build_residual_fitted_plot,
    build_time_series_chart,
    build_violinplot,
)
from src.visualization.themes import (
    COLOR_ACCENT,
    COLOR_GRID,
    COLOR_MUTED,
    PALETTE,
    apply_dark_theme,
)

__all__ = [
    "COLOR_ACCENT",
    "COLOR_GRID",
    "COLOR_MUTED",
    "PALETTE",
    "apply_dark_theme",
    "build_bivariate_scatter",
    "build_boxplot",
    "build_categorical_bar",
    "build_correlation_heatmap",
    "build_crosstab_heatmap",
    "build_grouped_bar",
    "build_histogram",
    "build_qq_plot",
    "build_residual_fitted_plot",
    "build_time_series_chart",
    "build_violinplot",
]
