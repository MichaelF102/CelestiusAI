"""Reusable Plotly chart builders adhering to consistent styling and responsive layouts."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.visualization.themes import (
    COLOR_ACCENT,
    PALETTE,
    apply_dark_theme,
)


def build_histogram(
    series: pd.Series,
    col_name: str,
    nbins: int = 30,
    show_mean_median: bool = True,
) -> go.Figure:
    """Build an interactive distribution histogram with mean and median markers."""
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        fig = go.Figure()
        return apply_dark_theme(fig, title=f"Histogram — {col_name} (No Valid Data)")

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=clean,
            nbinsx=nbins,
            name="Count",
            marker={"color": "rgba(56, 189, 248, 0.7)", "line": {"color": "#38bdf8", "width": 1}},
            hovertemplate="Range: %{x}<br>Count: %{y:,}<extra></extra>",
        )
    )

    if show_mean_median:
        mean_val = float(clean.mean())
        median_val = float(clean.median())
        fig.add_vline(
            x=mean_val,
            line_dash="dash",
            line_color="#f43f5e",
            line_width=2,
            annotation_text=f"Mean: {mean_val:,.2f}",
            annotation_position="top right",
            annotation_font={"color": "#f43f5e", "size": 10},
        )
        fig.add_vline(
            x=median_val,
            line_dash="dot",
            line_color="#10b981",
            line_width=2,
            annotation_text=f"Median: {median_val:,.2f}",
            annotation_position="top left",
            annotation_font={"color": "#10b981", "size": 10},
        )

    title = f"Distribution: {col_name} (N = {len(clean):,})"
    return apply_dark_theme(fig, title=title, x_title=col_name, y_title="Frequency")


def build_boxplot(series: pd.Series, col_name: str) -> go.Figure:
    """Build a horizontal box plot emphasizing IQR and potential outliers."""
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        fig = go.Figure()
        return apply_dark_theme(fig, title=f"Box Plot — {col_name} (No Valid Data)")

    fig = go.Figure()
    fig.add_trace(
        go.Box(
            x=clean,
            name=col_name,
            boxpoints="outliers",
            jitter=0.2,
            marker={"color": "#f43f5e", "size": 4, "outliercolor": "#f43f5e"},
            line={"color": COLOR_ACCENT, "width": 1.5},
            fillcolor="rgba(56, 189, 248, 0.25)",
            hovertemplate="Value: %{x:,.2f}<extra></extra>",
        )
    )

    title = f"Tukey Box Plot: {col_name}"
    return apply_dark_theme(fig, title=title, height=280, x_title=col_name)


def build_violinplot(series: pd.Series, col_name: str) -> go.Figure:
    """Build a violin plot showing kernel density estimate and quartile lines."""
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        fig = go.Figure()
        return apply_dark_theme(fig, title=f"Violin Plot — {col_name} (No Valid Data)")

    fig = go.Figure()
    fig.add_trace(
        go.Violin(
            y=clean,
            name=col_name,
            box_visible=True,
            meanline_visible=True,
            fillcolor="rgba(139, 92, 246, 0.3)",
            line_color="#a855f7",
            hovertemplate="Value: %{y:,.2f}<extra></extra>",
        )
    )

    title = f"Violin Density & Quartiles: {col_name}"
    return apply_dark_theme(fig, title=title, y_title=col_name)


def build_categorical_bar(
    categories: list[dict[str, object]],
    col_name: str,
    max_bars: int = 15,
) -> go.Figure:
    """Build an interactive horizontal bar chart for categorical distributions."""
    if not categories:
        fig = go.Figure()
        return apply_dark_theme(fig, title=f"Category Distribution — {col_name} (Empty)")

    display_cats = categories[:max_bars]
    labels = [str(c.get("label", "")) for c in display_cats]
    counts = [int(c.get("count", 0)) for c in display_cats]
    pcts = [float(c.get("percentage", 0.0)) for c in display_cats]

    # Invert so highest count appears at the top of the horizontal bar chart
    labels.reverse()
    counts.reverse()
    pcts.reverse()

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=labels,
            x=counts,
            orientation="h",
            marker={
                "color": pcts,
                "colorscale": [
                    [0.0, "rgba(56, 189, 248, 0.4)"],
                    [1.0, "rgba(56, 189, 248, 0.95)"],
                ],
                "line": {"color": "#38bdf8", "width": 1},
            },
            customdata=pcts,
            hovertemplate="Category: <b>%{y}</b><br>Count: %{x:,}<br>Share: %{customdata:.1f}%<extra></extra>",
        )
    )

    title = f"Top Frequencies: {col_name}"
    return apply_dark_theme(fig, title=title, x_title="Count", y_title="Category")


def build_correlation_heatmap(
    corr_df: pd.DataFrame,
    title: str = "Correlation Matrix",
) -> go.Figure:
    """Build a correlation heatmap with centered diverging color scale and 2-decimal annotations."""
    if corr_df.empty or len(corr_df.columns) < 2:
        fig = go.Figure()
        return apply_dark_theme(fig, title="Correlation Matrix (Insufficient Variables)")

    z_values = np.round(corr_df.values, 2)
    cols = list(corr_df.columns)

    fig = go.Figure(
        data=go.Heatmap(
            z=z_values,
            x=cols,
            y=cols,
            zmin=-1.0,
            zmax=1.0,
            colorscale=[
                [0.0, "#f43f5e"],  # -1.0 (Strong negative)
                [0.5, "#0f172a"],  #  0.0 (Neutral)
                [1.0, "#10b981"],  # +1.0 (Strong positive)
            ],
            text=z_values,
            texttemplate="%{text:.2f}",
            textfont={"size": 11, "color": "#f8fafc"},
            hoverongaps=False,
            hovertemplate="%{y} × %{x}<br>Coefficient: <b>%{z:.3f}</b><extra></extra>",
        )
    )

    return apply_dark_theme(fig, title=title, height=480)


def build_bivariate_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str | None = None,
    size_col: str | None = None,
    add_trendline: bool = False,
    sample_limit: int = 2500,
) -> tuple[go.Figure, bool]:
    """Build a bivariate scatter plot with optional color, size, and descriptive trendline.

    Returns (Figure, was_sampled_bool).
    """
    needed_cols = [x_col, y_col]
    if color_col and color_col in df.columns:
        needed_cols.append(color_col)
    if size_col and size_col in df.columns:
        needed_cols.append(size_col)

    clean_df = df[needed_cols].dropna()
    was_sampled = False

    if len(clean_df) > sample_limit:
        clean_df = clean_df.sample(n=sample_limit, random_state=42)
        was_sampled = True

    if clean_df.empty:
        fig = go.Figure()
        return apply_dark_theme(fig, title=f"Scatter: {x_col} vs {y_col} (No Valid Records)"), False

    trend_mode = "ols" if (add_trendline and len(clean_df) >= 5) else None

    kwargs = {
        "data_frame": clean_df,
        "x": x_col,
        "y": y_col,
        "color": color_col if (color_col and color_col in clean_df.columns) else None,
        "size": size_col if (size_col and size_col in clean_df.columns) else None,
        "trendline": trend_mode,
        "color_discrete_sequence": PALETTE,
        "template": "plotly_dark",
    }
    # Filter None values to keep plotly express happy
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    fig = px.scatter(**kwargs)

    # If trendline was rendered, refine its visual appearance
    if trend_mode:
        fig.update_traces(
            line={"color": "#f59e0b", "width": 2, "dash": "dash"},
            selector={"mode": "lines"},
        )

    title = f"{y_col} vs. {x_col}"
    if was_sampled:
        title += f" [Sampled to {sample_limit:,} records]"

    apply_dark_theme(fig, title=title, x_title=x_col, y_title=y_col)
    return fig, was_sampled


def build_grouped_bar(
    groups: list[dict[str, object]],
    group_col: str,
    measure_col: str,
    agg_name: str,
) -> go.Figure:
    """Build a ranked horizontal bar chart comparing aggregated metrics across groups."""
    if not groups:
        fig = go.Figure()
        return apply_dark_theme(fig, title=f"{agg_name} {measure_col} by {group_col} (Empty)")

    labels = [str(g.get("group", "")) for g in groups]
    values = [float(g.get("value", 0.0)) for g in groups]
    samples = [int(g.get("sample_size", 0)) for g in groups]

    labels.reverse()
    values.reverse()
    samples.reverse()

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=labels,
            x=values,
            orientation="h",
            marker={"color": "#38bdf8", "line": {"color": "#0284c7", "width": 1}},
            customdata=samples,
            hovertemplate="Group: <b>%{y}</b><br>Value: %{x:,.2f}<br>Sample Size: N = %{customdata:,}<extra></extra>",
        )
    )

    title = f"{agg_name.capitalize()} '{measure_col}' by '{group_col}'"
    return apply_dark_theme(
        fig, title=title, x_title=f"{agg_name.capitalize()} {measure_col}", y_title=group_col
    )


def build_time_series_chart(
    agg_df: pd.DataFrame,
    time_col: str,
    measure_col: str,
    agg_name: str = "mean",
) -> go.Figure:
    """Build an interactive line chart with area fill for chronological trend inspection."""
    if agg_df.empty:
        fig = go.Figure()
        return apply_dark_theme(fig, title="Time Series (Empty)")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=agg_df[time_col],
            y=agg_df[measure_col],
            mode="lines+markers",
            line={"color": "#38bdf8", "width": 2},
            marker={"size": 5, "color": "#38bdf8"},
            fill="tozeroy",
            fillcolor="rgba(56, 189, 248, 0.1)",
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Value: %{y:,.2f}<extra></extra>",
        )
    )

    title = f"{agg_name.capitalize()} '{measure_col}' Trend over Time"
    return apply_dark_theme(fig, title=title, x_title="Date", y_title=measure_col)


def build_crosstab_heatmap(
    contingency_df: pd.DataFrame,
    row_var: str,
    col_var: str,
) -> go.Figure:
    """Build a 2D contingency heatmap showing co-occurrence between two discrete variables."""
    if contingency_df.empty:
        fig = go.Figure()
        return apply_dark_theme(fig, title="Crosstab Heatmap (Empty)")

    z_vals = contingency_df.values
    x_labels = [str(c) for c in contingency_df.columns]
    y_labels = [str(r) for r in contingency_df.index]

    fig = go.Figure(
        data=go.Heatmap(
            z=z_vals,
            x=x_labels,
            y=y_labels,
            colorscale=[
                [0.0, "#0f172a"],
                [1.0, "#38bdf8"],
            ],
            text=z_vals,
            texttemplate="%{text:,}",
            textfont={"size": 11, "color": "#f8fafc"},
            hovertemplate=f"{row_var}: %{{y}}<br>{col_var}: %{{x}}<br>Count: %{{z:,}}<extra></extra>",
        )
    )

    title = f"Co-occurrence Matrix: '{row_var}' × '{col_var}'"
    return apply_dark_theme(fig, title=title, x_title=col_var, y_title=row_var)


def build_qq_plot(qq_data: dict[str, list[float]], title: str = "Normal Q-Q Plot") -> go.Figure:
    """Build a quantile-quantile plot comparing sample quantiles against theoretical standard normal."""
    theo = qq_data.get("theoretical_quantiles", [])
    sample = qq_data.get("sample_quantiles", [])
    tx = qq_data.get("trend_line_x", [])
    ty = qq_data.get("trend_line_y", [])

    fig = go.Figure()
    if not theo or not sample:
        return apply_dark_theme(fig, title=title)

    if tx and ty:
        fig.add_trace(
            go.Scatter(
                x=tx,
                y=ty,
                mode="lines",
                name="Theoretical Line",
                line={"color": "#f43f5e", "width": 2, "dash": "dash"},
            )
        )

    fig.add_trace(
        go.Scatter(
            x=theo,
            y=sample,
            mode="markers",
            name="Sample Quantiles",
            marker={"color": "#38bdf8", "size": 6, "line": {"color": "#0284c7", "width": 1}},
            hovertemplate="Theoretical: %{x:.2f}<br>Sample: %{y:.2f}<extra></extra>",
        )
    )

    return apply_dark_theme(
        fig, title=title, x_title="Theoretical Quantiles (Normal)", y_title="Sample Quantiles"
    )


def build_residual_fitted_plot(
    fitted: np.ndarray, residuals: np.ndarray, title: str = "Residuals vs Fitted"
) -> go.Figure:
    """Build a residual vs fitted scatter plot with zero reference line for homoscedasticity inspection."""
    fig = go.Figure()
    if len(fitted) == 0 or len(residuals) == 0:
        return apply_dark_theme(fig, title=title)

    fig.add_hline(y=0.0, line_dash="dash", line_color="#f43f5e", line_width=1.5)

    fig.add_trace(
        go.Scatter(
            x=fitted,
            y=residuals,
            mode="markers",
            name="Residuals",
            marker={"color": "rgba(56, 189, 248, 0.7)", "size": 5},
            hovertemplate="Fitted: %{x:.2f}<br>Residual: %{y:.2f}<extra></extra>",
        )
    )

    return apply_dark_theme(
        fig, title=title, x_title="Fitted Values (ŷ)", y_title="Residuals (e = y - ŷ)"
    )
