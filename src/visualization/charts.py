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


def build_group_comparison_distribution(
    df: pd.DataFrame,
    group_col: str,
    measure_col: str,
    groups: list[str] | None = None,
    title: str | None = None,
) -> go.Figure:
    """Build a side-by-side violin + box distribution plot comparing groups with data jitter."""
    needed = [group_col, measure_col]
    clean = df[needed].dropna()
    clean[group_col] = clean[group_col].astype(str)
    if groups:
        clean = clean[clean[group_col].isin(groups)]

    if clean.empty:
        fig = go.Figure()
        return apply_dark_theme(fig, title="Group Distribution (No Valid Data)")

    palette_colors = [
        {"line": "#38bdf8", "fill": "rgba(56, 189, 248, 0.25)"},
        {"line": "#f43f5e", "fill": "rgba(244, 63, 94, 0.25)"},
        {"line": "#10b981", "fill": "rgba(16, 185, 129, 0.25)"},
        {"line": "#a855f7", "fill": "rgba(168, 85, 247, 0.25)"},
        {"line": "#f59e0b", "fill": "rgba(245, 158, 11, 0.25)"},
    ]
    fig = go.Figure()

    unique_groups = groups if groups else list(clean[group_col].unique())
    for i, grp in enumerate(unique_groups):
        grp_data = clean[clean[group_col] == grp][measure_col]
        if grp_data.empty:
            continue
        color_info = palette_colors[i % len(palette_colors)]
        fig.add_trace(
            go.Violin(
                x=clean[clean[group_col] == grp][group_col],
                y=grp_data,
                name=str(grp),
                box_visible=True,
                meanline_visible=True,
                points="outliers",
                jitter=0.2,
                scalemode="count",
                line_color=color_info["line"],
                fillcolor=color_info["fill"],
                marker={"size": 4, "opacity": 0.7},
                hovertemplate=f"<b>{grp}</b><br>Value: %{{y:,.2f}}<extra></extra>",
            )
        )

    chart_title = title or f"Group Distribution Comparison: '{measure_col}' by '{group_col}'"
    return apply_dark_theme(fig, title=chart_title, x_title=group_col, y_title=measure_col)


def build_difference_forest_plot(
    point_estimate: float,
    ci_lower: float,
    ci_upper: float,
    estimand_label: str = "Mean Difference",
    group1: str = "",
    group2: str = "",
    is_significant: bool = True,
    title: str | None = None,
) -> go.Figure:
    """Build a horizontal forest plot showing effect estimate and 95% Confidence Interval vs Null."""
    fig = go.Figure()

    # Zero reference line (Null Hypothesis of 0 difference)
    fig.add_vline(
        x=0.0,
        line_dash="dash",
        line_color="#f43f5e",
        line_width=2,
        annotation_text="Null (Zero Difference)",
        annotation_position="top left",
        annotation_font={"color": "#f43f5e", "size": 11},
    )

    err_plus = max(0.0, ci_upper - point_estimate)
    err_minus = max(0.0, point_estimate - ci_lower)
    marker_color = "#10b981" if is_significant else "#f59e0b"

    fig.add_trace(
        go.Scatter(
            x=[point_estimate],
            y=[estimand_label],
            error_x={
                "type": "data",
                "symmetric": False,
                "array": [err_plus],
                "arrayminus": [err_minus],
                "color": marker_color,
                "thickness": 3,
                "width": 10,
            },
            mode="markers",
            marker={"size": 12, "color": marker_color, "symbol": "diamond"},
            name=estimand_label,
            hovertemplate=(
                f"<b>{estimand_label}</b><br>"
                f"Estimate: %{{x:,.3f}}<br>"
                f"95% CI: [{ci_lower:,.3f}, {ci_upper:,.3f}]<extra></extra>"
            ),
        )
    )

    # Annotate direction
    sub_title = title or f"Confidence Interval & Significance: {estimand_label}"
    if group1 and group2:
        sub_title += f" ({group1} vs {group2})"

    fig.update_layout(height=260)
    return apply_dark_theme(fig, title=sub_title, x_title=f"{estimand_label} (with 95% CI)")


def build_ecdf_comparison_plot(
    df: pd.DataFrame,
    group_col: str,
    measure_col: str,
    groups: list[str] | None = None,
    title: str | None = None,
) -> go.Figure:
    """Build an empirical cumulative distribution function (eCDF) plot comparing groups."""
    needed = [group_col, measure_col]
    clean = df[needed].dropna()
    clean[group_col] = clean[group_col].astype(str)
    if groups:
        clean = clean[clean[group_col].isin(groups)]

    if clean.empty:
        fig = go.Figure()
        return apply_dark_theme(fig, title="eCDF (No Valid Data)")

    palette_colors = ["#38bdf8", "#f43f5e", "#10b981", "#a855f7", "#f59e0b"]
    fig = go.Figure()

    unique_groups = groups if groups else list(clean[group_col].unique())
    for i, grp in enumerate(unique_groups):
        grp_vals = clean[clean[group_col] == grp][measure_col].sort_values().values
        if len(grp_vals) == 0:
            continue
        y_pct = (np.arange(len(grp_vals)) + 1) / len(grp_vals) * 100.0
        color = palette_colors[i % len(palette_colors)]

        fig.add_trace(
            go.Scatter(
                x=grp_vals,
                y=y_pct,
                mode="lines",
                name=f"{grp} (N={len(grp_vals):,})",
                line={"color": color, "width": 2.5},
                hovertemplate=f"<b>{grp}</b><br>Value: %{{x:,.2f}}<br>Cumulative: %{{y:.1f}}%<extra></extra>",
            )
        )

    chart_title = title or f"Empirical Cumulative Distribution (eCDF): '{measure_col}'"
    return apply_dark_theme(
        fig,
        title=chart_title,
        x_title=measure_col,
        y_title="Cumulative Percentile (%)",
    )


def build_normality_overlay_plot(
    series: pd.Series,
    group_name: str,
    var_name: str,
    title: str | None = None,
) -> go.Figure:
    """Build a histogram of empirical density with theoretical Gaussian bell curve overlaid."""
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean) < 3:
        fig = go.Figure()
        return apply_dark_theme(fig, title=f"Normality Check: '{var_name}' (Insufficient Data)")

    mean_val = float(clean.mean())
    std_val = float(clean.std(ddof=1)) if len(clean) > 1 else 1.0

    fig = go.Figure()

    # Empirical Density Histogram
    fig.add_trace(
        go.Histogram(
            x=clean,
            histnorm="probability density",
            name="Sample Data",
            marker={"color": "rgba(56, 189, 248, 0.45)", "line": {"color": "#38bdf8", "width": 1}},
            hovertemplate="Range: %{x}<br>Density: %{y:.5f}<extra></extra>",
        )
    )

    # Theoretical Normal Curve
    if std_val > 1e-9:
        x_min, x_max = float(clean.min()), float(clean.max())
        x_range = max(1e-6, x_max - x_min)
        x_grid = np.linspace(x_min - 0.1 * x_range, x_max + 0.1 * x_range, 200)
        norm_pdf = (1.0 / (std_val * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * ((x_grid - mean_val) / std_val) ** 2
        )

        fig.add_trace(
            go.Scatter(
                x=x_grid,
                y=norm_pdf,
                mode="lines",
                name="Theoretical Gaussian",
                line={"color": "#f43f5e", "width": 2.5, "dash": "dash"},
                hovertemplate="Value: %{x:,.2f}<br>Normal PDF: %{y:.5f}<extra></extra>",
            )
        )

    chart_title = title or f"Normality Check: '{var_name}' — {group_name} (vs. Normal Bell Curve)"
    return apply_dark_theme(fig, title=chart_title, x_title=var_name, y_title="Probability Density")


def build_regression_predicted_actual(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    outcome_name: str = "Outcome",
    title: str = "Predicted vs. Actual",
) -> go.Figure:
    """Build a predicted vs actual scatter plot with 45-degree ideal fit line."""
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)

    fig = go.Figure()
    if len(y_t) == 0:
        return apply_dark_theme(fig, title=title)

    min_val = min(float(np.min(y_t)), float(np.min(y_p)))
    max_val = max(float(np.max(y_t)), float(np.max(y_p)))

    # 45-degree ideal fit line
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            name="Ideal (1:1 Fit)",
            line={"color": "#f43f5e", "dash": "dash", "width": 2},
        )
    )

    fig.add_trace(
        go.Scatter(
            x=y_t,
            y=y_p,
            mode="markers",
            name="Observations",
            marker={"color": "rgba(56, 189, 248, 0.7)", "size": 6},
            hovertemplate="Actual: %{x:,.2f}<br>Predicted: %{y:,.2f}<extra></extra>",
        )
    )

    return apply_dark_theme(
        fig,
        title=title,
        x_title=f"Actual {outcome_name}",
        y_title=f"Predicted {outcome_name}",
    )


def build_regression_coef_forest_plot(
    coefficients: list[dict[str, object]],
    title: str = "Regression Coefficients & 95% Confidence Intervals",
) -> go.Figure:
    """Build a horizontal forest plot of regression beta coefficients with 95% CIs."""
    fig = go.Figure()
    if not coefficients:
        return apply_dark_theme(fig, title=title)

    vars_list = [str(c.get("variable", "")) for c in coefficients]
    estimates = [float(c.get("estimate", 0.0)) for c in coefficients]
    ci_lowers = [float(c.get("ci_lower", 0.0)) for c in coefficients]
    ci_uppers = [float(c.get("ci_upper", 0.0)) for c in coefficients]
    p_values = [float(c.get("p_value", 1.0)) for c in coefficients]

    # Invert so first predictor is at top
    vars_list.reverse()
    estimates.reverse()
    ci_lowers.reverse()
    ci_uppers.reverse()
    p_values.reverse()

    fig.add_vline(
        x=0.0,
        line_dash="dash",
        line_color="#94a3b8",
        line_width=1.5,
        annotation_text="Zero Effect (β=0)",
        annotation_position="top left",
    )

    err_plus = [max(0.0, u - e) for e, u in zip(estimates, ci_uppers, strict=False)]
    err_minus = [max(0.0, e - low) for e, low in zip(estimates, ci_lowers, strict=False)]
    colors = ["#10b981" if p < 0.05 else "#94a3b8" for p in p_values]

    fig.add_trace(
        go.Scatter(
            x=estimates,
            y=vars_list,
            error_x={
                "type": "data",
                "symmetric": False,
                "array": err_plus,
                "arrayminus": err_minus,
                "color": "#38bdf8",
                "thickness": 2,
                "width": 6,
            },
            mode="markers",
            marker={"size": 10, "color": colors},
            customdata=p_values,
            hovertemplate="<b>%{y}</b><br>Estimate (β): %{x:+,.3f}<br>p-value: %{customdata:.4f}<extra></extra>",
        )
    )

    return apply_dark_theme(
        fig,
        title=title,
        height=max(280, 50 * len(vars_list)),
        x_title="Coefficient Estimate (β with 95% CI)",
        y_title="Predictor Variable",
    )
