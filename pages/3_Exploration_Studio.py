"""Page 3: Exploration Studio — Interactive Distributions, Correlations, and Grounded EDA Insights."""

import pandas as pd
import plotly.express as px
import streamlit as st
from src.analytics.analysis_models import EDAFinding
from src.analytics.descriptive import (
    compute_categorical_summary,
    compute_numeric_summary,
)
from src.analytics.distributions import (
    calculate_empirical_quantiles,
    diagnose_distribution_shape,
)
from src.analytics.eda_service import EDAService
from src.analytics.grouped_analysis import (
    analyze_grouped_disparity,
)
from src.analytics.interpretation import (
    generate_category_concentration_finding,
    generate_correlation_finding,
    generate_grouped_disparity_finding,
    generate_skewness_finding,
)
from src.analytics.relationships import (
    compute_cramers_v,
    compute_pairwise_correlations,
)
from src.analytics.temporal import analyze_temporal_series
from src.config.settings import get_settings
from src.core.session_state import (
    KEY_CURRENT_DATASET,
    KEY_GENERATED_FINDINGS,
    init_session_state,
    is_dataset_loaded,
)
from src.core.ui_components import (
    render_empty_state_banner,
    render_page_header,
    render_sidebar_status,
)
from src.ingestion.loader import ManagedDataset
from src.ingestion.schema import infer_dataset_schema
from src.visualization.charts import (
    build_bivariate_scatter,
    build_boxplot,
    build_categorical_bar,
    build_correlation_heatmap,
    build_crosstab_heatmap,
    build_grouped_bar,
    build_histogram,
    build_time_series_chart,
    build_violinplot,
)

settings = get_settings()
st.set_page_config(
    page_title=f"{settings.app_name} | Exploration Studio",
    page_icon="📈",
    layout="wide",
)

init_session_state()
render_sidebar_status()

render_page_header(
    title="📈 Exploration Studio",
    subtitle="Interactive distributions, bivariate correlations, segment disparities, and grounded analytical narratives.",
    badge="Module 2 • Exploratory Analysis & Interpretations",
)

if not is_dataset_loaded(st.session_state):
    render_empty_state_banner("Exploration Studio", phase_label="Phase 3")
    st.page_link(
        "pages/2_Data_Quality.py", label="Go to Data Quality Lab to Upload a Dataset", icon="📥"
    )
    st.stop()

# Retrieve active ManagedDataset
dataset_obj = st.session_state.get(KEY_CURRENT_DATASET)
if isinstance(dataset_obj, ManagedDataset):
    raw_df = dataset_obj.working_df
    schemas = dataset_obj.schema_info or infer_dataset_schema(raw_df)
    dataset_name = dataset_obj.filename
elif isinstance(dataset_obj, pd.DataFrame):
    raw_df = dataset_obj
    schemas = infer_dataset_schema(raw_df)
    dataset_name = "Active DataFrame"
else:
    st.error("Invalid dataset representation in session state.")
    st.stop()

if raw_df.empty:
    st.warning("The active working dataset contains 0 records.")
    st.stop()

# ==============================================================================
# INTERACTIVE FILTERS & WORKING SUBSET
# ==============================================================================
with st.sidebar.expander("🔍 Interactive Workspace Filters", expanded=False):
    st.caption("Apply filters to explore specific subsets across all tabs.")
    filter_active = False
    filtered_df = raw_df.copy()

    # Numeric Range Filter
    num_filter_cols = [
        c for c, s in schemas.items() if s.inferred_type == "numeric" and c in raw_df.columns
    ]
    if num_filter_cols:
        sel_num_col = st.selectbox("Filter by Numeric Column:", ["None", *num_filter_cols], index=0)
        if sel_num_col != "None":
            col_min = float(raw_df[sel_num_col].min(skipna=True))
            col_max = float(raw_df[sel_num_col].max(skipna=True))
            if col_min < col_max:
                selected_range = st.slider(
                    f"Range for '{sel_num_col}':",
                    min_value=col_min,
                    max_value=col_max,
                    value=(col_min, col_max),
                )
                filtered_df = filtered_df[
                    (filtered_df[sel_num_col] >= selected_range[0])
                    & (filtered_df[sel_num_col] <= selected_range[1])
                ]
                filter_active = True

    # Categorical Filter
    cat_filter_cols = [
        c
        for c, s in schemas.items()
        if s.inferred_type == "categorical" and 2 <= s.distinct_count <= 25 and c in raw_df.columns
    ]
    if cat_filter_cols:
        sel_cat_col = st.selectbox(
            "Filter by Category Column:", ["None", *cat_filter_cols], index=0
        )
        if sel_cat_col != "None":
            unique_vals = list(raw_df[sel_cat_col].dropna().unique())
            selected_cats = st.multiselect(
                f"Allowed categories in '{sel_cat_col}':",
                options=unique_vals,
                default=unique_vals,
            )
            if len(selected_cats) < len(unique_vals):
                filtered_df = filtered_df[filtered_df[sel_cat_col].isin(selected_cats)]
                filter_active = True

    if filter_active:
        st.info(
            f"Filtered subset: **{len(filtered_df):,}** of **{len(raw_df):,}** rows ({len(filtered_df) / len(raw_df) * 100:.1f}%)"
        )
        if st.button("Reset Filters", use_container_width=True):
            st.rerun()

df = filtered_df

# Column classification
numeric_cols = [
    c
    for c, s in schemas.items()
    if s.inferred_type == "numeric" and not s.is_identifier and c in df.columns
]
categorical_cols = [
    c
    for c, s in schemas.items()
    if s.inferred_type == "categorical" and not s.is_identifier and c in df.columns
]
datetime_cols = [c for c, s in schemas.items() if s.inferred_type == "datetime" and c in df.columns]
identifier_cols = [c for c, s in schemas.items() if s.is_identifier and c in df.columns]


# Helper: Render Grounded 6-Layer Interpretation Card
def render_interpretation_card(finding: EDAFinding) -> None:
    """Render a structured, calculation-grounded 6-layer interpretation card."""
    severity_colors = {
        "critical": "🔴",
        "high": "🟠",
        "medium": "🟡",
        "low": "🔵",
    }
    icon = severity_colors.get(finding.severity, "ℹ️")

    st.markdown(
        f"""
        <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid #334155; border-left: 4px solid #38bdf8;
                    border-radius: 8px; padding: 14px 18px; margin-top: 14px; margin-bottom: 14px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <span style="font-size: 0.95rem; font-weight: 600; color: #f8fafc;">
                    {icon} {finding.title}
                </span>
                <span style="font-size: 0.75rem; text-transform: uppercase; background: #1e293b; color: #94a3b8;
                             padding: 2px 8px; border-radius: 4px; border: 1px solid #475569;">
                    {finding.category.replace("_", " ")}
                </span>
            </div>
            <p style="font-size: 0.88rem; color: #cbd5e1; margin-bottom: 8px; line-height: 1.5;">
                {finding.summary}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("🔍 Deep Interpretation Framework (6 Evidence Layers)", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**1. Observation:**\n\n{finding.observation}")
            st.markdown(f"**2. Quantitative Evidence:**\n\n`{finding.quantitative_evidence}`")
            st.markdown(f"**3. Statistical Context:**\n\n{finding.statistical_context}")
        with c2:
            st.markdown(f"**4. Practical Implication:**\n\n{finding.practical_implication}")
            st.markdown(f"**5. Limitations & Confounders:**\n\n{finding.limitations}")
            st.markdown(f"**6. Recommended Next Investigation:**\n\n{finding.next_investigation}")


# ==============================================================================
# MAIN TABS WORKSPACE
# ==============================================================================
(
    tab_overview,
    tab_var,
    tab_dist,
    tab_corr,
    tab_group,
    tab_multi,
    tab_insights,
    tab_custom,
) = st.tabs(
    [
        "📋 1. Dataset Overview",
        "🔍 2. Variable Explorer",
        "📊 3. Distributions",
        "🔗 4. Correlations",
        "👥 5. Grouped Disparity",
        "🌐 6. Multivariate Studio",
        "💡 7. Insights & Recommendations",
        "🎨 8. Custom Chart Studio",
    ]
)

# ------------------------------------------------------------------------------
# TAB 1: DATASET OVERVIEW
# ------------------------------------------------------------------------------
with tab_overview:
    st.subheader(f"Dataset Structure: {dataset_name}")
    if filter_active:
        st.caption(f"Currently viewing active filtered subset of {len(df):,} records.")

    m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)
    m_col1.metric("Total Rows", f"{len(df):,}")
    m_col2.metric("Total Columns", f"{len(df.columns):,}")
    m_col3.metric("Numeric Features", len(numeric_cols))
    m_col4.metric("Categorical Features", len(categorical_cols))
    m_col5.metric("Datetime Fields", len(datetime_cols))
    m_col6.metric("Identifiers", len(identifier_cols))

    st.markdown("---")
    ov_c1, ov_c2 = st.columns([1, 1])

    with ov_c1:
        st.markdown("#### Structural Integrity & Memory")
        total_cells = df.size
        total_nulls = int(df.isna().sum().sum())
        null_pct = (total_nulls / total_cells * 100) if total_cells > 0 else 0.0
        dup_count = int(df.duplicated().sum())
        mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)

        st.markdown(
            f"""
            - **Memory Consumption:** `{mem_mb:.2f} MB`
            - **Total Missing Cells:** `{total_nulls:,}` ({null_pct:.2f}% of all cells)
            - **Duplicate Rows:** `{dup_count:,}` ({dup_count / len(df) * 100:.2f}%)
            - **Identifier Columns:** `{", ".join(identifier_cols) if identifier_cols else "None detected"}`
            """
        )

        if identifier_cols:
            st.info(
                f"**Identifier Safeguard Active:** Columns `{', '.join(identifier_cols)}` have been tagged as unique record keys. "
                "They are excluded by default from correlation heatmaps and continuous distributions to prevent misleading statistical claims."
            )

    with ov_c2:
        st.markdown("#### Inferred Variable Types")
        type_breakdown = pd.DataFrame(
            [
                {
                    "Column": c,
                    "Inferred Type": s.inferred_type,
                    "Distinct Values": s.distinct_count,
                    "Missing %": f"{s.null_percentage:.1f}%",
                    "Status": "⚠️ High Nulls" if s.null_percentage > 20 else "✅ Normal",
                }
                for c, s in schemas.items()
                if c in df.columns
            ]
        )
        st.dataframe(type_breakdown, use_container_width=True, height=220)

# ------------------------------------------------------------------------------
# TAB 2: VARIABLE EXPLORER
# ------------------------------------------------------------------------------
with tab_var:
    st.subheader("Univariate Inspection & Variable Profiling")
    target_col = st.selectbox(
        "Select a column to inspect:", options=list(df.columns), key="var_explorer_col"
    )

    if target_col:
        col_series = df[target_col]
        col_schema = schemas.get(target_col)
        inferred_type = col_schema.inferred_type if col_schema else "numeric"

        st.caption(
            f"Classification: **{inferred_type.capitalize()}** | Missing: **{col_series.isna().sum():,}** ({col_series.isna().mean() * 100:.1f}%) | Distinct: **{col_series.nunique(dropna=True):,}**"
        )

        if inferred_type == "numeric":
            num_sum = compute_numeric_summary(col_series, target_col)
            shape_diag = diagnose_distribution_shape(col_series)

            # Key metrics
            v_col1, v_col2, v_col3, v_col4, v_col5 = st.columns(5)
            v_col1.metric("Mean", f"{num_sum.mean:,.2f}")
            v_col2.metric("Median", f"{num_sum.median:,.2f}")
            v_col3.metric("Std Dev", f"{num_sum.std:,.2f}")
            v_col4.metric("IQR", f"{num_sum.iqr:,.2f}")
            v_col5.metric("Outliers (Tukey)", f"{num_sum.outlier_count_iqr:,}")

            p_col1, p_col2 = st.columns([3, 2])
            with p_col1:
                hist_fig = build_histogram(col_series, target_col, nbins=30)
                st.plotly_chart(hist_fig, use_container_width=True, key=f"hist_var_{target_col}")
            with p_col2:
                box_fig = build_boxplot(col_series, target_col)
                st.plotly_chart(box_fig, use_container_width=True, key=f"box_var_{target_col}")

            # Interpret skewness
            fnd = generate_skewness_finding(num_sum, shape_diag)
            render_interpretation_card(fnd)

        elif inferred_type == "categorical":
            cat_sum = compute_categorical_summary(col_series, target_col)

            c_col1, c_col2, c_col3 = st.columns(3)
            c_col1.metric("Unique Categories", f"{cat_sum.distinct_count:,}")
            c_col2.metric("Dominant Category Share", f"{cat_sum.top_1_concentration_pct:.1f}%")
            c_col3.metric("Rare Categories (<1%)", f"{cat_sum.rare_categories_count:,}")

            bar_fig = build_categorical_bar(cat_sum.top_categories, target_col)
            st.plotly_chart(bar_fig, use_container_width=True, key=f"bar_cat_{target_col}")

            fnd_cat = generate_category_concentration_finding(cat_sum)
            render_interpretation_card(fnd_cat)

        elif inferred_type == "datetime":
            temp_res = analyze_temporal_series(df, target_col)
            d_col1, d_col2 = st.columns(2)
            d_col1.metric("Earliest Date", temp_res.earliest_date or "N/A")
            d_col2.metric("Latest Date", temp_res.latest_date or "N/A")

            # Check if there's a numeric measure to plot over time
            if numeric_cols:
                meas = st.selectbox("Measure to trend over time:", numeric_cols, key="dt_measure")
                daily_agg = (
                    df.groupby(pd.to_datetime(df[target_col], errors="coerce").dt.date)[meas]
                    .mean()
                    .reset_index()
                )
                daily_agg.columns = ["date", meas]
                ts_fig = build_time_series_chart(daily_agg, "date", meas, agg_name="mean")
                st.plotly_chart(
                    ts_fig, use_container_width=True, key=f"ts_plot_{target_col}_{meas}"
                )

        elif col_schema and col_schema.is_identifier:
            st.info(
                f"Column '{target_col}' is an identifier column. Uniqueness ratio: {col_schema.unique_ratio * 100:.1f}%. Naive statistics are suppressed."
            )

# ------------------------------------------------------------------------------
# TAB 3: DISTRIBUTION ANALYSIS
# ------------------------------------------------------------------------------
with tab_dist:
    st.subheader("Numeric Distribution Diagnostics & Shape Testing")
    if not numeric_cols:
        st.info("No continuous numeric variables detected for distribution testing.")
    else:
        dist_c1, dist_c2 = st.columns([1, 3])
        with dist_c1:
            dist_col = st.selectbox(
                "Select numeric variable:", options=numeric_cols, key="dist_col_sel"
            )
            bin_count = st.slider("Histogram Bins:", min_value=10, max_value=80, value=30, step=5)
            chart_mode = st.radio(
                "Visualization Mode:",
                ["Histogram + Markers", "Violin Density", "Empirical Quantiles"],
                index=0,
            )

        with dist_c2:
            s_clean = pd.to_numeric(df[dist_col], errors="coerce").dropna()
            num_sum = compute_numeric_summary(s_clean, dist_col)
            shape_diag = diagnose_distribution_shape(s_clean)

            if chart_mode == "Histogram + Markers":
                fig = build_histogram(s_clean, dist_col, nbins=bin_count)
                st.plotly_chart(
                    fig, use_container_width=True, key=f"dist_hist_{dist_col}_{bin_count}"
                )
            elif chart_mode == "Violin Density":
                fig = build_violinplot(s_clean, dist_col)
                st.plotly_chart(fig, use_container_width=True, key=f"dist_violin_{dist_col}")
            else:
                q_df = calculate_empirical_quantiles(s_clean)
                st.markdown(f"##### Empirical Quantiles — {dist_col}")
                st.dataframe(q_df, use_container_width=True)

        # 6-Point Distribution Diagnostics Box
        st.markdown("#### Distribution Diagnostics & Central Tendency Advisory")
        d_inf1, d_inf2 = st.columns(2)
        with d_inf1:
            skew_val = num_sum.skewness or 0.0
            kurt_val = num_sum.kurtosis or 0.0
            st.markdown(
                f"""
                - **Concentration:** IQR spans `[{num_sum.q25:,.2f}, {num_sum.q75:,.2f}]` (middle 50% of records).
                - **Skewness:** `{skew_val:+.2f}` — *{shape_diag.get("skewness_category", "symmetric").capitalize()}*.
                - **Kurtosis:** `{kurt_val:+.2f}` — *{shape_diag.get("tail_behavior", "normal tails").capitalize()}*.
                """
            )
        with d_inf2:
            pref_metric = shape_diag.get("preferred_central_metric", "median")
            div_pct = shape_diag.get("mean_median_divergence_pct", 0.0)
            st.markdown(
                f"""
                - **Mean vs. Median Divergence:** `{div_pct:.1f}%` difference.
                - **Advisory:** The **{pref_metric.upper()}** is the statistically sound metric of central tendency for this variable.
                - **Downstream Modeling:** Log transformation or robust scaling is recommended if used in linear models.
                """
            )

        fnd_dist = generate_skewness_finding(num_sum, shape_diag)
        render_interpretation_card(fnd_dist)

# ------------------------------------------------------------------------------
# TAB 4: CORRELATIONS & RELATIONSHIPS
# ------------------------------------------------------------------------------
with tab_corr:
    st.subheader("Bivariate Relationships & Correlation Intelligence")

    if len(numeric_cols) < 2:
        st.info("At least two continuous numeric variables are required for correlation analysis.")
    else:
        corr_c1, corr_c2 = st.columns([1, 2])
        with corr_c1:
            corr_method = st.radio(
                "Correlation Method:", ["pearson", "spearman"], index=0, horizontal=True
            )
            st.caption(
                "**Pearson:** Measures linear association (sensitive to outliers).\n\n"
                "**Spearman:** Measures monotonic rank association (robust to non-linear rank shifts)."
            )
            # Exclude identifiers and constants
            corr_features = st.multiselect(
                "Included numeric features:",
                options=numeric_cols,
                default=numeric_cols[:8],
            )

        with corr_c2:
            if len(corr_features) >= 2:
                corr_matrix = df[corr_features].corr(method=corr_method)
                heat_fig = build_correlation_heatmap(
                    corr_matrix, title=f"{corr_method.capitalize()} Correlation Heatmap"
                )
                st.plotly_chart(heat_fig, use_container_width=True, key=f"corr_heat_{corr_method}")

        st.markdown("---")
        st.markdown("#### Ranked Pairwise Associations & Bivariate Scatter")

        ranked_pairs = compute_pairwise_correlations(
            df, numeric_cols, method=corr_method, min_sample=10
        )
        if ranked_pairs:
            rp_c1, rp_c2 = st.columns([1, 1])
            with rp_c1:
                st.markdown(f"##### Top Pairwise Associations ({corr_method.capitalize()})")
                pair_data = [
                    {
                        "Variable 1": p.var1,
                        "Variable 2": p.var2,
                        "Coefficient": f"{p.coefficient:+.3f}",
                        "Strength": p.strength.capitalize(),
                        "N": f"{p.sample_size:,}",
                        "p-value": f"{p.p_value:.4f}" if p.p_value is not None else "N/A",
                    }
                    for p in ranked_pairs[:6]
                ]
                st.dataframe(pd.DataFrame(pair_data), use_container_width=True)

            with rp_c2:
                top_p = ranked_pairs[0]
                sel_pair_str = st.selectbox(
                    "Examine Pair in Scatter Plot:",
                    options=[f"{p.var1} × {p.var2}" for p in ranked_pairs[:8]],
                    index=0,
                )
                p_v1, p_v2 = sel_pair_str.split(" × ")
                add_trend = st.checkbox("Show Descriptive Trendline (OLS)", value=True)
                scatter_fig, was_sampled = build_bivariate_scatter(
                    df, p_v1, p_v2, add_trendline=add_trend
                )
                st.plotly_chart(scatter_fig, use_container_width=True, key=f"scatter_{p_v1}_{p_v2}")

            # Grounded 6-layer finding for top correlation
            fnd_corr = generate_correlation_finding(top_p)
            render_interpretation_card(fnd_corr)

        # Categorical Association alternative (Cramér's V)
        if len(categorical_cols) >= 2:
            st.markdown("---")
            st.markdown("#### Categorical Association (Cramér's V)")
            cat_c1, cat_c2 = st.columns(2)
            with cat_c1:
                c_v1 = st.selectbox("Category A:", categorical_cols, index=0)
            with cat_c2:
                c_v2 = st.selectbox(
                    "Category B:", [c for c in categorical_cols if c != c_v1], index=0
                )

            c_v_score, contingency_df = compute_cramers_v(df, c_v1, c_v2)
            st.info(
                f"**Cramér's V Association:** `{c_v_score:.3f}` (0.0 = complete independence, 1.0 = perfect co-occurrence)."
            )
            if not contingency_df.empty:
                ct_fig = build_crosstab_heatmap(contingency_df, c_v1, c_v2)
                st.plotly_chart(ct_fig, use_container_width=True, key=f"ct_fig_{c_v1}_{c_v2}")

# ------------------------------------------------------------------------------
# TAB 5: GROUPED ANALYSIS
# ------------------------------------------------------------------------------
with tab_group:
    st.subheader("Subgroup Disparity & Segment Comparison")
    if not categorical_cols or not numeric_cols:
        st.info(
            "Grouped analysis requires at least one categorical dimension and one quantitative measure."
        )
    else:
        grp_c1, grp_c2, grp_c3 = st.columns(3)
        with grp_c1:
            grp_col_sel = st.selectbox("Group by Dimension:", categorical_cols, key="grp_col_sel")
        with grp_c2:
            grp_meas_sel = st.selectbox("Quantitative Measure:", numeric_cols, key="grp_meas_sel")
        with grp_c3:
            grp_agg_sel = st.selectbox(
                "Aggregation Function:", ["mean", "median", "sum", "count", "std"], index=0
            )

        grp_result = analyze_grouped_disparity(df, grp_col_sel, grp_meas_sel, agg=grp_agg_sel)

        if grp_result and grp_result.groups:
            r_c1, r_c2 = st.columns([3, 2])
            with r_c1:
                grp_fig = build_grouped_bar(
                    grp_result.groups, grp_col_sel, grp_meas_sel, grp_agg_sel
                )
                st.plotly_chart(
                    grp_fig, use_container_width=True, key=f"grp_fig_{grp_col_sel}_{grp_meas_sel}"
                )

            with r_c2:
                st.markdown("##### Ranked Segment Metrics")
                grp_tbl = pd.DataFrame(
                    [
                        {
                            "Segment": g["group"],
                            f"{grp_agg_sel.capitalize()}": f"{g['value']:,.2f}",
                            "Records (N)": f"{g['sample_size']:,}",
                            "Share %": f"{g['share_pct']:.1f}%",
                        }
                        for g in grp_result.groups
                    ]
                )
                st.dataframe(grp_tbl, use_container_width=True, height=320)

            # 6-layer finding for group disparity
            fnd_grp = generate_grouped_disparity_finding(grp_result)
            render_interpretation_card(fnd_grp)

# ------------------------------------------------------------------------------
# TAB 6: MULTIVARIATE STUDIO
# ------------------------------------------------------------------------------
with tab_multi:
    st.subheader("Multivariate Exploration Studio")
    st.caption(
        "Visually inspect combinations of 3+ continuous and discrete dimensions simultaneously."
    )

    if len(numeric_cols) < 2:
        st.info("At least two continuous variables required for multivariate scatter plotting.")
    else:
        m_ctl1, m_ctl2, m_ctl3, m_ctl4 = st.columns(4)
        with m_ctl1:
            mx_col = st.selectbox("X-Axis Feature:", numeric_cols, index=0)
        with m_ctl2:
            my_col = st.selectbox(
                "Y-Axis Feature:", [c for c in numeric_cols if c != mx_col], index=0
            )
        with m_ctl3:
            m_color = st.selectbox(
                "Color Grouping (Optional):", ["None", *categorical_cols], index=0
            )
        with m_ctl4:
            m_size = st.selectbox(
                "Size Encoding (Optional):",
                ["None", *[c for c in numeric_cols if c not in (mx_col, my_col)]],
                index=0,
            )

        color_arg = None if m_color == "None" else m_color
        size_arg = None if m_size == "None" else m_size

        multi_fig, was_sampled = build_bivariate_scatter(
            df,
            mx_col,
            my_col,
            color_col=color_arg,
            size_col=size_arg,
            sample_limit=2500,
        )
        st.plotly_chart(multi_fig, use_container_width=True, key=f"multi_fig_{mx_col}_{my_col}")

        if was_sampled:
            st.caption(
                "⚡ **Performance Safeguard:** Visual sampled to 2,500 records to maintain responsive client rendering."
            )

# ------------------------------------------------------------------------------
# TAB 7: AUTOMATED INSIGHTS & CHART RECOMMENDATIONS
# ------------------------------------------------------------------------------
with tab_insights:
    st.subheader("Automated Dataset Findings & Deterministic Chart Recommendations")

    # Generate EDA Summary via EDAService
    if isinstance(dataset_obj, ManagedDataset):
        eda_summary = EDAService.generate_eda_summary(dataset_obj)
    else:
        # Wrap standalone df
        managed_temp = ManagedDataset(
            dataset_id="temp",
            filename=dataset_name,
            file_format="dataframe",
            file_size_bytes=int(df.memory_usage().sum()),
            fingerprint="temp",
            raw_df=df,
            working_df=df,
            schema_info=schemas,
            ingestion_report=None,  # type: ignore[arg-type]
        )
        eda_summary = EDAService.generate_eda_summary(managed_temp)

    # Store generated findings in session state for cross-module access
    st.session_state[KEY_GENERATED_FINDINGS] = eda_summary.findings

    st.markdown("#### 1. Grounded Exploratory Findings")
    if not eda_summary.findings:
        st.info(
            "No notable distributional anomalies, severe skewness, or extreme correlations detected."
        )
    else:
        for fnd in eda_summary.findings:
            render_interpretation_card(fnd)

    st.markdown("---")
    st.markdown("#### 2. Deterministic Chart Recommendations")
    st.caption(
        "Visualizations scored and ranked based on variable cardinality, statistical data types, and analytical objectives."
    )

    if not eda_summary.recommendations:
        st.info("No specific chart recommendations generated for this schema.")
    else:
        for rec in eda_summary.recommendations:
            st.markdown(
                f"""
                <div style="background: rgba(15, 23, 42, 0.5); border: 1px solid #334155; border-radius: 8px;
                            padding: 12px 16px; margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                        <span style="font-weight: 600; color: #38bdf8; font-size: 0.95rem;">
                            📊 {rec.title}
                        </span>
                        <span style="background: #0284c7; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: bold;">
                            Suitability Score: {rec.suitability_score:.0f}/100
                        </span>
                    </div>
                    <p style="font-size: 0.85rem; color: #cbd5e1; margin-bottom: 6px;">
                        <strong>Purpose:</strong> {rec.purpose}
                    </p>
                    <p style="font-size: 0.8rem; color: #94a3b8; margin-bottom: 2px;">
                        <strong>Why Appropriate:</strong> {rec.why_appropriate}
                    </p>
                    <p style="font-size: 0.8rem; color: #f59e0b; margin-bottom: 2px;">
                        <strong>Limitations:</strong> {rec.limitations}
                    </p>
                    <p style="font-size: 0.8rem; color: #10b981; margin-bottom: 0px;">
                        <strong>Suggested Follow-up:</strong> {rec.suggested_followup}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ------------------------------------------------------------------------------
# TAB 8: CUSTOM CHART & INSIGHT STUDIO
# ------------------------------------------------------------------------------
with tab_custom:
    st.subheader("🎨 Custom Chart & Insight Studio")
    st.caption(
        "Interactively configure, visualize, and analyze any relationship across your dataset with automated statistical interpretations."
    )

    all_cols = list(df.columns)
    c_builder, c_display = st.columns([1, 2])

    with c_builder:
        st.markdown("##### ⚙️ Chart Configuration")
        chart_type = st.selectbox(
            "Select Chart Type:",
            [
                "Bar Chart (Aggregated / Category)",
                "Scatter / Bubble Plot",
                "Line Chart (Trend / Temporal)",
                "Histogram / Frequency",
                "Box Plot (Segment Comparison)",
                "Violin Plot (Kernel Density)",
                "Pie / Donut Chart",
                "Treemap (Hierarchical Slices)",
                "2D Density Heatmap",
            ],
            key="custom_chart_type",
        )

        x_col = st.selectbox(
            "X-Axis / Category Column:", options=all_cols, index=0, key="custom_x_col"
        )

        # Y column selector depending on chart
        y_options = ["None", *all_cols]
        default_y_idx = 1 if len(numeric_cols) > 0 and numeric_cols[0] in y_options else 0
        y_col = st.selectbox(
            "Y-Axis / Measure Column:",
            options=y_options,
            index=default_y_idx,
            key="custom_y_col",
        )

        # Aggregation selector if Bar, Line, Pie, or Treemap
        agg_func = "None (Raw Values)"
        if chart_type in (
            "Bar Chart (Aggregated / Category)",
            "Line Chart (Trend / Temporal)",
            "Pie / Donut Chart",
            "Treemap (Hierarchical Slices)",
        ):
            if y_col != "None":
                agg_func = st.selectbox(
                    "Aggregation Function for Y:",
                    ["Sum", "Mean", "Median", "Count", "Min", "Max", "None (Raw Values)"],
                    index=1,
                    key="custom_agg_func",
                )
            else:
                agg_func = "Count"
                st.caption("ℹ️ No Y column selected: Plotting record counts per category.")

        # Optional color breakdown
        color_choices = [
            "None",
            *categorical_cols,
            *[c for c in numeric_cols if c not in (x_col, y_col)],
        ]
        color_col = st.selectbox(
            "Color / Grouping Dimension (Optional):",
            color_choices,
            index=0,
            key="custom_color_col",
        )

        # Optional size for scatter
        size_col = "None"
        if "Scatter" in chart_type:
            size_choices = ["None", *numeric_cols]
            size_col = st.selectbox(
                "Bubble Size Encoding (Optional):",
                size_choices,
                index=0,
                key="custom_size_col",
            )

        # Optional facet column
        facet_choices = [
            "None",
            *[c for c in categorical_cols if df[c].nunique() <= 8 and c != x_col],
        ]
        facet_col = st.selectbox(
            "Subplot Grid / Facet (Optional):",
            facet_choices,
            index=0,
            key="custom_facet_col",
        )

        # Visual styling
        with st.expander("🎨 Styling & Display Options", expanded=False):
            custom_title = st.text_input(
                "Chart Title:",
                value=f"{chart_type.split(' (')[0]}: {x_col}"
                + (f" vs {y_col}" if y_col != "None" else ""),
                key="custom_title_input",
            )
            color_theme = st.selectbox(
                "Color Palette:",
                ["Plotly Dark", "Viridis", "Blues", "Teal", "Plasma", "Sunsetdark", "RdBu_r"],
                index=0,
                key="custom_color_theme",
            )
            log_y = st.checkbox("Log scale Y-axis", value=False, key="custom_log_y")
            show_trend = False
            if "Scatter" in chart_type and y_col != "None":
                show_trend = st.checkbox("Fit OLS Trendline", value=True, key="custom_show_trend")
            orientation = "v"
            if "Bar" in chart_type or "Box" in chart_type:
                is_horiz = st.checkbox("Horizontal Orientation", value=False, key="custom_horiz")
                if is_horiz:
                    orientation = "h"

    with c_display:
        plot_df = df.copy()
        c_arg = None if color_col == "None" else color_col
        s_arg = None if size_col == "None" else size_col
        f_arg = None if facet_col == "None" else facet_col

        try:
            if agg_func not in ("None (Raw Values)", "None"):
                group_keys = [x_col]
                if c_arg and c_arg != x_col and c_arg in categorical_cols:
                    group_keys.append(c_arg)
                if f_arg and f_arg not in group_keys:
                    group_keys.append(f_arg)

                if agg_func == "Count":
                    plot_df = (
                        plot_df.groupby(group_keys, as_index=False)
                        .size()
                        .rename(columns={"size": "Record_Count"})
                    )
                    y_metric = "Record_Count"
                else:
                    if y_col != "None" and pd.api.types.is_numeric_dtype(plot_df[y_col]):
                        agg_map = {
                            "Sum": "sum",
                            "Mean": "mean",
                            "Median": "median",
                            "Min": "min",
                            "Max": "max",
                        }
                        func = agg_map.get(agg_func, "mean")
                        plot_df = plot_df.groupby(group_keys, as_index=False)[y_col].agg(func)
                        y_metric = y_col
                    else:
                        plot_df = (
                            plot_df.groupby(group_keys, as_index=False)
                            .size()
                            .rename(columns={"size": "Record_Count"})
                        )
                        y_metric = "Record_Count"
            else:
                y_metric = y_col if y_col != "None" else None

            # Render Chart by Type
            fig_custom = None
            template_name = "plotly_dark"
            palette_map = {
                "Plotly Dark": None,
                "Viridis": px.colors.sequential.Viridis,
                "Blues": px.colors.sequential.Blues,
                "Teal": px.colors.sequential.Teal,
                "Plasma": px.colors.sequential.Plasma,
                "Sunsetdark": px.colors.sequential.Sunsetdark,
                "RdBu_r": px.colors.diverging.RdBu_r,
            }
            colors_seq = palette_map.get(color_theme)

            if "Bar" in chart_type:
                fig_custom = px.bar(
                    plot_df,
                    x=x_col if orientation == "v" else y_metric,
                    y=y_metric if orientation == "v" else x_col,
                    color=c_arg,
                    facet_col=f_arg,
                    orientation=orientation,
                    title=custom_title,
                    template=template_name,
                    color_discrete_sequence=colors_seq,
                    color_continuous_scale=colors_seq,
                    log_y=log_y if orientation == "v" else False,
                    log_x=log_y if orientation == "h" else False,
                )
            elif "Scatter" in chart_type:
                if y_metric:
                    s_limit = min(len(plot_df), 3000)
                    sample_p = (
                        plot_df.sample(s_limit, random_state=42)
                        if len(plot_df) > s_limit
                        else plot_df
                    )
                    fig_custom = px.scatter(
                        sample_p,
                        x=x_col,
                        y=y_metric,
                        color=c_arg,
                        size=s_arg,
                        facet_col=f_arg,
                        trendline="ols"
                        if show_trend
                        and pd.api.types.is_numeric_dtype(sample_p[x_col])
                        and pd.api.types.is_numeric_dtype(sample_p[y_metric])
                        else None,
                        title=custom_title,
                        template=template_name,
                        color_discrete_sequence=colors_seq,
                        color_continuous_scale=colors_seq,
                        log_y=log_y,
                    )
                else:
                    st.warning("Scatter plot requires a Y-Axis column.")
            elif "Line" in chart_type:
                fig_custom = px.line(
                    plot_df,
                    x=x_col,
                    y=y_metric,
                    color=c_arg,
                    facet_col=f_arg,
                    title=custom_title,
                    template=template_name,
                    color_discrete_sequence=colors_seq,
                    log_y=log_y,
                )
            elif "Histogram" in chart_type:
                fig_custom = px.histogram(
                    plot_df,
                    x=x_col,
                    y=y_metric if y_metric and y_metric != "Record_Count" else None,
                    color=c_arg,
                    facet_col=f_arg,
                    marginal="box",
                    title=custom_title,
                    template=template_name,
                    color_discrete_sequence=colors_seq,
                    log_y=log_y,
                )
            elif "Box" in chart_type:
                fig_custom = px.box(
                    plot_df,
                    x=x_col if orientation == "v" else y_metric,
                    y=y_metric if orientation == "v" else x_col,
                    color=c_arg,
                    facet_col=f_arg,
                    orientation=orientation,
                    points="outliers",
                    title=custom_title,
                    template=template_name,
                    color_discrete_sequence=colors_seq,
                    log_y=log_y if orientation == "v" else False,
                )
            elif "Violin" in chart_type:
                fig_custom = px.violin(
                    plot_df,
                    x=x_col,
                    y=y_metric,
                    color=c_arg,
                    box=True,
                    points="outliers",
                    title=custom_title,
                    template=template_name,
                    color_discrete_sequence=colors_seq,
                    log_y=log_y,
                )
            elif "Pie" in chart_type:
                fig_custom = px.pie(
                    plot_df,
                    names=x_col,
                    values=y_metric if y_metric else None,
                    title=custom_title,
                    template=template_name,
                    color_discrete_sequence=colors_seq,
                    hole=0.4,
                )
            elif "Treemap" in chart_type:
                path_cols = [x_col]
                if c_arg and c_arg != x_col and c_arg in categorical_cols:
                    path_cols.append(c_arg)
                fig_custom = px.treemap(
                    plot_df,
                    path=path_cols,
                    values=y_metric if y_metric and y_metric != x_col else None,
                    title=custom_title,
                    template=template_name,
                    color_continuous_scale=colors_seq,
                )
            elif "2D Density" in chart_type:
                if y_metric:
                    fig_custom = px.density_heatmap(
                        plot_df,
                        x=x_col,
                        y=y_metric,
                        facet_col=f_arg,
                        title=custom_title,
                        template=template_name,
                        color_continuous_scale=colors_seq or "Viridis",
                    )
                else:
                    st.warning("2D Density Heatmap requires both X and Y columns.")

            if fig_custom is not None:
                fig_custom.update_layout(height=420, margin={"l": 20, "r": 20, "t": 50, "b": 30})
                st.plotly_chart(
                    fig_custom,
                    use_container_width=True,
                    key=f"cust_canvas_{chart_type[:4]}_{x_col}_{y_col}_{agg_func}_{color_col}",
                )

                # Statistical Insights Card
                st.markdown("#### 💡 Automated Analytical Insights")
                ins_c1, ins_c2, ins_c3, ins_c4 = st.columns(4)
                ins_c1.metric("Plotted Rows", f"{len(plot_df):,}")
                ins_c2.metric("Distinct X Values", f"{plot_df[x_col].nunique():,}")
                if (
                    y_metric
                    and y_metric in plot_df.columns
                    and pd.api.types.is_numeric_dtype(plot_df[y_metric])
                ):
                    y_s = plot_df[y_metric].dropna()
                    ins_c3.metric(f"Mean ({y_metric})", f"{y_s.mean():,.2f}")
                    ins_c4.metric(f"Sum ({y_metric})", f"{y_s.sum():,.2f}")
                else:
                    ins_c3.metric("Completeness", f"{(1 - df[x_col].isna().mean()) * 100:.1f}%")
                    ins_c4.metric(
                        "Memory Used", f"{plot_df.memory_usage(deep=True).sum() / 1024:.1f} KB"
                    )

                # Contextual Interpretation
                insight_text = ""
                if (
                    "Scatter" in chart_type
                    and y_metric
                    and pd.api.types.is_numeric_dtype(df[x_col])
                    and pd.api.types.is_numeric_dtype(df[y_metric])
                ):
                    clean_biv = df[[x_col, y_metric]].dropna()
                    if len(clean_biv) >= 3:
                        r_val = float(clean_biv[x_col].corr(clean_biv[y_metric]))
                        direction = "positive" if r_val > 0 else "negative"
                        strength = (
                            "strong"
                            if abs(r_val) >= 0.7
                            else ("moderate" if abs(r_val) >= 0.4 else "weak")
                        )
                        insight_text = (
                            f"**Correlation Analysis:** {x_col} and {y_metric} show a **{strength} {direction} correlation** "
                            f"($r = {r_val:+.3f}$, $R^2 = {r_val**2:.3f}$). Approximately **{r_val**2 * 100:.1f}%** of the variation in {y_metric} is statistically explained by {x_col}."
                        )
                elif (
                    agg_func not in ("None (Raw Values)", "None")
                    and y_metric
                    and pd.api.types.is_numeric_dtype(plot_df[y_metric])
                    and len(plot_df) >= 2
                ):
                    top_row = plot_df.sort_values(by=y_metric, ascending=False).iloc[0]
                    bot_row = plot_df.sort_values(by=y_metric, ascending=True).iloc[0]
                    top_val = float(top_row[y_metric])
                    bot_val = float(bot_row[y_metric])
                    ratio = (top_val / bot_val) if bot_val > 0 else 0.0
                    ratio_str = f" by a factor of **{ratio:.1f}×**." if ratio > 0 else "."
                    insight_text = (
                        f"**Segment Disparity:** The leading segment **'{top_row[x_col]}'** reaches **{top_val:,.2f}** "
                        f"({y_metric} {agg_func}), outperforming **'{bot_row[x_col]}'** ({bot_val:,.2f}){ratio_str}"
                    )
                elif "Histogram" in chart_type or "Box" in chart_type:
                    if pd.api.types.is_numeric_dtype(df[x_col]):
                        s_num = df[x_col].dropna()
                        skew = float(s_num.skew())
                        q25, med, q75 = s_num.quantile(0.25), s_num.median(), s_num.quantile(0.75)
                        shape_str = (
                            "right-skewed"
                            if skew > 0.5
                            else ("left-skewed" if skew < -0.5 else "symmetric")
                        )
                        insight_text = (
                            f"**Distribution Properties:** Median is **{med:,.2f}** with IQR spanning `[{q25:,.2f}, {q75:,.2f}]`. "
                            f"Distribution exhibits a **{shape_str}** shape (skewness = {skew:+.2f})."
                        )

                if insight_text:
                    st.info(insight_text)

                # Action buttons: Data download & Pin to Canvas
                st.markdown("---")
                btn_c1, btn_c2 = st.columns(2)
                with btn_c1:
                    csv_bytes = plot_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download Plotted Data (CSV)",
                        data=csv_bytes,
                        file_name=f"custom_plot_{x_col}_{y_metric or 'count'}.csv",
                        mime="text/csv",
                        key="btn_dl_custom_plot_data",
                    )
                with btn_c2:
                    if "saved_custom_charts" not in st.session_state:
                        st.session_state["saved_custom_charts"] = []

                    if st.button("📌 Pin this Chart to Saved Canvas", key="btn_pin_chart"):
                        st.session_state["saved_custom_charts"].append(
                            {
                                "title": custom_title,
                                "type": chart_type,
                                "summary": insight_text
                                or f"Plotted {x_col}"
                                + (f" against {y_metric}" if y_metric else ""),
                                "timestamp": pd.Timestamp.now().strftime("%H:%M:%S"),
                            }
                        )
                        st.success(f"Pinned '{custom_title}' to Canvas below!")
                        st.rerun()

        except Exception as ex:
            st.error(f"Could not render visual: {ex}")

    # Saved Charts Gallery
    if st.session_state.get("saved_custom_charts"):
        st.markdown("---")
        st.markdown("### 📌 Pinned Charts & Visual Canvas")
        st.caption(
            "Custom visual collection pinned during your active analytical exploration session."
        )
        gallery_cols = st.columns(min(3, len(st.session_state["saved_custom_charts"])))
        for idx, saved in enumerate(st.session_state["saved_custom_charts"]):
            col_idx = idx % len(gallery_cols)
            with gallery_cols[col_idx]:
                st.markdown(
                    f"""
                    <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid #475569; border-radius: 8px; padding: 12px; margin-bottom: 12px;">
                        <h5 style="color: #38bdf8; margin: 0 0 6px 0;">📊 {saved["title"]}</h5>
                        <p style="font-size: 0.8rem; color: #94a3b8; margin: 0 0 4px 0;"><strong>Type:</strong> {saved["type"]} • <em>Pinned at {saved["timestamp"]}</em></p>
                        <p style="font-size: 0.82rem; color: #e2e8f0; margin: 0;">{saved["summary"]}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        if st.button("🗑️ Clear Pinned Visuals Canvas", key="btn_clear_canvas"):
            st.session_state["saved_custom_charts"] = []
            st.rerun()

st.markdown("---")
st.page_link("app.py", label="← Return to Main Workspace", icon="🏠")
