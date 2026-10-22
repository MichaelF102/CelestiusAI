"""Page 4: Statistical Intelligence — Formal Hypothesis Testing, Effect Sizes, and Diagnostics."""

import pandas as pd
import streamlit as st
from src.analytics.assumption_checks import generate_qq_plot_data
from src.analytics.categorical_tests import run_chi_square_test
from src.analytics.correlation_inference import run_correlation_test
from src.analytics.group_comparisons import (
    run_multi_group_anova,
    run_paired_group_test,
    run_two_group_independent_test,
)
from src.analytics.multiple_testing import apply_multiple_testing_correction
from src.analytics.regression_inference import fit_ols_regression
from src.analytics.robustness import (
    evaluate_correlation_robustness,
    evaluate_two_group_robustness,
)
from src.analytics.statistical_interpretation import (
    render_regression_interpretation_markdown,
    render_statistical_interpretation_markdown,
)
from src.analytics.statistical_models import (
    HypothesisDirection,
    RegressionModelResult,
    StatisticalTestResult,
)
from src.analytics.statistical_service import StatisticalService
from src.config.settings import get_settings
from src.core.session_state import (
    KEY_CURRENT_DATASET,
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
    build_qq_plot,
)

settings = get_settings()
st.set_page_config(
    page_title=f"{settings.app_name} | Statistical Intelligence",
    page_icon="🧪",
    layout="wide",
)

init_session_state()
render_sidebar_status()

render_page_header(
    title="🧪 Statistical Intelligence",
    subtitle="Formal hypothesis testing, effect sizes, uncertainty quantification, and rigorous assumption verification.",
    badge="Module 3 • Statistical Inference",
)

if not is_dataset_loaded(st.session_state):
    render_empty_state_banner("Statistical Intelligence", phase_label="Phase 4")
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

df = raw_df

# Column classification
numeric_cols = [
    c
    for c, s in schemas.items()
    if s.inferred_type == "numeric" and not s.is_identifier and c in df.columns
]
categorical_cols = [
    c
    for c, s in schemas.items()
    if s.inferred_type == "categorical"
    and not s.is_identifier
    and 2 <= s.distinct_count <= 25
    and c in df.columns
]

# Session state key for statistical history
if "statistical_history" not in st.session_state:
    st.session_state["statistical_history"] = []

# ==============================================================================
# SECTION A: ANALYSIS SETUP
# ==============================================================================
st.subheader("1. Analysis Setup & Research Question")

col_mode, col_conf, col_alpha = st.columns([2, 1, 1])
with col_mode:
    analysis_type = st.selectbox(
        "Select Analytical Objective:",
        options=[
            ("two_group", "👥 Two-Group Comparison (Independent Samples)"),
            ("paired", "🔁 Paired Group Comparison (Repeated Measures)"),
            ("multi_group", "📊 Multi-Group Comparison (3+ Groups / ANOVA)"),
            ("categorical", "🏷️ Categorical Association (Chi-Square / Fisher)"),
            ("correlation", "🔗 Correlation Inference (Pearson / Spearman)"),
            ("regression", "📈 Ordinary Least Squares (OLS) Regression"),
        ],
        format_func=lambda opt: opt[1],
        index=0,
    )[0]

with col_conf:
    conf_level = st.selectbox(
        "Confidence Level:", [0.95, 0.90, 0.99], format_func=lambda x: f"{int(x * 100)}%", index=0
    )

with col_alpha:
    alpha_threshold = st.selectbox(
        "Significance Alpha (α):", [0.05, 0.01, 0.10], format_func=lambda x: f"α = {x}", index=0
    )

# Dynamic Variable Selectors
test_result: StatisticalTestResult | None = None
reg_result: RegressionModelResult | None = None

if analysis_type == "two_group":
    if not categorical_cols or not numeric_cols:
        st.warning(
            "Two-group comparison requires at least one categorical variable and one numeric variable."
        )
        st.stop()

    c_grp, c_meas, c_dir = st.columns(3)
    with c_grp:
        grp_col = st.selectbox("Grouping Variable (2 Categories):", categorical_cols, key="tg_grp")
        avail_cats = list(df[grp_col].dropna().astype(str).unique())
    with c_meas:
        meas_col = st.selectbox("Continuous Outcome Variable:", numeric_cols, key="tg_meas")
    with c_dir:
        direction: HypothesisDirection = st.selectbox(
            "Hypothesis Direction:",
            options=["two-sided", "greater", "less"],
            index=0,
        )

    g1_default = avail_cats[0] if len(avail_cats) > 0 else "Group 1"
    g2_default = (
        avail_cats[1] if len(avail_cats) > 1 else (avail_cats[0] if avail_cats else "Group 2")
    )

    cg1, cg2 = st.columns(2)
    with cg1:
        g1_name = st.selectbox("Group 1 (Baseline):", avail_cats, index=0)
    with cg2:
        g2_name = st.selectbox(
            "Group 2 (Comparison):", [c for c in avail_cats if c != g1_name] or avail_cats, index=0
        )

    # Test recommendation
    recommendation = StatisticalService.recommend_test("two_group", sample_size=len(df))
    st.info(
        f"💡 **Recommended Test:** {recommendation.recommended_test}\n\n*{recommendation.rationale}*"
    )

    override_method = st.selectbox(
        "Selected Test Method:", ["welch", "student", "mann_whitney"], index=0
    )

    if st.button("Run Statistical Analysis", type="primary", use_container_width=True):
        test_result = run_two_group_independent_test(
            df=df,
            group_col=grp_col,
            measure_col=meas_col,
            group1_name=g1_name,
            group2_name=g2_name,
            method=override_method,
            direction=direction,
            confidence_level=conf_level,
            alpha=alpha_threshold,
        )

elif analysis_type == "paired":
    if len(numeric_cols) < 2:
        st.warning("Paired comparison requires at least two numeric measurement columns.")
        st.stop()

    p_col1, p_col2, p_method = st.columns(3)
    with p_col1:
        pre_var = st.selectbox("Condition 1 (Pre / Baseline):", numeric_cols, index=0)
    with p_col2:
        post_var = st.selectbox(
            "Condition 2 (Post / Follow-up):", [c for c in numeric_cols if c != pre_var], index=0
        )
    with p_method:
        paired_method = st.selectbox("Paired Method:", ["paired_t", "wilcoxon"], index=0)

    recommendation = StatisticalService.recommend_test("paired", sample_size=len(df))
    st.info(
        f"💡 **Recommended Test:** {recommendation.recommended_test}\n\n*{recommendation.rationale}*"
    )

    if st.button("Run Paired Analysis", type="primary", use_container_width=True):
        test_result = run_paired_group_test(
            df=df,
            pre_col=pre_var,
            post_col=post_var,
            method=paired_method,
            confidence_level=conf_level,
            alpha=alpha_threshold,
        )

elif analysis_type == "multi_group":
    if not categorical_cols or not numeric_cols:
        st.warning(
            "Multi-group comparison requires a categorical grouping variable and a numeric metric."
        )
        st.stop()

    m_grp, m_meas = st.columns(2)
    with m_grp:
        mg_col = st.selectbox("Grouping Variable (3+ Groups):", categorical_cols, key="mg_grp")
    with m_meas:
        mm_col = st.selectbox("Continuous Outcome Variable:", numeric_cols, key="mg_meas")

    recommendation = StatisticalService.recommend_test("multi_group", sample_size=len(df))
    st.info(
        f"💡 **Recommended Test:** {recommendation.recommended_test}\n\n*{recommendation.rationale}*"
    )

    if st.button("Run ANOVA Analysis", type="primary", use_container_width=True):
        test_result = run_multi_group_anova(
            df=df,
            group_col=mg_col,
            measure_col=mm_col,
            alpha=alpha_threshold,
        )

elif analysis_type == "categorical":
    if len(categorical_cols) < 2:
        st.warning("Categorical association testing requires at least two categorical variables.")
        st.stop()

    cat_c1, cat_c2 = st.columns(2)
    with cat_c1:
        cat_v1 = st.selectbox("Categorical Dimension A:", categorical_cols, index=0)
    with cat_c2:
        cat_v2 = st.selectbox(
            "Categorical Dimension B:", [c for c in categorical_cols if c != cat_v1], index=0
        )

    recommendation = StatisticalService.recommend_test("categorical", sample_size=len(df))
    st.info(
        f"💡 **Recommended Test:** {recommendation.recommended_test}\n\n*{recommendation.rationale}*"
    )

    if st.button("Run Chi-Square Test", type="primary", use_container_width=True):
        test_result = run_chi_square_test(
            df=df,
            var1=cat_v1,
            var2=cat_v2,
            alpha=alpha_threshold,
        )

elif analysis_type == "correlation":
    if len(numeric_cols) < 2:
        st.warning("Correlation testing requires at least two continuous numeric variables.")
        st.stop()

    cr_c1, cr_c2, cr_m = st.columns(3)
    with cr_c1:
        cr_v1 = st.selectbox("Variable X:", numeric_cols, index=0)
    with cr_c2:
        cr_v2 = st.selectbox("Variable Y:", [c for c in numeric_cols if c != cr_v1], index=0)
    with cr_m:
        cr_method = st.selectbox("Method:", ["pearson", "spearman"], index=0)

    recommendation = StatisticalService.recommend_test("correlation", sample_size=len(df))
    st.info(
        f"💡 **Recommended Test:** {recommendation.recommended_test}\n\n*{recommendation.rationale}*"
    )

    if st.button("Run Correlation Test", type="primary", use_container_width=True):
        test_result = run_correlation_test(
            df=df,
            var1=cr_v1,
            var2=cr_v2,
            method=cr_method,
            confidence_level=conf_level,
            alpha=alpha_threshold,
        )

elif analysis_type == "regression":
    if len(numeric_cols) < 1:
        st.warning("Regression requires at least one numeric outcome variable.")
        st.stop()

    reg_c1, reg_c2 = st.columns([1, 2])
    with reg_c1:
        reg_y = st.selectbox("Continuous Outcome (y):", numeric_cols, key="reg_y")
    with reg_c2:
        available_preds = [c for c in (numeric_cols + categorical_cols) if c != reg_y]
        reg_x = st.multiselect(
            "Predictor Variables (X):", available_preds, default=available_preds[:3]
        )

    recommendation = StatisticalService.recommend_test("regression", sample_size=len(df))
    st.info(
        f"💡 **Recommended Test:** {recommendation.recommended_test}\n\n*{recommendation.rationale}*"
    )

    if st.button("Fit OLS Regression", type="primary", use_container_width=True) and reg_x:
        try:
            reg_result = fit_ols_regression(
                df=df,
                outcome_col=reg_y,
                predictor_cols=reg_x,
                confidence_level=conf_level,
                alpha=alpha_threshold,
            )
        except Exception as e:
            st.error(f"Regression estimation error: {e}")

# Save to history if executed
if test_result:
    st.session_state["active_test_result"] = test_result
    st.session_state["statistical_history"].append(test_result)
if reg_result:
    st.session_state["active_reg_result"] = reg_result

active_test = st.session_state.get("active_test_result")
active_reg = st.session_state.get("active_reg_result")

# ==============================================================================
# SECTION B: RESULTS OVERVIEW
# ==============================================================================
if active_test and analysis_type != "regression":
    st.markdown("---")
    st.subheader("2. Inferential Results & Evidence")

    st.markdown(
        f"""
        <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid #334155; border-left: 4px solid #38bdf8;
                    border-radius: 8px; padding: 14px 18px; margin-bottom: 16px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <span style="font-size: 1.05rem; font-weight: 600; color: #f8fafc;">
                    {active_test.test_name}
                </span>
                <span style="background: #1e293b; color: #38bdf8; padding: 3px 10px; border-radius: 4px; border: 1px solid #475569; font-size: 0.8rem; font-weight: bold;">
                    {active_test.p_value_display}
                </span>
            </div>
            <p style="font-size: 0.9rem; color: #cbd5e1; margin-bottom: 0;">
                {active_test.summary}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    r_col1, r_col2, r_col3, r_col4 = st.columns(4)
    r_col1.metric(
        "Test Statistic", f"{active_test.statistic_name}: {active_test.statistic_value:,.3f}"
    )
    r_col2.metric("p-value", active_test.p_value_display)
    if active_test.confidence_interval:
        ci = active_test.confidence_interval
        r_col3.metric(
            f"{int(ci.confidence_level * 100)}% Confidence Interval",
            f"[{ci.lower_bound:,.2f}, {ci.upper_bound:,.2f}]",
        )
    else:
        r_col3.metric("Sample Size (N)", f"{active_test.total_n:,}")

    if active_test.effect_size:
        eff = active_test.effect_size
        r_col4.metric(f"Effect Size ({eff.name})", f"{eff.estimate:+.2f} ({eff.magnitude})")
    else:
        r_col4.metric("Degrees of Freedom", str(active_test.degrees_of_freedom or "N/A"))

    # Tabs for Deep Evidence
    res_tab_interp, res_tab_diags, res_tab_robust = st.tabs(
        [
            "📖 8-Layer Grounded Interpretation",
            "🔬 Assumption Diagnostics",
            "⚖️ Robustness & Sensitivity",
        ]
    )

    with res_tab_interp:
        st.markdown(render_statistical_interpretation_markdown(active_test))

    with res_tab_diags:
        st.markdown("#### Diagnostic Verification of Assumptions")
        for check in active_test.assumption_checks:
            badge = "✅" if check.is_satisfied else "⚠️"
            st.markdown(
                f"**{badge} {check.name}** (`{check.statistic_name} = {check.statistic_value}`)"
            )
            st.write(check.message)
            st.caption(f"**Guidance:** {check.recommendation}")
            st.markdown("---")

        # Q-Q plot if numeric variable evaluated
        if active_test.selected_variables:
            target_num = active_test.selected_variables[-1]
            if target_num in df.columns and pd.api.types.is_numeric_dtype(df[target_num]):
                qq_data = generate_qq_plot_data(df[target_num])
                qq_fig = build_qq_plot(qq_data, title=f"Normal Q-Q Plot — '{target_num}'")
                st.plotly_chart(qq_fig, use_container_width=True)

    with res_tab_robust:
        st.markdown("#### Specification Curve & Sensitivity Analysis")
        st.caption(
            "Compares findings against defensible alternative methods to verify that conclusions do not hinge on arbitrary choices."
        )

        if analysis_type == "two_group" and active_test.groups and len(active_test.groups) >= 2:
            robust_comparisons = evaluate_two_group_robustness(
                df=df,
                group_col=active_test.selected_variables[0],
                measure_col=active_test.selected_variables[1],
                group1_name=active_test.groups[0],
                group2_name=active_test.groups[1],
                alpha=alpha_threshold,
            )
            robust_rows = [
                {
                    "Specification": r.specification_name,
                    "Sample (N)": f"{r.sample_size:,}",
                    "Estimate": f"{r.point_estimate:,.2f}",
                    "p-value": f"{r.p_value:.4f}" if r.p_value is not None else "N/A",
                    "Effect Size": f"{r.effect_size_estimate:+.2f}"
                    if r.effect_size_estimate is not None
                    else "N/A",
                    "Conclusion Stable?": "✅ Stable" if r.conclusion_stable else "⚠️ Diverges",
                    "Notes": r.notes,
                }
                for r in robust_comparisons
            ]
            st.dataframe(pd.DataFrame(robust_rows), use_container_width=True)
        elif analysis_type == "correlation":
            robust_comparisons = evaluate_correlation_robustness(
                df=df,
                var1=active_test.selected_variables[0],
                var2=active_test.selected_variables[1],
                alpha=alpha_threshold,
            )
            robust_rows = [
                {
                    "Specification": r.specification_name,
                    "Sample (N)": f"{r.sample_size:,}",
                    "Correlation (r)": f"{r.point_estimate:+.3f}",
                    "p-value": f"{r.p_value:.4f}" if r.p_value is not None else "N/A",
                    "Conclusion Stable?": "✅ Stable" if r.conclusion_stable else "⚠️ Diverges",
                    "Notes": r.notes,
                }
                for r in robust_comparisons
            ]
            st.dataframe(pd.DataFrame(robust_rows), use_container_width=True)
        else:
            st.info("Sensitivity matrix not applicable for current analysis type.")

elif active_reg and analysis_type == "regression":
    st.markdown("---")
    st.subheader("2. Regression Model Summary & Coefficients")

    rg_c1, rg_c2, rg_c3, rg_c4 = st.columns(4)
    rg_c1.metric("R² (Variance Explained)", f"{active_reg.r_squared * 100:.1f}%")
    rg_c2.metric("Adjusted R²", f"{active_reg.adj_r_squared * 100:.1f}%")
    rg_c3.metric("F-Statistic", f"{active_reg.f_statistic:,.2f}")
    rg_c4.metric(
        "Model p-value",
        "< 0.001" if active_reg.f_pvalue < 0.001 else f"= {active_reg.f_pvalue:.4f}",
    )

    st.markdown("#### Estimated Coefficients & Parameter Uncertainty")
    coeff_rows = [
        {
            "Predictor": c.variable,
            "Estimate (β)": f"{c.estimate:+,.3f}",
            "Std Error": f"{c.std_error:,.3f}",
            "t-stat": f"{c.t_statistic:+,.2f}",
            "p-value": "< 0.001" if c.p_value < 0.001 else f"{c.p_value:.4f}",
            "95% CI": f"[{c.ci_lower:+,.2f}, {c.ci_upper:+,.2f}]",
            "VIF": f"{c.vif:.2f}" if c.vif is not None else "1.00",
            "Significance": "✅ p < 0.05" if c.p_value < 0.05 else "Not Sig",
        }
        for c in active_reg.coefficients
    ]
    st.dataframe(pd.DataFrame(coeff_rows), use_container_width=True)

    reg_tab_interp, reg_tab_diags = st.tabs(
        ["📖 Regression Interpretation", "🔬 Model Diagnostics"]
    )
    with reg_tab_interp:
        st.markdown(render_regression_interpretation_markdown(active_reg))

    with reg_tab_diags:
        st.markdown("#### OLS Diagnostic Audits")
        for d in active_reg.diagnostics:
            badge = "✅" if d.is_satisfied else "⚠️"
            st.markdown(f"**{badge} {d.name}**")
            st.write(d.message)
            st.caption(f"**Guideline:** {d.recommendation}")
            st.markdown("---")

# ==============================================================================
# SECTION C: MULTIPLE-TESTING CORRECTION LAB
# ==============================================================================
if len(st.session_state["statistical_history"]) >= 2:
    st.markdown("---")
    st.subheader("3. Multiple-Testing Correction Lab (Family-Wise Error Rate & FDR)")
    st.caption(
        "When evaluating multiple statistical tests simultaneously, false positive rates inflate rapidly. Apply multiplicity adjustment across the active test family."
    )

    m_method = st.selectbox(
        "Correction Procedure:",
        options=["fdr_bh", "bonferroni", "holm", "none"],
        format_func=lambda m: {
            "fdr_bh": "Benjamini-Hochberg (False Discovery Rate - Recommended)",
            "bonferroni": "Bonferroni (Strict Family-Wise Error Rate)",
            "holm": "Holm-Bonferroni (Step-Down FWER)",
            "none": "Unadjusted (Exploratory Only)",
        }.get(m, m),
        index=0,
    )

    adjusted_family = apply_multiple_testing_correction(
        results=st.session_state["statistical_history"],
        method=m_method,
        family_name="Session Analysis Family",
    )

    family_rows = [
        {
            "Analysis ID": r.analysis_id,
            "Test": r.test_name,
            "Variables": ", ".join(r.selected_variables),
            "Raw p-value": f"{r.p_value:.4f}",
            "Adjusted p-value": f"{r.adjusted_p_value:.4f}"
            if r.adjusted_p_value is not None
            else "N/A",
            "Significant (α = 0.05)": "✅ Yes"
            if (r.adjusted_p_value or r.p_value) < 0.05
            else "❌ No",
            "Adjustment Details": r.adjustment_method or "None",
        }
        for r in adjusted_family
    ]
    st.dataframe(pd.DataFrame(family_rows), use_container_width=True)

st.markdown("---")
st.page_link("app.py", label="← Return to Main Workspace", icon="🏠")
st.page_link("pages/3_Exploration_Studio.py", label="← Back to Exploration Studio", icon="📈")
