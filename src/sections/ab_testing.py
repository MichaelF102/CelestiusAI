"""Section: Experiments & A/B Testing - plan, analyze, and avoid peeking mistakes."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.core.code_snippets import load_kwargs
from src.core.session_state import KEY_CURRENT_DATASET
from src.core.ui_components import render_code_button
from src.datascience import experiments as ex

SUBTITLE = "Size your test, analyze the result with frequentist and Bayesian views, and avoid peeking traps."


def render() -> None:
    """Render this section inside its parent page."""

    dataset = st.session_state.get(KEY_CURRENT_DATASET)

    tab_plan, tab_analyze, tab_peek = st.tabs(["📐 Plan", "📊 Analyze", "👀 Peeking guard"])

    with tab_plan:
        kind = st.radio("Metric type", ["Conversion rate", "Average value"], horizontal=True)
        c1, c2, c3 = st.columns(3)
        alpha = c1.select_slider("Significance level (α)", [0.01, 0.05, 0.10], value=0.05)
        power = c2.select_slider("Power", [0.7, 0.8, 0.9, 0.95], value=0.8)
        try:
            if kind == "Conversion rate":
                base = c3.number_input("Baseline rate", 0.001, 0.999, 0.10, 0.01, format="%.3f")
                target = st.number_input(
                    "Smallest rate worth detecting", 0.001, 0.999, 0.12, 0.01, format="%.3f"
                )
                n = ex.sample_size_proportions(base, target, alpha, power)
                st.metric(
                    "Rows needed per group",
                    f"{n:,}",
                    help="Control and treatment each need this many.",
                )
                st.caption(
                    f"Total ≈ {2 * n:,}. Detecting {abs(target - base):.1%} points ({abs(target / base - 1):.0%} relative)."
                )
                render_code_button(
                    "Sample size", ex.power_snippet(base, target, alpha, power), key="code_power"
                )
                xs = [int(n * f) for f in (0.25, 0.5, 0.75, 1, 1.25, 1.5)]
                fig = go.Figure(
                    go.Scatter(
                        x=xs,
                        y=[ex.power_for_proportions(base, target, x, alpha) for x in xs],
                        mode="lines+markers",
                    )
                )
                fig.update_layout(
                    template="plotly_dark",
                    height=280,
                    xaxis_title="Rows per group",
                    yaxis_title="Power",
                    margin={"l": 20, "r": 20, "t": 20, "b": 20},
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                delta = c3.number_input("Smallest difference worth detecting", value=5.0)
                sd = st.number_input(
                    "Standard deviation of the metric", min_value=0.0001, value=20.0
                )
                n = ex.sample_size_means(delta, sd, alpha, power)
                st.metric("Rows needed per group", f"{n:,}")
        except ex.ExperimentError as err:
            st.error(str(err))

    with tab_analyze:
        mode = st.radio(
            "Input", ["From the loaded dataset", "Enter summary numbers"], horizontal=True
        )
        a_alpha = st.select_slider("Significance level (α) ", [0.01, 0.05, 0.10], value=0.05)
        result = None
        code = None
        if mode == "Enter summary numbers":
            k1, k2 = st.columns(2)
            cs, cn = (
                k1.number_input("Control conversions", 0, value=500),
                k1.number_input("Control users", 1, value=5000),
            )
            ts, tn = (
                k2.number_input("Treatment conversions", 0, value=560),
                k2.number_input("Treatment users", 1, value=5000),
            )
            try:
                result = ex.analyze_conversion(int(cs), int(cn), int(ts), int(tn), a_alpha)
                srm = ex.srm_check([int(cn), int(tn)])
            except ex.ExperimentError as err:
                st.error(str(err))
                srm = None
        elif dataset is None:
            st.info("Load a dataset first, or switch to summary numbers.")
            srm = None
        else:
            df = dataset.working_df
            g1, g2 = st.columns(2)
            group_col = g1.selectbox(
                "Variant column", [c for c in df.columns if 2 <= df[c].nunique() <= 10]
            )
            metric_col = g2.selectbox("Metric column", [c for c in df.columns if c != group_col])
            levels = sorted(df[group_col].dropna().unique().tolist(), key=str)
            v1, v2 = st.columns(2)
            control = v1.selectbox("Control", levels)
            treatment = v2.selectbox("Treatment", [lv for lv in levels if lv != control])
            try:
                result = ex.analyze_frame(df, group_col, metric_col, control, treatment, a_alpha)
                counts = df[group_col].value_counts()
                srm = ex.srm_check([int(counts[control]), int(counts[treatment])])
                code = ex.ab_test_snippet(
                    group_col,
                    metric_col,
                    control,
                    treatment,
                    result["type"] == "conversion",
                    **load_kwargs(dataset),
                )
            except ex.ExperimentError as err:
                st.error(str(err))
                srm = None

        if result:
            if srm and srm["mismatch"]:
                st.error(
                    f"⚠️ Sample ratio mismatch (p = {srm['p_value']:.4g}): the groups are not the size you intended. Fix the assignment before trusting results."
                )
            elif srm:
                st.caption(f"Sample-ratio check passed (p = {srm['p_value']:.3f}).")
            conv = result["type"] == "conversion"
            m1, m2, m3, m4 = st.columns(4)
            m1.metric(
                "Control",
                f"{result['control_rate']:.2%}" if conv else f"{result['control_mean']:.3f}",
            )
            m2.metric(
                "Treatment",
                f"{result['treatment_rate']:.2%}" if conv else f"{result['treatment_mean']:.3f}",
                delta=f"{result['rel_lift']:+.1%}",
            )
            m3.metric("p-value", f"{result['p_value']:.4f}")
            m4.metric("P(treatment better)", f"{result['prob_treatment_better']:.1%}")
            level = int((1 - a_alpha) * 100)
            verdict = (
                "statistically significant"
                if result["significant"]
                else "not statistically significant"
            )
            st.info(
                f"The difference is **{result['abs_lift']:+.4f}** ({level}% CI {result['ci_low']:+.4f} to {result['ci_high']:+.4f}) and is **{verdict}** at α = {a_alpha}."
                + (
                    f" Expected loss if you ship: {result['expected_loss_if_ship']:.4%}."
                    if conv
                    else f" Effect size (Cohen's d): {result['cohens_d']:.2f}."
                )
            )
            if not result["significant"]:
                st.caption(
                    "Not significant is not proof of no effect: check whether the test had enough power (Plan tab)."
                )
            if code:
                render_code_button("A/B test analysis", code, key="code_ab")

    with tab_peek:
        st.markdown(
            "Checking a running test and stopping at the first *p < α* inflates false positives. "
            "Simulate it on data with **no real difference**:"
        )
        p1, p2 = st.columns(2)
        looks = p1.slider("Number of times you peek", 2, 15, 5)
        peek_alpha = p2.select_slider("α", [0.01, 0.05, 0.10], value=0.05, key="peek_alpha")
        if st.button("Run simulation"):
            with st.spinner("Simulating 2,000 experiments..."):
                sim = ex.simulate_peeking(looks, alpha=peek_alpha)
            s1, s2 = st.columns(2)
            s1.metric(
                "False positives with one look", f"{sim['single_look_false_positive_rate']:.1%}"
            )
            s2.metric("False positives when peeking", f"{sim['peeking_false_positive_rate']:.1%}")
        st.markdown("##### Corrected threshold per look")
        rows = [
            {"Method": m.title(), "Per-look α": ex.adjusted_alpha(peek_alpha, looks, m)}
            for m in ("bonferroni", "pocock")
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption(
            "Pocock values are tabulated for α = 0.05; other α fall back to Bonferroni (conservative)."
        )
    return
