"""Section: Causal & Uplift - difference-in-differences, propensity matching, uplift modeling."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.core.code_ds import did_snippet, matching_snippet, uplift_snippet
from src.core.code_snippets import load_kwargs
from src.core.ui_components import (
    render_code_button,
    require_dataset,
)
from src.datascience import catalog
from src.datascience import causal as cs

SUBTITLE = "Estimate what an intervention actually caused, and find who it works on. Observational data needs care."


def render() -> None:
    """Render this section inside its parent page."""

    dataset = require_dataset("Causal & Uplift")
    df = dataset.working_df
    LOAD = load_kwargs(dataset)
    num_cols = df.select_dtypes("number").columns.tolist()
    st.warning(
        "Correlation is not causation. These methods reduce, but cannot remove, bias from things you did not measure. Randomized experiments (the Experiments page) remain the gold standard."
    )

    two_level = [c for c in df.columns if df[c].dropna().nunique() == 2]
    tab_did, tab_psm, tab_up = st.tabs(
        ["📆 Difference-in-differences", "🔗 Propensity matching", "🎯 Uplift"]
    )

    def guarded(fn, *a, **kw):
        try:
            return fn(*a, **kw)
        except cs.CausalError as err:
            st.error(str(err))
        except Exception as err:
            st.error(f"Could not run the analysis: {err}")
        return None

    # ------------------------------------------------------------------ DiD
    with tab_did:
        st.caption(
            "Compare how a treated group changed before vs after, against how a control group changed over the same time."
        )
        if not two_level:
            st.info("Needs a two-value group column (treated/control).")
        else:
            d1, d2, d3 = st.columns(3)
            outcome = d1.selectbox("Outcome", num_cols, key="cs_did_y")
            group = d2.selectbox("Group column", two_level, key="cs_did_g")
            levels = sorted(df[group].dropna().unique(), key=str)
            treated = d3.selectbox("Treated group", levels, index=len(levels) - 1, key="cs_did_t")
            e1, e2, e3 = st.columns(3)
            time_col = e1.selectbox(
                "Time column",
                [c for c in df.columns if c not in (outcome, group)],
                key="cs_did_time",
            )
            times = df[time_col].dropna()
            default_start = (
                float(times.median())
                if pd.api.types.is_numeric_dtype(times)
                else str(pd.to_datetime(times, errors="coerce").dropna().median().date())
                if pd.to_datetime(times, errors="coerce").notna().any()
                else ""
            )
            start_text = e2.text_input(
                "Treatment starts at", value=str(default_start), key="cs_did_start"
            )
            cluster = e3.selectbox(
                "Cluster standard errors by",
                ["(none)"] + [c for c in df.columns if c not in (outcome, group, time_col)],
                key="cs_did_cluster",
            )
            covs = st.multiselect(
                "Covariates (optional)",
                [c for c in df.columns if c not in (outcome, group, time_col)],
                key="cs_did_covs",
            )
            start: float | str = (
                float(start_text)
                if pd.api.types.is_numeric_dtype(times)
                and start_text.replace(".", "", 1).replace("-", "", 1).isdigit()
                else start_text
            )
            if st.button("Estimate the effect", type="primary", key="cs_did_run"):
                st.session_state["cs_did"] = guarded(
                    cs.did,
                    df,
                    outcome,
                    group,
                    time_col,
                    start,
                    treated,
                    covs,
                    None if cluster == "(none)" else cluster,
                )
            r = st.session_state.get("cs_did")
            if r:
                a, b, c = st.columns(3)
                a.metric(
                    "Estimated effect",
                    f"{r['effect']:+.3f}",
                    help="Extra change in the treated group beyond the control group's change.",
                )
                b.metric("95% interval", f"{r['ci_low']:+.2f} to {r['ci_high']:+.2f}")
                c.metric("p-value", f"{r['p_value']:.3g}")
                for note in r["notes"]:
                    (st.warning if "violated" in note else st.info)(note)
                st.dataframe(r["means"].style.format("{:.3f}"), use_container_width=True)
                ev = r["event_study"].melt(
                    id_vars="_period", var_name="group", value_name="mean_outcome"
                )
                fig = px.line(
                    ev,
                    x="_period",
                    y="mean_outcome",
                    color="group",
                    markers=True,
                    template="plotly_dark",
                    title="Average outcome over time (lines should move together before the treatment)",
                )
                st.plotly_chart(fig, use_container_width=True)
                render_code_button(
                    "Difference-in-differences",
                    lambda o: did_snippet(
                        outcome, group, time_col, start, treated, options=o, **LOAD
                    ),
                    key="code_cs_did",
                )

    # ------------------------------------------------------------------ PSM
    with tab_psm:
        st.caption(
            "Match each treated row to a similar untreated row (same chance of being treated) and compare their outcomes."
        )
        if not two_level:
            st.info("Needs a two-value treatment column.")
        else:
            p1, p2, p3 = st.columns(3)
            treatment = p1.selectbox("Treatment column", two_level, key="cs_psm_t")
            tl = sorted(df[treatment].dropna().unique(), key=str)
            tval = p2.selectbox("Treated value", tl, index=len(tl) - 1, key="cs_psm_tval")
            outcome = p3.selectbox(
                "Outcome", [c for c in num_cols if c != treatment], key="cs_psm_y"
            )
            covs = st.multiselect(
                "Covariates to balance (things that influence both treatment and outcome)",
                [c for c in df.columns if c not in (treatment, outcome)],
                default=[c for c in num_cols if c not in (treatment, outcome)][:4],
                key="cs_psm_covs",
            )
            caliper = st.slider(
                "Caliper (SD of the propensity logit)", 0.05, 0.6, 0.2, 0.05, key="cs_psm_cal"
            )
            if st.button("Match and estimate", type="primary", disabled=not covs, key="cs_psm_run"):
                st.session_state["cs_psm"] = guarded(
                    cs.propensity_matching, df, treatment, outcome, covs, tval, caliper
                )
            r = st.session_state.get("cs_psm")
            if r:
                a, b, c, d = st.columns(4)
                a.metric("Effect on the treated (ATT)", f"{r['att']:+.3f}")
                b.metric("95% interval", f"{r['ci_low']:+.2f} to {r['ci_high']:+.2f}")
                c.metric(
                    "Naive difference",
                    f"{r['naive_difference']:+.3f}",
                    help="What you would conclude without adjusting. Large gaps vs the ATT mean confounding.",
                )
                d.metric("Matched", f"{r['matched_pairs']:,} of {r['treated_total']:,}")
                for note in r["notes"]:
                    st.warning(note)
                bal = r["balance"].melt(
                    id_vars="covariate",
                    value_vars=["smd_before", "smd_after"],
                    var_name="when",
                    value_name="smd",
                )
                fig = px.bar(
                    bal,
                    x="smd",
                    y="covariate",
                    color="when",
                    barmode="group",
                    orientation="h",
                    template="plotly_dark",
                    title="Covariate balance (|SMD| < 0.1 is balanced)",
                )
                fig.add_vline(x=0.1, line_dash="dot")
                fig.add_vline(x=-0.1, line_dash="dot")
                st.plotly_chart(fig, use_container_width=True)
                st.caption(
                    f"Propensity model AUC {r['ps_auc']:.2f} (how well covariates predict treatment). Overlap of scores:"
                )
                st.dataframe(r["overlap"].style.format("{:.3f}"), use_container_width=True)
                render_code_button(
                    "Propensity-score matching",
                    lambda o: matching_snippet(
                        treatment, outcome, covs, tval, caliper, options=o, **LOAD
                    ),
                    key="code_cs_psm",
                )

    # ------------------------------------------------------------------ uplift
    with tab_up:
        st.caption(
            "Uplift models predict the *extra* chance of a good outcome caused by the treatment, per person: who to target, who to leave alone."
        )
        if not two_level:
            st.info("Needs a two-value treatment column.")
        else:
            u1, u2, u3 = st.columns(3)
            treatment = u1.selectbox("Treatment column", two_level, key="cs_up_t")
            tl = sorted(df[treatment].dropna().unique(), key=str)
            tval = u2.selectbox("Treated value", tl, index=len(tl) - 1, key="cs_up_tval")
            outcome = u3.selectbox(
                "Outcome (0/1 or numeric)", [c for c in df.columns if c != treatment], key="cs_up_y"
            )
            feats = st.multiselect(
                "Features that describe each person",
                [c for c in df.columns if c not in (treatment, outcome)],
                default=[c for c in df.columns if c not in (treatment, outcome)][:5],
                key="cs_up_feats",
            )
            model = st.selectbox(
                "Model",
                ["Random Forest", "Hist Gradient Boosting", "Linear / Logistic Regression"],
                format_func=lambda n: catalog.display_name(catalog.get(n), "binary_classification"),
                key="cs_up_model",
            )
            if st.button(
                "Fit the uplift model", type="primary", disabled=not feats, key="cs_up_run"
            ):
                with st.spinner("Fitting separate models for treated and control..."):
                    st.session_state["cs_up"] = guarded(
                        cs.uplift_model, df, treatment, outcome, feats, model, tval
                    )
            r = st.session_state.get("cs_up")
            if r:
                a, b, c = st.columns(3)
                a.metric("Average effect (observed)", f"{r['ate_observed']:+.3f}")
                b.metric(
                    "Top 3 deciles",
                    f"{r['top3_observed_uplift']:+.3f}",
                    help="Observed effect among the people the model ranks highest.",
                )
                c.metric(
                    "AUUC",
                    f"{r['auuc']:.2f}",
                    help="Area between the Qini curve and random targeting.",
                )
                q = r["qini"]
                fig = go.Figure()
                fig.add_scatter(x=q["fraction"], y=q["qini"], name="model")
                fig.add_scatter(
                    x=q["fraction"], y=q["random"], name="random targeting", line={"dash": "dot"}
                )
                fig.update_layout(
                    template="plotly_dark",
                    title="Qini curve (higher than the dotted line = the model finds responders)",
                    xaxis_title="fraction of people targeted",
                    yaxis_title="incremental outcomes",
                )
                st.plotly_chart(fig, use_container_width=True)
                dec = r["by_decile"]
                st.plotly_chart(
                    px.bar(
                        dec,
                        x="decile",
                        y=["mean_predicted_uplift", "observed_uplift"],
                        barmode="group",
                        template="plotly_dark",
                        title="Predicted vs observed uplift by score decile (1 = highest)",
                    ),
                    use_container_width=True,
                )
                if r["top3_observed_uplift"] > r["ate_observed"] + 0.02:
                    st.success(
                        f"Targeting the top 30% yields about {r['top3_observed_uplift'] / max(abs(r['ate_observed']), 1e-9):.1f}× the average effect per person."
                    )
                else:
                    st.info(
                        "The model does not separate responders from non-responders: the treatment effect looks uniform, or there is too little data."
                    )
                render_code_button(
                    "Uplift model",
                    lambda o: uplift_snippet(
                        treatment, outcome, feats, tval, model, options=o, **LOAD
                    ),
                    key="code_cs_up",
                )
    return
