"""Section: Survival Lab - time-to-event analysis (churn timing, failures, recovery)."""

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.core.code_ds import survival_snippet
from src.core.code_snippets import load_kwargs
from src.core.ui_components import (
    render_code_button,
    require_dataset,
)
from src.datascience import survival as sv

SUBTITLE = "Model how long until something happens (churn, failure, purchase), including rows where it has not happened yet."


def render() -> None:
    """Render this section inside its parent page."""

    dataset = require_dataset("Survival Lab")
    df = dataset.working_df
    LOAD = load_kwargs(dataset)
    num_cols = df.select_dtypes("number").columns.tolist()
    binary = [
        c
        for c in df.columns
        if df[c].dropna().nunique() == 2 and set(df[c].dropna().unique()) <= {0, 1}
    ]
    if not num_cols or not binary:
        st.warning(
            "Needs a numeric duration column and a 0/1 event column (1 = the event happened, 0 = not yet)."
        )
        return

    c1, c2, c3 = st.columns(3)
    duration = c1.selectbox("Duration (time observed)", num_cols, key="sv_dur")
    event = c2.selectbox(
        "Event happened (0/1)", [c for c in binary if c != duration], key="sv_event"
    )
    group = c3.selectbox(
        "Compare groups (optional)",
        ["(none)"]
        + [c for c in df.columns if c not in (duration, event) and 2 <= df[c].nunique() <= 8],
        key="sv_group",
    )
    st.caption(
        f"{int(df[event].sum()):,} events among {len(df):,} rows ({df[event].mean():.0%}). Rows with 0 are *censored*: still waiting, which is information, not missing data."
    )

    tab_km, tab_cox = st.tabs(["📉 Kaplan-Meier", "🧮 Cox model"])
    with tab_km:
        try:
            km = sv.kaplan_meier(df, duration, event, None if group == "(none)" else group)
            show_ci = st.checkbox("Show confidence bands", value=True, key="sv_ci")
            fig = go.Figure()
            for name, part in km["curves"].groupby("group"):
                fig.add_scatter(
                    x=part["time"], y=part["survival"], mode="lines", line_shape="hv", name=name
                )
                if show_ci:
                    fig.add_scatter(
                        x=list(part["time"]) + list(part["time"][::-1]),
                        y=list(part["ci_high"]) + list(part["ci_low"][::-1]),
                        fill="toself",
                        line={"width": 0},
                        opacity=0.2,
                        showlegend=False,
                    )
            fig.add_hline(y=0.5, line_dash="dot", annotation_text="median")
            fig.update_layout(
                template="plotly_dark",
                title="Probability of still being event-free",
                xaxis_title=duration,
                yaxis_title="survival",
                yaxis_range=[0, 1.02],
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(
                km["summary"].style.format(
                    {"median_survival": "{:.2f}", "event_rate": "{:.1%}"}, na_rep="not reached"
                ),
                hide_index=True,
                use_container_width=True,
            )
            if km["logrank"]:
                p = km["logrank"]["p_value"]
                (st.success if p < 0.05 else st.info)(
                    f"Log-rank test: χ² = {km['logrank']['chi2']:.2f}, p = {p:.3g}. "
                    + (
                        "The groups' survival differs."
                        if p < 0.05
                        else "No reliable difference between groups."
                    )
                )
            render_code_button(
                "Survival analysis",
                lambda o: survival_snippet(
                    duration, event, None if group == "(none)" else group, options=o, **LOAD
                ),
                key="code_sv_km",
            )
        except sv.SurvivalError as err:
            st.error(str(err))

    with tab_cox:
        feats = st.multiselect(
            "Features",
            [c for c in df.columns if c not in (duration, event)],
            default=[c for c in df.columns if c not in (duration, event) and df[c].nunique() > 1][
                :4
            ],
            key="sv_feats",
        )
        if feats:
            try:
                cox = sv.cox_ph(df, duration, event, feats)
                a, b, c = st.columns(3)
                a.metric(
                    "Concordance",
                    f"{cox['concordance']:.3f}",
                    help="0.5 = random ranking, 1 = perfect.",
                )
                b.metric("Rows", f"{cox['n']:,}")
                c.metric("Events", f"{cox['events']:,}")
                for note in cox["notes"]:
                    st.warning(note)
                t = cox["table"]
                fig = go.Figure(
                    go.Scatter(
                        x=t["hazard_ratio"],
                        y=t.index,
                        mode="markers",
                        error_x={
                            "type": "data",
                            "symmetric": False,
                            "array": t["hr_high"] - t["hazard_ratio"],
                            "arrayminus": t["hazard_ratio"] - t["hr_low"],
                        },
                    )
                )
                fig.add_vline(x=1, line_dash="dot")
                fig.update_layout(
                    template="plotly_dark",
                    title="Hazard ratios (>1 = event happens sooner, <1 = later)",
                    xaxis_type="log",
                    height=max(250, 40 * len(t)),
                )
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(
                    t.style.format(
                        {
                            "coef": "{:.3f}",
                            "hazard_ratio": "{:.3f}",
                            "hr_low": "{:.3f}",
                            "hr_high": "{:.3f}",
                            "p_value": "{:.3g}",
                            "hr_per_sd": "{:.3f}",
                        }
                    ),
                    use_container_width=True,
                )
                if len(cox["ph_check"]):
                    with st.expander("Proportional-hazards check"):
                        st.dataframe(
                            cox["ph_check"].style.format(
                                {"correlation_with_time": "{:.3f}", "p_value": "{:.3g}"}
                            ),
                            hide_index=True,
                            use_container_width=True,
                        )
                st.markdown("##### Predicted survival for a profile")
                profile = {}
                cols = st.columns(3)
                for i, name in enumerate(cox["columns"]):
                    with cols[i % 3]:
                        profile[name] = st.number_input(name, value=0.0, key=f"sv_prof_{name}")
                horizon = float(np.nanpercentile(cox["durations"], 90))
                pred = sv.predict_survival(cox, profile, np.linspace(0, horizon, 60))
                st.plotly_chart(
                    px.line(
                        pred,
                        x="time",
                        y="survival",
                        template="plotly_dark",
                        range_y=[0, 1.02],
                        title="Survival curve for the profile above",
                    ),
                    use_container_width=True,
                )
                render_code_button(
                    "Cox model",
                    lambda o: survival_snippet(
                        duration,
                        event,
                        None if group == "(none)" else group,
                        feats,
                        options=o,
                        **LOAD,
                    ),
                    key="code_sv_cox",
                )
            except sv.SurvivalError as err:
                st.error(str(err))
        else:
            st.info("Pick at least one feature.")
    return
