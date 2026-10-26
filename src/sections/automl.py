"""Section: AutoML Lab - one-click pipeline with every step explained."""

import plotly.express as px
import streamlit as st

from src.core.code_snippets import load_kwargs
from src.core.ds_ui import experiment_log, record_run, store_fitted_model
from src.core.session_state import KEY_AUTOML_RESULT
from src.core.ui_components import (
    render_code_button,
    require_dataset,
)
from src.datascience import automl as am
from src.datascience.modeling import SCORING_CHOICES, ModelingError, infer_task

SUBTITLE = (
    "Pick a target; the lab removes leaks, races models, tunes, stacks, and explains each decision."
)


def render() -> None:
    """Render this section inside its parent page."""

    dataset = require_dataset("AutoML Lab")
    df = dataset.working_df
    LOAD = load_kwargs(dataset)

    c1, c2 = st.columns([1, 3])
    target = c1.selectbox("Target", list(df.columns), key="am_target")
    pool = [c for c in df.columns if c != target]
    features = c2.multiselect(
        "Features (leave all selected to let AutoML decide)", pool, default=pool, key="am_features"
    )
    task = infer_task(df[target].dropna())

    with st.expander("⚙️ Settings", expanded=False):
        s1, s2, s3, s4 = st.columns(4)
        budget = s1.slider("Time budget (s)", 15, 300, 60, 15, key="am_budget")
        cv = s2.slider("CV folds", 2, 8, 4, key="am_cv")
        n_race = s3.slider("Models to race", 3, 14, 8, key="am_nrace")
        top_k = s4.slider("Models to tune", 0, 4, 2, key="am_topk")
        o1, o2, o3, o4 = st.columns(4)
        do_ens = o1.checkbox("Try stacking", value=True, key="am_ens")
        exclude = o2.checkbox(
            "Drop identifier-like columns",
            value=True,
            key="am_excl",
            help="Leakage suspects (names containing the target, near-perfect correlates) are always dropped.",
        )
        optional = o3.checkbox("Include XGBoost / LightGBM if installed", value=True, key="am_opt")
        scoring = o4.selectbox(
            "Optimize",
            list(SCORING_CHOICES[task]),
            format_func=SCORING_CHOICES[task].get,
            key="am_scoring",
        )

    if st.button("🚀 Run AutoML", type="primary", disabled=not features, key="am_run"):
        with st.status("Running the pipeline...", expanded=True) as status:
            try:
                result = am.run_automl(
                    df,
                    target,
                    features,
                    task,
                    time_budget_s=budget,
                    cv=cv,
                    n_race=n_race,
                    top_k=top_k,
                    tune=top_k > 0,
                    try_ensemble=do_ens,
                    exclude_flagged=exclude,
                    scoring=scoring,
                    include_optional=optional,
                )
                for step in result.steps:
                    st.write(f"✅ **{step.name}**: {step.detail}")
                status.update(
                    label=f"Done in {result.total_seconds:.0f}s: {result.chosen}", state="complete"
                )
                st.session_state[KEY_AUTOML_RESULT] = (target, result)
            except ModelingError as err:
                status.update(label="Could not run", state="error")
                st.error(str(err))

    stored = st.session_state.get(KEY_AUTOML_RESULT)
    if stored and stored[0] == target:
        res: am.AutoMLResult = stored[1]
        a, b, c, d = st.columns(4)
        a.metric("Chosen model", res.chosen.split(" (")[0])
        b.metric(f"CV {res.scoring}", f"{res.chosen_cv:.4f}")
        c.metric(
            "Held-out", ", ".join(f"{k}={v:.2f}" for k, v in list(res.model.metrics.items())[:2])
        )
        d.metric("Total time", f"{res.total_seconds:.0f}s")
        st.info(res.chosen_reason)
        if res.dropped:
            with st.expander(f"🛡️ {len(res.dropped)} feature(s) removed automatically"):
                for col, why in res.dropped:
                    st.markdown(f"- `{col}`: {why}")
        t_board, t_tune, t_steps, t_report = st.tabs(
            ["🏁 Leaderboard", "🎚️ Tuning", "🪜 Steps", "📄 Report & code"]
        )
        with t_board:
            ok = res.leaderboard[res.leaderboard["status"] == "ok"]
            st.dataframe(
                res.leaderboard.style.format(
                    {
                        "cv_mean": "{:.4f}",
                        "cv_std": "{:.4f}",
                        "train_mean": "{:.4f}",
                        "gap": "{:.3f}",
                        "seconds": "{:.1f}",
                    },
                    na_rep="-",
                ),
                hide_index=True,
                use_container_width=True,
            )
            if len(ok):
                st.plotly_chart(
                    px.bar(
                        ok,
                        x="model",
                        y="cv_mean",
                        error_y="cv_std",
                        color="overfit",
                        template="plotly_dark",
                    ),
                    use_container_width=True,
                )
            if res.ensemble_members:
                st.caption("Stacked: " + ", ".join(res.ensemble_members))
        with t_tune:
            if res.tuned:
                import pandas as pd

                t = pd.DataFrame(
                    [
                        {
                            "model": x["model"],
                            "default CV": x["default_cv"],
                            "tuned CV": x["tuned_cv"],
                            "gain": x["tuned_cv"] - x["default_cv"],
                            "best parameters": str(x["params"]),
                        }
                        for x in res.tuned
                    ]
                )
                st.dataframe(
                    t.style.format(
                        {"default CV": "{:.4f}", "tuned CV": "{:.4f}", "gain": "{:+.4f}"}
                    ),
                    hide_index=True,
                    use_container_width=True,
                )
            else:
                st.caption("No models were tuned in this run.")
        with t_steps:
            for i, step in enumerate(res.steps, 1):
                st.markdown(f"**{i}. {step.name}** · {step.seconds:.1f}s  \n{step.detail}")
        with t_report:
            report = res.report_markdown()
            st.markdown(report)
            st.download_button(
                "⬇️ Report (Markdown)",
                report,
                "automl_report.md",
                "text/markdown",
                key="am_report_dl",
            )
            render_code_button(
                f"AutoML: {res.chosen}",
                lambda o: am.automl_code(res, options=o, **LOAD),
                key="code_am",
            )
        k1, k2 = st.columns(2)
        if k1.button("Send the model to Deployment / Explainer / Registry", key="am_send"):
            name = store_fitted_model(res.model, f"{res.chosen} (AutoML)")
            st.success(f"Saved as '{name}'.")
        if k2.button("Log this run to the Experiment Tracker", key="am_log"):
            record_run(
                f"AutoML: {res.chosen}",
                "automl",
                res.model.estimator_name,
                res.task,
                target,
                res.features_used,
                {**res.model.metrics, "cv_score": res.chosen_cv},
                params=res.chosen_params,
                seed=res.seed,
                duration_s=res.total_seconds,
                dataset=dataset,
                notes=res.chosen_reason,
            )
            st.toast(f"{len(experiment_log().runs)} run(s) logged", icon="✅")
    else:
        st.info("Choose a target and press **Run AutoML**. Nothing runs until you do.")
    return
