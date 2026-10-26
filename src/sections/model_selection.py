"""Section: Model Selection - recommendations, model races, nested CV, comparisons, and curves."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.core.code_ds import nested_cv_snippet, race_snippet
from src.core.code_snippets import load_kwargs
from src.core.ds_ui import model_card_box, model_multiselect, pick_target_features, record_run
from src.core.ui_components import (
    render_code_button,
    require_dataset,
)
from src.datascience import catalog
from src.datascience import selection as sel
from src.datascience.modeling import SCORING_CHOICES, ModelingError

SUBTITLE = "Get a guided shortlist, race models fairly, check honest performance, and compare them statistically."


def render() -> None:
    """Render this section inside its parent page."""

    dataset = require_dataset("Model Selection")
    df = dataset.working_df
    LOAD = load_kwargs(dataset)

    target, features, task = pick_target_features(df, "ms")
    if not features:
        st.info("Pick at least one feature.")
        return

    tab_assist, tab_race, tab_nested, tab_curves, tab_cards = st.tabs(
        ["🧭 Assistant", "🏁 Race", "🔬 Nested CV", "📈 Curves", "📚 Model cards"]
    )

    # ------------------------------------------------------------------ assistant
    with tab_assist:
        if st.button("Analyze my data", type="primary", key="ms_profile_btn"):
            with st.spinner("Profiling and running a quick linear vs. boosted test..."):
                try:
                    st.session_state["ms_profile"] = sel.profile_for_selection(
                        df, target, features, task
                    )
                except ModelingError as err:
                    st.error(str(err))
        profile = st.session_state.get("ms_profile")
        if profile is not None and profile.task == task:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Rows", f"{profile.n_rows:,}")
            m2.metric(
                "Features",
                f"{profile.n_features}",
                help=f"{profile.n_numeric} numeric, {profile.n_categorical} categorical",
            )
            m3.metric("Missing", f"{profile.missing_pct:.1f}%")
            m4.metric(
                "Non-linear signal",
                f"{profile.nonlinear_gain:+.3f}",
                help="Boosted model score minus linear model score on a quick test. Positive = curved/interacting patterns.",
            )
            notes = []
            if profile.imbalance_ratio and profile.imbalance_ratio > 4:
                notes.append(
                    f"Classes are imbalanced ({profile.imbalance_ratio:.0f}:1). Accuracy will mislead; see the Classification Toolkit."
                )
            if profile.high_cardinality:
                notes.append(
                    f"{profile.high_cardinality} categorical feature(s) have more than 30 values."
                )
            if profile.target_skew and abs(profile.target_skew) > 2:
                notes.append(
                    f"The target is strongly skewed ({profile.target_skew:.1f}); consider a log transform or a Gamma model."
                )
            for note in notes:
                st.warning(note)
            recs = sel.recommend_models(profile)
            st.session_state["ms_recs"] = [r.name for r in recs]
            st.markdown("#### Recommended shortlist")
            for r in recs[:8]:
                with st.expander(
                    f"{r.score:.0f} · {catalog.display_name(catalog.get(r.name), task)}"
                ):
                    for reason in r.reasons:
                        st.markdown(f"- {reason}")
                    model_card_box(r.name, task)
            st.caption(
                "Scores start at 50; every adjustment is listed with its reason. They rank candidates to *try*, not winners."
            )
        else:
            st.info("Click **Analyze my data** for a ranked, explained shortlist.")

    # ------------------------------------------------------------------ race
    with tab_race:
        recs = st.session_state.get("ms_recs") or []
        default = (["Baseline"] + [r for r in recs if r != "Baseline"][:5]) if recs else None
        models = model_multiselect(task, "ms_models", default, "Models to race")
        c1, c2, c3, c4 = st.columns(4)
        strategies = {
            k: v
            for k, v in sel.CV_STRATEGIES.items()
            if k not in ("stratified", "repeated_stratified") or task != "regression"
        }
        strategy = c1.selectbox(
            "Validation",
            list(strategies),
            format_func=strategies.get,
            index=list(strategies).index(sel.default_cv_strategy(task)),
            key="ms_cv",
        )
        folds = c2.slider("Folds", 2, 10, 5, key="ms_folds")
        repeats = c3.slider(
            "Repeats", 1, 5, 2, key="ms_repeats", disabled=not strategy.startswith("repeated")
        )
        scoring = c4.selectbox(
            "Score",
            list(SCORING_CHOICES[task]),
            format_func=SCORING_CHOICES[task].get,
            key="ms_scoring",
        )
        group_col = (
            st.selectbox(
                "Group column (keeps each group in one fold)",
                [c for c in df.columns if c != target],
                key="ms_group",
            )
            if strategy == "group"
            else None
        )
        budget = st.slider("Time budget (seconds, 0 = none)", 0, 300, 0, 10, key="ms_budget")
        if st.button("Run the race", type="primary", disabled=not models, key="ms_run"):
            with st.spinner("Cross-validating on identical folds..."):
                try:
                    st.session_state["ms_race"] = sel.race(
                        df,
                        target,
                        features,
                        models,
                        task,
                        strategy,
                        folds,
                        repeats,
                        scoring,
                        time_budget_s=budget or None,
                        groups=df[group_col] if group_col else None,
                    )
                    st.session_state["ms_race_meta"] = (
                        target,
                        features,
                        task,
                        strategy,
                        folds,
                        scoring,
                    )
                except ModelingError as err:
                    st.error(str(err))
        race = st.session_state.get("ms_race")
        if race is not None and st.session_state.get("ms_race_meta", (None,))[0] == target:
            table = race.table
            show_cols = [
                c
                for c in (
                    "model",
                    "family",
                    "cv_mean",
                    "cv_std",
                    "train_mean",
                    "gap",
                    "overfit",
                    "seconds",
                    "status",
                )
                if c in table.columns
            ]
            st.dataframe(
                table[show_cols].style.format(
                    {
                        "cv_mean": "{:.4f}",
                        "cv_std": "{:.4f}",
                        "train_mean": "{:.4f}",
                        "gap": "{:.3f}",
                        "seconds": "{:.1f}",
                    },
                    na_rep="-",
                ),
                use_container_width=True,
                hide_index=True,
            )
            ok = table[table["status"] == "ok"]
            if len(ok):
                fig = px.bar(
                    ok,
                    x="model",
                    y="cv_mean",
                    error_y="cv_std",
                    color="overfit",
                    template="plotly_dark",
                    title=f"Cross-validated {SCORING_CHOICES[task][race.scoring]}",
                    color_discrete_map={"low": "#10b981", "moderate": "#eab308", "high": "#ef4444"},
                )
                st.plotly_chart(fig, use_container_width=True)
                pick, why = sel.one_se_choice(table)
                st.success(f"**Suggested:** {pick}. {why}")
                if (table["overfit"] == "high").any():
                    st.warning(
                        "Models with a high train-vs-validation gap are memorizing the training data; prefer simpler settings or more data."
                    )
                base = ok[ok["model"] == "Baseline"]
                if len(base) and ok.iloc[0]["model"] == "Baseline":
                    st.error(
                        "No model beats the baseline: the features may carry no signal for this target."
                    )
            if len(race.fold_scores) >= 2:
                with st.expander("Is the difference real? (corrected resampled t-test p-values)"):
                    pw = sel.pairwise_table(race)
                    st.plotly_chart(
                        px.imshow(
                            pw,
                            text_auto=".3f",
                            color_continuous_scale="RdYlGn_r",
                            zmin=0,
                            zmax=0.2,
                            template="plotly_dark",
                            title="p-value that ROW differs from COLUMN (small = reliable difference)",
                        ),
                        use_container_width=True,
                    )
                    a, b = st.columns(2)
                    names = list(race.fold_scores)
                    m_a, m_b = (
                        a.selectbox("Model A", names, key="ms_cmp_a"),
                        b.selectbox("Model B", names, index=min(1, len(names) - 1), key="ms_cmp_b"),
                    )
                    if m_a != m_b:
                        res = sel.compare_models(
                            race.fold_scores[m_a],
                            race.fold_scores[m_b],
                            race.n_train_per_fold,
                            race.n_test_per_fold,
                        )
                        st.info(
                            f"Mean difference (A - B): {res['mean_diff']:+.4f} · p = {res['p_value']:.3f} · verdict: **{res['verdict']}**"
                        )
            r1, r2 = st.columns(2)
            if r1.button("Log the best model to the Experiment Tracker", key="ms_log") and len(ok):
                best = ok.iloc[0]
                record_run(
                    f"Race winner: {best['model']}",
                    "race",
                    best["model"],
                    task,
                    target,
                    features,
                    {"cv_mean": best["cv_mean"], "cv_std": best["cv_std"]},
                    cv_scores=list(race.fold_scores[best["model"]]),
                    dataset=dataset,
                )
                st.toast("Logged", icon="✅")
            with r2:
                render_code_button(
                    "Model race",
                    lambda o: race_snippet(
                        models, target, features, task, folds, race.scoring, options=o, **LOAD
                    ),
                    key="code_ms_race",
                )

    # ------------------------------------------------------------------ nested CV
    with tab_nested:
        st.caption(
            "Tuning on the same folds you report inflates the score. Nested CV repeats the tuning *inside* every outer fold, so the result is honest."
        )
        tunable = catalog.tunable(task)
        nm = st.selectbox(
            "Model",
            tunable,
            format_func=lambda n: catalog.display_name(catalog.get(n), task),
            key="ms_nested_model",
        )
        n1, n2, n3 = st.columns(3)
        outer, inner, n_iter = (
            n1.slider("Outer folds", 3, 8, 5, key="ms_outer"),
            n2.slider("Inner folds", 2, 5, 3, key="ms_inner"),
            n3.slider("Search trials", 3, 20, 8, key="ms_iter"),
        )
        if st.button("Run nested CV", key="ms_nested_run"):
            with st.spinner("Tuning inside every outer fold..."):
                try:
                    st.session_state["ms_nested"] = sel.nested_cv(
                        df, target, features, nm, task, outer, inner, n_iter
                    )
                except ModelingError as err:
                    st.error(str(err))
        nested = st.session_state.get("ms_nested")
        if nested is not None and nested.model == nm:
            a, b, c = st.columns(3)
            a.metric(
                "Honest score", f"{nested.mean:.4f}", help=f"± {nested.std:.4f} across outer folds"
            )
            b.metric("Inner search score", f"{nested.non_nested_best:.4f}")
            c.metric(
                "Optimism",
                f"{nested.optimism:+.4f}",
                help="How much the ordinary tuned score over-promises.",
            )
            st.plotly_chart(
                px.bar(
                    x=[f"fold {i + 1}" for i in range(len(nested.outer_scores))],
                    y=nested.outer_scores,
                    template="plotly_dark",
                    labels={"x": "", "y": "score"},
                ),
                use_container_width=True,
            )
            st.dataframe(pd.DataFrame(nested.best_params), use_container_width=True)
            st.caption(
                "If the best parameters change a lot between folds, the tuning is unstable and the gain is probably noise."
            )
        render_code_button(
            "Nested CV",
            lambda o: nested_cv_snippet(
                nm, target, features, task, outer, inner, n_iter, options=o, **LOAD
            ),
            key="code_ms_nested",
        )

    # ------------------------------------------------------------------ curves
    with tab_curves:
        cm = st.selectbox(
            "Model",
            [s.name for s in catalog.available_models(task, include_baseline=False)],
            format_func=lambda n: catalog.display_name(catalog.get(n), task),
            key="ms_curve_model",
        )
        k1, k2 = st.columns(2)
        if k1.button("Learning curve", key="ms_lc_btn"):
            with st.spinner("Training on growing subsets..."):
                try:
                    st.session_state["ms_lc"] = (
                        cm,
                        sel.learning_curve_table(df, target, features, cm, task),
                    )
                except ModelingError as err:
                    st.error(str(err))
        space = catalog.param_space(cm, task)
        param = k2.selectbox("Hyperparameter", list(space) or ["(none)"], key="ms_vc_param")
        if st.button("Validation curve", disabled=not space, key="ms_vc_btn"):
            with st.spinner("Refitting across values..."):
                try:
                    st.session_state["ms_vc"] = (
                        cm,
                        param,
                        sel.validation_curve_table(
                            df, target, features, cm, param, space[param], task
                        ),
                    )
                except ModelingError as err:
                    st.error(str(err))
        lc = st.session_state.get("ms_lc")
        if lc and lc[0] == cm:
            t = lc[1]
            fig = go.Figure()
            fig.add_scatter(
                x=t["train_size"], y=t["train_mean"], name="training", mode="lines+markers"
            )
            fig.add_scatter(
                x=t["train_size"],
                y=t["val_mean"],
                name="validation",
                mode="lines+markers",
                error_y={"type": "data", "array": t["val_std"]},
            )
            fig.update_layout(
                template="plotly_dark",
                title="Learning curve",
                xaxis_title="training rows",
                yaxis_title="score",
            )
            st.plotly_chart(fig, use_container_width=True)
            gap, slope = (
                t["train_mean"].iloc[-1] - t["val_mean"].iloc[-1],
                t["val_mean"].iloc[-1] - t["val_mean"].iloc[-2],
            )
            st.info(
                "Validation is still rising: more data should help."
                if slope > 0.01
                else (
                    "Large gap between training and validation: overfitting. Simplify the model or add data."
                    if gap > 0.1
                    else "The curves have flattened: more data of the same kind will not help much."
                )
            )
        vc = st.session_state.get("ms_vc")
        if vc and vc[0] == cm:
            t = vc[2]
            fig = go.Figure()
            fig.add_scatter(x=t["value"], y=t["train_mean"], name="training", mode="lines+markers")
            fig.add_scatter(
                x=t["value"],
                y=t["val_mean"],
                name="validation",
                mode="lines+markers",
                error_y={"type": "data", "array": t["val_std"]},
            )
            fig.update_layout(
                template="plotly_dark",
                title=f"Validation curve: {vc[1]}",
                xaxis_title=vc[1],
                yaxis_title="score",
            )
            st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------ cards
    with tab_cards:
        fam = st.multiselect("Family", sorted({s.family for s in catalog.CATALOG}), key="ms_fam")
        for spec in catalog.available_models(task):
            if fam and spec.family not in fam:
                continue
            with st.expander(f"{catalog.display_name(spec, task)} · {spec.family}"):
                model_card_box(spec.name, task)
                space = catalog.param_space(spec.name, task)
                if space:
                    st.caption("Tunable: " + ", ".join(f"`{k}`" for k in space))
        unavailable = [s for s in catalog.CATALOG if not s.available]
        if unavailable:
            st.caption(
                "Not installed: "
                + ", ".join(f"{s.name} (`pip install {s.package}`)" for s in unavailable)
            )
    return
