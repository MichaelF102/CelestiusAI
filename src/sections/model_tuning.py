"""Section: Model Tuning & Comparison - cross-validated leaderboard, tuning, and side-by-side results."""

import pandas as pd
import plotly.express as px
import streamlit as st

from src.core.code_snippets import load_kwargs, model_training_snippet
from src.core.ds_ui import model_multiselect, record_run
from src.core.session_state import KEY_FITTED_MODELS
from src.core.ui_components import render_code_button, require_dataset
from src.datascience import catalog
from src.datascience import modeling as ml

SUBTITLE = "Cross-validate candidates, tune the best one, and compare experiments side by side."


def render() -> None:
    """Render this section inside its parent page."""

    dataset = require_dataset("Model Tuning")
    df = dataset.working_df
    models: dict = st.session_state.get(KEY_FITTED_MODELS) or {}

    s1, s2 = st.columns([1, 2])
    target = s1.selectbox("Target", list(df.columns))
    features = s2.multiselect(
        "Features",
        [c for c in df.columns if c != target],
        default=[c for c in df.columns if c != target and not c.lower().endswith("id")][:10],
    )
    try:
        task = ml.infer_task(df[target])
    except Exception:
        task = ml.REGRESSION
    t1, t2, t3, t4 = st.columns(4)
    task = t1.selectbox(
        "Task",
        [ml.REGRESSION, ml.BINARY, ml.MULTICLASS],
        index=[ml.REGRESSION, ml.BINARY, ml.MULTICLASS].index(task),
        format_func=lambda t: t.replace("_", " ").title(),
    )
    scoring = t2.selectbox(
        "Optimize", list(ml.SCORING_CHOICES[task]), format_func=ml.SCORING_CHOICES[task].get
    )
    cv = t3.slider("CV folds", 2, 10, 5)
    seed = t4.number_input("Random seed", 0, 9999, 42)

    tab_board, tab_tune, tab_compare = st.tabs(
        ["🏆 Leaderboard", "🎚️ Tune", "⚖️ Compare experiments"]
    )

    with tab_board:
        st.caption(
            "Every model is scored with the same cross-validation folds; preprocessing is fitted inside each fold."
        )
        board_models = model_multiselect(
            task,
            "mt_board_models",
            ["Baseline", "Linear / Logistic Regression", "Random Forest", "Hist Gradient Boosting"],
            "Models to compare (all 20+ catalog models available)",
        )
        if st.button(
            "Run cross-validation", type="primary", disabled=not (features and board_models)
        ):
            with st.spinner("Cross-validating candidates..."):
                try:
                    table, _ = ml.leaderboard(
                        df, target, features, task, cv, scoring, int(seed), models=board_models
                    )
                    st.session_state["mt_board"] = table
                except ml.ModelingError as err:
                    st.error(str(err))
                except Exception as err:
                    st.error(f"Could not train: {err}")
        board = st.session_state.get("mt_board")
        if board is not None:
            st.dataframe(
                board.style.format({"cv_mean": "{:.4f}", "cv_std": "{:.4f}"}),
                use_container_width=True,
                hide_index=True,
            )
            fig = px.bar(
                board,
                x="model",
                y="cv_mean",
                error_y="cv_std",
                template="plotly_dark",
                title=f"Cross-validated {ml.SCORING_CHOICES[task][scoring]}",
            )
            st.plotly_chart(fig, use_container_width=True)
            top = board.iloc[0]
            base_row = board[board["model"].str.startswith("Baseline")]
            if len(base_row) and top["model"] != base_row.iloc[0]["model"]:
                st.success(
                    f"**{top['model']}** leads, {top['cv_mean'] - base_row.iloc[0]['cv_mean']:+.3f} vs. the baseline."
                )
            elif len(base_row):
                st.warning(
                    "No model beats the baseline: the features may not carry signal for this target."
                )
        if features:
            render_code_button(
                "Predictive model comparison",
                model_training_snippet(
                    target, features, task, 0.2, int(seed), **load_kwargs(dataset)
                ),
                key="code_board",
            )

    with tab_tune:
        tunable = catalog.tunable(task)
        model_name = st.selectbox(
            "Model to tune",
            tunable,
            index=tunable.index("Random Forest") if "Random Forest" in tunable else 0,
            format_func=lambda n: catalog.display_name(catalog.get(n), task),
        )
        space = ml.space_for(model_name, task)
        with st.expander("Search space"):
            st.json({k.removeprefix("model__"): v for k, v in space.items()})
        methods = {
            "random": "Random search",
            "grid": "Grid search",
            "halving": "Successive halving (fast: drops weak settings early)",
            "optuna": "Bayesian optimization (Optuna)",
        }
        if not ml.optuna_available():
            methods.pop("optuna")
        m1, m2 = st.columns(2)
        method = m1.radio("Search", list(methods), format_func=methods.get, horizontal=False)
        n_iter = m2.slider("Trials / candidates", 4, 60, 12, disabled=method == "grid")
        if st.button("Tune", type="primary", disabled=not features):
            with st.spinner("Searching hyperparameters..."):
                try:
                    res = ml.tune_model(
                        df,
                        target,
                        features,
                        model_name,
                        task,
                        method,
                        n_iter,
                        cv,
                        scoring,
                        seed=int(seed),
                    )
                    key = f"{res.model.name} #{len(models) + 1}"
                    models[key] = res.model
                    st.session_state[KEY_FITTED_MODELS] = models
                    st.session_state["mt_last"] = (key, res)
                    record_run(
                        f"Tuned {model_name} ({method})",
                        "tuning",
                        model_name,
                        task,
                        target,
                        features,
                        {
                            **res.test_metrics,
                            "cv_score": res.best_cv_score,
                            "default_cv_score": res.default_cv_score,
                        },
                        params=res.best_params,
                        seed=int(seed),
                        dataset=dataset,
                    )
                except ml.ModelingError as err:
                    st.error(str(err))
                except Exception as err:
                    st.error(f"Tuning failed: {err}")
        last = st.session_state.get("mt_last")
        if last:
            key, res = last
            a, b, c = st.columns(3)
            a.metric("Default CV score", f"{res.default_cv_score:.4f}")
            b.metric(
                "Tuned CV score",
                f"{res.best_cv_score:.4f}",
                delta=f"{res.best_cv_score - res.default_cv_score:+.4f}",
            )
            c.metric("Saved as", key)
            st.markdown(
                "**Best parameters:** "
                + ", ".join(f"`{k}={v}`" for k, v in res.best_params.items())
            )
            st.markdown("**Untouched test split**")
            st.dataframe(
                pd.DataFrame([res.test_metrics]).style.format("{:.4f}"),
                hide_index=True,
                use_container_width=True,
            )
            st.dataframe(res.results.head(10), use_container_width=True, hide_index=True)
            if res.best_cv_score - res.default_cv_score < 0.002:
                st.caption("Tuning gave little gain here; the defaults were already near-optimal.")
            try:
                imp = ml.permutation_importance_table(res.model)
                st.plotly_chart(
                    px.bar(
                        imp.head(15).iloc[::-1],
                        x="importance",
                        y="feature",
                        orientation="h",
                        template="plotly_dark",
                        title="Permutation importance (held-out data)",
                    ),
                    use_container_width=True,
                )
                st.caption("Importance shows predictive usefulness, not causation.")
            except Exception:
                pass
            render_code_button(
                "Tune " + model_name,
                ml.tuning_code(
                    target,
                    features,
                    task,
                    model_name,
                    space,
                    method,
                    n_iter,
                    cv,
                    scoring,
                    int(seed),
                    **load_kwargs(dataset),
                ),
                key="code_tune",
            )

    with tab_compare:
        if not models:
            st.info("Tune a model to add it here. Each tuning run is stored as an experiment.")
        else:
            rows = [
                {
                    "experiment": k,
                    "model": m.estimator_name,
                    "target": m.target,
                    "features": len(m.features),
                    "rows": m.rows_used,
                    "trained": m.trained_at,
                    **m.metrics,
                }
                for k, m in models.items()
            ]
            st.dataframe(
                pd.DataFrame(rows).style.format(precision=4, na_rep="-"),
                use_container_width=True,
                hide_index=True,
            )
            pick = st.multiselect("Compare parameters", list(models), default=list(models)[:3])
            if pick:
                st.dataframe(
                    pd.DataFrame(
                        {k: {p: str(v) for p, v in models[k].params.items()} for k in pick}
                    ).fillna("-"),
                    use_container_width=True,
                )
            if st.button("Clear experiments"):
                st.session_state[KEY_FITTED_MODELS] = {}
                st.rerun()
            st.page_link("pages/26_MLOps.py", label="Deploy a model →")
    return
