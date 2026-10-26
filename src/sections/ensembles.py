"""Section: Ensembles - bagging, voting, and stacking compared with their base models."""

import plotly.express as px
import streamlit as st

from src.core.code_ds import ensemble_snippet
from src.core.code_snippets import load_kwargs
from src.core.ds_ui import model_multiselect, pick_target_features, record_run
from src.core.ui_components import (
    render_code_button,
    require_dataset,
)
from src.datascience import catalog
from src.datascience import ensembles as en
from src.datascience import selection as sel
from src.datascience.modeling import SCORING_CHOICES, ModelingError

SUBTITLE = "Combine models by bagging, voting, or stacking, and see whether the combination really beats its members."


def render() -> None:
    """Render this section inside its parent page."""

    dataset = require_dataset("Ensembles")
    df = dataset.working_df
    LOAD = load_kwargs(dataset)
    target, features, task = pick_target_features(df, "en")
    if not features:
        return
    is_reg = task == "regression"

    st.markdown(
        "**Bagging** averages many copies of one model trained on bootstrap samples (reduces variance). "
        "**Voting** averages different models. **Stacking** trains a meta-model to weigh them. "
        "Ensembles help most when the base models make *different* mistakes."
    )
    bases = model_multiselect(
        task,
        "en_bases",
        [
            "Ridge / L2 Logistic" if is_reg else "Linear / Logistic Regression",
            "Decision Tree",
            "K-Nearest Neighbors",
            "Random Forest",
        ],
        "Base models",
        include_baseline=False,
    )
    kinds_all = {k: v for k, v in en.ENSEMBLE_KINDS.items() if not (is_reg and k == "voting_hard")}
    kinds = st.multiselect(
        "Ensembles to build",
        list(kinds_all),
        default=["voting_soft", "stacking"],
        format_func=kinds_all.get,
        key="en_kinds",
    )
    c1, c2, c3 = st.columns(3)
    bag_base = (
        c1.selectbox("Bagging base model", bases or ["Decision Tree"], key="en_bag_base")
        if "bagging" in kinds
        else None
    )
    meta = (
        c2.selectbox(
            "Stacking meta-model",
            [s.name for s in catalog.available_models(task, include_baseline=False)],
            index=0,
            key="en_meta",
            format_func=lambda n: catalog.display_name(catalog.get(n), task),
        )
        if "stacking" in kinds
        else None
    )
    cv = c3.slider("Folds", 2, 8, 4, key="en_cv")
    scoring = st.selectbox(
        "Score",
        list(SCORING_CHOICES[task]),
        format_func=SCORING_CHOICES[task].get,
        key="en_scoring",
    )

    if st.button(
        "Run the comparison", type="primary", disabled=len(bases) < 2 or not kinds, key="en_run"
    ):
        specs = []
        for k in kinds:
            base_list = [bag_base] if k == "bagging" else bases
            specs.append(
                en.EnsembleSpec(
                    en.ENSEMBLE_KINDS[k].split(" (")[0] + f" [{k}]",
                    k,
                    base_list,
                    meta if k == "stacking" else None,
                    15,
                )
            )
        with st.spinner("Cross-validating base models and ensembles..."):
            try:
                st.session_state["en_study"] = (
                    target,
                    specs,
                    en.ensemble_study(df, target, features, bases, specs, task, cv, scoring),
                )
            except ModelingError as err:
                st.error(str(err))
            except Exception as err:
                st.error(f"Could not build the ensembles: {err}")

    stored = st.session_state.get("en_study")
    if stored and stored[0] == target:
        _, specs, study = stored
        t = study.table
        st.dataframe(
            t.style.format(
                {
                    "cv_mean": "{:.4f}",
                    "cv_std": "{:.4f}",
                    "fit_seconds": "{:.1f}",
                    "gain_vs_best_base": "{:+.4f}",
                },
                na_rep="-",
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.plotly_chart(
            px.bar(
                t,
                x="model",
                y="cv_mean",
                color="type",
                error_y="cv_std",
                template="plotly_dark",
                title="Cross-validated score",
            ),
            use_container_width=True,
        )
        s = study.summary
        if s.get("gain") is not None:
            if s["gain"] > 0.005:
                st.success(f"The best ensemble beats the best single model by {s['gain']:+.4f}.")
            else:
                st.warning(
                    f"The best ensemble gains only {s['gain']:+.4f}: a single model is simpler and about as good."
                )
        st.info(en.diversity_comment(s))
        if study.diversity is not None:
            d1, d2 = st.columns(2)
            d1.plotly_chart(
                px.imshow(
                    study.diversity,
                    text_auto=".2f",
                    zmin=-1,
                    zmax=1,
                    color_continuous_scale="RdBu_r",
                    template="plotly_dark",
                    title="Error correlation (lower = more diverse)",
                ),
                use_container_width=True,
            )
            if study.disagreement is not None and not study.disagreement.isna().all().all():
                d2.plotly_chart(
                    px.imshow(
                        study.disagreement,
                        text_auto=".2f",
                        color_continuous_scale="Blues",
                        template="plotly_dark",
                        title="How often two models disagree",
                    ),
                    use_container_width=True,
                )
        ens_rows = t[t["type"] != "base"]
        if len(ens_rows):
            best_ens, best_base = ens_rows.iloc[0]["model"], t[t["type"] == "base"].iloc[0]["model"]
            n_te = 1
            verdict = (
                sel.compare_models(
                    study.fold_scores[best_ens], study.fold_scores[best_base], 3 * n_te, n_te
                )
                if len(study.fold_scores[best_ens]) >= 3
                else None
            )
            if verdict:
                st.caption(
                    f"Best ensemble vs best base: p = {verdict['p_value']:.3f} ({verdict['verdict']}); few folds make this test conservative."
                )
            r1, r2 = st.columns(2)
            spec = (
                next(sp for sp in specs if sp.name == best_ens)
                if any(sp.name == best_ens for sp in specs)
                else specs[0]
            )
            with r1:
                render_code_button(
                    f"Ensemble: {spec.kind}",
                    lambda o: ensemble_snippet(
                        spec.kind,
                        spec.base_models,
                        target,
                        features,
                        task,
                        spec.meta,
                        cv=cv,
                        options=o,
                        **LOAD,
                    ),
                    key="code_en",
                )
            if r2.button("Log the best ensemble to the Experiment Tracker", key="en_log"):
                row = ens_rows.iloc[0]
                record_run(
                    row["model"],
                    "ensemble",
                    row["model"],
                    task,
                    target,
                    features,
                    {"cv_mean": row["cv_mean"], "cv_std": row["cv_std"]},
                    cv_scores=list(study.fold_scores[row["model"]]),
                    dataset=dataset,
                )
                st.toast("Logged", icon="✅")
    else:
        st.info("Pick at least two base models and one ensemble type, then run the comparison.")
    return
