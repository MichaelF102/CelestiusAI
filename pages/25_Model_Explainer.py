"""Page 25: Model Explainer - global and local explanations, what-if, counterfactuals, fairness, drift."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from src.core.code_ds import drift_snippet, explainer_snippet, fairness_snippet
from src.core.code_snippets import load_kwargs
from src.core.ds_ui import model_multiselect, pick_target_features, store_fitted_model
from src.core.session_state import KEY_FITTED_MODELS
from src.core.ui_components import (
    page_setup,
    render_code_button,
    render_code_export,
    require_dataset,
)
from src.datascience import explain as ex
from src.datascience import modeling as ml
from src.datascience.modeling import ModelingError

page_setup(
    "Model Explainer",
    "💡",
    "Understand what a model learned: global effects, single predictions, what-ifs, fairness, and drift.",
)
dataset = require_dataset("Model Explainer")
df = dataset.working_df
LOAD = load_kwargs(dataset)
models: dict = st.session_state.get(KEY_FITTED_MODELS) or {}

with st.expander("🎛️ Model to explain", expanded=not models):
    if models:
        st.caption(
            "Choose a model from Model Tuning, AutoML, or Deployment, or train a new one below."
        )
    target, features, task = pick_target_features(df, "xp")
    pick = model_multiselect(
        task, "xp_train_model", ["Random Forest"], "Train this model", include_baseline=False
    )
    if st.button("Train a model to explain", disabled=not (features and pick), key="xp_train"):
        try:
            fitted = ml.fit_model(df, target, features, pick[0], task)
            store_fitted_model(fitted, f"{pick[0]} (explainer)")
            st.rerun()
        except ModelingError as err:
            st.error(str(err))

models = st.session_state.get(KEY_FITTED_MODELS) or {}
if not models:
    st.info("Train a model above to begin.")
    st.stop()
key = st.selectbox("Explain", list(models), key="xp_model_key")
model: ml.FittedModel = models[key]
if model.holdout is None:
    st.warning(
        "This model was refit on all rows and has no held-out data. Pick one trained with a held-out split."
    )
    st.stop()
X_te, y_te = model.holdout
positive = model.classes[-1] if model.classes else None
st.caption(
    f"{model.estimator_name} · {model.task.replace('_', ' ')} · target `{model.target}` · {len(X_te):,} held-out rows · metrics: "
    + ", ".join(f"{k}={v:.3f}" for k, v in model.metrics.items())
)

t_glob, t_loc, t_fair, t_drift = st.tabs(
    ["🌍 Global", "🔎 One prediction", "⚖️ Fairness", "🌊 Drift"]
)

# ------------------------------------------------------------------ global
with t_glob:
    if st.button("Compute permutation importance", key="xp_imp_btn"):
        with st.spinner("Shuffling each feature on held-out rows..."):
            st.session_state["xp_imp"] = (key, ml.permutation_importance_table(model))
    imp = st.session_state.get("xp_imp")
    if imp and imp[0] == key:
        t = imp[1]
        fig = go.Figure(
            go.Bar(
                x=t["importance"][::-1],
                y=t["feature"][::-1],
                orientation="h",
                error_x={"type": "data", "array": t["std"][::-1]},
            )
        )
        fig.update_layout(
            template="plotly_dark",
            title="Drop in score when the feature is shuffled (held-out)",
            height=max(260, 32 * len(t)),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Importance measures predictive usefulness, not cause and effect.")
    feat = st.selectbox(
        "Feature effect (partial dependence + ICE)", model.features, key="xp_pdp_feat"
    )
    try:
        r = ex.pdp_ice(model.pipeline, X_te, feat)
        grid = [str(g) for g in r["grid"]] if r["categorical"] else r["grid"]
        fig = go.Figure()
        if not r["categorical"]:
            for curve in r["ice"][:25]:
                fig.add_scatter(
                    x=grid,
                    y=curve,
                    mode="lines",
                    line={"color": "rgba(150,150,150,0.25)"},
                    showlegend=False,
                )
        fig.add_scatter(
            x=grid,
            y=r["average"],
            mode="lines+markers" if r["categorical"] else "lines",
            line={"color": "#ef4444", "width": 3},
            name="average effect",
        )
        fig.update_layout(
            template="plotly_dark",
            title=f"How {feat} changes the prediction",
            xaxis_title=feat,
            yaxis_title="prediction" if model.task == "regression" else f"P({positive})",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Grey lines are individual rows (ICE): if they fan out, the effect depends on other features."
        )
    except Exception as err:
        st.info(f"Could not compute the effect: {err}")
    nums = [c for c in model.features if pd.api.types.is_numeric_dtype(X_te[c])]
    if len(nums) >= 2:
        i1, i2 = st.columns(2)
        f1, f2 = (
            i1.selectbox("Interaction: feature 1", nums, key="xp_i1"),
            i2.selectbox("feature 2", [c for c in nums if c != nums[0]] or nums, key="xp_i2"),
        )
        if f1 != f2 and st.button("Measure interaction", key="xp_int_btn"):
            h = ex.interaction_strength(model.pipeline, X_te, f1, f2)
            st.metric(
                "Interaction strength (H)",
                f"{h:.2f}",
                help="0 = the two effects simply add; higher = the effect of one depends on the other.",
            )
    render_code_button(
        "Explain the model",
        lambda o: explainer_snippet(
            model.estimator_name, model.target, model.features, model.task, feat, options=o, **LOAD
        ),
        key="code_xp",
    )

# ------------------------------------------------------------------ local
with t_loc:
    idx = st.selectbox("Held-out row", list(X_te.index[:300]), key="xp_row")
    row = X_te.loc[idx]
    st.dataframe(row.to_frame("value").T, use_container_width=True)
    pred = model.pipeline.predict(pd.DataFrame([row]))[0]
    actual = y_te.loc[idx]
    a, b = st.columns(2)
    a.metric("Prediction", f"{pred:,.3f}" if isinstance(pred, (float, np.floating)) else str(pred))
    b.metric(
        "Actual", f"{actual:,.3f}" if isinstance(actual, (float, np.floating)) else str(actual)
    )
    perms = st.slider("Accuracy of the explanation (more = slower)", 20, 300, 80, 20, key="xp_perm")
    if st.button("Explain this prediction", type="primary", key="xp_shap_btn"):
        with st.spinner("Estimating Shapley values..."):
            st.session_state["xp_shap"] = (
                key,
                idx,
                ex.shapley_local(model.pipeline, row, X_te, perms, positive),
            )
    shap = st.session_state.get("xp_shap")
    if shap and shap[0] == key and shap[1] == idx:
        res = shap[2]
        t = res["table"]
        st.caption(
            f"Starting point (average prediction): {res['baseline']:.3f} → this row: {res['prediction']:.3f}. Each bar is how much that feature moved the prediction."
        )
        fig = go.Figure(
            go.Bar(
                x=t["contribution"][::-1],
                y=[f"{f} = {v}" for f, v in zip(t["feature"][::-1], t["value"][::-1], strict=True)],
                orientation="h",
                marker_color=["#10b981" if c > 0 else "#ef4444" for c in t["contribution"][::-1]],
            )
        )
        fig.update_layout(
            template="plotly_dark",
            height=max(260, 34 * len(t)),
            title="Contribution to this prediction",
        )
        st.plotly_chart(fig, use_container_width=True)
        if ex.shap_available():
            with st.expander("SHAP (KernelExplainer) for comparison"):
                sh = ex.shap_local(model.pipeline, row, X_te, positive)
                if sh:
                    st.dataframe(
                        sh["table"].style.format({"shap": "{:+.4f}"}),
                        hide_index=True,
                        use_container_width=True,
                    )
        else:
            st.caption("Install `shap` for a second, library-based explanation.")
    st.markdown("##### What if...?")
    edits = {}
    cols = st.columns(3)
    for i, f in enumerate(model.features):
        info = model.feature_info.get(f, {})
        with cols[i % 3]:
            if info.get("kind") == "numeric":
                lo, hi = float(info["min"]), float(info["max"])
                edits[f] = st.number_input(
                    f,
                    value=float(row[f]),
                    min_value=None,
                    max_value=None,
                    key=f"xp_wi_{f}",
                    help=f"training range {lo:g} to {hi:g}",
                )
            else:
                choices = info.get("choices", [str(row[f])])
                edits[f] = st.selectbox(
                    f,
                    choices,
                    index=choices.index(str(row[f])) if str(row[f]) in choices else 0,
                    key=f"xp_wi_{f}",
                )
    changed = {
        f: v
        for f, v in edits.items()
        if str(v) != str(row[f]) and not (isinstance(v, float) and float(row[f]) == v)
    }
    if changed:
        w = ex.what_if(model.pipeline, row.to_dict(), changed, positive)
        st.info(
            f"Changing {', '.join(changed)} moves the prediction from **{w['before']}** to **{w['after']}**"
            + (
                f" (probability {w['proba_before']:.2f} → {w['proba_after']:.2f})."
                if "proba_before" in w
                else "."
            )
        )
    st.markdown("##### Counterfactual: what is the smallest change that gets a different result?")
    if model.task == "regression":
        desired = st.number_input(
            "Desired prediction at least / at most",
            value=float(pred) * 1.1 if pred else 1.0,
            key="xp_cf_target",
        )
    else:
        desired = st.selectbox("Desired class", model.classes, key="xp_cf_class")
    steps_n = st.slider("Search resolution", 5, 25, 12, key="xp_cf_steps")
    if st.button("Search for a counterfactual", key="xp_cf_btn"):
        with st.spinner("Searching one- and two-feature changes..."):
            cf = ex.counterfactual(
                model.pipeline,
                row.to_dict(),
                X_te,
                desired,
                model.features,
                2,
                steps_n,
                regression_target=model.task == "regression",
            )
        if cf is None:
            st.warning(
                "No change in up to two numeric features reaches that result. Try a closer target or a higher resolution."
            )
        elif not cf["changes"]:
            st.success("The model already predicts that result for this row.")
        else:
            st.success(
                "Smallest change found: "
                + "; ".join(f"{f}: {row[f]:.3g} → {v:.3g}" for f, v in cf["changes"].items())
                + f" (prediction becomes {cf['prediction']})."
            )
            st.caption(
                "This describes the model, not the real world: changing a feature may not be possible or may not cause the outcome."
            )

# ------------------------------------------------------------------ fairness
with t_fair:
    if model.task == "regression":
        st.info("Group fairness metrics here are for classification models.")
    else:
        group = st.selectbox(
            "Compare groups by",
            [c for c in df.columns if c != model.target and 2 <= df[c].nunique() <= 12],
            key="xp_group",
        )
        if group:
            pred_all = pd.Series(model.pipeline.predict(X_te), index=X_te.index)
            rep = ex.fairness_report(y_te, pred_all, df.loc[X_te.index, group], positive)
            a, b, c = st.columns(3)
            a.metric(
                "Disparate impact",
                f"{rep['disparate_impact']:.2f}",
                help="Lowest group selection rate ÷ highest. Below 0.80 is a common warning.",
            )
            b.metric("True-positive-rate gap", f"{rep['tpr_gap']:.0%}")
            c.metric("False-positive-rate gap", f"{rep['fpr_gap']:.0%}")
            for w in rep["warnings"]:
                st.warning(w)
            if not rep["warnings"]:
                st.success(
                    "No disparities flagged at the checked thresholds (this is not proof of fairness)."
                )
            st.dataframe(
                rep["table"].style.format(
                    dict.fromkeys(
                        ("base_rate", "selection_rate", "tpr", "fpr", "precision", "accuracy"),
                        "{:.2f}",
                    ),
                    na_rep="-",
                ),
                hide_index=True,
                use_container_width=True,
            )
            st.plotly_chart(
                px.bar(
                    rep["table"].melt(id_vars="group", value_vars=["selection_rate", "tpr", "fpr"]),
                    x="group",
                    y="value",
                    color="variable",
                    barmode="group",
                    template="plotly_dark",
                ),
                use_container_width=True,
            )
            render_code_button(
                "Fairness by group",
                lambda o: fairness_snippet(
                    group, model.target, model.features, model.estimator_name, options=o, **LOAD
                ),
                key="code_xp_fair",
            )

# ------------------------------------------------------------------ drift
with t_drift:
    st.caption(
        "Before scoring a new file, check whether it still looks like the data the model was trained on."
    )
    up = st.file_uploader("New data (CSV)", type=["csv"], key="xp_drift_file")
    use_self = st.checkbox(
        "Demo: compare the held-out rows with the full dataset", key="xp_drift_demo"
    )
    new = pd.read_csv(up) if up is not None else (X_te if use_self else None)
    if new is not None:
        rep = ex.drift_report(df, new, model.features)
        st.dataframe(
            rep.style.format({"psi": "{:.3f}", "p_value": "{:.2g}"}, na_rep="-"),
            hide_index=True,
            use_container_width=True,
        )
        st.plotly_chart(
            px.bar(
                rep,
                x="feature",
                y="psi",
                color="status",
                template="plotly_dark",
                color_discrete_map={
                    "stable": "#10b981",
                    "moderate shift": "#eab308",
                    "major shift": "#ef4444",
                    "missing in new data": "#94a3b8",
                },
            ),
            use_container_width=True,
        )
        if (rep["status"] == "major shift").any():
            st.error(
                "Major shift in: "
                + ", ".join(rep.loc[rep["status"] == "major shift", "feature"])
                + ". Predictions may be unreliable."
            )
        check = ml.validate_scoring_frame(model, new)
        for level, msg in check["issues"]:
            (st.error if level == "error" else st.warning if level == "warning" else st.info)(msg)
    render_code_button(
        "Data drift (PSI)",
        lambda o: drift_snippet(model.features, options=o, **LOAD),
        key="code_xp_drift",
    )

render_code_export()
