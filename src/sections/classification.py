"""Section: Classification Toolkit - imbalance, evaluation, thresholds, calibration, multi-class."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.core.code_ds import classification_snippet
from src.core.code_snippets import load_kwargs
from src.core.ds_ui import model_card_box, pick_target_features, record_run
from src.core.ui_components import (
    render_code_button,
    require_dataset,
)
from src.datascience import catalog
from src.datascience import classification as cl
from src.datascience.modeling import ModelingError

SUBTITLE = "Handle imbalance, evaluate beyond accuracy, choose a cost-based threshold, and check calibration."


def render() -> None:
    """Render this section inside its parent page."""

    dataset = require_dataset("Classification Toolkit")
    df = dataset.working_df
    LOAD = load_kwargs(dataset)
    target, features, task = pick_target_features(df, "cl", task_filter="classification")
    if task == "regression":
        st.warning("The chosen target looks continuous. Pick a target with a few distinct classes.")
        return
    if not features:
        return

    models = [s.name for s in catalog.available_models(task, include_baseline=False)]
    model = st.selectbox(
        "Model",
        models,
        index=models.index("Random Forest") if "Random Forest" in models else 0,
        format_func=lambda n: catalog.display_name(catalog.get(n), task),
        key="cl_model",
    )
    with st.expander("About this model"):
        model_card_box(model, task)
    counts = df[target].value_counts()
    st.caption(
        "Class balance: "
        + " · ".join(f"{k}: {v:,} ({v / counts.sum():.0%})" for k, v in counts.items())
    )

    tab_imb, tab_eval, tab_thr, tab_cal, tab_multi = st.tabs(
        ["⚖️ Imbalance", "📊 Evaluation", "🎚️ Threshold & cost", "📐 Calibration", "🔀 Multi-class"]
    )

    with tab_imb:
        st.caption(
            "Accuracy rewards guessing the majority class. These strategies trade accuracy for catching the minority; resampling happens only inside each training fold."
        )
        chosen = st.multiselect(
            "Strategies",
            list(cl.RESAMPLING),
            default=list(cl.RESAMPLING),
            format_func=cl.RESAMPLING.get,
            key="cl_strats",
        )
        if st.button("Compare strategies", type="primary", key="cl_imb_run", disabled=not chosen):
            with st.spinner("Cross-validating each strategy..."):
                try:
                    st.session_state["cl_imb"] = (
                        model,
                        cl.imbalance_experiment(df, target, features, model, chosen),
                    )
                except ModelingError as err:
                    st.error(str(err))
        imb = st.session_state.get("cl_imb")
        if imb and imb[0] == model:
            t = imb[1]
            fmt = {c: "{:.3f}" for c in t.columns if c not in ("strategy", "note")}
            st.dataframe(t.style.format(fmt, na_rep="-"), use_container_width=True, hide_index=True)
            metric_cols = [
                c
                for c in (
                    "accuracy",
                    "balanced_accuracy",
                    "f1_minority",
                    "f1_macro",
                    "roc_auc",
                    "avg_precision",
                )
                if c in t.columns
            ]
            long = t.melt(
                id_vars="strategy", value_vars=metric_cols, var_name="metric", value_name="value"
            ).dropna()
            st.plotly_chart(
                px.bar(
                    long,
                    x="strategy",
                    y="value",
                    color="metric",
                    barmode="group",
                    template="plotly_dark",
                ),
                use_container_width=True,
            )
            if "none" in set(t["strategy"]):
                row = t[t["strategy"] == "none"].iloc[0]
                if row.get("accuracy", 0) - row.get("balanced_accuracy", 0) > 0.1:
                    st.warning(
                        f"Without resampling, accuracy ({row['accuracy']:.2f}) is far above balanced accuracy ({row['balanced_accuracy']:.2f}): the model mostly predicts the majority class."
                    )

    with tab_eval:
        strategy = st.selectbox(
            "Training strategy",
            ["none", "class_weight", "oversample", "undersample", "smote"],
            format_func=cl.RESAMPLING.get,
            key="cl_eval_strat",
        )
        pos_options = sorted(df[target].dropna().unique(), key=str)
        positive = (
            st.selectbox("Positive class", pos_options, index=len(pos_options) - 1, key="cl_pos")
            if len(pos_options) == 2
            else None
        )
        if st.button("Fit and evaluate", type="primary", key="cl_eval_run"):
            with st.spinner("Training on 75% and testing on the rest..."):
                try:
                    h = cl.holdout_fit(df, target, features, model, strategy, positive=positive)
                    st.session_state["cl_holdout"] = (
                        target,
                        model,
                        strategy,
                        h,
                        cl.evaluate_holdout(h),
                    )
                except ModelingError as err:
                    st.error(str(err))
        stored = st.session_state.get("cl_holdout")
        if stored and stored[0] == target and stored[1] == model:
            _, _, strat_used, h, ev = stored
            a, b, c, d = st.columns(4)
            a.metric("Accuracy", f"{ev['accuracy']:.3f}")
            b.metric("Balanced accuracy", f"{ev['balanced_accuracy']:.3f}")
            if "roc_auc" in ev:
                c.metric("ROC AUC", f"{ev['roc_auc']:.3f}")
                d.metric(
                    "Avg precision",
                    f"{ev['avg_precision']:.3f}",
                    help=f"Random guessing scores the prevalence ({ev['prevalence']:.2f}).",
                )
            else:
                c.metric("Macro F1", f"{ev['macro_f1']:.3f}")
                d.metric("Weighted F1", f"{ev['weighted_f1']:.3f}")
            if "roc_auc" in ev:
                r1, r2 = st.columns(2)
                roc = go.Figure(
                    go.Scatter(x=ev["roc_curve"]["fpr"], y=ev["roc_curve"]["tpr"], name="model")
                )
                roc.add_scatter(x=[0, 1], y=[0, 1], line={"dash": "dot"}, name="chance")
                roc.update_layout(
                    template="plotly_dark",
                    title=f"ROC (AUC {ev['roc_auc']:.3f}, KS {ev['ks']:.3f})",
                    xaxis_title="false positive rate",
                    yaxis_title="true positive rate",
                    height=320,
                )
                r1.plotly_chart(roc, use_container_width=True)
                pr = go.Figure(
                    go.Scatter(x=ev["pr_curve"]["recall"], y=ev["pr_curve"]["precision"])
                )
                pr.add_hline(y=ev["prevalence"], line_dash="dot")
                pr.update_layout(
                    template="plotly_dark",
                    title="Precision-recall",
                    xaxis_title="recall",
                    yaxis_title="precision",
                    height=320,
                )
                r2.plotly_chart(pr, use_container_width=True)
            m1, m2 = st.columns(2)
            mode = m1.radio(
                "Confusion matrix", ["counts", "row rates"], horizontal=True, key="cl_cm_mode"
            )
            m1.plotly_chart(
                px.imshow(
                    ev["confusion"] if mode == "counts" else ev["confusion_rates"],
                    text_auto=".2f" if mode != "counts" else True,
                    color_continuous_scale="Blues",
                    template="plotly_dark",
                ),
                use_container_width=True,
            )
            m2.markdown("**Per-class report**")
            m2.dataframe(
                ev["report"].style.format(
                    {"precision": "{:.3f}", "recall": "{:.3f}", "f1": "{:.3f}"}
                ),
                use_container_width=True,
                hide_index=True,
            )
            if "gain_lift" in ev:
                gl = ev["gain_lift"]
                g1, g2 = st.columns(2)
                g1.plotly_chart(
                    px.line(
                        gl,
                        x="group",
                        y="cum_gain",
                        markers=True,
                        template="plotly_dark",
                        title="Cumulative gain (share of positives found)",
                    ),
                    use_container_width=True,
                )
                g2.plotly_chart(
                    px.bar(
                        gl,
                        x="group",
                        y="lift",
                        template="plotly_dark",
                        title="Lift by score decile (1 = highest scores)",
                    ),
                    use_container_width=True,
                )
                st.caption(
                    f"The top 20% of scores contain {gl['cum_gain'].iloc[1]:.0%} of all positives (random would be 20%)."
                )
            e1, e2 = st.columns(2)
            with e1:
                render_code_button(
                    "Classification toolkit",
                    lambda o: classification_snippet(
                        model, target, features, strat_used, options=o, **LOAD
                    ),
                    key="code_cl",
                )
            if e2.button("Log this evaluation to the Experiment Tracker", key="cl_log"):
                keys = [
                    k
                    for k in (
                        "accuracy",
                        "balanced_accuracy",
                        "roc_auc",
                        "avg_precision",
                        "macro_f1",
                    )
                    if k in ev
                ]
                record_run(
                    f"{model} ({strat_used})",
                    "manual",
                    model,
                    task,
                    target,
                    features,
                    {k: ev[k] for k in keys},
                    dataset=dataset,
                )
                st.toast("Logged", icon="✅")

    stored = st.session_state.get("cl_holdout")
    ready = bool(
        stored and stored[0] == target and stored[1] == model and len(stored[3].classes) == 2
    )
    with tab_thr:
        if not ready:
            st.info("Fit a binary model on the Evaluation tab first.")
        else:
            h = stored[3]
            pos = (h.y_test == h.positive).to_numpy()
            score = h.proba[:, h.classes.index(h.positive)]
            st.caption(
                "Set what each kind of mistake costs. The cheapest cutoff is usually not 0.5."
            )
            k1, k2, k3, k4 = st.columns(4)
            cost_fp, cost_fn = (
                k1.number_input("Cost of a false positive", 0.0, 1e6, 1.0, key="cl_cfp"),
                k2.number_input("Cost of a false negative", 0.0, 1e6, 5.0, key="cl_cfn"),
            )
            gain_tp, gain_tn = (
                k3.number_input("Benefit of a true positive", 0.0, 1e6, 0.0, key="cl_btp"),
                k4.number_input("Benefit of a true negative", 0.0, 1e6, 0.0, key="cl_btn"),
            )
            res = cl.optimal_threshold(pos, score, cost_fp, cost_fn, gain_tp, gain_tn)
            m1, m2, m3 = st.columns(3)
            m1.metric("Cheapest threshold", f"{res['threshold']:.2f}")
            m2.metric(
                "Total cost",
                f"{res['cost']:,.1f}",
                delta=f"{-res['saving_vs_0.5']:,.1f} vs 0.50",
                delta_color="inverse",
            )
            m3.metric("Recall at that cutoff", f"{res['recall']:.1%}")
            table = res["table"]
            chosen = st.slider(
                "Threshold to inspect",
                0.01,
                0.99,
                float(round(res["threshold"], 2)),
                0.01,
                key="cl_thr",
            )
            row = table.iloc[(table["threshold"] - chosen).abs().argmin()]
            fig = go.Figure()
            fig.add_scatter(x=table["threshold"], y=table["cost"], name="total cost")
            fig.add_vline(x=chosen, line_dash="dot")
            fig.update_layout(
                template="plotly_dark",
                title="Cost by threshold",
                xaxis_title="threshold",
                yaxis_title="cost",
                height=300,
            )
            st.plotly_chart(fig, use_container_width=True)
            p1, p2 = st.columns(2)
            p1.plotly_chart(
                px.line(
                    table,
                    x="threshold",
                    y=["precision", "recall", "f1"],
                    template="plotly_dark",
                    title="Precision / recall trade-off",
                ),
                use_container_width=True,
            )
            p2.dataframe(
                pd.DataFrame(
                    {
                        "": ["actual positive", "actual negative"],
                        "predicted positive": [int(row["tp"]), int(row["fp"])],
                        "predicted negative": [int(row["fn"]), int(row["tn"])],
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )

    with tab_cal:
        if not ready:
            st.info("Fit a binary model on the Evaluation tab first.")
        else:
            h = stored[3]
            pos = (h.y_test == h.positive).to_numpy()
            score = h.proba[:, h.classes.index(h.positive)]
            summary = cl.calibration_summary(pos, score)
            a, b = st.columns(2)
            a.metric(
                "Brier score",
                f"{summary['brier']:.4f}",
                help="Lower is better; 0.25 is a coin flip on balanced data.",
            )
            b.metric("Expected calibration error", f"{summary['ece']:.4f}")
            curve = summary["curve"]
            fig = go.Figure(
                go.Scatter(
                    x=curve["mean_predicted"],
                    y=curve["fraction_positive"],
                    mode="lines+markers",
                    name="model",
                )
            )
            fig.add_scatter(x=[0, 1], y=[0, 1], line={"dash": "dot"}, name="perfect")
            fig.update_layout(
                template="plotly_dark",
                title="Reliability curve",
                xaxis_title="predicted probability",
                yaxis_title="observed frequency",
                height=340,
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Points below the diagonal mean the model is over-confident; above, under-confident."
            )
            method = st.radio(
                "Recalibrate with", ["sigmoid", "isotonic"], horizontal=True, key="cl_cal_method"
            )
            if st.button("Recalibrate and compare", key="cl_cal_run"):
                try:
                    res = cl.calibrate_holdout(h, method)
                    st.metric(
                        "Brier before → after",
                        f"{res['brier_before']:.4f} → {res['brier_after']:.4f}",
                        delta=f"{res['brier_after'] - res['brier_before']:+.4f}",
                        delta_color="inverse",
                    )
                    st.success(
                        "Calibration improved the probabilities."
                        if res["improved"]
                        else "Calibration did not help here; the model was already reasonably calibrated."
                    )
                except ModelingError as err:
                    st.error(str(err))

    with tab_multi:
        if len(counts) < 3:
            st.info("This target has only two classes. Multi-class strategies apply to 3+ classes.")
        else:
            st.caption(
                "One-vs-rest trains one model per class; one-vs-one trains a model per pair of classes. Compared by cross-validated macro F1."
            )
            if st.button("Compare strategies", key="cl_multi_run"):
                from sklearn.model_selection import StratifiedKFold, cross_val_score
                from sklearn.pipeline import Pipeline

                from src.datascience.modeling import build_preprocessor

                Xs = df[features]
                ys = df[target]
                rows = []
                with st.spinner("Cross-validating..."):
                    for kind in ("native", "ovr", "ovo"):
                        est = cl.multiclass_wrapper(model, task, kind)
                        pipe = Pipeline([("prep", build_preprocessor(Xs)), ("m", est)])
                        scores = cross_val_score(
                            pipe,
                            Xs,
                            ys,
                            cv=StratifiedKFold(3, shuffle=True, random_state=0),
                            scoring="f1_macro",
                        )
                        rows.append(
                            {"strategy": kind, "macro_f1": scores.mean(), "std": scores.std(ddof=1)}
                        )
                st.dataframe(
                    pd.DataFrame(rows).style.format({"macro_f1": "{:.4f}", "std": "{:.4f}"}),
                    hide_index=True,
                    use_container_width=True,
                )
    return
