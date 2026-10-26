"""Section: Model Deployment - export a model and score new rows or files."""

import pandas as pd
import streamlit as st

from src.core.session_state import KEY_FITTED_MODELS, KEY_PREDICTION_LOG
from src.core.ui_components import render_code_button, require_dataset
from src.datascience import deployment as dep
from src.datascience import explain as ex
from src.datascience import modeling as ml
from src.datascience import tracking as tr
from src.workspace.registry import ModelRegistry, RegistryError, model_card_markdown

SUBTITLE = (
    "Package a trained model with its preprocessing, then score single records or whole files."
)


def render() -> None:
    """Render this section inside its parent page."""

    dataset = require_dataset("Model Deployment")
    df = dataset.working_df
    models: dict = st.session_state.get(KEY_FITTED_MODELS) or {}

    with st.expander("➕ Train a quick model", expanded=not models):
        q1, q2 = st.columns([1, 2])
        target = q1.selectbox("Target", list(df.columns), key="dep_target")
        features = q2.multiselect(
            "Features",
            [c for c in df.columns if c != target],
            default=[c for c in df.columns if c != target and not c.lower().endswith("id")][:10],
            key="dep_feats",
        )
        try:
            task = ml.infer_task(df[target])
        except Exception:
            task = ml.REGRESSION
        name = st.selectbox(
            "Model", [m for m in ml.candidate_estimators(task) if not m.startswith("Baseline")]
        )
        if st.button("Train", type="primary", disabled=not features):
            try:
                fitted = ml.fit_model(df, target, features, name, task)
                models[f"{fitted.name} (quick) #{len(models) + 1}"] = fitted
                st.session_state[KEY_FITTED_MODELS] = models
                st.rerun()
            except ml.ModelingError as err:
                st.error(str(err))
            except Exception as err:
                st.error(f"Training failed: {err}")

    if not models:
        st.info("Train a model above, or tune one on the Model Tuning page.")
        return

    key = st.selectbox("Model to deploy", list(models))
    model: ml.FittedModel = models[key]
    m1, m2, m3 = st.columns(3)
    m1.metric("Task", model.task.replace("_", " ").title())
    m2.metric("Features", len(model.features))
    m3.metric("Trained on rows", f"{model.rows_used:,}")
    st.caption("Holdout metrics: " + ", ".join(f"{k} = {v:.3f}" for k, v in model.metrics.items()))

    tab_one, tab_batch, tab_export, tab_monitor = st.tabs(
        ["🧍 Score one record", "📄 Score a file", "📦 Export & registry", "📈 Monitoring"]
    )

    with tab_one:
        values: dict = {}
        cols = st.columns(3)
        for i, feat in enumerate(model.features):
            info = model.feature_info[feat]
            with cols[i % 3]:
                if info["kind"] == "numeric":
                    lo, hi = info["min"], info["max"]
                    values[feat] = st.number_input(
                        feat,
                        value=info["default"],
                        help=f"Training range {lo:g} to {hi:g}",
                        key=f"in_{feat}",
                    )
                    if not lo <= values[feat] <= hi:
                        st.caption("⚠️ Outside the training range")
                else:
                    values[feat] = st.selectbox(
                        feat,
                        info["choices"],
                        index=info["choices"].index(info["default"]),
                        key=f"in_{feat}",
                    )
        if st.button("Predict", type="primary"):
            out, _ = ml.score_frame(model, pd.DataFrame([values]))
            pred = out[f"predicted_{model.target}"].iloc[0]
            st.metric(
                f"Predicted {model.target}",
                f"{pred:,.3f}" if isinstance(pred, float) else str(pred),
            )
            proba = {c: out[c].iloc[0] for c in out.columns if c.startswith("proba_")}
            if proba:
                st.bar_chart(pd.Series(proba))

    with tab_batch:
        up = st.file_uploader("CSV with the feature columns", type=["csv"], key="score_file")
        use_loaded = st.checkbox("…or score the loaded dataset")
        source = pd.read_csv(up) if up is not None else (df if use_loaded else None)
        if source is not None:
            check = ml.validate_scoring_frame(model, source)
            for level, msg in check["issues"]:
                (st.error if level == "error" else st.warning if level == "warning" else st.info)(
                    msg
                )
            if len(check["out_of_range"]):
                with st.expander("Values outside the training range"):
                    st.dataframe(check["out_of_range"], hide_index=True, use_container_width=True)
            if len(check["unseen_categories"]):
                with st.expander("Categories never seen in training"):
                    st.dataframe(
                        check["unseen_categories"], hide_index=True, use_container_width=True
                    )
            shared = [f for f in model.features if f in source.columns and f in df.columns]
            if shared and source is not df:
                drift = ex.drift_report(df, source, shared)
                major = drift[drift["status"] == "major shift"]
                if len(major):
                    st.error(
                        "Major drift versus the training data in: "
                        + ", ".join(major["feature"])
                        + ". Treat these predictions with caution."
                    )
                with st.expander("Drift report (PSI)"):
                    st.dataframe(
                        drift.style.format({"psi": "{:.3f}", "p_value": "{:.2g}"}, na_rep="-"),
                        hide_index=True,
                        use_container_width=True,
                    )
            if not check["ok"]:
                return
            scored, warnings = ml.score_frame(model, source)
            for w in warnings:
                st.warning(w)
            st.dataframe(scored.head(100), use_container_width=True)
            d1, d2 = st.columns(2)
            d1.download_button(
                "⬇️ Download predictions (CSV)",
                scored.to_csv(index=False),
                "predictions.csv",
                "text/csv",
            )
            if d2.button("Keep these predictions for monitoring", key="dep_log_btn"):
                score_col = next((c for c in scored.columns if c.startswith("proba_")), None)
                id_series = source["prediction_id"] if "prediction_id" in source else None
                st.session_state[KEY_PREDICTION_LOG] = tr.log_predictions(
                    st.session_state.get(KEY_PREDICTION_LOG),
                    key,
                    source[[f for f in model.features if f in source.columns]],
                    scored[f"predicted_{model.target}"],
                    ids=id_series,
                    proba=scored[score_col] if score_col else None,
                )
                st.toast(
                    f"{len(st.session_state[KEY_PREDICTION_LOG]):,} predictions logged", icon="✅"
                )

    with tab_export:
        st.download_button(
            "⬇️ Download model (.joblib)",
            ml.export_model_bytes(model),
            "model.joblib",
            "application/octet-stream",
            type="primary",
        )
        st.warning(
            "`.joblib` files are Python pickles: only load files you created or fully trust."
        )
        render_code_button("Score new data", ml.scoring_script(model), key="code_deploy")
        st.markdown("##### Serve it as an API")
        st.caption(
            "A FastAPI app with /health and /predict. Put model.joblib next to app.py, then run `uvicorn app:app`."
        )
        api_code = dep.fastapi_template(model)
        render_code_button(
            "FastAPI service", api_code, key="code_deploy_api", label="</> app.py (FastAPI)"
        )
        st.download_button("⬇️ app.py", api_code, "app.py", "text/x-python", key="dep_api_dl")
        st.download_button(
            "⬇️ requirements.txt",
            dep.requirements_text(model),
            "requirements.txt",
            "text/plain",
            key="dep_req_dl",
        )
        st.markdown("##### Model card & registry")
        card = model_card_markdown(
            model,
            key.split(" (")[0],
            1,
            "draft",
            extras={
                "data_fingerprint": getattr(dataset, "fingerprint", ""),
                "data_version": getattr(dataset, "version", 1),
            },
        )
        with st.expander("Preview the model card"):
            st.markdown(card)
        st.download_button(
            "⬇️ Model card (Markdown)", card, "MODEL_CARD.md", "text/markdown", key="dep_card_dl"
        )
        reg_name = st.text_input("Registry name", value=key.split(" (")[0], key="dep_reg_name")
        if st.button(
            "Register this model (versioned)", key="dep_register", disabled=not reg_name.strip()
        ):
            try:
                meta = ModelRegistry().register(
                    model,
                    reg_name,
                    extras={
                        "data_fingerprint": getattr(dataset, "fingerprint", ""),
                        "data_version": getattr(dataset, "version", 1),
                    },
                )
                st.success(
                    f"Registered **{meta['name']}** v{meta['version']}. Manage stages on the Experiment Tracker page."
                )
            except RegistryError as err:
                st.error(str(err))
        if model.holdout is not None and st.button("Refit on all rows for production"):
            models[f"{model.name} [refit on all data]"] = ml.refit_on_all(model, df)
            st.session_state[KEY_FITTED_MODELS] = models
            st.rerun()

    with tab_monitor:
        log = st.session_state.get(KEY_PREDICTION_LOG)
        if log is None or log.empty:
            st.info(
                "Keep predictions from the **Score a file** tab, then upload real outcomes here to watch live performance."
            )
        else:
            st.metric("Predictions logged", f"{len(log):,}")
            st.dataframe(log.head(50), use_container_width=True)
            st.download_button(
                "⬇️ Prediction log (CSV)",
                log.to_csv(index=False),
                "prediction_log.csv",
                "text/csv",
                key="dep_log_dl",
            )
            up_out = st.file_uploader(
                "Outcomes file (CSV with prediction_id and the real result)",
                type=["csv"],
                key="dep_outcomes",
            )
            if up_out is not None:
                outcomes = pd.read_csv(up_out)
                id_col = st.selectbox("Id column", outcomes.columns, key="dep_id_col")
                out_col = st.selectbox(
                    "Outcome column",
                    [c for c in outcomes.columns if c != id_col],
                    key="dep_out_col",
                )
                try:
                    labeled = tr.attach_outcomes(
                        log,
                        outcomes.rename(columns={id_col: "prediction_id"}),
                        "prediction_id",
                        out_col,
                    )
                    perf = tr.live_performance(labeled, model.task, model.metrics, 0.1)
                    a, b = st.columns(2)
                    a.metric("Predictions with outcomes", f"{perf['n_labeled']:,}")
                    for name, val in perf["metrics"].items():
                        b.metric(
                            name,
                            f"{val:.3f}",
                            delta=f"{perf.get('vs_baseline', {}).get(name, 0):+.3f} vs holdout"
                            if name in perf.get("vs_baseline", {})
                            else None,
                        )
                    for alert in perf["alerts"]:
                        st.warning(alert)
                except tr.TrackingError as err:
                    st.error(str(err))
            if st.button("Clear the log", key="dep_log_clear"):
                st.session_state[KEY_PREDICTION_LOG] = None
                st.rerun()
    return
