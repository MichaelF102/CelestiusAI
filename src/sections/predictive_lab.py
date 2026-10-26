"""Section: Predictive Lab — Leakage-Safe Model Training, Comparison, Diagnostics & Forecasting."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config.settings import get_settings
from src.core.code_snippets import load_kwargs, model_training_snippet
from src.core.session_state import (
    KEY_CURRENT_DATASET,
    KEY_PREDICTIVE_RECORDS,
    get_model_registry,
    is_dataset_loaded,
)
from src.core.ui_components import (
    render_code_button,
    render_empty_state_banner,
)
from src.ingestion.loader import ManagedDataset
from src.ingestion.schema import infer_dataset_schema
from src.predictive import (
    ModelRegistry,
    PredictionTask,
    ValidationStrategy,
    evaluate_target,
    train_predictive_models,
)

SUBTITLE = "Leakage-safe predictive modeling, baseline benchmarking, diagnostics, permutation explainability, and forecasting."


def render() -> None:
    """Render this section inside its parent page."""

    get_settings()

    # Ensure ModelRegistry in session state
    model_registry: ModelRegistry = get_model_registry(st.session_state)

    if not is_dataset_loaded(st.session_state):
        render_empty_state_banner("Predictive Analytics Lab", phase_label="Phase 5")
        st.page_link(
            "pages/2_Data_Quality.py", label="Go to Data Quality Lab to Upload a Dataset", icon="📥"
        )
        return

    dataset_obj = st.session_state.get(KEY_CURRENT_DATASET)
    if isinstance(dataset_obj, ManagedDataset):
        raw_df = dataset_obj.working_df
        dataset_obj.schema_info or infer_dataset_schema(raw_df)
        dataset_id = str(dataset_obj.dataset_id)
        dataset_version = getattr(dataset_obj, "version", 1)
    elif isinstance(dataset_obj, pd.DataFrame):
        raw_df = dataset_obj
        infer_dataset_schema(raw_df)
        dataset_id = "default_dataset"
        dataset_version = 1
    else:
        st.error("Invalid dataset representation in session state.")
        return

    if raw_df.empty or len(raw_df) < 10:
        st.warning("The dataset contains too few records (< 10) for predictive model training.")
        return

    # Interactive Tabs
    tab_setup, tab_models, tab_diag, tab_explain, tab_registry = st.tabs(
        [
            "1. Task & Setup",
            "2. Model Training & Comparison",
            "3. Diagnostics & Residuals",
            "4. Explainability & Uncertainty",
            "5. Model Registry & Provenance",
        ]
    )

    # -------------------------------------------------------------
    # TAB 1: Task Setup & Split Configuration
    # -------------------------------------------------------------
    with tab_setup:
        st.subheader("1. Target Variable & Task Definition")
        all_columns = list(raw_df.columns)

        col_target, col_task = st.columns([1, 1])

        with col_target:
            target_col = st.selectbox(
                "Select Target Variable to Predict:",
                options=all_columns,
                index=len(all_columns) - 1,
                help="Choose the outcome variable to model.",
            )

        # Evaluate Target
        rec = evaluate_target(raw_df, target_col)

        with col_task:
            task_options = [
                PredictionTask.REGRESSION,
                PredictionTask.BINARY_CLASSIFICATION,
                PredictionTask.MULTICLASS_CLASSIFICATION,
                PredictionTask.TIME_SERIES_FORECASTING,
            ]
            recommended_idx = (
                task_options.index(rec.recommended_task)
                if rec.recommended_task in task_options
                else 0
            )
            task_type = st.selectbox(
                "Prediction Task Type:",
                options=task_options,
                index=recommended_idx,
                format_func=lambda t: t.value.replace("_", " ").title(),
                help="Inferred based on target variable distribution and type.",
            )

        if rec.recommended_task:
            st.info(f"💡 **Recommendation:** {rec.reason}")
        if rec.warnings:
            for w in rec.warnings:
                st.warning(f"⚠️ {w}")

        if not rec.is_suitable:
            st.error(f"Selected target cannot be learned meaningfully: {rec.reason}")
            return

        st.markdown("---")
        st.subheader("2. Predictor Variables & Leakage Controls")

        # Filter out potential IDs or target itself
        default_predictors = [
            c for c in all_columns if c != target_col and not c.lower().endswith("id")
        ]

        col_preds, col_split = st.columns([1.2, 0.8])

        with col_preds:
            selected_predictors = st.multiselect(
                "Select Predictor Features:",
                options=[c for c in all_columns if c != target_col],
                default=default_predictors,
                help="Preprocessing pipeline fits exclusively on training data.",
            )

            time_col = None
            if task_type == PredictionTask.TIME_SERIES_FORECASTING:
                datetime_cols = [
                    c
                    for c in all_columns
                    if "date" in c.lower()
                    or "time" in c.lower()
                    or pd.api.types.is_datetime64_any_dtype(raw_df[c])
                ]
                time_col = st.selectbox(
                    "Select Timestamp / Date Column:",
                    options=datetime_cols if datetime_cols else all_columns,
                    help="Used for chronological ordering and walk-forward validation.",
                )

        with col_split:
            test_pct = st.slider(
                "Holdout Test Split Proportion (%):",
                min_value=10,
                max_value=40,
                value=20,
                step=5,
                help="Percentage of data strictly reserved for unbiased holdout evaluation.",
            )
            val_strategy = (
                ValidationStrategy.CHRONOLOGICAL_SPLIT
                if task_type == PredictionTask.TIME_SERIES_FORECASTING
                else ValidationStrategy.TRAIN_TEST_SPLIT
            )
            st.caption(f"Validation Strategy: **{val_strategy.value}**")

    # -------------------------------------------------------------
    # TAB 2: Model Training & Comparison
    # -------------------------------------------------------------
    with tab_models:
        st.subheader("Model Training & Benchmark Matrix")
        st.markdown(
            "Candidate models and defensible naive baselines are evaluated on the exact same holdout split. "
            "All data preprocessing (scaling, imputation, encoding) is fitted **strictly on training observations**."
        )

        if task_type != PredictionTask.TIME_SERIES_FORECASTING and selected_predictors:
            render_code_button(
                f"Predictive model: {target_col}",
                model_training_snippet(
                    target_col,
                    selected_predictors,
                    task_type.value,
                    test_pct / 100.0,
                    **load_kwargs(dataset_obj),
                ),
                key="code_predictive",
            )

        if st.button(
            "🚀 Train Models & Evaluate Baselines", type="primary", use_container_width=True
        ):
            if not selected_predictors and task_type != PredictionTask.TIME_SERIES_FORECASTING:
                st.error("Please select at least one predictor feature.")
            else:
                with st.spinner("Executing leakage-safe pipeline and evaluating models..."):
                    records = train_predictive_models(
                        df=raw_df,
                        target_col=target_col,
                        feature_cols=selected_predictors,
                        task_type=task_type,
                        dataset_id=dataset_id,
                        dataset_version=dataset_version,
                        time_col=time_col,
                        validation_strategy=val_strategy,
                        test_size=test_pct / 100.0,
                    )
                    model_registry.register_models(records)
                    st.session_state[KEY_PREDICTIVE_RECORDS] = records
                    st.success(f"Successfully trained and evaluated {len(records)} models.")

        current_records = st.session_state.get(KEY_PREDICTIVE_RECORDS, [])

        if current_records:
            best_model = current_records[0]
            st.info(
                f"🏆 **Top Performer:** **{best_model.model_name}** ({best_model.family.value}) "
                f"| Train Duration: {best_model.training_duration_ms / 1000.0:.2f}s"
            )

            table_data = []
            for r in current_records:
                if r.regression_metrics:
                    primary_score = f"RMSE: {r.regression_metrics.rmse:.3f}"
                    secondary_score = f"MAE: {r.regression_metrics.mae:.3f}"
                    third_score = f"R²: {r.regression_metrics.r_squared:.3f}"
                elif r.classification_metrics:
                    primary_score = f"F1: {r.classification_metrics.f1_macro:.3f}"
                    secondary_score = f"Acc: {r.classification_metrics.accuracy:.3f}"
                    third_score = (
                        f"ROC-AUC: {r.classification_metrics.roc_auc:.3f}"
                        if r.classification_metrics.roc_auc is not None
                        else "N/A"
                    )
                elif r.forecasting_metrics:
                    primary_score = f"RMSE: {r.forecasting_metrics.rmse:.3f}"
                    secondary_score = f"MAE: {r.forecasting_metrics.mae:.3f}"
                    third_score = f"sMAPE: {r.forecasting_metrics.smape:.2f}%"
                else:
                    primary_score = "N/A"
                    secondary_score = "N/A"
                    third_score = "N/A"

                test_n = r.split_info.test_size if r.split_info else "Holdout"

                table_data.append(
                    {
                        "Model": r.model_name,
                        "Family": r.family.value,
                        "Primary Metric": primary_score,
                        "Metric 2": secondary_score,
                        "Metric 3": third_score,
                        "Status": r.fit_status,
                        "Duration": f"{r.training_duration_ms / 1000.0:.2f}s",
                        "Split": f"Test N={test_n}",
                    }
                )

            st.dataframe(pd.DataFrame(table_data), use_container_width=True)
        else:
            st.info(
                "No models trained yet for this session. Configure your task and click 'Train Models' above."
            )

    # -------------------------------------------------------------
    # TAB 3: Diagnostics & Residuals
    # -------------------------------------------------------------
    with tab_diag:
        st.subheader("Holdout Model Diagnostics")

        if not current_records:
            st.info("Train models in Tab 2 to view residual diagnostics and calibration charts.")
        else:
            selected_model_name = st.selectbox(
                "Select Model to Inspect:",
                options=[r.model_name for r in current_records],
                index=0,
            )
            model_rec = next(r for r in current_records if r.model_name == selected_model_name)
            diag = model_rec.diagnostics

            if not diag:
                st.warning("Diagnostics unavailable for this model.")
            elif model_rec.task == PredictionTask.TIME_SERIES_FORECASTING:
                st.markdown("#### Chronological Forecast vs Actuals")
                fc = model_rec.forecast
                if fc and fc.future_timestamps and fc.point_forecasts:
                    fig_ts = go.Figure()
                    fig_ts.add_trace(
                        go.Scatter(
                            x=fc.future_timestamps,
                            y=fc.point_forecasts,
                            mode="lines+markers",
                            name="Point Forecast",
                            line={"dash": "dash", "color": "#38bdf8"},
                        )
                    )

                    if fc.lower_confidence and fc.upper_confidence:
                        fig_ts.add_trace(
                            go.Scatter(
                                x=fc.future_timestamps,
                                y=fc.upper_confidence,
                                mode="lines",
                                line={"width": 0},
                                showlegend=False,
                            )
                        )
                        fig_ts.add_trace(
                            go.Scatter(
                                x=fc.future_timestamps,
                                y=fc.lower_confidence,
                                mode="lines",
                                fill="tonexty",
                                fillcolor="rgba(56, 189, 248, 0.15)",
                                name="90% Conformal Interval",
                            )
                        )

                    fig_ts.update_layout(
                        title="Future Projections & Uncertainty Bounds",
                        xaxis_title="Date",
                        yaxis_title="Target",
                    )
                    st.plotly_chart(fig_ts, use_container_width=True)

            elif model_rec.task == PredictionTask.REGRESSION:
                col_d1, col_d2 = st.columns(2)

                with col_d1:
                    st.markdown("#### Actual vs. Predicted (Holdout)")
                    if diag.actual_values and diag.predicted_values:
                        pts = pd.DataFrame(
                            {"actual": diag.actual_values, "predicted": diag.predicted_values}
                        )
                        fig_avp = px.scatter(
                            pts,
                            x="actual",
                            y="predicted",
                            opacity=0.7,
                            title="Actual vs. Predicted",
                        )
                        min_val = min(pts["actual"].min(), pts["predicted"].min())
                        max_val = max(pts["actual"].max(), pts["predicted"].max())
                        fig_avp.add_shape(
                            type="line",
                            x0=min_val,
                            y0=min_val,
                            x1=max_val,
                            y1=max_val,
                            line={"color": "red", "dash": "dash"},
                        )
                        st.plotly_chart(fig_avp, use_container_width=True)

                with col_d2:
                    st.markdown("#### Residuals vs. Predicted")
                    if diag.residuals and diag.predicted_values:
                        res_df = pd.DataFrame(
                            {"predicted": diag.predicted_values, "residual": diag.residuals}
                        )
                        fig_res = px.scatter(
                            res_df, x="predicted", y="residual", opacity=0.7, title="Residual Plot"
                        )
                        fig_res.add_hline(y=0, line_dash="dash", line_color="red")
                        st.plotly_chart(fig_res, use_container_width=True)

            else:  # Classification
                col_c1, col_c2 = st.columns(2)

                with col_c1:
                    st.markdown("#### Confusion Matrix")
                    cls_m = model_rec.classification_metrics
                    if cls_m and cls_m.confusion_matrix and cls_m.class_labels:
                        cm_df = pd.DataFrame(
                            cls_m.confusion_matrix,
                            index=cls_m.class_labels,
                            columns=cls_m.class_labels,
                        )
                        fig_cm = px.imshow(
                            cm_df,
                            text_auto=True,
                            color_continuous_scale="Blues",
                            title="Confusion Matrix",
                        )
                        st.plotly_chart(fig_cm, use_container_width=True)

                with col_c2:
                    st.markdown("#### ROC Curve")
                    if diag.roc_curve and "fpr" in diag.roc_curve and "tpr" in diag.roc_curve:
                        roc_df = pd.DataFrame(
                            {"fpr": diag.roc_curve["fpr"], "tpr": diag.roc_curve["tpr"]}
                        )
                        fig_roc = px.line(roc_df, x="fpr", y="tpr", title="ROC Curve")
                        fig_roc.add_shape(
                            type="line",
                            x0=0,
                            y0=0,
                            x1=1,
                            y1=1,
                            line={"dash": "dash", "color": "gray"},
                        )
                        st.plotly_chart(fig_roc, use_container_width=True)

    # -------------------------------------------------------------
    # TAB 4: Explainability & Uncertainty
    # -------------------------------------------------------------
    with tab_explain:
        st.subheader("Feature Importance & Uncertainty Quantification")

        if not current_records:
            st.info("Train models in Tab 2 to view feature importance and prediction intervals.")
        else:
            best_model = current_records[0]
            st.markdown(f"**Explaining Model:** `{best_model.model_name}`")

            exp = best_model.feature_importance
            if exp and exp.feature_names and exp.importances:
                fi_df = pd.DataFrame(
                    {
                        "Feature": exp.feature_names,
                        "Importance": exp.importances,
                    }
                ).sort_values(by="Importance", ascending=True)

                fig_fi = px.bar(
                    fi_df,
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    title=f"Feature Importance ({exp.method})",
                    color="Importance",
                    color_continuous_scale="Viridis",
                )
                st.plotly_chart(fig_fi, use_container_width=True)

                st.caption(f"💡 {exp.caveat}")
            else:
                st.info("Feature importance not computed for this model.")

            # Uncertainty intervals
            unc = best_model.uncertainty
            if unc:
                st.markdown("---")
                st.markdown("#### Prediction Uncertainty Intervals (Conformal)")
                u1, u2 = st.columns(2)
                u1.metric("Nominal Confidence", f"{unc.confidence_level * 100:.0f}%")
                u2.metric("Interval Method", unc.interval_method)

                for a in unc.assumptions:
                    st.caption(f"• {a}")

    # -------------------------------------------------------------
    # TAB 5: Model Registry & Provenance
    # -------------------------------------------------------------
    with tab_registry:
        st.subheader("Model Registry & Dataset Provenance")
        st.markdown(
            "Every trained model records exact dataset version IDs, random seeds, and split parameters. "
            "If the active working dataset is cleaned or modified, models are automatically marked as **stale**."
        )

        registered_models = model_registry.list_models(
            dataset_id=dataset_id, active_version=dataset_version
        )

        if not registered_models:
            st.info("No models registered yet.")
        else:
            reg_rows = []
            for rm in registered_models:
                reg_rows.append(
                    {
                        "Model ID": rm.model_id,
                        "Name": rm.model_name,
                        "Task": rm.task.value,
                        "Target": rm.target_column,
                        "Dataset v": rm.dataset_version,
                        "Stale": "⚠️ Yes" if rm.is_stale else "✅ Current",
                        "Created At": rm.created_at[:19].replace("T", " ") + " UTC",
                    }
                )
            st.dataframe(pd.DataFrame(reg_rows), use_container_width=True)
    return
