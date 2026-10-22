"""Predictive Model Training Service.

Coordinates data splitting, leakage-safe preprocessing, baseline estimation,
candidate model training, metric evaluation, diagnostics, and explainability.
Ensures zero target leakage and strictly training-only fits.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge

from src.predictive.diagnostics import ModelDiagnostics
from src.predictive.evaluation import ModelEvaluator
from src.predictive.explainability import ModelExplainability
from src.predictive.forecasting import TimeSeriesForecastingEngine
from src.predictive.predictive_models import (
    ModelFamily,
    PredictionTask,
    TrainedModelRecord,
    ValidationStrategy,
)
from src.predictive.preprocessing import LeakageSafePreprocessor
from src.predictive.split_strategy import SplitStrategyService
from src.predictive.uncertainty import UncertaintyEstimator

logger = logging.getLogger(__name__)


def train_predictive_models(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str] | None = None,
    task_type: PredictionTask = PredictionTask.REGRESSION,
    dataset_id: str = "dataset",
    dataset_version: int = 1,
    time_col: str | None = None,
    validation_strategy: ValidationStrategy = ValidationStrategy.TRAIN_TEST_SPLIT,
    test_size: float = 0.2,
    random_seed: int = 42,
    max_train_samples: int = 15000,
) -> list[TrainedModelRecord]:
    """Train baselines and candidate models on the specified task and dataset.

    Returns a list of TrainedModelRecord objects, sorted by primary metric.
    """
    # 1. Special case: Time Series Forecasting
    if task_type == PredictionTask.TIME_SERIES_FORECASTING:
        if not time_col:
            raise ValueError("Time column is required for time-series forecasting.")
        clean_ts = TimeSeriesForecastingEngine.prepare_time_series(df, time_col, target_col)
        naive_rec, candidate_rec = TimeSeriesForecastingEngine.fit_and_evaluate_forecast(
            series_df=clean_ts,
            time_column=time_col,
            target_column=target_col,
            horizon=max(7, int(len(clean_ts) * test_size)),
            dataset_id=dataset_id,
            dataset_version=dataset_version,
        )
        # Mark recommendation based on RMSE
        if candidate_rec.forecasting_metrics and naive_rec.forecasting_metrics:
            if candidate_rec.forecasting_metrics.rmse < naive_rec.forecasting_metrics.rmse:
                candidate_rec.is_recommended = True
            else:
                naive_rec.is_recommended = True
        return [candidate_rec, naive_rec]

    # 2. Filter / Sample data if very large
    work_df = df.copy()
    if len(work_df) > max_train_samples:
        work_df = work_df.sample(n=max_train_samples, random_state=random_seed)

    if feature_cols:
        cols_to_keep = list(set(feature_cols + [target_col]))
        work_df = work_df[[c for c in cols_to_keep if c in work_df.columns]]

    # 3. Data Splitting
    X_train, X_test, y_train, y_test, split_info = SplitStrategyService.partition_data(
        df=work_df,
        target_column=target_col,
        task=task_type,
        strategy=validation_strategy,
        test_ratio=test_size,
        random_seed=random_seed,
    )

    # 4. Leakage-safe Preprocessing: Identify numeric vs categorical
    num_cols = [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c])]
    cat_cols = [c for c in X_train.columns if c not in num_cols]

    preprocessor = LeakageSafePreprocessor(numeric_features=num_cols, categorical_features=cat_cols)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    proc_feature_names = preprocessor.get_feature_names_out()

    # Targets as numpy arrays
    if task_type in (
        PredictionTask.BINARY_CLASSIFICATION,
        PredictionTask.MULTICLASS_CLASSIFICATION,
    ):
        y_tr = np.asarray(y_train).ravel()
        y_te = np.asarray(y_test).ravel()
    else:
        y_tr = np.asarray(y_train, dtype=float).ravel()
        y_te = np.asarray(y_test, dtype=float).ravel()

    # 5. Model Candidate Portfolio
    models_to_train: list[tuple[str, ModelFamily, Any, bool]] = []

    if task_type == PredictionTask.REGRESSION:
        models_to_train.append(
            (
                "Baseline: Mean Dummy",
                ModelFamily.BASELINE_DUMMY,
                DummyRegressor(strategy="mean"),
                True,
            )
        )
        models_to_train.append(
            ("Linear Regression", ModelFamily.LINEAR_MODEL, LinearRegression(), False)
        )
        models_to_train.append(
            (
                "Ridge Regression",
                ModelFamily.RIDGE_REGRESSION,
                Ridge(alpha=1.0, random_state=random_seed),
                False,
            )
        )
        models_to_train.append(
            (
                "Random Forest Regressor",
                ModelFamily.RANDOM_FOREST,
                RandomForestRegressor(
                    n_estimators=50, max_depth=10, random_state=random_seed, n_jobs=-1
                ),
                False,
            )
        )
        models_to_train.append(
            (
                "HistGradientBoosting Regressor",
                ModelFamily.HIST_GRADIENT_BOOSTING,
                HistGradientBoostingRegressor(max_iter=50, max_depth=6, random_state=random_seed),
                False,
            )
        )

    else:
        models_to_train.append(
            (
                "Baseline: Majority Class Dummy",
                ModelFamily.BASELINE_DUMMY,
                DummyClassifier(strategy="most_frequent"),
                True,
            )
        )
        models_to_train.append(
            (
                "Logistic Regression",
                ModelFamily.LINEAR_MODEL,
                LogisticRegression(max_iter=500, random_state=random_seed),
                False,
            )
        )
        models_to_train.append(
            (
                "Random Forest Classifier",
                ModelFamily.RANDOM_FOREST,
                RandomForestClassifier(
                    n_estimators=50, max_depth=10, random_state=random_seed, n_jobs=-1
                ),
                False,
            )
        )
        models_to_train.append(
            (
                "HistGradientBoosting Classifier",
                ModelFamily.HIST_GRADIENT_BOOSTING,
                HistGradientBoostingClassifier(max_iter=50, max_depth=6, random_state=random_seed),
                False,
            )
        )

    records: list[TrainedModelRecord] = []

    for name, family, estimator, is_baseline in models_to_train:
        t0 = time.time()
        fit_status = "SUCCESS"
        warnings_list: list[str] = []
        reg_metrics = None
        cls_metrics = None
        diagnostics = None
        feature_importance = None
        uncertainty = None

        try:
            estimator.fit(X_train_proc, y_tr)
            train_duration_ms = (time.time() - t0) * 1000.0

            y_pred = estimator.predict(X_test_proc)
            y_prob = None
            if hasattr(estimator, "predict_proba"):
                try:
                    y_prob = estimator.predict_proba(X_test_proc)
                except Exception as ex:
                    logger.debug("predict_proba failed for %s: %s", name, ex)

            # Evaluate Metrics & Diagnostics
            if task_type == PredictionTask.REGRESSION:
                reg_metrics = ModelEvaluator.evaluate_regression(y_te, y_pred)
                diagnostics = ModelDiagnostics.generate_regression_diagnostics(y_te, y_pred)
                # Conformal prediction interval
                y_train_pred = estimator.predict(X_train_proc)
                uncertainty = UncertaintyEstimator.estimate_regression_intervals(
                    y_train_true=y_tr,
                    y_train_pred=y_train_pred,
                    y_test_pred=y_pred,
                    confidence_level=0.90,
                )
            else:
                cls_metrics = ModelEvaluator.evaluate_classification(y_te, y_pred, y_prob)
                diagnostics = ModelDiagnostics.generate_classification_diagnostics(
                    y_te, y_pred, y_prob
                )

            # Explainability: held-out permutation importance
            feature_importance = ModelExplainability.compute_importance(
                model=estimator,
                X_test=X_test_proc,
                y_test=y_te,
                feature_names=proc_feature_names,
                evaluation_split="test",
            )

        except Exception as exc:
            logger.error("Failed fitting %s: %s", name, exc, exc_info=True)
            fit_status = f"FAILED: {exc}"
            train_duration_ms = (time.time() - t0) * 1000.0
            warnings_list.append(str(exc))

        rec = TrainedModelRecord(
            model_name=name,
            family=family,
            task=task_type,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            target_column=target_col,
            predictor_columns=list(X_train.columns),
            split_info=split_info,
            regression_metrics=reg_metrics,
            classification_metrics=cls_metrics,
            diagnostics=diagnostics,
            feature_importance=feature_importance,
            uncertainty=uncertainty,
            training_duration_ms=round(train_duration_ms, 2),
            random_seed=random_seed,
            is_baseline=is_baseline,
            fit_status=fit_status,
            warnings=warnings_list,
        )
        records.append(rec)

    # Sort candidates
    if task_type == PredictionTask.REGRESSION:
        records.sort(
            key=lambda r: r.regression_metrics.rmse if r.regression_metrics else float("inf")
        )
    else:
        records.sort(
            key=lambda r: r.classification_metrics.f1_macro if r.classification_metrics else -1.0,
            reverse=True,
        )

    # Mark top non-baseline as recommended
    non_baselines = [r for r in records if not r.is_baseline and r.fit_status == "SUCCESS"]
    if non_baselines:
        non_baselines[0].is_recommended = True
    elif records:
        records[0].is_recommended = True

    return records
