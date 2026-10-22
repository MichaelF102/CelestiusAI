"""Unit tests for Predictive Analytics Lab (Prompt 7).

Tests target selection, leakage-safe preprocessing, splitting strategies,
baseline comparison, regression, classification, forecasting, diagnostics,
explainability, and uncertainty intervals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src.predictive import (
    LeakageSafePreprocessor,
    ModelRegistry,
    NaiveForecaster,
    PredictionTask,
    SeasonalNaiveForecaster,
    SplitStrategyService,
    TargetSelector,
    ValidationStrategy,
    train_predictive_models,
)


@pytest.fixture
def regression_data() -> pd.DataFrame:
    """Synthetic dataset for regression testing."""
    np.random.seed(42)
    n = 100
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.uniform(10, 20, n)
    cat = np.random.choice(["A", "B", "C"], n)
    y = 3.5 * x1 - 1.2 * x2 + np.random.normal(0, 0.2, n)
    return pd.DataFrame({"x1": x1, "x2": x2, "cat": cat, "target": y})


@pytest.fixture
def classification_data() -> pd.DataFrame:
    """Synthetic dataset for binary and multiclass classification testing."""
    np.random.seed(42)
    n = 100
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(5, 2, n)
    # Binary
    y_bin = (x1 + x2 > 5).astype(int)
    # Multiclass
    y_multi = np.where(x1 < -0.5, "Class_A", np.where(x1 > 0.5, "Class_B", "Class_C"))
    return pd.DataFrame({"x1": x1, "x2": x2, "y_bin": y_bin, "y_multi": y_multi})


@pytest.fixture
def time_series_data() -> pd.DataFrame:
    """Synthetic chronological dataset for forecasting."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    trend = np.linspace(10, 50, 60)
    seasonal = 5 * np.sin(np.linspace(0, 8 * np.pi, 60))
    noise = np.random.normal(0, 1, 60)
    sales = trend + seasonal + noise
    return pd.DataFrame({"date": dates, "sales": sales})


def test_evaluate_target_recommends_regression(regression_data):
    rec = TargetSelector.evaluate_target(regression_data, "target")
    assert rec.is_suitable is True
    assert rec.recommended_task == PredictionTask.REGRESSION


def test_evaluate_target_recommends_classification(classification_data):
    rec_bin = TargetSelector.evaluate_target(classification_data, "y_bin")
    assert rec_bin.is_suitable is True
    assert rec_bin.recommended_task == PredictionTask.BINARY_CLASSIFICATION

    rec_multi = TargetSelector.evaluate_target(classification_data, "y_multi")
    assert rec_multi.is_suitable is True
    assert rec_multi.recommended_task == PredictionTask.MULTICLASS_CLASSIFICATION


def test_evaluate_target_rejects_constant_or_invalid():
    df_const = pd.DataFrame({"const": [5] * 30, "x": range(30)})
    rec_const = TargetSelector.evaluate_target(df_const, "const")
    assert rec_const.is_suitable is False

    df_missing = pd.DataFrame({"missing": [np.nan] * 30, "x": range(30)})
    rec_miss = TargetSelector.evaluate_target(df_missing, "missing")
    assert rec_miss.is_suitable is False


def test_leakage_safe_preprocessor(regression_data):
    """Verify that preprocessing is fit exclusively on training data and handles unseen categories."""
    train_df = regression_data.iloc[:80][["x1", "x2", "cat"]]
    test_df = regression_data.iloc[80:][["x1", "x2", "cat"]].copy()
    test_df.loc[test_df.index[0], "cat"] = "Unseen_Category_Z"

    prep = LeakageSafePreprocessor(numeric_features=["x1", "x2"], categorical_features=["cat"])
    X_train_proc = prep.fit_transform(train_df)
    assert X_train_proc.shape[0] == 80

    # Transforming unseen categories should not fail due to handle_unknown="ignore"
    X_test_proc = prep.transform(test_df)
    assert X_test_proc.shape[0] == 20
    assert not np.isnan(X_test_proc).any()


def test_split_dataset_no_leakage(regression_data):
    X_train, X_test, y_train, y_test, split_info = SplitStrategyService.partition_data(
        df=regression_data,
        target_column="target",
        task=PredictionTask.REGRESSION,
        strategy=ValidationStrategy.TRAIN_TEST_SPLIT,
        test_ratio=0.2,
        random_seed=42,
    )
    assert X_train.shape[0] == 80
    assert X_test.shape[0] == 20
    # Confirm no overlap in index
    assert len(set(X_train.index).intersection(set(X_test.index))) == 0


def test_train_predictive_models_regression(regression_data):
    records = train_predictive_models(
        df=regression_data,
        target_col="target",
        feature_cols=["x1", "x2", "cat"],
        task_type=PredictionTask.REGRESSION,
        dataset_id="test_ds",
        dataset_version=1,
    )
    assert len(records) >= 3
    baseline_rec = next(r for r in records if r.is_baseline)
    best_candidate = records[0]
    assert best_candidate.regression_metrics is not None
    assert baseline_rec.regression_metrics is not None
    assert best_candidate.regression_metrics.rmse <= baseline_rec.regression_metrics.rmse
    assert best_candidate.diagnostics is not None
    assert best_candidate.feature_importance is not None
    assert best_candidate.uncertainty is not None


def test_train_predictive_models_classification(classification_data):
    records = train_predictive_models(
        df=classification_data,
        target_col="y_bin",
        feature_cols=["x1", "x2"],
        task_type=PredictionTask.BINARY_CLASSIFICATION,
        dataset_id="test_ds",
        dataset_version=1,
    )
    assert len(records) >= 2
    best_rec = records[0]
    assert best_rec.classification_metrics is not None
    assert best_rec.classification_metrics.accuracy is not None
    assert best_rec.classification_metrics.f1_macro is not None
    assert best_rec.diagnostics is not None


def test_forecasting_models(time_series_data):
    records = train_predictive_models(
        df=time_series_data,
        target_col="sales",
        feature_cols=[],
        task_type=PredictionTask.TIME_SERIES_FORECASTING,
        dataset_id="ts_ds",
        dataset_version=1,
        time_col="date",
        test_size=0.2,
    )
    assert len(records) >= 1
    fc_rec = records[0]
    assert fc_rec.task == PredictionTask.TIME_SERIES_FORECASTING
    assert fc_rec.forecasting_metrics is not None
    assert fc_rec.forecast is not None


def test_naive_forecaster():
    y_hist = np.array([10.0, 12.0, 15.0])
    forecaster = NaiveForecaster()
    forecaster.fit(y_hist)
    preds = forecaster.predict(horizon=3)
    assert np.allclose(preds, [15.0, 15.0, 15.0])

    s_forecaster = SeasonalNaiveForecaster(season_length=2)
    s_forecaster.fit(np.array([1.0, 2.0, 1.1, 2.1]))
    s_preds = s_forecaster.predict(horizon=4)
    assert np.allclose(s_preds, [1.1, 2.1, 1.1, 2.1])


def test_model_registry_staleness():
    registry = ModelRegistry()
    records = train_predictive_models(
        df=pd.DataFrame({"x": range(20), "y": [float(i) for i in range(20)]}),
        target_col="y",
        feature_cols=["x"],
        task_type=PredictionTask.REGRESSION,
        dataset_id="ds_1",
        dataset_version=1,
    )
    registry.register_models(records)

    models_current = registry.list_models(dataset_id="ds_1", active_version=1)
    assert models_current[0].is_stale is False

    models_stale = registry.list_models(dataset_id="ds_1", active_version=2)
    assert models_stale[0].is_stale is True
