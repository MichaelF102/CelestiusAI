"""Typed Pydantic data schemas for the Predictive Analytics Lab.

Defines schemas for:
- Prediction tasks (regression, binary/multiclass classification, forecasting)
- Validation and splitting strategies
- Preprocessing configurations
- Evaluation metrics and diagnostic curves
- Feature importance and uncertainty quantification
- Model registry records with dataset version provenance
"""

import uuid
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class PredictionTask(StrEnum):
    """Machine learning prediction task categories."""

    REGRESSION = "regression"
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    TIME_SERIES_FORECASTING = "time_series_forecasting"


class ValidationStrategy(StrEnum):
    """Data partitioning and validation approaches."""

    TRAIN_TEST_SPLIT = "train_test_split"
    STRATIFIED_SPLIT = "stratified_split"
    K_FOLD_CV = "k_fold_cv"
    STRATIFIED_K_FOLD_CV = "stratified_k_fold_cv"
    CHRONOLOGICAL_SPLIT = "chronological_split"


class ModelFamily(StrEnum):
    """Supported machine learning model architectures."""

    BASELINE_DUMMY = "baseline_dummy"
    LINEAR_MODEL = "linear_model"
    RIDGE_REGRESSION = "ridge_regression"
    RANDOM_FOREST = "random_forest"
    HIST_GRADIENT_BOOSTING = "hist_gradient_boosting"
    NAIVE_FORECAST = "naive_forecast"
    SEASONAL_NAIVE_FORECAST = "seasonal_naive_forecast"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"


class TargetRecommendation(BaseModel):
    """Validation outcome and task suggestion for a candidate target variable."""

    target_column: str
    recommended_task: PredictionTask
    is_suitable: bool
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    reason: str
    unique_values_count: int
    missing_count: int
    missing_pct: float
    candidate_predictors: list[str] = Field(default_factory=list)
    excluded_columns: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class DatasetSplitInfo(BaseModel):
    """Metadata describing the data partitioning procedure."""

    strategy: ValidationStrategy
    train_size: int
    test_size: int
    test_ratio: float
    is_stratified: bool = False
    random_seed: int = 42
    time_column: str | None = None
    split_boundary: str | None = None
    group_column: str | None = None


class RegressionMetrics(BaseModel):
    """Evaluation metrics for continuous regression tasks."""

    mae: float
    rmse: float
    r_squared: float
    median_absolute_error: float
    sample_size: int
    explanation: str = "Test-set evaluation metrics. Note: R² can be negative on unseen holdout data if the model predicts worse than the training mean."


class ClassificationMetrics(BaseModel):
    """Evaluation metrics for discrete classification tasks."""

    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    roc_auc: float | None = None
    pr_auc: float | None = None
    log_loss: float | None = None
    confusion_matrix: list[list[int]] = Field(default_factory=list)
    class_labels: list[str] = Field(default_factory=list)
    class_prevalence: dict[str, float] = Field(default_factory=dict)
    sample_size: int
    is_imbalanced: bool = False
    warning: str | None = None


class ForecastingMetrics(BaseModel):
    """Evaluation metrics for time-series forecasting."""

    mae: float
    rmse: float
    smape: float
    mase: float | None = None
    horizon: int
    sample_size: int
    zero_values_detected: bool = False
    notes: str = "sMAPE uses symmetric denominator (|actual| + |pred|)/2 to avoid division-by-zero instability."


class DiagnosticsData(BaseModel):
    """Data points for model evaluation visual diagnostics."""

    actual_values: list[float] = Field(default_factory=list)
    predicted_values: list[float] = Field(default_factory=list)
    residuals: list[float] = Field(default_factory=list)
    roc_curve: dict[str, list[float]] | None = None  # fpr, tpr, thresholds
    calibration_curve: dict[str, list[float]] | None = None  # prob_true, prob_pred
    timestamps: list[str] | None = None


class FeatureImportanceResult(BaseModel):
    """Quantified predictor contribution to model performance."""

    feature_names: list[str] = Field(default_factory=list)
    importances: list[float] = Field(default_factory=list)
    method: str = Field(description="'held_out_permutation', 'tree_gini', or 'linear_coefficients'")
    evaluation_split: str = "test"
    correlated_feature_warnings: list[str] = Field(default_factory=list)
    caveat: str = "Predictive feature importance indicates association and predictive utility, NOT causal influence."


class UncertaintyResult(BaseModel):
    """Prediction interval or reliability assessment."""

    interval_method: str = Field(description="'residual_quantile_conformal' or 'standard_error'")
    confidence_level: float = 0.90
    lower_bounds: list[float] = Field(default_factory=list)
    upper_bounds: list[float] = Field(default_factory=list)
    coverage_rate: float | None = None
    assumptions: list[str] = Field(default_factory=list)


class ForecastResult(BaseModel):
    """Time-series projections over a future horizon."""

    horizon: int
    future_timestamps: list[str] = Field(default_factory=list)
    point_forecasts: list[float] = Field(default_factory=list)
    lower_confidence: list[float] = Field(default_factory=list)
    upper_confidence: list[float] = Field(default_factory=list)
    model_name: str
    frequency_detected: str = "irregular"


class TrainedModelRecord(BaseModel):
    """Complete provenance artifact for a trained and evaluated model."""

    model_id: str = Field(default_factory=lambda: f"model_{uuid.uuid4().hex[:8]}")
    model_name: str
    family: ModelFamily
    task: PredictionTask
    dataset_id: str
    dataset_version: int
    target_column: str
    predictor_columns: list[str] = Field(default_factory=list)
    split_info: DatasetSplitInfo | None = None
    regression_metrics: RegressionMetrics | None = None
    classification_metrics: ClassificationMetrics | None = None
    forecasting_metrics: ForecastingMetrics | None = None
    diagnostics: DiagnosticsData | None = None
    feature_importance: FeatureImportanceResult | None = None
    uncertainty: UncertaintyResult | None = None
    forecast: ForecastResult | None = None
    training_duration_ms: float = 0.0
    random_seed: int = 42
    is_baseline: bool = False
    is_recommended: bool = False
    is_stale: bool = False
    fit_status: str = "SUCCESS"
    warnings: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
