"""Predictive Analytics Engine Package."""

from src.predictive.baselines import (
    BaselineModels,
    DummyClassifierBaseline,
    DummyRegressorBaseline,
    NaiveForecaster,
    SeasonalNaiveForecaster,
)
from src.predictive.diagnostics import ModelDiagnostics
from src.predictive.evaluation import ModelEvaluator
from src.predictive.explainability import ModelExplainability
from src.predictive.forecasting import TimeSeriesForecastingEngine
from src.predictive.model_registry import ModelRegistry
from src.predictive.predictive_models import (
    ClassificationMetrics,
    DatasetSplitInfo,
    DiagnosticsData,
    FeatureImportanceResult,
    ForecastingMetrics,
    ForecastResult,
    ModelFamily,
    PredictionTask,
    RegressionMetrics,
    TargetRecommendation,
    TrainedModelRecord,
    UncertaintyResult,
    ValidationStrategy,
)
from src.predictive.preprocessing import LeakageSafePreprocessor
from src.predictive.split_strategy import SplitStrategyService
from src.predictive.target_selection import TargetSelector
from src.predictive.training_service import train_predictive_models
from src.predictive.uncertainty import UncertaintyEstimator

# Convenient top-level functional aliases
evaluate_target = TargetSelector.evaluate_target
partition_data = SplitStrategyService.partition_data

__all__ = [
    "PredictionTask",
    "ValidationStrategy",
    "ModelFamily",
    "TargetRecommendation",
    "DatasetSplitInfo",
    "RegressionMetrics",
    "ClassificationMetrics",
    "ForecastingMetrics",
    "DiagnosticsData",
    "FeatureImportanceResult",
    "UncertaintyResult",
    "ForecastResult",
    "TrainedModelRecord",
    "TargetSelector",
    "evaluate_target",
    "LeakageSafePreprocessor",
    "SplitStrategyService",
    "partition_data",
    "BaselineModels",
    "DummyRegressorBaseline",
    "DummyClassifierBaseline",
    "NaiveForecaster",
    "SeasonalNaiveForecaster",
    "ModelEvaluator",
    "ModelDiagnostics",
    "ModelExplainability",
    "UncertaintyEstimator",
    "TimeSeriesForecastingEngine",
    "train_predictive_models",
    "ModelRegistry",
]
