"""Time-Series Forecasting Engine.

Implements chronological data preparation, regular frequency detection/aggregation,
naive and exponential smoothing statistical forecasts, and prediction intervals.
"""

import logging
from datetime import timedelta

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from src.predictive.baselines import BaselineModels
from src.predictive.evaluation import ModelEvaluator
from src.predictive.predictive_models import (
    ForecastResult,
    ModelFamily,
    PredictionTask,
    TrainedModelRecord,
)

logger = logging.getLogger(__name__)


class TimeSeriesForecastingEngine:
    """Manages time-series parsing, frequency aggregation, model fitting, and evaluation."""

    @staticmethod
    def prepare_time_series(
        df: pd.DataFrame,
        time_column: str,
        target_column: str,
        aggregation: str = "mean",
    ) -> pd.DataFrame:
        """Sort chronologically, parse timestamps, and aggregate duplicate dates."""
        clean = df[[time_column, target_column]].dropna().copy()
        clean[time_column] = pd.to_datetime(clean[time_column], errors="coerce")
        clean = clean.dropna(subset=[time_column])
        clean = clean.sort_values(by=time_column)

        # Aggregate duplicates if any
        if clean[time_column].duplicated().any():
            clean = clean.groupby(time_column)[target_column].agg(aggregation).reset_index()

        return clean.reset_index(drop=True)

    @staticmethod
    def fit_and_evaluate_forecast(
        series_df: pd.DataFrame,
        time_column: str,
        target_column: str,
        horizon: int = 14,
        season_length: int = 7,
        dataset_id: str = "dataset",
        dataset_version: int = 1,
    ) -> tuple[TrainedModelRecord, TrainedModelRecord]:
        """Fit Exponential Smoothing model and compare against Naive baseline on holdout."""
        y = series_df[target_column].values.astype(float)
        dates = series_df[time_column].dt.strftime("%Y-%m-%d").tolist()
        n = len(y)

        if n <= horizon + 5:
            raise ValueError(
                f"Insufficient time-series observations (N={n}) for forecast horizon H={horizon}."
            )

        # Train / Test chronological split
        train_y = y[:-horizon]
        test_y = y[-horizon:]
        test_dates = dates[-horizon:]

        # 1. Baseline: Naive Forecast
        naive_preds = BaselineModels.forecast_naive(train_y, horizon)
        naive_metrics = ModelEvaluator.evaluate_forecasting(
            y_true=test_y,
            y_pred=naive_preds,
            y_train_history=train_y,
            season_length=season_length,
        )

        naive_record = TrainedModelRecord(
            model_name="Naive Baseline Forecast",
            family=ModelFamily.NAIVE_FORECAST,
            task=PredictionTask.TIME_SERIES_FORECASTING,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            target_column=target_column,
            split_info=None,  # type: ignore[arg-type]
            forecasting_metrics=naive_metrics,
            is_baseline=True,
            is_recommended=False,
            forecast=ForecastResult(
                horizon=horizon,
                future_timestamps=test_dates,
                point_forecasts=[round(float(v), 2) for v in naive_preds],
                lower_confidence=[round(float(v * 0.85), 2) for v in naive_preds],
                upper_confidence=[round(float(v * 1.15), 2) for v in naive_preds],
                model_name="Naive Baseline",
            ),
        )

        # 2. Statistical Candidate: Holt-Winters Exponential Smoothing
        candidate_preds: np.ndarray
        try:
            # Check if seasonal length is supported
            seasonal_type = (
                "add" if len(train_y) >= 2 * season_length and season_length > 1 else None
            )
            hw = ExponentialSmoothing(
                train_y,
                trend="add",
                seasonal=seasonal_type,
                seasonal_periods=season_length if seasonal_type else None,
            ).fit()
            candidate_preds = hw.forecast(horizon)
        except Exception as err:
            logger.warning(
                "ExponentialSmoothing fit failed (%s). Falling back to Seasonal Naive.", err
            )
            candidate_preds = BaselineModels.forecast_seasonal_naive(
                train_y, horizon, season_length
            )

        candidate_metrics = ModelEvaluator.evaluate_forecasting(
            y_true=test_y,
            y_pred=candidate_preds,
            y_train_history=train_y,
            season_length=season_length,
        )

        # Generate future projection beyond the dataset
        last_dt = pd.to_datetime(series_df[time_column].iloc[-1])
        future_dts = [
            (last_dt + timedelta(days=i + 1)).strftime("%Y-%m-%d") for i in range(horizon)
        ]

        # Future point forecast
        try:
            hw_full = ExponentialSmoothing(y, trend="add").fit()
            future_points = hw_full.forecast(horizon)
        except Exception:
            future_points = BaselineModels.forecast_naive(y, horizon)

        # Conformal uncertainty bounds based on test residual standard error
        test_residuals = np.abs(test_y - candidate_preds)
        error_margin = float(np.quantile(test_residuals, 0.90)) if len(test_residuals) > 0 else 1.0

        lower_conf = [round(float(max(0.0, p - error_margin)), 2) for p in future_points]
        upper_conf = [round(float(p + error_margin), 2) for p in future_points]

        candidate_record = TrainedModelRecord(
            model_name="Holt-Winters Exponential Smoothing",
            family=ModelFamily.EXPONENTIAL_SMOOTHING,
            task=PredictionTask.TIME_SERIES_FORECASTING,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            target_column=target_column,
            split_info=None,  # type: ignore[arg-type]
            forecasting_metrics=candidate_metrics,
            is_baseline=False,
            is_recommended=bool(candidate_metrics.rmse <= naive_metrics.rmse),
            forecast=ForecastResult(
                horizon=horizon,
                future_timestamps=future_dts,
                point_forecasts=[round(float(v), 2) for v in future_points],
                lower_confidence=lower_conf,
                upper_confidence=upper_conf,
                model_name="Exponential Smoothing",
            ),
        )

        return naive_record, candidate_record
