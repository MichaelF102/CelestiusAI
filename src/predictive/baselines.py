"""Baseline prediction models for benchmark comparison.

Ensures that every candidate machine learning model is evaluated against
a defensible naive baseline on the identical test partition:
- Regression: DummyRegressor predicting training mean and median
- Classification: DummyClassifier predicting most frequent class
- Forecasting: Naive ($y_{t+h} = y_t$) and Seasonal Naive ($y_{t+h} = y_{t+h-s}$)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor


class BaselineModels:
    """Provides baseline estimators for regression, classification, and forecasting."""

    @staticmethod
    def get_regression_baseline(strategy: str = "mean") -> DummyRegressor:
        """Return DummyRegressor predicting training mean or median."""
        return DummyRegressor(strategy=strategy)

    @staticmethod
    def get_classification_baseline(strategy: str = "most_frequent") -> DummyClassifier:
        """Return DummyClassifier predicting majority class."""
        return DummyClassifier(strategy=strategy)

    @staticmethod
    def forecast_naive(history: pd.Series | np.ndarray, horizon: int) -> np.ndarray:
        """Naive forecast: projects the last observed historical value across all future horizons."""
        vals = np.asarray(history)
        if len(vals) == 0:
            return np.zeros(horizon)
        last_val = float(vals[-1])
        return np.full(shape=horizon, fill_value=last_val)

    @staticmethod
    def forecast_seasonal_naive(
        history: pd.Series | np.ndarray,
        horizon: int,
        season_length: int = 7,
    ) -> np.ndarray:
        """Seasonal naive forecast: projects value observed exactly one seasonal period prior."""
        vals = np.asarray(history)
        if len(vals) < season_length:
            return BaselineModels.forecast_naive(history, horizon)

        forecasts = []
        for h in range(horizon):
            idx = -(season_length - (h % season_length))
            forecasts.append(float(vals[idx]))
        return np.array(forecasts)


class DummyRegressorBaseline(DummyRegressor):
    """Convenience alias for DummyRegressor baseline."""

    def __init__(self, strategy: str = "mean", **kwargs):
        super().__init__(strategy=strategy, **kwargs)


class DummyClassifierBaseline(DummyClassifier):
    """Convenience alias for DummyClassifier baseline."""

    def __init__(self, strategy: str = "most_frequent", **kwargs):
        super().__init__(strategy=strategy, **kwargs)


class NaiveForecaster:
    """Estimator-like wrapper for naive time-series forecasting."""

    def __init__(self) -> None:
        self.history: np.ndarray = np.array([])

    def fit(self, y: pd.Series | np.ndarray) -> NaiveForecaster:
        self.history = np.asarray(y)
        return self

    def predict(self, horizon: int) -> np.ndarray:
        return BaselineModels.forecast_naive(self.history, horizon)


class SeasonalNaiveForecaster:
    """Estimator-like wrapper for seasonal naive time-series forecasting."""

    def __init__(self, season_length: int = 7) -> None:
        self.season_length = season_length
        self.history: np.ndarray = np.array([])

    def fit(self, y: pd.Series | np.ndarray) -> SeasonalNaiveForecaster:
        self.history = np.asarray(y)
        return self

    def predict(self, horizon: int) -> np.ndarray:
        return BaselineModels.forecast_seasonal_naive(self.history, horizon, self.season_length)
