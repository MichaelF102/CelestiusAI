"""Prediction Uncertainty and Interval Estimation.

Computes distribution-free conformal / empirical residual-quantile prediction intervals
for continuous regression forecasts. Clearly distinguishes prediction intervals
(uncertainty in future individual observations) from parameter confidence intervals.
"""

import logging

import numpy as np
import pandas as pd

from src.predictive.predictive_models import UncertaintyResult

logger = logging.getLogger(__name__)


class UncertaintyEstimator:
    """Computes empirical prediction intervals and coverage metrics."""

    @staticmethod
    def estimate_regression_intervals(
        y_train_true: pd.Series | np.ndarray,
        y_train_pred: pd.Series | np.ndarray,
        y_test_pred: pd.Series | np.ndarray,
        confidence_level: float = 0.90,
    ) -> UncertaintyResult:
        """Construct prediction intervals for test predictions using empirical residual quantiles."""
        yt = np.asarray(y_train_true, dtype=float)
        yp = np.asarray(y_train_pred, dtype=float)
        residuals = np.abs(yt - yp)

        # Conformal / empirical quantile bound
        q_level = min(0.99, max(0.50, confidence_level))
        quantile_error = float(np.quantile(residuals, q_level))

        test_preds = np.asarray(y_test_pred, dtype=float)
        lower_bounds = [round(float(p - quantile_error), 4) for p in test_preds]
        upper_bounds = [round(float(p + quantile_error), 4) for p in test_preds]

        return UncertaintyResult(
            interval_method="residual_quantile_conformal",
            confidence_level=confidence_level,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            assumptions=[
                f"Empirical conformal quantile estimated at alpha={1.0 - confidence_level:.2f}.",
                "Assumes training and future observations are exchangeable (identically distributed).",
            ],
        )
