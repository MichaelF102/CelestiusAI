"""Model Diagnostics Data Generation.

Produces structured diagnostics:
- Actual vs. Predicted curves and residuals
- Confusion matrices
- ROC curves (FPR, TPR) for probabilistic classifiers
- Calibration curves for probability reliability assessment
"""

import logging

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve

from src.predictive.predictive_models import DiagnosticsData

logger = logging.getLogger(__name__)


class ModelDiagnostics:
    """Generates structured diagnostics data for regression, classification, and forecasting."""

    @staticmethod
    def generate_regression_diagnostics(
        y_true: pd.Series | np.ndarray,
        y_pred: pd.Series | np.ndarray,
        max_points: int = 500,
    ) -> DiagnosticsData:
        """Extract actual vs predicted points and residual distribution."""
        yt = np.asarray(y_true, dtype=float)
        yp = np.asarray(y_pred, dtype=float)
        residuals = yt - yp

        if len(yt) > max_points:
            # Subsample evenly for UI performance
            indices = np.linspace(0, len(yt) - 1, max_points, dtype=int)
            yt = yt[indices]
            yp = yp[indices]
            residuals = residuals[indices]

        return DiagnosticsData(
            actual_values=[round(float(v), 4) for v in yt],
            predicted_values=[round(float(v), 4) for v in yp],
            residuals=[round(float(v), 4) for v in residuals],
        )

    @staticmethod
    def generate_classification_diagnostics(
        y_true: pd.Series | np.ndarray,
        y_pred: pd.Series | np.ndarray,
        y_prob: np.ndarray | None = None,
    ) -> DiagnosticsData:
        """Extract ROC and calibration curve data for probabilistic binary classification."""
        yt = np.asarray(y_true)
        np.asarray(y_pred)
        unique_classes = sorted(set(yt))

        roc_data = None
        calib_data = None

        if y_prob is not None and len(unique_classes) == 2 and y_prob.ndim == 2:
            try:
                # Map positive class to 1
                pos_class = unique_classes[1]
                y_binary = (yt == pos_class).astype(int)
                pos_probs = y_prob[:, 1]

                fpr, tpr, thresholds = roc_curve(y_binary, pos_probs)
                roc_data = {
                    "fpr": [round(float(x), 4) for x in fpr],
                    "tpr": [round(float(x), 4) for x in tpr],
                    "thresholds": [round(float(x), 4) for x in thresholds],
                }

                prob_true, prob_pred = calibration_curve(
                    y_binary, pos_probs, n_bins=8, strategy="uniform"
                )
                calib_data = {
                    "prob_true": [round(float(x), 4) for x in prob_true],
                    "prob_pred": [round(float(x), 4) for x in prob_pred],
                }
            except Exception as err:
                logger.debug("Diagnostics curve calculation skipped: %s", err)

        return DiagnosticsData(
            actual_values=[float(i) for i in range(min(50, len(yt)))],  # Sample index
            predicted_values=[],
            residuals=[],
            roc_curve=roc_data,
            calibration_curve=calib_data,
        )

    @staticmethod
    def generate_forecasting_diagnostics(
        y_true: pd.Series | np.ndarray,
        y_pred: pd.Series | np.ndarray,
        timestamps: list[str] | None = None,
    ) -> DiagnosticsData:
        """Extract chronological actuals, forecasts, and forecast errors."""
        yt = np.asarray(y_true, dtype=float)
        yp = np.asarray(y_pred, dtype=float)
        residuals = yt - yp

        return DiagnosticsData(
            actual_values=[round(float(v), 4) for v in yt],
            predicted_values=[round(float(v), 4) for v in yp],
            residuals=[round(float(v), 4) for v in residuals],
            timestamps=timestamps,
        )
