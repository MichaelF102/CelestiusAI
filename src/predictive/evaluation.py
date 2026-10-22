"""Evaluation metrics computation for regression, classification, and forecasting.

Provides comprehensive task-specific metrics with safe edge-case handling for
constant targets, zero denominators, and imbalanced class representations.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)

from src.predictive.predictive_models import (
    ClassificationMetrics,
    ForecastingMetrics,
    RegressionMetrics,
)

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Computes validated evaluation metrics across all supported predictive tasks."""

    @staticmethod
    def evaluate_regression(
        y_true: pd.Series | np.ndarray,
        y_pred: pd.Series | np.ndarray,
    ) -> RegressionMetrics:
        """Calculate MAE, RMSE, R-squared, and Median Absolute Error."""
        yt = np.asarray(y_true, dtype=float)
        yp = np.asarray(y_pred, dtype=float)

        mae = float(mean_absolute_error(yt, yp))
        rmse = float(root_mean_squared_error(yt, yp))
        medae = float(median_absolute_error(yt, yp))

        # Check for constant y_true where R² is technically undefined
        if np.all(yt == yt[0]):
            r2 = 0.0
            explanation = "Target is constant; R² defaults to 0.0."
        else:
            r2 = float(r2_score(yt, yp))
            explanation = (
                "Standard test-set metrics. Note: R² can be negative on holdout data if the model "
                "performs worse than simply predicting the training set mean."
            )

        return RegressionMetrics(
            mae=round(mae, 4),
            rmse=round(rmse, 4),
            r_squared=round(r2, 4),
            median_absolute_error=round(medae, 4),
            sample_size=len(yt),
            explanation=explanation,
        )

    @staticmethod
    def evaluate_classification(
        y_true: pd.Series | np.ndarray,
        y_pred: pd.Series | np.ndarray,
        y_prob: np.ndarray | None = None,
    ) -> ClassificationMetrics:
        """Calculate Accuracy, Precision, Recall, F1, Confusion Matrix, and optional ROC-AUC."""
        yt = np.asarray(y_true)
        yp = np.asarray(y_pred)

        unique_classes = sorted(set(yt) | set(yp))
        class_str_labels = [str(c) for c in unique_classes]

        acc = float(accuracy_score(yt, yp))
        prec = float(precision_score(yt, yp, average="macro", zero_division=0))
        rec = float(recall_score(yt, yp, average="macro", zero_division=0))
        f1 = float(f1_score(yt, yp, average="macro", zero_division=0))

        cm = confusion_matrix(yt, yp, labels=unique_classes).tolist()

        # Class prevalence
        total_obs = len(yt)
        prevalence: dict[str, float] = {}
        counts = pd.Series(yt).value_counts()
        for c in unique_classes:
            prevalence[str(c)] = round(float(counts.get(c, 0) / total_obs), 4)

        # Check for severe imbalance (any class < 15%)
        is_imbalanced = any(p < 0.15 for p in prevalence.values()) and len(unique_classes) > 1
        warning = None
        if is_imbalanced:
            warning = "Severe class imbalance detected. Accuracy may be misleading; inspect Macro F1 score and Confusion Matrix."

        # Compute ROC-AUC and Log Loss if valid probabilities are supplied
        roc_auc: float | None = None
        loss_val: float | None = None

        if y_prob is not None:
            try:
                if len(unique_classes) == 2 and y_prob.ndim == 2:
                    # Binary classification
                    roc_auc = round(float(roc_auc_score(yt, y_prob[:, 1])), 4)
                elif len(unique_classes) > 2 and y_prob.ndim == 2:
                    # Multiclass classification
                    roc_auc = round(
                        float(roc_auc_score(yt, y_prob, multi_class="ovr", average="macro")),
                        4,
                    )
            except Exception as err:
                logger.debug("ROC-AUC computation skipped: %s", err)

            try:
                loss_val = round(float(log_loss(yt, y_prob, labels=unique_classes)), 4)
            except Exception as err:
                logger.debug("Log-loss computation skipped: %s", err)

        return ClassificationMetrics(
            accuracy=round(acc, 4),
            precision_macro=round(prec, 4),
            recall_macro=round(rec, 4),
            f1_macro=round(f1, 4),
            roc_auc=roc_auc,
            log_loss=loss_val,
            confusion_matrix=cm,
            class_labels=class_str_labels,
            class_prevalence=prevalence,
            sample_size=total_obs,
            is_imbalanced=is_imbalanced,
            warning=warning,
        )

    @staticmethod
    def evaluate_forecasting(
        y_true: pd.Series | np.ndarray,
        y_pred: pd.Series | np.ndarray,
        y_train_history: pd.Series | np.ndarray | None = None,
        season_length: int = 1,
    ) -> ForecastingMetrics:
        """Calculate MAE, RMSE, sMAPE, and MASE for time-series forecasts."""
        yt = np.asarray(y_true, dtype=float)
        yp = np.asarray(y_pred, dtype=float)

        mae = float(mean_absolute_error(yt, yp))
        rmse = float(root_mean_squared_error(yt, yp))

        # Safe sMAPE: 200% * |y - yhat| / (|y| + |yhat| + epsilon)
        denom = np.abs(yt) + np.abs(yp)
        eps = 1e-8
        smape = float(np.mean(2.0 * np.abs(yt - yp) / np.maximum(denom, eps)) * 100.0)

        # MASE (Mean Absolute Scaled Error) relative to naive in-sample differences
        mase: float | None = None
        if y_train_history is not None and len(y_train_history) > season_length:
            y_hist = np.asarray(y_train_history, dtype=float)
            diffs = np.abs(y_hist[season_length:] - y_hist[:-season_length])
            scale = float(np.mean(diffs))
            if scale > eps:
                mase = round(mae / scale, 4)

        has_zeros = bool(np.any(yt == 0.0))

        return ForecastingMetrics(
            mae=round(mae, 4),
            rmse=round(rmse, 4),
            smape=round(smape, 2),
            mase=mase,
            horizon=len(yt),
            sample_size=len(yt),
            zero_values_detected=has_zeros,
        )
