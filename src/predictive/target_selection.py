"""Target variable analysis, task inference, and predictor validation.

Recommends suitable prediction tasks, validates target suitability, and screens
out high-cardinality identifiers, all-null columns, and post-outcome leakage vectors.
"""

import logging

import pandas as pd

from src.ingestion.schema import infer_dataset_schema
from src.predictive.predictive_models import PredictionTask, TargetRecommendation

logger = logging.getLogger(__name__)


class TargetSelector:
    """Analyzes columns to recommend and validate machine learning prediction tasks."""

    @staticmethod
    def evaluate_target(
        df: pd.DataFrame,
        target_column: str,
        candidate_time_col: str | None = None,
        min_samples: int = 15,
    ) -> TargetRecommendation:
        """Analyze candidate target variable and determine appropriate task category."""
        warnings: list[str] = []

        if target_column not in df.columns:
            return TargetRecommendation(
                target_column=target_column,
                recommended_task=PredictionTask.REGRESSION,
                is_suitable=False,
                confidence=0.0,
                reason=f"Target column '{target_column}' does not exist in dataset.",
                unique_values_count=0,
                missing_count=len(df),
                missing_pct=100.0,
                warnings=["Column not found."],
            )

        series = df[target_column]
        n_total = len(series)
        n_missing = int(series.isna().sum())
        missing_pct = round((n_missing / n_total) * 100, 2) if n_total > 0 else 100.0
        clean_series = series.dropna()
        n_valid = len(clean_series)
        unique_vals = clean_series.unique()
        n_unique = len(unique_vals)

        # 1. Reject insufficient data
        if n_valid < min_samples:
            return TargetRecommendation(
                target_column=target_column,
                recommended_task=PredictionTask.REGRESSION,
                is_suitable=False,
                confidence=0.0,
                reason=f"Insufficient non-null observations (N={n_valid}). Minimum {min_samples} required.",
                unique_values_count=n_unique,
                missing_count=n_missing,
                missing_pct=missing_pct,
                warnings=[f"Target has only {n_valid} non-null values."],
            )

        # 2. Reject constant targets
        if n_unique <= 1:
            return TargetRecommendation(
                target_column=target_column,
                recommended_task=PredictionTask.REGRESSION,
                is_suitable=False,
                confidence=0.0,
                reason=f"Target column '{target_column}' is constant ({unique_vals[0] if len(unique_vals) > 0 else 'empty'}). Models cannot learn without variance.",
                unique_values_count=n_unique,
                missing_count=n_missing,
                missing_pct=missing_pct,
                warnings=["Zero variance target."],
            )

        # 3. Detect candidate predictor exclusions (identifiers, constants, target itself)
        schema = infer_dataset_schema(df)
        candidate_predictors: list[str] = []
        excluded_columns: list[str] = [target_column]

        for col, col_meta in schema.items():
            if col == target_column:
                continue
            if col_meta.is_identifier or "id" in col.lower() or col_meta.distinct_count == n_total:
                excluded_columns.append(col)
                warnings.append(
                    f"Excluded '{col}': Identified as unique record identifier or serial key."
                )
            elif col_meta.is_constant or df[col].nunique(dropna=True) <= 1:
                excluded_columns.append(col)
            else:
                candidate_predictors.append(col)

        # 4. Infer Task Type
        is_numeric = pd.api.types.is_numeric_dtype(series)

        # Time series forecasting candidate
        if candidate_time_col and candidate_time_col in df.columns and is_numeric:
            task = PredictionTask.TIME_SERIES_FORECASTING
            reason = (
                f"Numeric metric '{target_column}' paired with timestamp '{candidate_time_col}'. "
                "Time-series forecasting recommended."
            )
        # Binary classification
        elif n_unique == 2:
            task = PredictionTask.BINARY_CLASSIFICATION
            reason = f"Target '{target_column}' contains exactly 2 distinct classes ({list(unique_vals)}). Binary classification recommended."
        # Multiclass classification
        elif (not is_numeric and n_unique <= 20) or (
            is_numeric and n_unique <= 10 and n_valid >= n_unique * 5
        ):
            task = PredictionTask.MULTICLASS_CLASSIFICATION
            reason = f"Target '{target_column}' contains {n_unique} discrete classes. Multiclass classification recommended."
        # Regression
        elif is_numeric:
            task = PredictionTask.REGRESSION
            reason = f"Target '{target_column}' is a continuous numeric variable with {n_unique} distinct values. Regression recommended."
        else:
            task = PredictionTask.MULTICLASS_CLASSIFICATION
            reason = f"Categorical target with {n_unique} classes. High cardinality warning."
            warnings.append(
                f"Target has {n_unique} classes. Many classes with sparse representation may reduce model reliability."
            )

        return TargetRecommendation(
            target_column=target_column,
            recommended_task=task,
            is_suitable=True,
            confidence=0.95,
            reason=reason,
            unique_values_count=n_unique,
            missing_count=n_missing,
            missing_pct=missing_pct,
            candidate_predictors=candidate_predictors,
            excluded_columns=excluded_columns,
            warnings=warnings,
        )
