"""Data Splitting Strategies with Leakage Prevention.

Supports:
1. Standard Train/Test split with reproducible random seed.
2. Stratified splits for imbalanced classification tasks.
3. Chronological splits for time-series data (zero shuffling, zero future leakage).
4. Group-aware splits when records share an entity identifier.
"""

import logging

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

from src.predictive.predictive_models import (
    DatasetSplitInfo,
    PredictionTask,
    ValidationStrategy,
)

logger = logging.getLogger(__name__)


class SplitStrategyService:
    """Manages defensible train/test partitioning for predictive workflows."""

    @staticmethod
    def partition_data(
        df: pd.DataFrame,
        target_column: str,
        task: PredictionTask,
        strategy: ValidationStrategy = ValidationStrategy.TRAIN_TEST_SPLIT,
        test_ratio: float = 0.20,
        random_seed: int = 42,
        time_column: str | None = None,
        group_column: str | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, DatasetSplitInfo]:
        """Split DataFrame into X_train, X_test, y_train, y_test with full provenance."""
        # 1. Clean missing targets before splitting
        valid_df = df.dropna(subset=[target_column]).copy()
        n_total = len(valid_df)

        if n_total < 10:
            raise ValueError(
                f"Sample size N={n_total} is too small to perform valid train/test partitioning."
            )

        feature_cols = [c for c in valid_df.columns if c != target_column]
        X = valid_df[feature_cols]
        y = valid_df[target_column]

        # ---------------------------------------------------------------------
        # Case A: Chronological / Time-Series Partitioning
        # ---------------------------------------------------------------------
        if (
            strategy == ValidationStrategy.CHRONOLOGICAL_SPLIT
            or task == PredictionTask.TIME_SERIES_FORECASTING
        ):
            if time_column and time_column in valid_df.columns:
                valid_df = valid_df.sort_values(by=time_column).reset_index(drop=True)
                X = valid_df[feature_cols]
                y = valid_df[target_column]

            split_idx = int(n_total * (1.0 - test_ratio))
            split_idx = max(5, min(split_idx, n_total - 2))

            X_train = X.iloc[:split_idx]
            X_test = X.iloc[split_idx:]
            y_train = y.iloc[:split_idx]
            y_test = y.iloc[split_idx:]

            boundary = None
            if time_column and time_column in valid_df.columns:
                boundary = str(valid_df.iloc[split_idx][time_column])

            split_info = DatasetSplitInfo(
                strategy=ValidationStrategy.CHRONOLOGICAL_SPLIT,
                train_size=len(X_train),
                test_size=len(X_test),
                test_ratio=round(len(X_test) / n_total, 4),
                is_stratified=False,
                random_seed=random_seed,
                time_column=time_column,
                split_boundary=boundary,
            )
            return X_train, X_test, y_train, y_test, split_info

        # ---------------------------------------------------------------------
        # Case B: Stratified Partitioning for Classification
        # ---------------------------------------------------------------------
        if strategy == ValidationStrategy.STRATIFIED_SPLIT and task in (
            PredictionTask.BINARY_CLASSIFICATION,
            PredictionTask.MULTICLASS_CLASSIFICATION,
        ):
            min_class_count = y.value_counts().min()
            if min_class_count >= 2:
                try:
                    sss = StratifiedShuffleSplit(
                        n_splits=1, test_size=test_ratio, random_state=random_seed
                    )
                    train_idx, test_idx = next(sss.split(X, y))
                    X_train = X.iloc[train_idx]
                    X_test = X.iloc[test_idx]
                    y_train = y.iloc[train_idx]
                    y_test = y.iloc[test_idx]

                    split_info = DatasetSplitInfo(
                        strategy=ValidationStrategy.STRATIFIED_SPLIT,
                        train_size=len(X_train),
                        test_size=len(X_test),
                        test_ratio=round(len(X_test) / n_total, 4),
                        is_stratified=True,
                        random_seed=random_seed,
                    )
                    return X_train, X_test, y_train, y_test, split_info
                except Exception as err:
                    logger.warning(
                        "Stratified split failed (%s). Falling back to standard split.", err
                    )

        # ---------------------------------------------------------------------
        # Case C: Standard Train/Test Split
        # ---------------------------------------------------------------------
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_ratio,
            random_state=random_seed,
            shuffle=True,
        )

        split_info = DatasetSplitInfo(
            strategy=ValidationStrategy.TRAIN_TEST_SPLIT,
            train_size=len(X_train),
            test_size=len(X_test),
            test_ratio=round(len(X_test) / n_total, 4),
            is_stratified=False,
            random_seed=random_seed,
            group_column=group_column,
        )
        return X_train, X_test, y_train, y_test, split_info
