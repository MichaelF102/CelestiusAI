"""Leakage-Safe Preprocessing and Column Transformation Pipelines.

Ensures:
1. Imputation, scaling, and categorical encoders are fitted strictly on X_train.
2. Missing values are imputed consistently (median for numeric, mode for categorical).
3. OneHotEncoder ignores unknown categories at inference time to prevent runtime crashes.
4. Feature names are preserved and recoverable after transformation for explainability.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


class LeakageSafePreprocessor:
    """Builds and fits scikit-learn preprocessing pipelines strictly on training data."""

    def __init__(
        self,
        numeric_features: list[str],
        categorical_features: list[str],
    ) -> None:
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.pipeline: ColumnTransformer | None = None
        self.feature_names_out_: list[str] = []

    def fit_transform(self, X_train: pd.DataFrame) -> np.ndarray:
        """Fit preprocessor exclusively on training partition and transform it."""
        transformers = []

        if self.numeric_features:
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )
            transformers.append(("numeric", num_pipeline, self.numeric_features))

        if self.categorical_features:
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "onehot",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            sparse_output=False,
                            drop=None,
                        ),
                    ),
                ]
            )
            transformers.append(("categorical", cat_pipeline, self.categorical_features))

        if not transformers:
            raise ValueError("No valid numeric or categorical features supplied for preprocessing.")

        self.pipeline = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            verbose_feature_names_out=False,
        )

        transformed = self.pipeline.fit_transform(X_train)

        # Cache output feature names
        try:
            self.feature_names_out_ = list(self.pipeline.get_feature_names_out())
        except Exception:
            self.feature_names_out_ = [f"feat_{i}" for i in range(transformed.shape[1])]

        return transformed

    def transform(self, X_test: pd.DataFrame) -> np.ndarray:
        """Transform test or unseen data using previously fitted parameters (zero test leakage)."""
        if self.pipeline is None:
            raise RuntimeError(
                "Preprocessor has not been fitted yet. Call fit_transform(X_train) first."
            )
        return self.pipeline.transform(X_test)

    def get_feature_names_out(self) -> list[str]:
        """Return transformed feature names after preprocessing."""
        return self.feature_names_out_

    def get_transformed_feature_names(self) -> list[str]:
        """Alias for get_feature_names_out."""
        return self.get_feature_names_out()

    @staticmethod
    def extract_datetime_features(df: pd.DataFrame, datetime_col: str) -> pd.DataFrame:
        """Derive standard non-leaking calendar features from timestamp series."""
        df_copy = df.copy()
        dt_series = pd.to_datetime(df_copy[datetime_col], errors="coerce")
        df_copy[f"{datetime_col}_year"] = dt_series.dt.year.fillna(2000).astype(int)
        df_copy[f"{datetime_col}_month"] = dt_series.dt.month.fillna(1).astype(int)
        df_copy[f"{datetime_col}_day"] = dt_series.dt.day.fillna(1).astype(int)
        df_copy[f"{datetime_col}_dayofweek"] = dt_series.dt.dayofweek.fillna(0).astype(int)
        df_copy[f"{datetime_col}_is_weekend"] = (
            df_copy[f"{datetime_col}_dayofweek"].isin([5, 6]).astype(int)
        )
        return df_copy
