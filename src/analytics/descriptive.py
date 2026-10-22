"""Comprehensive descriptive statistics for numerical, categorical, datetime, and boolean variables."""

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.analytics.analysis_models import CategoricalSummary, NumericSummary
from src.core.logging_config import get_logger

logger = get_logger("analytics.descriptive")


def calculate_numeric_summary(
    series: pd.Series,
    col_name: str | None = None,
) -> NumericSummary:
    """Calculate comprehensive, non-crashing descriptive metrics for a numeric Series."""
    name = (
        col_name
        if col_name is not None
        else (str(series.name) if series.name is not None else "numeric_col")
    )
    total_len = len(series)
    clean = pd.to_numeric(series.dropna(), errors="coerce")
    clean = clean[np.isfinite(clean)]
    non_null_count = len(clean)

    missing_cnt = total_len - non_null_count
    missing_pct = round((missing_cnt / total_len) * 100.0, 2) if total_len > 0 else 0.0

    if non_null_count == 0:
        return NumericSummary(
            column_name=name,
            count=0,
            missing_count=missing_cnt,
            missing_percentage=missing_pct,
            mean=0.0,
            median=0.0,
            std=0.0,
            variance=0.0,
            min_value=0.0,
            max_value=0.0,
            q25=0.0,
            q75=0.0,
            iqr=0.0,
            skewness=None,
            kurtosis=None,
            zero_count=0,
            negative_count=0,
            outlier_count_iqr=0,
        )

    mean_val = float(clean.mean())
    median_val = float(clean.median())
    std_val = float(clean.std(ddof=1)) if non_null_count > 1 else 0.0
    var_val = float(clean.var(ddof=1)) if non_null_count > 1 else 0.0
    min_val = float(clean.min())
    max_val = float(clean.max())

    q25 = float(clean.quantile(0.25))
    q75 = float(clean.quantile(0.75))
    iqr = float(q75 - q25)

    skew_val: float | None = None
    kurt_val: float | None = None
    if non_null_count >= 5 and std_val > 1e-9:
        try:
            skew_val = round(float(stats.skew(clean, bias=False)), 4)
            kurt_val = round(float(stats.kurtosis(clean, bias=False)), 4)
        except Exception:
            skew_val = round(float(clean.skew()), 4)
            kurt_val = round(float(clean.kurt()), 4)

    zero_count = int((clean == 0).sum())
    neg_count = int((clean < 0).sum())

    # Count outliers outside 1.5 * IQR
    lower_b = q25 - 1.5 * iqr
    upper_b = q75 + 1.5 * iqr
    outlier_count = int(((clean < lower_b) | (clean > upper_b)).sum()) if iqr > 0 else 0

    return NumericSummary(
        column_name=name,
        count=non_null_count,
        missing_count=missing_cnt,
        missing_percentage=missing_pct,
        mean=round(mean_val, 4),
        median=round(median_val, 4),
        std=round(std_val, 4),
        variance=round(var_val, 4),
        min_value=round(min_val, 4),
        max_value=round(max_val, 4),
        q25=round(q25, 4),
        q75=round(q75, 4),
        iqr=round(iqr, 4),
        skewness=skew_val,
        kurtosis=kurt_val,
        zero_count=zero_count,
        negative_count=neg_count,
        outlier_count_iqr=outlier_count,
    )


def calculate_categorical_summary(
    series: pd.Series,
    col_name: str | None = None,
    max_top_categories: int = 10,
) -> CategoricalSummary:
    """Calculate frequency metrics, concentration ratios, and rare categories for discrete columns."""
    name = (
        col_name
        if col_name is not None
        else (str(series.name) if series.name is not None else "categorical_col")
    )
    total_len = len(series)
    non_null_series = series.dropna().astype(str)
    non_null_count = len(non_null_series)
    missing_cnt = total_len - non_null_count

    if non_null_count == 0:
        return CategoricalSummary(
            column_name=name,
            total_count=0,
            missing_count=missing_cnt,
            distinct_count=0,
        )

    val_counts = non_null_series.value_counts()
    distinct_cnt = len(val_counts)

    top_cats: list[dict[str, Any]] = []
    for val, cnt in val_counts.head(max_top_categories).items():
        top_cats.append(
            {
                "label": str(val)[:50],
                "count": int(cnt),
                "percentage": round((cnt / non_null_count) * 100.0, 2),
            }
        )

    top_1_share = (
        round((val_counts.iloc[0] / non_null_count) * 100.0, 2) if distinct_cnt > 0 else 0.0
    )
    top_3_share = (
        round((val_counts.head(3).sum() / non_null_count) * 100.0, 2) if distinct_cnt > 0 else 0.0
    )

    rare_threshold = max(1, int(non_null_count * 0.01))
    rare_count = int((val_counts < rare_threshold).sum())
    empty_strings = int((non_null_series.str.strip() == "").sum())

    return CategoricalSummary(
        column_name=name,
        total_count=non_null_count,
        missing_count=missing_cnt,
        distinct_count=distinct_cnt,
        top_categories=top_cats,
        top_1_concentration_pct=top_1_share,
        top_3_concentration_pct=top_3_share,
        rare_categories_count=rare_count,
        empty_strings_count=empty_strings,
    )


def compute_all_numeric_summaries(
    df: pd.DataFrame,
    numeric_cols: list[str],
) -> dict[str, NumericSummary]:
    """Compute numeric summaries for multiple columns."""
    results: dict[str, NumericSummary] = {}
    for col in numeric_cols:
        if col in df.columns:
            results[col] = calculate_numeric_summary(df[col], col)
    return results


# Aliases for descriptive functions
compute_numeric_summary = calculate_numeric_summary
compute_categorical_summary = calculate_categorical_summary
