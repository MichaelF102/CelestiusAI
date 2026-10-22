"""Orchestrator for comprehensive dataset profiling and quality issue consolidation."""

from typing import Any

import numpy as np
import pandas as pd

from src.ingestion.models import ColumnSchema
from src.ingestion.schema import infer_dataset_schema
from src.profiling.consistency import analyze_consistency
from src.profiling.duplicates import analyze_duplicates
from src.profiling.missing_values import analyze_missing_values
from src.profiling.models import DatasetProfile
from src.profiling.outliers import analyze_outliers
from src.profiling.quality_score import calculate_quality_score


def profile_numeric_column(series: pd.Series) -> dict[str, Any]:
    """Calculate descriptive statistics, percentiles, skewness, and non-finite counts."""
    clean = series.dropna().astype(float)
    non_null_count = len(clean)

    if non_null_count == 0:
        return {"non_null_count": 0}

    mean_val = float(clean.mean())
    median_val = float(clean.median())
    std_val = float(clean.std(ddof=1)) if non_null_count > 1 else 0.0
    var_val = float(clean.var(ddof=1)) if non_null_count > 1 else 0.0
    min_val = float(clean.min())
    max_val = float(clean.max())
    q25 = float(clean.quantile(0.25))
    q75 = float(clean.quantile(0.75))
    iqr = q75 - q25

    # Safe skewness and kurtosis
    skew_val: float | None = None
    kurt_val: float | None = None
    if non_null_count >= 5 and std_val > 0:
        try:
            skew_val = round(float(clean.skew()), 4)
            kurt_val = round(float(clean.kurt()), 4)
        except Exception:
            pass

    zero_count = int((clean == 0).sum())
    neg_count = int((clean < 0).sum())
    inf_count = int(np.isinf(clean).sum())

    return {
        "non_null_count": non_null_count,
        "mean": round(mean_val, 4),
        "median": round(median_val, 4),
        "std": round(std_val, 4),
        "variance": round(var_val, 4),
        "min": round(min_val, 4),
        "max": round(max_val, 4),
        "range": round(max_val - min_val, 4),
        "q25": round(q25, 4),
        "q75": round(q75, 4),
        "iqr": round(iqr, 4),
        "skewness": skew_val,
        "kurtosis": kurt_val,
        "zero_count": zero_count,
        "negative_count": neg_count,
        "infinite_count": inf_count,
    }


def profile_categorical_column(series: pd.Series) -> dict[str, Any]:
    """Calculate frequencies, concentration, rare categories, and whitespace issues."""
    clean = series.dropna().astype(str)
    non_null_count = len(clean)

    if non_null_count == 0:
        return {"non_null_count": 0}

    val_counts = clean.value_counts()
    distinct_count = len(val_counts)

    # Top 5 most frequent
    top_5 = [
        {
            "value": str(val)[:40],
            "count": int(cnt),
            "percentage": round((cnt / non_null_count) * 100.0, 2),
        }
        for val, cnt in val_counts.head(5).items()
    ]

    top_1_share = (
        round((val_counts.iloc[0] / non_null_count) * 100.0, 2) if distinct_count > 0 else 0.0
    )
    top_3_share = (
        round((val_counts.head(3).sum() / non_null_count) * 100.0, 2) if distinct_count > 0 else 0.0
    )

    # Rare categories (< 1% frequency)
    rare_count = int((val_counts < (non_null_count * 0.01)).sum())

    # Whitespace and empty string count
    empty_str_count = int((clean.str.strip() == "").sum())

    return {
        "non_null_count": non_null_count,
        "distinct_count": distinct_count,
        "top_frequent": top_5,
        "top_1_concentration_pct": top_1_share,
        "top_3_concentration_pct": top_3_share,
        "rare_category_count": rare_count,
        "empty_string_count": empty_str_count,
    }


def profile_datetime_column(series: pd.Series) -> dict[str, Any]:
    """Calculate chronological boundaries, span in days, and unparseable counts."""
    # Attempt parsing if not already datetime
    if not pd.api.types.is_datetime64_any_dtype(series):
        dt_series = pd.to_datetime(series, errors="coerce")
    else:
        dt_series = series

    valid = dt_series.dropna()
    if len(valid) == 0:
        return {"valid_dates_count": 0, "unparseable_count": int(series.notna().sum())}

    min_ts = valid.min()
    max_ts = valid.max()
    span_days = (max_ts - min_ts).days if hasattr((max_ts - min_ts), "days") else None

    return {
        "valid_dates_count": len(valid),
        "unparseable_count": int(dt_series.isna().sum()) - int(series.isna().sum()),
        "min_timestamp": str(min_ts),
        "max_timestamp": str(max_ts),
        "span_days": span_days,
    }


def profile_dataset(
    df: pd.DataFrame,
    dataset_id: str = "dataset_active",
    schemas: dict[str, ColumnSchema] | None = None,
) -> DatasetProfile:
    """Generate a complete independent data profile, audit report, and quality score."""
    row_count = len(df)
    col_count = len(df.columns)
    total_cells = row_count * col_count
    total_missing_cells = int(df.isna().sum().sum())
    missing_pct = round((total_missing_cells / total_cells) * 100.0, 2) if total_cells > 0 else 0.0
    mem_usage = int(df.memory_usage(deep=True).sum()) if row_count > 0 else 0

    if schemas is None:
        schemas = infer_dataset_schema(df)

    # 1. Missingness analysis
    missing_stats, missing_issues = analyze_missing_values(df, schemas)

    # 2. Duplicate analysis
    dup_stats, dup_issues = analyze_duplicates(df)
    duplicate_rows = int(dup_stats.get("exact_duplicate_rows", 0))
    duplicate_pct = float(dup_stats.get("exact_duplicate_percentage", 0.0))

    # 3. Outlier analysis
    outlier_stats, outlier_issues = analyze_outliers(df, schemas)

    # 4. Consistency analysis
    consistency_stats, consistency_issues = analyze_consistency(df, schemas)

    # Consolidate issues
    all_issues = missing_issues + dup_issues + outlier_issues + consistency_issues

    # Sort issues by severity (critical, high, medium, low)
    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    all_issues.sort(key=lambda i: severity_order.get(i.severity, 4))

    # 5. Quality scoring
    quality_score = calculate_quality_score(
        total_cells=total_cells,
        missing_cells=total_missing_cells,
        total_rows=row_count,
        duplicate_rows=duplicate_rows,
        issues=all_issues,
    )

    # 6. Column-level specific profiles
    numeric_profiles: dict[str, dict[str, Any]] = {}
    categorical_profiles: dict[str, dict[str, Any]] = {}
    datetime_profiles: dict[str, dict[str, Any]] = {}

    constant_cols = [c for c, s in schemas.items() if s.is_constant]
    near_constant_cols = [c for c, s in schemas.items() if s.is_near_constant]
    id_cols = [c for c, s in schemas.items() if s.is_identifier]

    for col, schema in schemas.items():
        if col not in df.columns:
            continue
        series = df[col]
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]

        if schema.inferred_type == "numeric":
            numeric_profiles[col] = profile_numeric_column(series)
        elif schema.inferred_type in ("categorical", "text", "boolean"):
            categorical_profiles[col] = profile_categorical_column(series)
        elif schema.inferred_type == "datetime":
            datetime_profiles[col] = profile_datetime_column(series)

    return DatasetProfile(
        dataset_id=dataset_id,
        row_count=row_count,
        column_count=col_count,
        memory_usage_bytes=mem_usage,
        total_missing_cells=total_missing_cells,
        total_cells=total_cells,
        missing_cell_percentage=missing_pct,
        duplicate_row_count=duplicate_rows,
        duplicate_row_percentage=duplicate_pct,
        constant_columns=constant_cols,
        near_constant_columns=near_constant_cols,
        identifier_columns=id_cols,
        column_schemas=schemas,
        numeric_profiles=numeric_profiles,
        categorical_profiles=categorical_profiles,
        datetime_profiles=datetime_profiles,
        quality_issues=all_issues,
        quality_score=quality_score,
    )
