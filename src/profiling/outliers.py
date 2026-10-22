"""Outlier and anomaly detection using IQR, Z-Score, and Modified Z-Score (MAD)."""

import uuid

import numpy as np
import pandas as pd

from src.ingestion.models import ColumnSchema
from src.profiling.models import QualityIssue


def detect_outliers_iqr(series: pd.Series, k: float = 1.5) -> dict[str, object]:
    """Detect outliers using the Interquartile Range (IQR) rule."""
    clean = series.dropna().astype(float)
    if len(clean) < 5:
        return {"count": 0, "percentage": 0.0, "lower_bound": None, "upper_bound": None}

    q1 = float(clean.quantile(0.25))
    q3 = float(clean.quantile(0.75))
    iqr = q3 - q1

    if iqr == 0:
        return {"count": 0, "percentage": 0.0, "lower_bound": q1, "upper_bound": q3}

    lower_bound = round(q1 - k * iqr, 4)
    upper_bound = round(q3 + k * iqr, 4)

    outliers = clean[(clean < lower_bound) | (clean > upper_bound)]
    count = int(len(outliers))
    pct = round((count / len(clean)) * 100.0, 2)

    return {
        "count": count,
        "percentage": pct,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "outlier_indices": outliers.index.tolist()[:100],  # sample up to 100 indices
    }


def detect_outliers_zscore(series: pd.Series, threshold: float = 3.0) -> dict[str, object]:
    """Detect outliers using standard Z-scores."""
    clean = series.dropna().astype(float)
    if len(clean) < 10:
        return {"count": 0, "percentage": 0.0, "lower_bound": None, "upper_bound": None}

    mean = float(clean.mean())
    std = float(clean.std(ddof=1))

    if std == 0 or np.isnan(std):
        return {"count": 0, "percentage": 0.0, "lower_bound": mean, "upper_bound": mean}

    lower_bound = round(mean - threshold * std, 4)
    upper_bound = round(mean + threshold * std, 4)

    outliers = clean[(clean < lower_bound) | (clean > upper_bound)]
    count = int(len(outliers))
    pct = round((count / len(clean)) * 100.0, 2)

    return {
        "count": count,
        "percentage": pct,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "outlier_indices": outliers.index.tolist()[:100],
    }


def detect_outliers_modified_zscore(series: pd.Series, threshold: float = 3.5) -> dict[str, object]:
    """Detect outliers using robust Modified Z-scores based on Median Absolute Deviation (MAD)."""
    clean = series.dropna().astype(float)
    if len(clean) < 10:
        return {"count": 0, "percentage": 0.0, "lower_bound": None, "upper_bound": None}

    median = float(clean.median())
    abs_deviations = (clean - median).abs()
    mad = float(abs_deviations.median())

    if mad == 0:
        # If MAD is 0, fall back to IQR or report 0 outliers
        return {"count": 0, "percentage": 0.0, "lower_bound": median, "upper_bound": median}

    # Modified Z-score = 0.6745 * (x - median) / MAD
    mod_z = 0.6745 * abs_deviations / mad
    outliers = clean[mod_z > threshold]
    count = int(len(outliers))
    pct = round((count / len(clean)) * 100.0, 2)

    lower_bound = round(median - (threshold * mad / 0.6745), 4)
    upper_bound = round(median + (threshold * mad / 0.6745), 4)

    return {
        "count": count,
        "percentage": pct,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "outlier_indices": outliers.index.tolist()[:100],
    }


def analyze_outliers(
    df: pd.DataFrame,
    schemas: dict[str, ColumnSchema],
) -> tuple[dict[str, dict[str, object]], list[QualityIssue]]:
    """Scan all eligible numeric columns for statistical outliers and return metrics and issues."""
    outlier_stats: dict[str, dict[str, object]] = {}
    issues: list[QualityIssue] = []

    for col_name, schema in schemas.items():
        if schema.inferred_type != "numeric" or schema.is_identifier or schema.is_constant:
            continue

        if col_name not in df.columns:
            continue

        series = df[col_name]
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]

        iqr_res = detect_outliers_iqr(series)
        z_res = detect_outliers_zscore(series)
        mad_res = detect_outliers_modified_zscore(series)

        outlier_stats[col_name] = {
            "iqr": iqr_res,
            "zscore": z_res,
            "modified_zscore": mad_res,
        }

        # Issue generation: Flag if IQR outliers exceed 3% and z-score outliers exist
        iqr_count = int(iqr_res["count"])  # type: ignore[arg-type]
        iqr_pct = float(iqr_res["percentage"])  # type: ignore[arg-type]

        if iqr_count > 0 and iqr_pct >= 3.0:
            severity = "high" if iqr_pct >= 10.0 else "medium"
            issues.append(
                QualityIssue(
                    issue_id=f"iss_outlier_{uuid.uuid4().hex[:6]}",
                    category="outliers",
                    severity=severity,
                    title=f"Statistical outliers detected in '{col_name}' ({iqr_count} rows, {iqr_pct}%)",
                    affected_columns=[col_name],
                    affected_row_count=iqr_count,
                    affected_cell_count=iqr_count,
                    measured_evidence=(
                        f"{iqr_count} values lie beyond 1.5×IQR bounds "
                        f"[{iqr_res['lower_bound']}, {iqr_res['upper_bound']}]."
                    ),
                    explanation="Extreme numerical values can distort parametric statistics like means and variances.",
                    recommended_action="Cap values at 1st and 99th percentiles or filter invalid records.",
                    potential_consequences="Outliers may inflate regression residuals and bias hypothesis test p-values.",
                    can_preview=True,
                    suggested_action_type="cap_outliers",
                    suggested_action_params={
                        "column_name": col_name,
                        "lower_percentile": 0.01,
                        "upper_percentile": 0.99,
                    },
                )
            )

    return outlier_stats, issues
