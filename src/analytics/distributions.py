"""Distribution shape, skewness, kurtosis, and typical value diagnostics."""

from typing import Any

import numpy as np
import pandas as pd

from src.analytics.analysis_models import NumericSummary
from src.analytics.descriptive import calculate_numeric_summary


def evaluate_distribution_shape(summary: NumericSummary) -> dict[str, Any]:
    """Diagnose distribution asymmetry, tails, typical value representation, and analytical risks."""
    skew = summary.skewness
    if skew is None:
        skew_class = "unknown"
        skew_category = "symmetric"
        is_skewed = False
        skew_description = "Sample size too small to evaluate skewness reliably."
    elif abs(skew) < 0.5:
        skew_class = "approximately_symmetric"
        skew_category = "symmetric"
        is_skewed = False
        skew_description = "The distribution is approximately symmetric around the center."
    elif 0.5 <= skew <= 1.0:
        skew_class = "moderately_right_skewed"
        skew_category = "moderately_skewed"
        is_skewed = True
        skew_description = "Moderately right-skewed: a positive tail extends towards higher values."
    elif skew > 1.0:
        skew_class = "highly_right_skewed"
        skew_category = "highly_skewed"
        is_skewed = True
        skew_description = "Highly right-skewed: a small number of large values pull the distribution to the right."
    elif -1.0 <= skew <= -0.5:
        skew_class = "moderately_left_skewed"
        skew_category = "moderately_skewed"
        is_skewed = True
        skew_description = "Moderately left-skewed: a negative tail extends towards lower values."
    else:
        skew_class = "highly_left_skewed"
        skew_category = "highly_skewed"
        is_skewed = True
        skew_description = (
            "Highly left-skewed: low extreme values pull the distribution to the left."
        )

    # Mean vs. Median divergence
    mean_val = summary.mean
    median_val = summary.median
    divergence_pct: float = 0.0
    if abs(median_val) > 1e-6:
        divergence_pct = round(((mean_val - median_val) / abs(median_val)) * 100.0, 2)

    # Guidance on central tendency
    if abs(divergence_pct) >= 10.0 or is_skewed:
        preferred_metric = "median"
        metric_rationale = (
            f"The mean ({mean_val:,.2f}) diverges by {abs(divergence_pct):.1f}% from the median ({median_val:,.2f}). "
            "Because the variable is asymmetrical, the median represents a more typical observation."
        )
    else:
        preferred_metric = "mean"
        metric_rationale = (
            f"The mean ({mean_val:,.2f}) and median ({median_val:,.2f}) are closely aligned, "
            "indicating that either metric reliably represents central tendency."
        )

    # Kurtosis / Tail thickness
    kurt = summary.kurtosis
    if kurt is None:
        tail_class = "unknown"
        tail_behavior = "normal tails"
        tail_description = "Insufficient observations to estimate kurtosis."
    elif kurt > 1.5:
        tail_class = "heavy_tailed_leptokurtic"
        tail_behavior = "heavy tails (leptokurtic)"
        tail_description = "Heavy-tailed (leptokurtic): greater probability of extreme outlier events than a normal distribution."
    elif kurt < -1.0:
        tail_class = "light_tailed_platykurtic"
        tail_behavior = "light tails (platykurtic)"
        tail_description = "Light-tailed (platykurtic): values are dispersed flatter without prominent central peaks."
    else:
        tail_class = "mesokurtic_normal_like"
        tail_behavior = "normal-like tails"
        tail_description = "Normal-like tail behavior with moderate dispersion."

    return {
        "column_name": summary.column_name,
        "is_skewed": is_skewed,
        "skewness_value": skew,
        "skewness_category": skew_category,
        "skewness_classification": skew_class,
        "skewness_description": skew_description,
        "kurtosis_value": kurt,
        "tail_behavior": tail_behavior,
        "tail_classification": tail_class,
        "tail_description": tail_description,
        "preferred_central_metric": preferred_metric,
        "metric_rationale": metric_rationale,
        "mean_median_divergence_pct": divergence_pct,
        "has_outliers": summary.outlier_count_iqr > 0,
        "outlier_count": summary.outlier_count_iqr,
    }


def diagnose_distribution_shape(
    data: pd.Series | NumericSummary,
) -> dict[str, Any]:
    """Diagnose distribution shape directly from a Series or precomputed NumericSummary."""
    if isinstance(data, NumericSummary):
        return evaluate_distribution_shape(data)
    summary = calculate_numeric_summary(data)
    return evaluate_distribution_shape(summary)


def calculate_empirical_quantiles(
    series: pd.Series,
    quantiles: list[float] | None = None,
) -> pd.DataFrame:
    """Calculate empirical quantiles across a numeric series."""
    if quantiles is None:
        quantiles = [0.01, 0.05, 0.10, 0.20, 0.25, 0.40, 0.50, 0.60, 0.75, 0.80, 0.90, 0.95, 0.99]

    clean = pd.to_numeric(series.dropna(), errors="coerce")
    clean = clean[np.isfinite(clean)]

    if clean.empty:
        return pd.DataFrame(columns=["Percentile", "Value"])

    q_vals = clean.quantile(quantiles)
    rows = []
    for q in quantiles:
        rows.append(
            {
                "Percentile": f"p{int(q * 100):02d}",
                "Value": round(float(q_vals.loc[q]), 4),
            }
        )
    return pd.DataFrame(rows)
