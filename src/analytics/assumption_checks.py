"""Diagnostic verification of normality, homoscedasticity, and distributional assumptions."""

import numpy as np
import pandas as pd
from scipy import stats

from src.analytics.statistical_models import AssumptionCheckResult


def check_normality_shapiro(
    series: pd.Series,
    alpha: float = 0.05,
) -> AssumptionCheckResult:
    """Evaluate normality using the Shapiro-Wilk test with sample size safeguards."""
    clean = pd.to_numeric(series, errors="coerce").dropna().values
    clean = clean[np.isfinite(clean)]
    n = len(clean)

    if n < 3:
        return AssumptionCheckResult(
            name="Normality (Shapiro-Wilk)",
            statistic_name="W",
            statistic_value=None,
            p_value=None,
            is_satisfied=False,
            severity="warning",
            message=f"Insufficient observations (N = {n}) to test normality (minimum 3 required).",
            recommendation="Collect more data or use non-parametric rank methods.",
        )

    # Shapiro-Wilk is valid up to N=5000
    test_sample = (
        clean if n <= 5000 else np.random.default_rng(42).choice(clean, size=5000, replace=False)
    )
    stat, p_val = stats.shapiro(test_sample)
    stat_val = round(float(stat), 4)
    p_float = float(p_val)

    is_normal = p_float >= alpha
    severity = "info" if is_normal else ("warning" if n < 100 else "info")

    if is_normal:
        msg = (
            f"Shapiro-Wilk W = {stat_val:.4f}, p = {p_float:.4f} (>= {alpha}). "
            "Data does not show statistically significant departure from normality."
        )
        rec = "Parametric procedures are reasonable under distributional assumptions."
    else:
        msg = (
            f"Shapiro-Wilk W = {stat_val:.4f}, p = {p_float:.4f} (< {alpha}). "
            "Distribution departs significantly from standard Gaussian symmetry."
        )
        if n >= 50:
            rec = (
                "With N >= 50, the Central Limit Theorem provides robustness for mean-based tests. "
                "However, inspect skewness or consider rank tests if outliers are severe."
            )
        else:
            rec = "Consider non-parametric alternatives (e.g., Mann-Whitney U or Wilcoxon) or bootstrap intervals."

    return AssumptionCheckResult(
        name="Normality (Shapiro-Wilk)",
        statistic_name="W",
        statistic_value=stat_val,
        p_value=round(p_float, 6),
        is_satisfied=is_normal,
        severity=severity,
        message=msg,
        recommendation=rec,
    )


def check_normality_dagostino(
    series: pd.Series,
    alpha: float = 0.05,
) -> AssumptionCheckResult:
    """Evaluate normality using D'Agostino and Pearson's omnibus test of skewness & kurtosis."""
    clean = pd.to_numeric(series, errors="coerce").dropna().values
    clean = clean[np.isfinite(clean)]
    n = len(clean)

    if n < 8:
        return AssumptionCheckResult(
            name="Normality (D'Agostino-Pearson)",
            statistic_name="K2",
            statistic_value=None,
            p_value=None,
            is_satisfied=False,
            severity="info",
            message=f"Sample size N = {n} is too small for D'Agostino's test (minimum 8 required).",
            recommendation="Use Shapiro-Wilk or visual Q-Q plots for smaller samples.",
        )

    stat, p_val = stats.normaltest(clean)
    stat_val = round(float(stat), 4)
    p_float = float(p_val)
    is_normal = p_float >= alpha

    msg = (
        f"Omnibus K2 = {stat_val:.4f}, p = {p_float:.4f} ({'>=' if is_normal else '<'} {alpha}). "
        f"{'No significant deviation from normality.' if is_normal else 'Departure from normal skew/kurtosis detected.'}"
    )
    rec = (
        "Normal assumptions acceptable."
        if is_normal
        else "Inspect Q-Q plots to evaluate whether deviations are tail-driven or global."
    )

    return AssumptionCheckResult(
        name="Normality (D'Agostino-Pearson)",
        statistic_name="K2",
        statistic_value=stat_val,
        p_value=round(p_float, 6),
        is_satisfied=is_normal,
        severity="info" if is_normal else "warning",
        message=msg,
        recommendation=rec,
    )


def check_homoscedasticity_levene(
    groups: list[pd.Series | np.ndarray],
    group_names: list[str] | None = None,
    alpha: float = 0.05,
    center: str = "median",
) -> AssumptionCheckResult:
    """Evaluate equality of variances using Levene's test (Brown-Forsythe median center default)."""
    clean_groups = []
    for g in groups:
        arr = pd.to_numeric(pd.Series(g), errors="coerce").dropna().values
        arr = arr[np.isfinite(arr)]
        if len(arr) >= 2:
            clean_groups.append(arr)

    if len(clean_groups) < 2:
        return AssumptionCheckResult(
            name="Equal Variances (Brown-Forsythe / Levene)",
            statistic_name="W",
            statistic_value=None,
            p_value=None,
            is_satisfied=True,
            severity="info",
            message="Fewer than two valid groups with N >= 2 available to evaluate variance homogeneity.",
            recommendation="Ensure at least two groups have sufficient non-null observations.",
        )

    stat, p_val = stats.levene(*clean_groups, center=center)
    stat_val = round(float(stat), 4)
    p_float = float(p_val)
    equal_var = p_float >= alpha

    names_str = ", ".join(group_names) if group_names else f"{len(clean_groups)} groups"
    if equal_var:
        msg = (
            f"Brown-Forsythe W = {stat_val:.4f}, p = {p_float:.4f} (>= {alpha}) across {names_str}. "
            "Equal variance assumption is reasonably supported."
        )
        rec = (
            "Standard pooled-variance parametric tests (Student's t, classic ANOVA) are defensible."
        )
    else:
        msg = (
            f"Brown-Forsythe W = {stat_val:.4f}, p = {p_float:.4f} (< {alpha}) across {names_str}. "
            "Variances between groups are significantly unequal (heteroscedasticity)."
        )
        rec = "Use Welch's t-test or Welch's ANOVA, which do not assume equal group variances."

    return AssumptionCheckResult(
        name="Equal Variances (Brown-Forsythe / Levene)",
        statistic_name="W",
        statistic_value=stat_val,
        p_value=round(p_float, 6),
        is_satisfied=equal_var,
        severity="info" if equal_var else "warning",
        message=msg,
        recommendation=rec,
    )


def generate_qq_plot_data(series: pd.Series) -> dict[str, list[float]]:
    """Compute theoretical normal quantiles and sample ordered values for Q-Q diagnostic plotting."""
    clean = pd.to_numeric(series, errors="coerce").dropna().values
    clean = clean[np.isfinite(clean)]
    if len(clean) < 3:
        return {"theoretical_quantiles": [], "sample_quantiles": []}

    (osm, osr), (slope, intercept, _) = stats.probplot(clean, dist="norm")
    return {
        "theoretical_quantiles": [round(float(x), 4) for x in osm],
        "sample_quantiles": [round(float(y), 4) for y in osr],
        "trend_line_x": [round(float(osm[0]), 4), round(float(osm[-1]), 4)],
        "trend_line_y": [
            round(float(slope * osm[0] + intercept), 4),
            round(float(slope * osm[-1] + intercept), 4),
        ],
    }
