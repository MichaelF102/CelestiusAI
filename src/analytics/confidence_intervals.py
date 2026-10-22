"""Confidence interval estimation for means, mean differences, proportions, and correlations."""

import numpy as np
import pandas as pd
from scipy import stats

from src.analytics.statistical_models import ConfidenceIntervalResult


def calculate_mean_ci(
    series: pd.Series | np.ndarray,
    confidence_level: float = 0.95,
) -> ConfidenceIntervalResult:
    """Calculate Student's t confidence interval for a single continuous mean."""
    clean = pd.to_numeric(pd.Series(series), errors="coerce").dropna().values
    clean = clean[np.isfinite(clean)]
    n = len(clean)

    if n < 2:
        val = float(clean[0]) if n == 1 else 0.0
        return ConfidenceIntervalResult(
            estimand="Population Mean (μ)",
            point_estimate=val,
            lower_bound=val,
            upper_bound=val,
            confidence_level=confidence_level,
            method="Degenerate (N < 2)",
        )

    mean_val = float(np.mean(clean))
    std_val = float(np.std(clean, ddof=1))
    se = std_val / np.sqrt(n)

    alpha = 1.0 - confidence_level
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=n - 1))
    margin = t_crit * se

    return ConfidenceIntervalResult(
        estimand="Population Mean (μ)",
        point_estimate=round(mean_val, 4),
        lower_bound=round(mean_val - margin, 4),
        upper_bound=round(mean_val + margin, 4),
        confidence_level=confidence_level,
        method=f"Student's t-interval (df = {n - 1})",
    )


def calculate_difference_in_means_ci(
    group1: pd.Series | np.ndarray,
    group2: pd.Series | np.ndarray,
    confidence_level: float = 0.95,
    equal_var: bool = False,
) -> ConfidenceIntervalResult:
    """Calculate Welch-Satterthwaite (or pooled Student's) CI for difference in two independent means."""
    g1 = pd.to_numeric(pd.Series(group1), errors="coerce").dropna().values
    g2 = pd.to_numeric(pd.Series(group2), errors="coerce").dropna().values
    g1 = g1[np.isfinite(g1)]
    g2 = g2[np.isfinite(g2)]

    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return ConfidenceIntervalResult(
            estimand="Difference in Means (μ₁ - μ₂)",
            point_estimate=0.0,
            lower_bound=0.0,
            upper_bound=0.0,
            confidence_level=confidence_level,
            method="Degenerate (N < 2 in groups)",
        )

    m1, m2 = float(np.mean(g1)), float(np.mean(g2))
    v1, v2 = float(np.var(g1, ddof=1)), float(np.var(g2, ddof=1))
    diff = m1 - m2
    alpha = 1.0 - confidence_level

    if equal_var:
        df_pooled = n1 + n2 - 2
        sp2 = ((n1 - 1) * v1 + (n2 - 1) * v2) / df_pooled
        se = np.sqrt(sp2 * (1.0 / n1 + 1.0 / n2))
        t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=df_pooled))
        method_name = f"Student's pooled t-interval (df = {df_pooled})"
    else:
        # Welch-Satterthwaite approximation
        se1 = v1 / n1
        se2 = v2 / n2
        se = np.sqrt(se1 + se2)
        numerator = (se1 + se2) ** 2
        denominator = (se1**2 / (n1 - 1)) + (se2**2 / (n2 - 1)) if (n1 > 1 and n2 > 1) else 1.0
        df_welch = max(1.0, numerator / denominator)
        t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=df_welch))
        method_name = f"Welch-Satterthwaite t-interval (df = {df_welch:.1f})"

    margin = t_crit * se
    return ConfidenceIntervalResult(
        estimand="Difference in Means (μ₁ - μ₂)",
        point_estimate=round(diff, 4),
        lower_bound=round(diff - margin, 4),
        upper_bound=round(diff + margin, 4),
        confidence_level=confidence_level,
        method=method_name,
    )


def calculate_paired_difference_ci(
    differences: pd.Series | np.ndarray,
    confidence_level: float = 0.95,
) -> ConfidenceIntervalResult:
    """Calculate Student's t confidence interval for mean of paired differences."""
    res = calculate_mean_ci(differences, confidence_level=confidence_level)
    return ConfidenceIntervalResult(
        estimand="Paired Mean Difference (μ_D)",
        point_estimate=res.point_estimate,
        lower_bound=res.lower_bound,
        upper_bound=res.upper_bound,
        confidence_level=confidence_level,
        method=f"Paired {res.method}",
    )


def calculate_proportion_ci(
    successes: int,
    trials: int,
    confidence_level: float = 0.95,
) -> ConfidenceIntervalResult:
    """Calculate Wilson score confidence interval for a single proportion."""
    if trials <= 0:
        return ConfidenceIntervalResult(
            estimand="Proportion (p)",
            point_estimate=0.0,
            lower_bound=0.0,
            upper_bound=0.0,
            confidence_level=confidence_level,
            method="Invalid trials (N <= 0)",
        )

    p_hat = successes / trials
    alpha = 1.0 - confidence_level
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))

    denominator = 1.0 + (z**2 / trials)
    centre = (p_hat + (z**2 / (2.0 * trials))) / denominator
    margin = (
        z * np.sqrt((p_hat * (1.0 - p_hat) / trials) + (z**2 / (4.0 * trials**2)))
    ) / denominator

    lower = max(0.0, centre - margin)
    upper = min(1.0, centre + margin)

    return ConfidenceIntervalResult(
        estimand="Proportion (p)",
        point_estimate=round(p_hat, 4),
        lower_bound=round(lower, 4),
        upper_bound=round(upper, 4),
        confidence_level=confidence_level,
        method="Wilson score interval",
    )


def calculate_correlation_ci(
    r: float,
    n: int,
    confidence_level: float = 0.95,
) -> ConfidenceIntervalResult:
    """Calculate Fisher's z-transformation confidence interval for Pearson correlation."""
    if n <= 3 or abs(r) >= 1.0:
        return ConfidenceIntervalResult(
            estimand="Pearson Correlation (ρ)",
            point_estimate=r,
            lower_bound=r,
            upper_bound=r,
            confidence_level=confidence_level,
            method="Degenerate (N <= 3 or |r| >= 1)",
        )

    # Fisher z-transformation: z = arctanh(r)
    z = np.arctanh(r)
    se_z = 1.0 / np.sqrt(n - 3)

    alpha = 1.0 - confidence_level
    z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))

    z_lower = z - z_crit * se_z
    z_upper = z + z_crit * se_z

    # Inverse transform: tanh(z)
    r_lower = float(np.tanh(z_lower))
    r_upper = float(np.tanh(z_upper))

    return ConfidenceIntervalResult(
        estimand="Pearson Correlation (ρ)",
        point_estimate=round(r, 4),
        lower_bound=round(r_lower, 4),
        upper_bound=round(r_upper, 4),
        confidence_level=confidence_level,
        method=f"Fisher's z-transformation (N = {n})",
    )


def calculate_bootstrap_ci(
    data: pd.Series | np.ndarray,
    statistic_func=np.median,
    estimand_name: str = "Median",
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    random_seed: int = 42,
) -> ConfidenceIntervalResult:
    """Compute empirical percentile bootstrap confidence interval with reproducible seed."""
    clean = pd.to_numeric(pd.Series(data), errors="coerce").dropna().values
    clean = clean[np.isfinite(clean)]
    n = len(clean)

    if n < 3:
        pt = float(clean[0]) if n > 0 else 0.0
        return ConfidenceIntervalResult(
            estimand=estimand_name,
            point_estimate=pt,
            lower_bound=pt,
            upper_bound=pt,
            confidence_level=confidence_level,
            method="Degenerate (N < 3)",
        )

    rng = np.random.default_rng(random_seed)
    # Resample with replacement
    boot_indices = rng.integers(0, n, size=(n_resamples, n))
    boot_samples = clean[boot_indices]
    boot_stats = np.apply_along_axis(statistic_func, 1, boot_samples)

    point_est = float(statistic_func(clean))
    alpha = 1.0 - confidence_level
    lower_pct = (alpha / 2.0) * 100.0
    upper_pct = (1.0 - alpha / 2.0) * 100.0

    ci_lower = float(np.percentile(boot_stats, lower_pct))
    ci_upper = float(np.percentile(boot_stats, upper_pct))

    return ConfidenceIntervalResult(
        estimand=estimand_name,
        point_estimate=round(point_est, 4),
        lower_bound=round(ci_lower, 4),
        upper_bound=round(ci_upper, 4),
        confidence_level=confidence_level,
        method=f"Percentile Bootstrap (B = {n_resamples:,}, seed = {random_seed})",
    )
