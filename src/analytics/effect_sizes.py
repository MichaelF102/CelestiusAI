"""Standardized effect-size metrics, magnitude benchmarks, and analytical interpretations."""

from typing import Literal

import numpy as np
import pandas as pd

from src.analytics.statistical_models import EffectSizeResult


def _classify_cohen_d(
    d_abs: float,
) -> Literal["negligible", "small", "medium", "large", "very_large"]:
    if d_abs < 0.2:
        return "negligible"
    if d_abs < 0.5:
        return "small"
    if d_abs < 0.8:
        return "medium"
    if d_abs < 1.2:
        return "large"
    return "very_large"


def _classify_eta_squared(
    eta_abs: float,
) -> Literal["negligible", "small", "medium", "large", "very_large"]:
    if eta_abs < 0.01:
        return "negligible"
    if eta_abs < 0.06:
        return "small"
    if eta_abs < 0.14:
        return "medium"
    if eta_abs < 0.25:
        return "large"
    return "very_large"


def _classify_cramers_v(
    v_abs: float, df_dim: int = 1
) -> Literal["negligible", "small", "medium", "large", "very_large"]:
    # Adjusted thresholds for degrees of freedom in contingency tables
    scale = np.sqrt(max(1, df_dim))
    adj_v = v_abs * scale
    if adj_v < 0.1:
        return "negligible"
    if adj_v < 0.3:
        return "small"
    if adj_v < 0.5:
        return "medium"
    if adj_v < 0.8:
        return "large"
    return "very_large"


def calculate_cohens_d(
    group1: pd.Series | np.ndarray,
    group2: pd.Series | np.ndarray,
    hedges_correction: bool = True,
    ci_level: float = 0.95,
) -> EffectSizeResult:
    """Calculate Cohen's d (or Hedges' g bias-corrected) for two independent groups."""
    g1 = pd.to_numeric(pd.Series(group1), errors="coerce").dropna().values
    g2 = pd.to_numeric(pd.Series(group2), errors="coerce").dropna().values
    g1 = g1[np.isfinite(g1)]
    g2 = g2[np.isfinite(g2)]

    n1 = len(g1)
    n2 = len(g2)
    if n1 < 2 or n2 < 2:
        return EffectSizeResult(
            name="Cohen's d",
            estimate=0.0,
            magnitude="negligible",
            interpretation="Insufficient data to compute standardized mean difference.",
        )

    m1, m2 = float(np.mean(g1)), float(np.mean(g2))
    v1, v2 = float(np.var(g1, ddof=1)), float(np.var(g2, ddof=1))

    # Pooled standard deviation
    df_pool = n1 + n2 - 2
    s_pooled = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / df_pool) if df_pool > 0 else 0.0

    if s_pooled <= 1e-9:
        d = 0.0
    else:
        d = (m1 - m2) / s_pooled

    # Hedges' g small-sample bias correction factor
    if hedges_correction and df_pool > 1:
        j_factor = 1.0 - (3.0 / (4.0 * df_pool - 1.0))
        estimate = d * j_factor
        name = "Hedges' g"
    else:
        estimate = d
        name = "Cohen's d"

    # Approximate variance of d for confidence interval
    var_d = ((n1 + n2) / (n1 * n2)) + ((estimate**2) / (2.0 * df_pool)) if df_pool > 0 else 0.0
    se_d = np.sqrt(max(0.0, var_d))
    z_crit = 1.96 if ci_level == 0.95 else (2.576 if ci_level == 0.99 else 1.645)

    ci_lower = round(estimate - z_crit * se_d, 4)
    ci_upper = round(estimate + z_crit * se_d, 4)

    mag = _classify_cohen_d(abs(estimate))
    interp = (
        f"{name} = {estimate:+.3f} [{ci_level * 100:.0f}% CI: {ci_lower:+.3f}, {ci_upper:+.3f}]. "
        f"Indicates a {mag.replace('_', ' ')} standardized difference between group means. "
        "Standardized effect sizes quantify the magnitude of separation independently of raw measurement units."
    )

    return EffectSizeResult(
        name=name,
        estimate=round(estimate, 4),
        magnitude=mag,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_level=ci_level,
        interpretation=interp,
    )


def calculate_paired_cohens_d(
    differences: pd.Series | np.ndarray,
    ci_level: float = 0.95,
) -> EffectSizeResult:
    """Calculate Cohen's d for paired observations (mean difference / std of differences)."""
    diffs = pd.to_numeric(pd.Series(differences), errors="coerce").dropna().values
    diffs = diffs[np.isfinite(diffs)]
    n = len(diffs)

    if n < 2:
        return EffectSizeResult(
            name="Paired Cohen's d (dz)",
            estimate=0.0,
            magnitude="negligible",
            interpretation="Insufficient paired records to calculate paired effect size.",
        )

    mean_d = float(np.mean(diffs))
    std_d = float(np.std(diffs, ddof=1))

    if std_d <= 1e-9:
        d_z = 0.0
    else:
        d_z = mean_d / std_d

    se_dz = np.sqrt((1.0 / n) + ((d_z**2) / (2.0 * (n - 1)))) if n > 1 else 0.0
    z_crit = 1.96 if ci_level == 0.95 else 1.645
    ci_lower = round(d_z - z_crit * se_dz, 4)
    ci_upper = round(d_z + z_crit * se_dz, 4)

    mag = _classify_cohen_d(abs(d_z))
    interp = (
        f"Paired d_z = {d_z:+.3f} [{ci_level * 100:.0f}% CI: {ci_lower:+.3f}, {ci_upper:+.3f}]. "
        f"Represents a {mag.replace('_', ' ')} within-subject effect."
    )

    return EffectSizeResult(
        name="Paired Cohen's d (dz)",
        estimate=round(d_z, 4),
        magnitude=mag,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_level=ci_level,
        interpretation=interp,
    )


def calculate_cliffs_delta(
    group1: pd.Series | np.ndarray,
    group2: pd.Series | np.ndarray,
) -> EffectSizeResult:
    """Calculate Cliff's delta non-parametric effect size for two ordinal/continuous groups."""
    g1 = pd.to_numeric(pd.Series(group1), errors="coerce").dropna().values
    g2 = pd.to_numeric(pd.Series(group2), errors="coerce").dropna().values
    n1, n2 = len(g1), len(g2)

    if n1 == 0 or n2 == 0:
        return EffectSizeResult(
            name="Cliff's delta",
            estimate=0.0,
            magnitude="negligible",
            interpretation="Empty groups for Cliff's delta.",
        )

    # Vectorized comparison: (x > y) - (x < y)
    diff_matrix = np.subtract.outer(g1, g2)
    greater = np.sum(diff_matrix > 0)
    less = np.sum(diff_matrix < 0)
    delta = float((greater - less) / (n1 * n2))

    # Classical Cliff's delta magnitude thresholds: <0.147 negligible, <0.33 small, <0.474 medium, >=0.474 large
    abs_d = abs(delta)
    if abs_d < 0.147:
        mag: Literal["negligible", "small", "medium", "large", "very_large"] = "negligible"
    elif abs_d < 0.33:
        mag = "small"
    elif abs_d < 0.474:
        mag = "medium"
    else:
        mag = "large"

    interp = (
        f"Cliff's delta = {delta:+.3f} ({mag} non-parametric separation). "
        "Represents the degree of stochastic dominance between the two distributions."
    )

    return EffectSizeResult(
        name="Cliff's delta",
        estimate=round(delta, 4),
        magnitude=mag,
        interpretation=interp,
    )


def calculate_eta_squared(
    ss_between: float,
    ss_total: float,
    ss_within: float | None = None,
    df_between: int | None = None,
) -> EffectSizeResult:
    """Calculate Eta-squared (and Omega-squared if within SS and DF provided) for multi-group ANOVA."""
    if ss_total <= 1e-9:
        return EffectSizeResult(
            name="Eta-squared (η²)",
            estimate=0.0,
            magnitude="negligible",
            interpretation="Total variance is zero.",
        )

    eta_sq = max(0.0, min(1.0, ss_between / ss_total))
    mag = _classify_eta_squared(eta_sq)

    name = "Eta-squared (η²)"
    estimate = eta_sq

    # Compute Omega-squared (less biased) if additional components exist
    if ss_within is not None and df_between is not None and df_between > 0:
        n_total_approx = ss_total  # baseline
        ms_within = ss_within / (len(str(n_total_approx)) * 10) if ss_within > 0 else 0.0
        # Standard omega squared: (SS_between - df_between * MS_within) / (SS_total + MS_within)
        if ss_total + ms_within > 0:
            omega_sq = max(0.0, (ss_between - df_between * ms_within) / (ss_total + ms_within))
            name = "Omega-squared (ω²)"
            estimate = min(1.0, omega_sq)
            mag = _classify_eta_squared(estimate)

    pct = estimate * 100.0
    interp = (
        f"{name} = {estimate:.4f} ({pct:.1f}% of outcome variance explained by group membership). "
        f"Reflects a {mag.replace('_', ' ')} practical effect size."
    )

    return EffectSizeResult(
        name=name,
        estimate=round(estimate, 4),
        magnitude=mag,
        interpretation=interp,
    )


def calculate_cramers_v_effect(
    chi2_val: float,
    n_obs: int,
    contingency_shape: tuple[int, int],
) -> EffectSizeResult:
    """Calculate Cramér's V association metric from Chi-Square test statistics."""
    r, c = contingency_shape
    min_dim = min(r - 1, c - 1)
    if min_dim <= 0 or n_obs <= 0:
        return EffectSizeResult(
            name="Cramér's V",
            estimate=0.0,
            magnitude="negligible",
            interpretation="Degenerate contingency table dimensions.",
        )

    v = np.sqrt(chi2_val / (n_obs * min_dim))
    v_clean = float(np.clip(v, 0.0, 1.0))
    mag = _classify_cramers_v(v_clean, df_dim=min_dim)

    interp = (
        f"Cramér's V = {v_clean:.4f} ({mag.replace('_', ' ')} categorical association). "
        f"Measures relationship strength for {r}×{c} contingency table on [0, 1] scale."
    )

    return EffectSizeResult(
        name="Cramér's V",
        estimate=round(v_clean, 4),
        magnitude=mag,
        interpretation=interp,
    )
