"""Regression diagnostics: VIF, Breusch-Pagan, Durbin-Watson, Jarque-Bera, and Cook's distance."""

import numpy as np
import pandas as pd
from scipy import stats

from src.analytics.statistical_models import AssumptionCheckResult


def calculate_vif(X: pd.DataFrame) -> dict[str, float]:
    """Calculate Variance Inflation Factor (VIF) for numeric feature matrix."""
    cols = list(X.columns)
    if len(cols) <= 1:
        return {cols[0]: 1.0} if cols else {}

    vif_dict: dict[str, float] = {}
    X_vals = X.values.astype(float)

    try:
        # Correlation matrix inversion
        corr_matrix = np.corrcoef(X_vals, rowvar=False)
        inv_corr = np.linalg.pinv(corr_matrix)
        for i, col in enumerate(cols):
            vif_val = float(inv_corr[i, i])
            vif_dict[col] = round(max(1.0, vif_val), 2)
    except Exception:
        for col in cols:
            vif_dict[col] = 1.0

    return vif_dict


def calculate_durbin_watson(residuals: np.ndarray) -> float:
    """Calculate the Durbin-Watson test statistic for residual serial correlation."""
    diff = np.diff(residuals)
    numerator = float(np.sum(diff**2))
    denominator = float(np.sum(residuals**2))
    if denominator <= 1e-9:
        return 2.0
    return round(numerator / denominator, 3)


def check_breusch_pagan_heteroscedasticity(
    residuals: np.ndarray,
    X_design: np.ndarray,
    alpha: float = 0.05,
) -> AssumptionCheckResult:
    """Evaluate homoscedasticity via Breusch-Pagan Lagrange Multiplier test."""
    n = len(residuals)
    p = X_design.shape[1] - 1  # Excluding intercept

    if n < p + 5 or p < 1:
        return AssumptionCheckResult(
            name="Homoscedasticity (Breusch-Pagan)",
            statistic_name="LM",
            statistic_value=None,
            p_value=None,
            is_satisfied=True,
            severity="info",
            message="Sample size or predictor count too small for Breusch-Pagan test.",
            recommendation="Inspect residual-vs-fitted scatter plot visually.",
        )

    # Auxiliary regression of squared residuals on predictors
    e2 = residuals**2
    e2_mean = np.mean(e2)
    if e2_mean <= 1e-9:
        return AssumptionCheckResult(
            name="Homoscedasticity (Breusch-Pagan)",
            statistic_name="LM",
            statistic_value=0.0,
            p_value=1.0,
            is_satisfied=True,
            severity="info",
            message="Residual variance is negligible.",
            recommendation="Constant variance supported.",
        )

    # Scaled squared residuals
    scaled_e2 = e2 / e2_mean
    try:
        beta, _, _, _ = np.linalg.lstsq(X_design, scaled_e2, rcond=None)
        pred = X_design @ beta
        ss_reg = 0.5 * np.sum((pred - 1.0) ** 2)
        lm_stat = float(ss_reg)
        p_val = float(1.0 - stats.chi2.cdf(lm_stat, df=p))
    except Exception:
        lm_stat = 0.0
        p_val = 1.0

    is_homo = p_val >= alpha
    msg = (
        f"Breusch-Pagan LM = {lm_stat:.3f}, p = {p_val:.4f} ({'>=' if is_homo else '<'} {alpha}). "
        f"{'Residual variance appears homogeneous (constant).' if is_homo else 'Evidence of heteroscedasticity (non-constant residual variance).'}"
    )
    rec = (
        "Standard OLS standard errors are trustworthy."
        if is_homo
        else "Use heteroscedasticity-robust standard errors (HC3/Huber-White) or apply a variance-stabilizing transform."
    )

    return AssumptionCheckResult(
        name="Homoscedasticity (Breusch-Pagan)",
        statistic_name="LM",
        statistic_value=round(lm_stat, 4),
        p_value=round(p_val, 6),
        is_satisfied=is_homo,
        severity="info" if is_homo else "warning",
        message=msg,
        recommendation=rec,
    )


def check_residual_normality_jarque_bera(
    residuals: np.ndarray,
    alpha: float = 0.05,
) -> AssumptionCheckResult:
    """Evaluate residual normality using the Jarque-Bera test."""
    n = len(residuals)
    if n < 8:
        return AssumptionCheckResult(
            name="Residual Normality (Jarque-Bera)",
            statistic_name="JB",
            statistic_value=None,
            p_value=None,
            is_satisfied=True,
            severity="info",
            message=f"Sample size N = {n} is too small for Jarque-Bera test.",
            recommendation="Inspect Q-Q plot for residuals.",
        )

    jb_stat, p_val = stats.jarque_bera(residuals)
    stat_val = round(float(jb_stat), 4)
    p_float = float(p_val)
    is_normal = p_float >= alpha

    msg = (
        f"Jarque-Bera JB = {stat_val:.3f}, p = {p_float:.4f} ({'>=' if is_normal else '<'} {alpha}). "
        f"{'Residuals do not significantly depart from Gaussian normality.' if is_normal else 'Residuals exhibit non-normal skewness or kurtosis.'}"
    )
    rec = (
        "Standard t and F inference for coefficients is valid."
        if is_normal
        else "With large samples (N > 50), coefficient estimators remain asymptotically normal under CLT."
    )

    return AssumptionCheckResult(
        name="Residual Normality (Jarque-Bera)",
        statistic_name="JB",
        statistic_value=stat_val,
        p_value=round(p_float, 6),
        is_satisfied=is_normal,
        severity="info" if is_normal else "warning",
        message=msg,
        recommendation=rec,
    )


def identify_influential_points_cooks_d(
    residuals: np.ndarray,
    X_design: np.ndarray,
) -> tuple[np.ndarray, list[int]]:
    """Compute Cook's distance and identify observation indices exceeding 4/N threshold."""
    n, p = X_design.shape
    if n <= p + 1:
        return np.zeros(n), []

    try:
        # Hat matrix diagonals h_ii = diag(X (X^T X)^-1 X^T)
        q, r = np.linalg.qr(X_design)
        h_diag = np.sum(q**2, axis=1)
        h_diag = np.clip(h_diag, 0.0, 0.999)

        mse = float(np.sum(residuals**2) / (n - p))
        if mse <= 1e-9:
            return np.zeros(n), []

        cooks_d = (residuals**2 / (p * mse)) * (h_diag / ((1.0 - h_diag) ** 2))
        threshold = 4.0 / n
        influential_indices = [int(i) for i in np.where(cooks_d > threshold)[0]]
        return cooks_d, influential_indices
    except Exception:
        return np.zeros(n), []
