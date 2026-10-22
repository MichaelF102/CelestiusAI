"""Ordinary Least Squares (OLS) regression modeling, inference, and multi-predictor diagnostics."""

import uuid

import numpy as np
import pandas as pd
from scipy import stats

from src.analytics.regression_diagnostics import (
    calculate_durbin_watson,
    calculate_vif,
    check_breusch_pagan_heteroscedasticity,
    check_residual_normality_jarque_bera,
    identify_influential_points_cooks_d,
)
from src.analytics.statistical_models import (
    AssumptionCheckResult,
    RegressionCoefficient,
    RegressionModelResult,
)


def fit_ols_regression(
    df: pd.DataFrame,
    outcome_col: str,
    predictor_cols: list[str],
    include_intercept: bool = True,
    confidence_level: float = 0.95,
    alpha: float = 0.05,
) -> RegressionModelResult:
    """Fit Ordinary Least Squares linear regression model with full inference and diagnostic checks."""
    needed_cols = [outcome_col, *predictor_cols]
    clean_df = df[needed_cols].dropna()

    if clean_df.empty:
        raise ValueError("No valid records remaining after dropping missing values.")

    # Convert outcome to numeric
    y = pd.to_numeric(clean_df[outcome_col], errors="coerce").values
    valid_mask = np.isfinite(y)
    clean_df = clean_df[valid_mask]
    y = y[valid_mask]

    n_obs = len(y)
    if n_obs < len(predictor_cols) + 2:
        raise ValueError(
            f"Sample size (N = {n_obs}) is insufficient for {len(predictor_cols)} predictors (minimum {len(predictor_cols) + 2} required)."
        )

    # Encode predictors (one-hot encode categoricals, keep numerics)
    encoded_dfs = []
    final_predictor_names = []

    for col in predictor_cols:
        col_series = clean_df[col]
        # Check if categorical or string
        if col_series.dtype == "object" or col_series.dtype.name in ("category", "string"):
            dummies = pd.get_dummies(col_series, prefix=col, drop_first=True, dtype=float)
            for d_col in dummies.columns:
                final_predictor_names.append(str(d_col))
            encoded_dfs.append(dummies)
        else:
            num_vals = pd.to_numeric(col_series, errors="coerce").fillna(0.0)
            final_predictor_names.append(col)
            encoded_dfs.append(pd.DataFrame({col: num_vals}))

    X_frame = pd.concat(encoded_dfs, axis=1)
    X_mat = X_frame.values.astype(float)

    if include_intercept:
        X_design = np.column_stack([np.ones(n_obs), X_mat])
        all_var_names = ["Intercept", *final_predictor_names]
    else:
        X_design = X_mat
        all_var_names = final_predictor_names

    p_params = X_design.shape[1]
    df_resid = max(1.0, float(n_obs - p_params))
    df_model = float(p_params - 1) if include_intercept else float(p_params)

    # Least squares estimation via SVD / pseudo-inverse for numerical stability
    beta, residuals_sum, rank, s = np.linalg.lstsq(X_design, y, rcond=None)
    y_pred = X_design @ beta
    residuals = y - y_pred

    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot <= 1e-9:
        r_squared = 1.0 if ss_res <= 1e-9 else 0.0
    else:
        r_squared = max(0.0, 1.0 - (ss_res / ss_tot))

    adj_r_squared = (
        1.0 - ((1.0 - r_squared) * (n_obs - 1) / df_resid) if df_resid > 0 else r_squared
    )
    adj_r_squared = max(0.0, adj_r_squared)

    # Residual variance and covariance of coefficients
    sigma2 = ss_res / df_resid if df_resid > 0 else 0.0
    try:
        inv_xtx = np.linalg.pinv(X_design.T @ X_design)
        cov_beta = sigma2 * inv_xtx
        se_beta = np.sqrt(np.diag(cov_beta))
    except Exception:
        se_beta = np.zeros(p_params)

    # F-statistic for overall model
    if df_model > 0 and sigma2 > 1e-9:
        ms_model = (ss_tot - ss_res) / df_model
        f_stat = float(ms_model / sigma2)
        f_pval = float(1.0 - stats.f.cdf(f_stat, dfn=df_model, dfd=df_resid))
    else:
        f_stat = 0.0
        f_pval = 1.0

    # Information criteria (AIC / BIC)
    ll = -0.5 * n_obs * (np.log(2.0 * np.pi) + np.log(max(1e-9, ss_res / n_obs)) + 1.0)
    aic = 2.0 * p_params - 2.0 * ll
    bic = np.log(n_obs) * p_params - 2.0 * ll

    # Compute VIF on raw predictor frame
    vif_dict = calculate_vif(X_frame)

    # Critical t for confidence interval
    t_crit = float(stats.t.ppf(1.0 - (1.0 - confidence_level) / 2.0, df=df_resid))

    # Coefficient objects
    coeff_list: list[RegressionCoefficient] = []
    for i, var_name in enumerate(all_var_names):
        b_val = float(beta[i])
        se_val = float(se_beta[i])
        t_stat = (b_val / se_val) if se_val > 1e-9 else 0.0
        p_val = float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=df_resid))) if df_resid > 0 else 1.0

        ci_l = b_val - t_crit * se_val
        ci_u = b_val + t_crit * se_val
        vif_val = vif_dict.get(var_name)

        coeff_list.append(
            RegressionCoefficient(
                variable=var_name,
                estimate=round(b_val, 4),
                std_error=round(se_val, 4),
                t_statistic=round(t_stat, 3),
                p_value=round(p_val, 6),
                ci_lower=round(ci_l, 4),
                ci_upper=round(ci_u, 4),
                vif=vif_val,
            )
        )

    # Diagnostics
    diag_bp = check_breusch_pagan_heteroscedasticity(residuals, X_design, alpha=alpha)
    diag_jb = check_residual_normality_jarque_bera(residuals, alpha=alpha)
    dw_stat = calculate_durbin_watson(residuals)
    _, influential_idx = identify_influential_points_cooks_d(residuals, X_design)

    # Multi-collinearity check
    high_vifs = [k for k, v in vif_dict.items() if v > 5.0]
    diag_vif = AssumptionCheckResult(
        name="Multicollinearity (VIF)",
        statistic_name="Max VIF",
        statistic_value=float(max(vif_dict.values())) if vif_dict else 1.0,
        p_value=None,
        is_satisfied=len(high_vifs) == 0,
        severity="warning" if high_vifs else "info",
        message=(
            f"VIF > 5 detected in {len(high_vifs)} predictor(s): {', '.join(high_vifs)}."
            if high_vifs
            else "All predictor VIF values are below 5.0 (minimal multicollinearity)."
        ),
        recommendation="Consider dropping redundant collinear features or using regularized regression."
        if high_vifs
        else "Predictors provide distinct variance.",
    )

    all_diags = [diag_bp, diag_jb, diag_vif]

    f_disp = "< 0.001" if f_pval < 0.001 else f"= {f_pval:.4f}"
    is_f_sig = f_pval < alpha

    interp_layers = {
        "Research Question": f"How well do predictors [{', '.join(predictor_cols)}] account for variation in '{outcome_col}'?",
        "Method & Rationale": (
            "Ordinary Least Squares (OLS) linear regression estimates the conditional mean of the outcome "
            "as a linear function of selected predictors."
        ),
        "Quantitative Result": (
            f"R² = {r_squared:.3f} (Adjusted R² = {adj_r_squared:.3f}). "
            f"Overall Model F({df_model:.0f}, {df_resid:.0f}) = {f_stat:,.2f}, p {f_disp} (N = {n_obs:,}). "
            f"Durbin-Watson = {dw_stat:.2f}."
        ),
        "Statistical Evidence": (
            f"The overall regression model is {'statistically significant' if is_f_sig else 'not statistically significant'} (p {f_disp}). "
            f"Predictors jointly account for {r_squared * 100:.1f}% of total variance in '{outcome_col}'."
        ),
        "Effect Size & Practical Importance": (
            f"Model explains {r_squared * 100:.1f}% of outcome variation. "
            f"Statistically significant individual predictors (p < {alpha}): "
            f"{', '.join([c.variable for c in coeff_list if c.p_value < alpha and c.variable != 'Intercept']) or 'None'}."
        ),
        "Assumptions & Limitations": (
            f"Heteroscedasticity: {diag_bp.message} Residual Normality: {diag_jb.message} "
            f"Multicollinearity: {diag_vif.message} "
            f"Flagged {len(influential_idx)} potentially influential observations exceeding Cook's D (4/N) threshold. "
            "CRITICAL: Regression coefficients represent conditional observational associations under this specific model specification, "
            "NOT causal impacts. High R² does not prove causal mechanisms or guarantee out-of-sample predictive validity."
        ),
        "Conclusion": (
            f"The regression model explains a statistically significant proportion of variance in '{outcome_col}'."
            if is_f_sig
            else f"The selected predictors do not explain statistically significant variance in '{outcome_col}' at α = {alpha}."
        ),
        "Next Steps": (
            "Inspect residual-versus-fitted diagnostic plots to evaluate linearity, check for non-linear terms or interaction effects, "
            "and conduct robustness checks with robust standard errors (HC3)."
        ),
    }

    summary_text = (
        f"OLS Regression: R² = {r_squared:.3f} (Adj R² = {adj_r_squared:.3f}), "
        f"F({df_model:.0f}, {df_resid:.0f}) = {f_stat:.2f}, p {f_disp} (N = {n_obs:,})."
    )

    return RegressionModelResult(
        analysis_id=f"reg_{uuid.uuid4().hex[:6]}",
        outcome_variable=outcome_col,
        predictor_variables=predictor_cols,
        n_observations=n_obs,
        df_model=df_model,
        df_residuals=df_resid,
        r_squared=round(r_squared, 4),
        adj_r_squared=round(adj_r_squared, 4),
        f_statistic=round(f_stat, 3),
        f_pvalue=round(f_pval, 6),
        aic=round(float(aic), 2),
        bic=round(float(bic), 2),
        coefficients=coeff_list,
        diagnostics=all_diags,
        summary=summary_text,
        interpretation_layers=interp_layers,
        limitations=[
            "Coefficients are conditional associations, NOT causal effects.",
            "Ordinary standard errors assume homoscedastic, independent errors.",
            f"{len(influential_idx)} observations exceed Cook's distance threshold (4/N).",
        ],
    )
