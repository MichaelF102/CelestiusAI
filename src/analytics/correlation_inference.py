"""Correlation inference with exact p-values, Fisher z-transform CIs, and non-causal interpretations."""

import uuid

import numpy as np
import pandas as pd
from scipy import stats

from src.analytics.confidence_intervals import (
    calculate_bootstrap_ci,
    calculate_correlation_ci,
)
from src.analytics.statistical_models import (
    CorrelationMethod,
    EffectSizeResult,
    StatisticalTestResult,
)


def _classify_correlation_magnitude(r_abs: float) -> str:
    if r_abs < 0.1:
        return "negligible"
    if r_abs < 0.3:
        return "small"
    if r_abs < 0.5:
        return "medium"
    if r_abs < 0.7:
        return "large"
    return "very_large"


def run_correlation_test(
    df: pd.DataFrame,
    var1: str,
    var2: str,
    method: CorrelationMethod = "pearson",
    confidence_level: float = 0.95,
    alpha: float = 0.05,
) -> StatisticalTestResult:
    """Execute formal correlation inference with exact p-values and confidence intervals."""
    clean = df[[var1, var2]].dropna()
    clean = clean.apply(pd.to_numeric, errors="coerce").dropna()
    clean = clean[np.isfinite(clean[var1]) & np.isfinite(clean[var2])]

    n_total = len(clean)
    method_name = method.capitalize()

    if n_total < 5 or clean[var1].std() == 0 or clean[var2].std() == 0:
        return StatisticalTestResult(
            analysis_id=f"corr_{uuid.uuid4().hex[:6]}",
            test_name=f"{method_name} Correlation Test",
            research_question=f"Is there a linear/monotonic association between '{var1}' and '{var2}'?",
            null_hypothesis=f"Population correlation between '{var1}' and '{var2}' is zero (ρ = 0).",
            alternative_hypothesis=f"Population correlation between '{var1}' and '{var2}' is non-zero (ρ ≠ 0).",
            selected_variables=[var1, var2],
            total_n=n_total,
            statistic_name=f"{method_name} r",
            statistic_value=0.0,
            degrees_of_freedom=max(0, n_total - 2),
            p_value=1.0,
            p_value_display="p = 1.000",
            summary="Insufficient records (N < 5) or constant variance to evaluate correlation.",
            practical_significance="Cannot be evaluated.",
            limitations=["Zero variance or fewer than 5 complete pairwise observations."],
        )

    s1 = clean[var1].values
    s2 = clean[var2].values

    if method == "spearman":
        corr_res = stats.spearmanr(s1, s2)
        r_val = float(corr_res.statistic)  # type: ignore[attr-defined]
        p_val = float(corr_res.pvalue)  # type: ignore[attr-defined]
        stat_name = "Spearman's rho (ρ)"
        # Use bootstrap for Spearman CI
        ci_res = calculate_bootstrap_ci(
            clean.index.values,
            statistic_func=lambda idx: stats.spearmanr(s1[idx], s2[idx]).statistic,  # type: ignore[attr-defined]
            estimand_name="Spearman rho",
            confidence_level=confidence_level,
        )
    else:
        corr_res = stats.pearsonr(s1, s2)
        r_val = float(corr_res.statistic)
        p_val = float(corr_res.pvalue)
        stat_name = "Pearson's r"
        ci_res = calculate_correlation_ci(r_val, n_total, confidence_level=confidence_level)

    df_val = n_total - 2
    p_float = float(p_val)
    p_disp = "< 0.001" if p_float < 0.001 else f"= {p_float:.4f}"
    is_significant = p_float < alpha

    r_abs = abs(r_val)
    mag = _classify_correlation_magnitude(r_abs)
    direction = "positive" if r_val > 0.02 else ("negative" if r_val < -0.02 else "zero")

    effect_size = EffectSizeResult(
        name=stat_name,
        estimate=round(r_val, 4),
        magnitude=mag,  # type: ignore[arg-type]
        ci_lower=ci_res.lower_bound,
        ci_upper=ci_res.upper_bound,
        ci_level=confidence_level,
        interpretation=f"Observed {direction} association with coefficient {r_val:+.3f} ({mag}).",
    )

    interp_layers = {
        "Research Question": f"Is '{var1}' statistically associated with '{var2}'?",
        "Method & Rationale": (
            f"{method_name} correlation tests whether bivariate co-movement deviates from random fluctuation. "
            f"Pearson assumes linear relationships; Spearman tests monotonic rank order."
        ),
        "Quantitative Result": (
            f"{stat_name} = {r_val:+.3f} [{confidence_level * 100:.0f}% CI: {ci_res.lower_bound:+.3f}, {ci_res.upper_bound:+.3f}], "
            f"t({df_val}) p {p_disp} (N = {n_total:,})."
        ),
        "Statistical Evidence": (
            f"With p {p_disp} ({'<' if is_significant else '>='} {alpha}), there is "
            f"{'statistically significant evidence of an association' if is_significant else 'insufficient evidence to reject the zero-correlation null hypothesis'}."
        ),
        "Effect Size & Practical Importance": (
            f"The association explains approximately {r_val**2 * 100:.1f}% of pairwise co-variance (r² = {r_val**2:.3f}). "
            f"This represents a {mag.replace('_', ' ')} magnitude effect."
        ),
        "Assumptions & Limitations": (
            "CRITICAL: Correlation does not establish or imply causation. The association may reflect omitted variable bias, "
            "simultaneity, or confounding by shared external drivers. Extreme outliers can heavily leverage Pearson coefficients."
        ),
        "Conclusion": (
            f"A statistically detectable {direction} association is observed between '{var1}' and '{var2}'."
            if is_significant
            else f"No statistically significant linear relationship detected between '{var1}' and '{var2}' at α = {alpha}."
        ),
        "Next Steps": (
            "Fit a multivariate regression model controlling for potential confounding factors, and inspect scatter plots for non-linear patterns."
        ),
    }

    summary_text = (
        f"{stat_name} = {r_val:+.3f} (p {p_disp}, N = {n_total:,}) [{confidence_level * 100:.0f}% CI: {ci_res.lower_bound:+.3f}, {ci_res.upper_bound:+.3f}]. "
        f"{'Statistically significant' if is_significant else 'Not statistically significant'} ({mag})."
    )

    return StatisticalTestResult(
        analysis_id=f"corr_{uuid.uuid4().hex[:6]}",
        test_name=f"{method_name} Correlation Test",
        research_question=f"Is there a linear/monotonic association between '{var1}' and '{var2}'?",
        null_hypothesis=f"Population correlation between '{var1}' and '{var2}' is zero (ρ = 0).",
        alternative_hypothesis=f"Population correlation between '{var1}' and '{var2}' is non-zero (ρ ≠ 0).",
        selected_variables=[var1, var2],
        total_n=n_total,
        missing_policy="drop_pairwise",
        excluded_count=len(df) - n_total,
        statistic_name=stat_name,
        statistic_value=round(r_val, 4),
        degrees_of_freedom=df_val,
        p_value=p_float,
        p_value_display=p_disp,
        confidence_interval=ci_res,
        effect_size=effect_size,
        summary=summary_text,
        interpretation_layers=interp_layers,
        practical_significance=f"{mag.capitalize()} {direction} association (r² = {r_val**2:.3f}).",
        limitations=[
            "Correlation does NOT imply causation.",
            "Vulnerable to lurking confounders and outlier leverage.",
        ],
        recommended_followups=[
            "Estimate partial correlation controlling for third variables.",
            "Inspect scatter plot for non-linear relationships.",
        ],
    )
