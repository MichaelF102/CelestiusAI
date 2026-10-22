"""Categorical association testing (Pearson Chi-Square, Fisher's Exact, Standardized Residuals)."""

import uuid

import numpy as np
import pandas as pd
from scipy import stats

from src.analytics.effect_sizes import calculate_cramers_v_effect
from src.analytics.statistical_models import AssumptionCheckResult, StatisticalTestResult


def run_chi_square_test(
    df: pd.DataFrame,
    var1: str,
    var2: str,
    alpha: float = 0.05,
    use_fisher_if_2x2_sparse: bool = True,
) -> StatisticalTestResult:
    """Perform Pearson Chi-Square test of independence (or Fisher's exact test for 2x2 sparse tables)."""
    clean = df[[var1, var2]].dropna()
    clean = clean.astype(str)
    n_total = len(clean)

    if n_total < 5 or clean[var1].nunique() < 2 or clean[var2].nunique() < 2:
        return StatisticalTestResult(
            analysis_id=f"cat_{uuid.uuid4().hex[:6]}",
            test_name="Pearson Chi-Square Test of Independence",
            research_question=f"Are '{var1}' and '{var2}' statistically associated?",
            null_hypothesis=f"Variables '{var1}' and '{var2}' are independent.",
            alternative_hypothesis=f"Variables '{var1}' and '{var2}' are dependent (associated).",
            selected_variables=[var1, var2],
            total_n=n_total,
            statistic_name="Chi-Square (χ²)",
            statistic_value=0.0,
            degrees_of_freedom=0,
            p_value=1.0,
            p_value_display="p = 1.000",
            summary="Insufficient discrete categories or observations to conduct Chi-Square test.",
            practical_significance="Cannot be assessed.",
            limitations=["Fewer than 2 distinct categories or insufficient observations."],
        )

    contingency = pd.crosstab(clean[var1], clean[var2])
    r_dim, c_dim = contingency.shape
    df_val = (r_dim - 1) * (c_dim - 1)

    chi2_stat, p_val, _, expected = stats.chi2_contingency(contingency)
    chi2_val = round(float(chi2_stat), 4)
    p_float = float(p_val)

    # Check Cochran's criteria: expected frequencies >= 5 in at least 80% of cells, and all >= 1
    exp_under_5 = np.sum(expected < 5)
    exp_under_1 = np.sum(expected < 1)
    total_cells = expected.size
    pct_under_5 = (exp_under_5 / total_cells) * 100.0 if total_cells > 0 else 0.0

    is_sparse = pct_under_5 > 20.0 or exp_under_1 > 0
    assumptions: list[AssumptionCheckResult] = []

    if is_sparse:
        assumptions.append(
            AssumptionCheckResult(
                name="Cochran's Expected Frequency Rule",
                statistic_name="Cells with E < 5",
                statistic_value=float(exp_under_5),
                p_value=None,
                is_satisfied=False,
                severity="warning",
                message=(
                    f"{exp_under_5} of {total_cells} cells ({pct_under_5:.1f}%) have expected counts < 5. "
                    "Asymptotic Chi-Square distribution may overstate significance."
                ),
                recommendation="Combine rare categories or inspect exact probabilities.",
            )
        )
    else:
        assumptions.append(
            AssumptionCheckResult(
                name="Cochran's Expected Frequency Rule",
                statistic_name="Cells with E < 5",
                statistic_value=float(exp_under_5),
                p_value=None,
                is_satisfied=True,
                severity="info",
                message=f"All cells have adequate expected frequencies ({pct_under_5:.1f}% < 5).",
                recommendation="Asymptotic Chi-Square inference is mathematically valid.",
            )
        )

    # If 2x2 table and sparse, use Fisher's exact test
    test_used = "Pearson Chi-Square Test of Independence"
    stat_name = "Chi-Square (χ²)"
    disp_stat = chi2_val

    if r_dim == 2 and c_dim == 2 and use_fisher_if_2x2_sparse and is_sparse:
        try:
            odds_ratio, p_fisher = stats.fisher_exact(contingency)
            test_used = "Fisher's Exact Test (2x2)"
            stat_name = "Odds Ratio"
            disp_stat = round(float(odds_ratio), 4)
            p_float = float(p_fisher)
        except Exception:
            pass

    # Standardized residuals: (Observed - Expected) / sqrt(Expected * (1 - row_prob) * (1 - col_prob))
    row_sums = contingency.sum(axis=1).values
    col_sums = contingency.sum(axis=0).values
    total_sum = n_total

    row_probs = (row_sums / total_sum).reshape(-1, 1)
    col_probs = (col_sums / total_sum).reshape(1, -1)
    variance_matrix = expected * (1.0 - row_probs) * (1.0 - col_probs)
    variance_matrix[variance_matrix <= 1e-9] = 1e-9

    std_residuals = (contingency.values - expected) / np.sqrt(variance_matrix)

    # Find dominant driver cell (|std_res| > 2 indicates significant departure)
    max_res_idx = np.unravel_index(np.argmax(np.abs(std_residuals)), std_residuals.shape)
    driver_row = contingency.index[max_res_idx[0]]
    driver_col = contingency.columns[max_res_idx[1]]
    driver_res = std_residuals[max_res_idx]

    # Effect size: Cramér's V
    effect_size = calculate_cramers_v_effect(chi2_val, n_total, (r_dim, c_dim))

    # P-value display
    p_disp = "< 0.001" if p_float < 0.001 else f"= {p_float:.4f}"
    is_significant = p_float < alpha

    # Interpretation layers
    interp_layers = {
        "Research Question": f"Is there a statistically significant association between '{var1}' and '{var2}'?",
        "Method & Rationale": (
            f"{test_used} evaluates whether the observed joint category frequencies deviate from "
            "independent co-occurrence under random chance."
        ),
        "Quantitative Result": (
            f"{stat_name} = {disp_stat:,.3f}, df = {df_val}, p {p_disp} (N = {n_total:,}). "
            f"Effect size: {effect_size.name} = {effect_size.estimate:.3f} ({effect_size.magnitude})."
        ),
        "Statistical Evidence": (
            f"With p {p_disp} ({'<' if is_significant else '>='} {alpha}), the sample data "
            f"{'rejects' if is_significant else 'fails to reject'} the null hypothesis of independence."
        ),
        "Effect Size & Practical Importance": (
            f"{effect_size.interpretation} "
            f"The pairing with the strongest deviation from independence is '{driver_row}' × '{driver_col}' "
            f"(standardized residual = {driver_res:+.2f})."
        ),
        "Assumptions & Limitations": (
            "Assumes independent sampling. Categorical association strictly measures co-occurrence and does NOT "
            "establish causality or the directional influence of one category on another."
        ),
        "Conclusion": (
            f"The data provides evidence of an association between '{var1}' and '{var2}' under the stated assumptions."
            if is_significant
            else f"No statistically significant association between '{var1}' and '{var2}' was detected at α = {alpha}."
        ),
        "Next Steps": (
            "Examine the standardized residual matrix to understand which subgroup pairs drive the association, "
            "or perform stratified logistic regression controlling for key confounders."
        ),
    }

    summary_text = (
        f"{test_used}: {stat_name} = {disp_stat:.2f}, df = {df_val}, p {p_disp}. "
        f"{'Statistically associated' if is_significant else 'No significant association detected'} "
        f"({effect_size.name} = {effect_size.estimate:.2f}, {effect_size.magnitude})."
    )

    return StatisticalTestResult(
        analysis_id=f"cat_{uuid.uuid4().hex[:6]}",
        test_name=test_used,
        research_question=f"Are '{var1}' and '{var2}' statistically associated?",
        null_hypothesis=f"Variables '{var1}' and '{var2}' are independent.",
        alternative_hypothesis=f"Variables '{var1}' and '{var2}' are dependent (associated).",
        selected_variables=[var1, var2],
        total_n=n_total,
        missing_policy="drop_pairwise",
        excluded_count=len(df) - n_total,
        statistic_name=stat_name,
        statistic_value=disp_stat,
        degrees_of_freedom=df_val,
        p_value=p_float,
        p_value_display=p_disp,
        effect_size=effect_size,
        assumption_checks=assumptions,
        summary=summary_text,
        interpretation_layers=interp_layers,
        practical_significance=f"{effect_size.magnitude.capitalize()} association ({effect_size.name} = {effect_size.estimate:.2f}).",
        limitations=[
            "Association does not demonstrate causal influence.",
            f"Cochran's cell rule: {pct_under_5:.1f}% cells with expected count < 5.",
        ],
        recommended_followups=[
            f"Inspect standardized residuals for '{driver_row}' × '{driver_col}'.",
            "Stratify by secondary demographic or operational dimensions.",
        ],
        method_metadata={
            "contingency_table": contingency.to_dict(),
            "expected_counts": expected.tolist(),
            "standardized_residuals": std_residuals.tolist(),
            "driver_cell": {
                "row": str(driver_row),
                "col": str(driver_col),
                "residual": round(float(driver_res), 3),
            },
        },
    )
