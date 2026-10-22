"""Hypothesis testing for two-group and multi-group continuous comparisons."""

import uuid
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

from src.analytics.assumption_checks import (
    check_homoscedasticity_levene,
    check_normality_shapiro,
)
from src.analytics.confidence_intervals import (
    calculate_difference_in_means_ci,
    calculate_paired_difference_ci,
)
from src.analytics.effect_sizes import (
    calculate_cliffs_delta,
    calculate_cohens_d,
    calculate_eta_squared,
    calculate_paired_cohens_d,
)
from src.analytics.statistical_models import (
    HypothesisDirection,
    StatisticalTestResult,
)


def run_two_group_independent_test(
    df: pd.DataFrame,
    group_col: str,
    measure_col: str,
    group1_name: str | None = None,
    group2_name: str | None = None,
    method: Literal["welch", "student", "mann_whitney"] = "welch",
    direction: HypothesisDirection = "two-sided",
    confidence_level: float = 0.95,
    alpha: float = 0.05,
) -> StatisticalTestResult:
    """Perform formal two-sample independent comparison (Welch's t, Student's t, or Mann-Whitney U)."""
    clean = df[[group_col, measure_col]].dropna()
    clean[measure_col] = pd.to_numeric(clean[measure_col], errors="coerce")
    clean = clean.dropna()

    available_groups = list(clean[group_col].astype(str).unique())
    if len(available_groups) < 2:
        return StatisticalTestResult(
            analysis_id=f"grp2_{uuid.uuid4().hex[:6]}",
            test_name="Two-Sample Comparison",
            research_question=f"Does '{measure_col}' differ between groups of '{group_col}'?",
            null_hypothesis="Group means/distributions are identical.",
            alternative_hypothesis="Group means/distributions differ.",
            selected_variables=[group_col, measure_col],
            total_n=len(clean),
            statistic_name="Statistic",
            statistic_value=0.0,
            p_value=1.0,
            p_value_display="p = 1.000",
            summary="Fewer than 2 distinct groups found in data.",
            practical_significance="Cannot be evaluated.",
            limitations=["Insufficient group categories."],
        )

    g1_label = (
        group1_name if group1_name and group1_name in available_groups else available_groups[0]
    )
    g2_label = (
        group2_name if group2_name and group2_name in available_groups else available_groups[1]
    )

    s1 = clean[clean[group_col].astype(str) == g1_label][measure_col].values
    s2 = clean[clean[group_col].astype(str) == g2_label][measure_col].values

    n1, n2 = len(s1), len(s2)
    n_total = n1 + n2

    if n1 < 2 or n2 < 2:
        return StatisticalTestResult(
            analysis_id=f"grp2_{uuid.uuid4().hex[:6]}",
            test_name="Two-Sample Comparison",
            research_question=f"Does '{measure_col}' differ between '{g1_label}' and '{g2_label}'?",
            null_hypothesis=f"Means of '{g1_label}' and '{g2_label}' are identical.",
            alternative_hypothesis=f"Means of '{g1_label}' and '{g2_label}' differ.",
            selected_variables=[group_col, measure_col],
            groups=[g1_label, g2_label],
            sample_sizes={g1_label: n1, g2_label: n2},
            total_n=n_total,
            statistic_name="Statistic",
            statistic_value=0.0,
            p_value=1.0,
            p_value_display="p = 1.000",
            summary=f"Insufficient observations (N1={n1}, N2={n2}). Minimum 2 per group required.",
            practical_significance="Cannot be evaluated.",
        )

    m1, m2 = float(np.mean(s1)), float(np.mean(s2))
    med1, med2 = float(np.median(s1)), float(np.median(s2))
    std1, std2 = float(np.std(s1, ddof=1)), float(np.std(s2, ddof=1))
    diff_means = m1 - m2
    rel_diff_pct = ((diff_means / abs(m2)) * 100.0) if abs(m2) > 1e-6 else None

    # Assumption checks
    norm1 = check_normality_shapiro(pd.Series(s1), alpha=alpha)
    norm2 = check_normality_shapiro(pd.Series(s2), alpha=alpha)
    homo = check_homoscedasticity_levene([s1, s2], [g1_label, g2_label], alpha=alpha)
    assumptions = [norm1, norm2, homo]

    # Map direction to SciPy alternative
    scipy_alt = "two-sided"
    if direction == "greater":
        scipy_alt = "greater"
    elif direction == "less":
        scipy_alt = "less"

    if method == "mann_whitney":
        test_used = "Mann-Whitney U Test (Rank Sum)"
        mw_res = stats.mannwhitneyu(s1, s2, alternative=scipy_alt)
        stat_val = round(float(mw_res.statistic), 4)
        p_val = float(mw_res.pvalue)
        stat_name = "Mann-Whitney U"
        df_display = None
        ci_res = calculate_difference_in_means_ci(
            s1, s2, confidence_level=confidence_level, equal_var=False
        )
        eff_res = calculate_cliffs_delta(s1, s2)

    elif method == "student":
        test_used = "Student's Independent-Samples t-test"
        tt_res = stats.ttest_ind(s1, s2, equal_var=True, alternative=scipy_alt)
        stat_val = round(float(tt_res.statistic), 4)
        p_val = float(tt_res.pvalue)
        stat_name = "Student's t"
        df_display = float(n1 + n2 - 2)
        ci_res = calculate_difference_in_means_ci(
            s1, s2, confidence_level=confidence_level, equal_var=True
        )
        eff_res = calculate_cohens_d(s1, s2, hedges_correction=True, ci_level=confidence_level)

    else:  # Welch (recommended default)
        test_used = "Welch's Heteroscedastic t-test"
        tt_res = stats.ttest_ind(s1, s2, equal_var=False, alternative=scipy_alt)
        stat_val = round(float(tt_res.statistic), 4)
        p_val = float(tt_res.pvalue)
        stat_name = "Welch's t"
        # Degrees of freedom from Welch formula
        se1 = (std1**2) / n1
        se2 = (std2**2) / n2
        df_welch = ((se1 + se2) ** 2) / ((se1**2 / (n1 - 1)) + (se2**2 / (n2 - 1)))
        df_display = round(float(df_welch), 2)
        ci_res = calculate_difference_in_means_ci(
            s1, s2, confidence_level=confidence_level, equal_var=False
        )
        eff_res = calculate_cohens_d(s1, s2, hedges_correction=True, ci_level=confidence_level)

    p_disp = "< 0.001" if p_val < 0.001 else f"= {p_val:.4f}"
    is_significant = p_val < alpha

    rel_str = f" ({rel_diff_pct:+.1f}%)" if rel_diff_pct is not None else ""
    interp_layers = {
        "Research Question": f"Does '{measure_col}' differ systematically between '{g1_label}' and '{g2_label}'?",
        "Method & Rationale": (
            f"{test_used} was selected to evaluate differences without assuming equal population variances. "
            f"Direction evaluated: {direction}."
        ),
        "Quantitative Result": (
            f"Mean difference = {diff_means:,.3f}{rel_str} "
            f"[{confidence_level * 100:.0f}% CI: {ci_res.lower_bound:,.3f}, {ci_res.upper_bound:,.3f}]. "
            f"Group 1 ('{g1_label}'): Mean = {m1:,.2f}, Median = {med1:,.2f}, N = {n1:,}. "
            f"Group 2 ('{g2_label}'): Mean = {m2:,.2f}, Median = {med2:,.2f}, N = {n2:,}. "
            f"{stat_name} = {stat_val:,.3f}, p {p_disp}."
        ),
        "Statistical Evidence": (
            f"With p {p_disp} ({'<' if is_significant else '>='} {alpha}), the sample evidence "
            f"{'is sufficient to reject the null hypothesis of equal central values' if is_significant else 'is insufficient to reject the null hypothesis'}."
        ),
        "Effect Size & Practical Importance": (
            f"Standardized magnitude: {eff_res.interpretation} "
            "Statistical detectability does not dictate practical business relevance; assess whether the absolute "
            f"spread of {abs(diff_means):,.2f} justifies specialized operational treatment."
        ),
        "Assumptions & Limitations": (
            f"Variance homogeneity check: {homo.message} "
            "Assumes independent sampling. Extreme outliers or skewed sub-distributions may impact arithmetic means."
        ),
        "Conclusion": (
            f"The data provides evidence of a difference in '{measure_col}' between '{g1_label}' and '{g2_label}' under the stated assumptions."
            if is_significant
            else f"No statistically significant difference in '{measure_col}' between '{g1_label}' and '{g2_label}' was detected at α = {alpha}."
        ),
        "Next Steps": (
            "Inspect boxplots to verify whether difference is driven by a small subgroup, or conduct a segmented multivariate regression."
        ),
    }

    summary_text = (
        f"{test_used}: {stat_name} = {stat_val:.3f}, p {p_disp}. "
        f"Mean diff = {diff_means:+.2f}{rel_str} [{confidence_level * 100:.0f}% CI: {ci_res.lower_bound:+.2f}, {ci_res.upper_bound:+.2f}]. "
        f"{eff_res.name} = {eff_res.estimate:+.2f} ({eff_res.magnitude})."
    )

    return StatisticalTestResult(
        analysis_id=f"grp2_{uuid.uuid4().hex[:6]}",
        test_name=test_used,
        research_question=f"Does '{measure_col}' differ between '{g1_label}' and '{g2_label}'?",
        null_hypothesis=f"Population central values of '{g1_label}' and '{g2_label}' are identical.",
        alternative_hypothesis=f"Population central values of '{g1_label}' and '{g2_label}' differ ({direction}).",
        selected_variables=[group_col, measure_col],
        groups=[g1_label, g2_label],
        sample_sizes={g1_label: n1, g2_label: n2},
        total_n=n_total,
        missing_policy="drop_pairwise",
        excluded_count=len(df) - n_total,
        statistic_name=stat_name,
        statistic_value=stat_val,
        degrees_of_freedom=df_display,
        p_value=round(p_val, 6),
        p_value_display=p_disp,
        confidence_interval=ci_res,
        effect_size=eff_res,
        assumption_checks=assumptions,
        summary=summary_text,
        interpretation_layers=interp_layers,
        practical_significance=f"Mean difference of {diff_means:+.2f} ({eff_res.magnitude} effect).",
        limitations=[
            "Independent observation assumption.",
            f"{homo.message}",
        ],
        recommended_followups=[
            "Compare with rank-based Mann-Whitney U for sensitivity.",
            "Inspect distribution tails via boxplots.",
        ],
        method_metadata={
            "mean_g1": m1,
            "mean_g2": m2,
            "median_g1": med1,
            "median_g2": med2,
            "std_g1": std1,
            "std_g2": std2,
            "raw_diff": diff_means,
            "relative_diff_pct": rel_diff_pct,
        },
    )


def run_paired_group_test(
    df: pd.DataFrame,
    pre_col: str,
    post_col: str,
    method: Literal["paired_t", "wilcoxon"] = "paired_t",
    confidence_level: float = 0.95,
    alpha: float = 0.05,
) -> StatisticalTestResult:
    """Perform formal paired comparison between two aligned measurements."""
    clean = df[[pre_col, post_col]].dropna()
    clean[pre_col] = pd.to_numeric(clean[pre_col], errors="coerce")
    clean[post_col] = pd.to_numeric(clean[post_col], errors="coerce")
    clean = clean.dropna()

    n = len(clean)
    if n < 3:
        return StatisticalTestResult(
            analysis_id=f"pair_{uuid.uuid4().hex[:6]}",
            test_name="Paired Comparison",
            research_question=f"Does measurement differ between '{pre_col}' and '{post_col}'?",
            null_hypothesis="Mean/median of paired differences is zero.",
            alternative_hypothesis="Mean/median of paired differences is non-zero.",
            selected_variables=[pre_col, post_col],
            total_n=n,
            statistic_name="Statistic",
            statistic_value=0.0,
            p_value=1.0,
            p_value_display="p = 1.000",
            summary="Insufficient complete pairs (N < 3).",
            practical_significance="Cannot be evaluated.",
        )

    diffs = clean[post_col].values - clean[pre_col].values
    mean_diff = float(np.mean(diffs))
    norm_diff = check_normality_shapiro(pd.Series(diffs), alpha=alpha)

    if method == "wilcoxon":
        test_used = "Wilcoxon Signed-Rank Test (Paired)"
        res = stats.wilcoxon(clean[post_col].values, clean[pre_col].values)
        stat_val = round(float(res.statistic), 4)
        p_val = float(res.pvalue)
        stat_name = "Wilcoxon W"
        df_display = None
        eff_res = calculate_paired_cohens_d(diffs, ci_level=confidence_level)
    else:
        test_used = "Paired Student's t-test"
        res = stats.ttest_rel(clean[post_col].values, clean[pre_col].values)
        stat_val = round(float(res.statistic), 4)
        p_val = float(res.pvalue)
        stat_name = "Paired t"
        df_display = float(n - 1)
        eff_res = calculate_paired_cohens_d(diffs, ci_level=confidence_level)

    ci_res = calculate_paired_difference_ci(diffs, confidence_level=confidence_level)
    p_disp = "< 0.001" if p_val < 0.001 else f"= {p_val:.4f}"
    is_significant = p_val < alpha

    interp_layers = {
        "Research Question": f"Do paired records change between '{pre_col}' and '{post_col}'?",
        "Method & Rationale": f"{test_used} evaluates within-subject change accounting for paired covariance.",
        "Quantitative Result": (
            f"Mean paired change = {mean_diff:+.3f} [{confidence_level * 100:.0f}% CI: {ci_res.lower_bound:+.3f}, {ci_res.upper_bound:+.3f}]. "
            f"{stat_name} = {stat_val:,.3f}, p {p_disp} (N = {n:,} pairs)."
        ),
        "Statistical Evidence": (
            f"With p {p_disp} ({'<' if is_significant else '>='} {alpha}), the within-pair shift is "
            f"{'statistically significant' if is_significant else 'not statistically distinguishable from zero'}."
        ),
        "Effect Size & Practical Importance": eff_res.interpretation,
        "Assumptions & Limitations": f"Normality of differences: {norm_diff.message}",
        "Conclusion": (
            f"Significant within-subject shift detected in '{post_col}' versus '{pre_col}'."
            if is_significant
            else f"No statistically significant paired shift detected at α = {alpha}."
        ),
        "Next Steps": "Inspect difference distribution to confirm shift is consistent across all subjects.",
    }

    summary_text = (
        f"{test_used}: Mean change = {mean_diff:+.2f} [{confidence_level * 100:.0f}% CI: {ci_res.lower_bound:+.2f}, {ci_res.upper_bound:+.2f}], "
        f"p {p_disp} (N = {n:,} pairs)."
    )

    return StatisticalTestResult(
        analysis_id=f"pair_{uuid.uuid4().hex[:6]}",
        test_name=test_used,
        research_question=f"Does measurement change between '{pre_col}' and '{post_col}'?",
        null_hypothesis="Mean paired difference is zero (μ_D = 0).",
        alternative_hypothesis="Mean paired difference is non-zero (μ_D ≠ 0).",
        selected_variables=[pre_col, post_col],
        total_n=n,
        statistic_name=stat_name,
        statistic_value=stat_val,
        degrees_of_freedom=df_display,
        p_value=round(p_val, 6),
        p_value_display=p_disp,
        confidence_interval=ci_res,
        effect_size=eff_res,
        assumption_checks=[norm_diff],
        summary=summary_text,
        interpretation_layers=interp_layers,
        practical_significance=f"Mean paired difference: {mean_diff:+.2f} ({eff_res.magnitude}).",
        limitations=["Assumes pairs are correctly aligned.", "Repeated measurement dependency."],
    )


def run_multi_group_anova(
    df: pd.DataFrame,
    group_col: str,
    measure_col: str,
    alpha: float = 0.05,
) -> StatisticalTestResult:
    """Perform One-Way ANOVA across three or more independent groups."""
    clean = df[[group_col, measure_col]].dropna()
    clean[measure_col] = pd.to_numeric(clean[measure_col], errors="coerce")
    clean = clean.dropna()

    groups_data = {}
    for g, sub_df in clean.groupby(group_col):
        vals = sub_df[measure_col].values
        if len(vals) >= 2:
            groups_data[str(g)] = vals

    k = len(groups_data)
    total_n = sum(len(v) for v in groups_data.values())

    if k < 2 or total_n < 5:
        return StatisticalTestResult(
            analysis_id=f"aov_{uuid.uuid4().hex[:6]}",
            test_name="One-Way ANOVA",
            research_question=f"Does '{measure_col}' vary across '{group_col}' segments?",
            null_hypothesis="All group population means are equal.",
            alternative_hypothesis="At least one group population mean differs.",
            selected_variables=[group_col, measure_col],
            total_n=total_n,
            statistic_name="F-statistic",
            statistic_value=0.0,
            p_value=1.0,
            p_value_display="p = 1.000",
            summary="Insufficient groups (k < 2) with N >= 2.",
            practical_significance="Cannot be evaluated.",
        )

    group_arrays = list(groups_data.values())
    f_stat, p_val = stats.f_oneway(*group_arrays)
    f_val = round(float(f_stat), 4)
    p_float = float(p_val)
    p_disp = "< 0.001" if p_float < 0.001 else f"= {p_float:.4f}"
    is_significant = p_float < alpha

    df_between = k - 1
    df_within = total_n - k

    # Compute SS between and total for Eta-squared
    grand_mean = clean[measure_col].mean()
    ss_between = sum(len(v) * ((np.mean(v) - grand_mean) ** 2) for v in group_arrays)
    ss_total = float(np.sum((clean[measure_col].values - grand_mean) ** 2))
    ss_within = ss_total - ss_between

    eff_size = calculate_eta_squared(
        ss_between=ss_between,
        ss_total=ss_total,
        ss_within=ss_within,
        df_between=df_between,
    )

    # Assumption checks
    homo = check_homoscedasticity_levene(group_arrays, list(groups_data.keys()), alpha=alpha)

    eff_res_desc = f"{eff_size.interpretation} The grouping explains {eff_size.estimate * 100:.1f}% of total metric variance."
    interp_layers = {
        "Research Question": f"Does '{measure_col}' differ significantly across the {k} groups of '{group_col}'?",
        "Method & Rationale": "One-Way ANOVA evaluates the omnibus hypothesis of equal means across multiple segments.",
        "Quantitative Result": (
            f"F({df_between}, {df_within}) = {f_val:,.3f}, p {p_disp} (k = {k} groups, Total N = {total_n:,}). "
            f"Effect size: {eff_size.name} = {eff_size.estimate:.4f} ({eff_size.magnitude})."
        ),
        "Statistical Evidence": (
            f"With p {p_disp} ({'<' if is_significant else '>='} {alpha}), the omnibus test "
            f"{'rejects the hypothesis of identical group means' if is_significant else 'fails to detect significant between-group divergence'}."
        ),
        "Effect Size & Practical Importance": eff_res_desc,
        "Assumptions & Limitations": f"Variance homogeneity check: {homo.message}",
        "Conclusion": (
            f"Statistically significant between-group differences exist across '{group_col}' segments."
            if is_significant
            else f"No statistically significant between-group divergence detected across '{group_col}' at α = {alpha}."
        ),
        "Next Steps": (
            "Run post-hoc pairwise comparisons with multiplicity adjustment (e.g. Tukey HSD / FDR) to isolate which specific pairs differ."
            if is_significant
            else "Group differences appear compatible with random sampling variation."
        ),
    }

    summary_text = (
        f"One-Way ANOVA: F({df_between}, {df_within}) = {f_val:.2f}, p {p_disp}. "
        f"{'Statistically significant difference between groups' if is_significant else 'No significant divergence'} "
        f"({eff_size.name} = {eff_size.estimate:.3f}, {eff_size.magnitude})."
    )

    return StatisticalTestResult(
        analysis_id=f"aov_{uuid.uuid4().hex[:6]}",
        test_name="One-Way ANOVA (Omnibus)",
        research_question=f"Does '{measure_col}' differ across the {k} groups of '{group_col}'?",
        null_hypothesis=f"All {k} group population means are equal.",
        alternative_hypothesis="At least one group population mean differs.",
        selected_variables=[group_col, measure_col],
        groups=list(groups_data.keys()),
        sample_sizes={g: len(v) for g, v in groups_data.items()},
        total_n=total_n,
        statistic_name="F-statistic",
        statistic_value=f_val,
        degrees_of_freedom=[float(df_between), float(df_within)],
        p_value=round(p_float, 6),
        p_value_display=p_disp,
        effect_size=eff_size,
        assumption_checks=[homo],
        summary=summary_text,
        interpretation_layers=interp_layers,
        practical_significance=eff_res_desc,
        limitations=[
            "Omnibus test does not reveal WHICH specific pairs differ.",
            f"{homo.message}",
        ],
        recommended_followups=[
            "Conduct post-hoc comparisons with FDR / Bonferroni adjustment.",
            "Inspect group-level distribution box plots.",
        ],
    )
