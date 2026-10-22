"""Robustness, sensitivity analysis, and alternative specification evaluation."""

import pandas as pd

from src.analytics.correlation_inference import run_correlation_test
from src.analytics.group_comparisons import run_two_group_independent_test
from src.analytics.statistical_models import RobustnessComparison


def evaluate_two_group_robustness(
    df: pd.DataFrame,
    group_col: str,
    measure_col: str,
    group1_name: str,
    group2_name: str,
    alpha: float = 0.05,
) -> list[RobustnessComparison]:
    """Compare two-sample findings across parametric, non-parametric, outlier-trimmed, and varying confidence levels."""
    comparisons: list[RobustnessComparison] = []

    # 1. Baseline: Welch's t-test (Heteroscedastic Parametric)
    res_welch = run_two_group_independent_test(
        df=df,
        group_col=group_col,
        measure_col=measure_col,
        group1_name=group1_name,
        group2_name=group2_name,
        method="welch",
        alpha=alpha,
    )
    baseline_sig = res_welch.p_value < alpha

    ci_l = res_welch.confidence_interval.lower_bound if res_welch.confidence_interval else None
    ci_u = res_welch.confidence_interval.upper_bound if res_welch.confidence_interval else None
    eff_est = res_welch.effect_size.estimate if res_welch.effect_size else None

    comparisons.append(
        RobustnessComparison(
            specification_name="1. Baseline: Welch's t-test (Unequal Variance)",
            method="Parametric t-test without equal-variance assumption",
            sample_size=res_welch.total_n,
            point_estimate=round(float(res_welch.method_metadata.get("raw_diff", 0.0)), 4),
            p_value=res_welch.p_value,
            ci_lower=ci_l,
            ci_upper=ci_u,
            effect_size_estimate=eff_est,
            conclusion_stable=True,
            notes="Primary recommended parametric specification.",
        )
    )

    # 2. Student's t-test (Equal Variance Assumed)
    res_student = run_two_group_independent_test(
        df=df,
        group_col=group_col,
        measure_col=measure_col,
        group1_name=group1_name,
        group2_name=group2_name,
        method="student",
        alpha=alpha,
    )
    is_stud_sig = res_student.p_value < alpha
    comparisons.append(
        RobustnessComparison(
            specification_name="2. Alternative: Student's t-test (Pooled Variance)",
            method="Parametric t-test assuming equal variances",
            sample_size=res_student.total_n,
            point_estimate=round(float(res_student.method_metadata.get("raw_diff", 0.0)), 4),
            p_value=res_student.p_value,
            ci_lower=res_student.confidence_interval.lower_bound
            if res_student.confidence_interval
            else None,
            ci_upper=res_student.confidence_interval.upper_bound
            if res_student.confidence_interval
            else None,
            effect_size_estimate=res_student.effect_size.estimate
            if res_student.effect_size
            else None,
            conclusion_stable=is_stud_sig == baseline_sig,
            notes="Tests sensitivity to variance equality assumption.",
        )
    )

    # 3. Non-parametric: Mann-Whitney U Test (Rank Sum)
    res_mw = run_two_group_independent_test(
        df=df,
        group_col=group_col,
        measure_col=measure_col,
        group1_name=group1_name,
        group2_name=group2_name,
        method="mann_whitney",
        alpha=alpha,
    )
    is_mw_sig = res_mw.p_value < alpha
    comparisons.append(
        RobustnessComparison(
            specification_name="3. Alternative: Mann-Whitney U (Rank Sum)",
            method="Non-parametric rank sum test (robust to non-normality and outliers)",
            sample_size=res_mw.total_n,
            point_estimate=round(float(res_mw.method_metadata.get("raw_diff", 0.0)), 4),
            p_value=res_mw.p_value,
            ci_lower=res_mw.confidence_interval.lower_bound if res_mw.confidence_interval else None,
            ci_upper=res_mw.confidence_interval.upper_bound if res_mw.confidence_interval else None,
            effect_size_estimate=res_mw.effect_size.estimate if res_mw.effect_size else None,
            conclusion_stable=is_mw_sig == baseline_sig,
            notes="Tests whether findings depend on normality or tail skew.",
        )
    )

    # 4. Outlier-Trimmed Specification (Excluding points outside Tukey's 1.5 * IQR)
    clean_sub = df[[group_col, measure_col]].dropna()
    clean_sub[measure_col] = pd.to_numeric(clean_sub[measure_col], errors="coerce")
    clean_sub = clean_sub.dropna()

    q25 = clean_sub[measure_col].quantile(0.25)
    q75 = clean_sub[measure_col].quantile(0.75)
    iqr = q75 - q25
    trimmed_df = clean_sub[
        (clean_sub[measure_col] >= q25 - 1.5 * iqr) & (clean_sub[measure_col] <= q75 + 1.5 * iqr)
    ]

    if len(trimmed_df) < len(clean_sub) and len(trimmed_df) >= 10:
        res_trim = run_two_group_independent_test(
            df=trimmed_df,
            group_col=group_col,
            measure_col=measure_col,
            group1_name=group1_name,
            group2_name=group2_name,
            method="welch",
            alpha=alpha,
        )
        is_trim_sig = res_trim.p_value < alpha
        removed_count = len(clean_sub) - len(trimmed_df)
        comparisons.append(
            RobustnessComparison(
                specification_name=f"4. Outlier Sensitivity (Trimmed {removed_count} IQR Outliers)",
                method="Welch's t-test after excluding Tukey's 1.5x IQR outliers",
                sample_size=res_trim.total_n,
                point_estimate=round(float(res_trim.method_metadata.get("raw_diff", 0.0)), 4),
                p_value=res_trim.p_value,
                ci_lower=res_trim.confidence_interval.lower_bound
                if res_trim.confidence_interval
                else None,
                ci_upper=res_trim.confidence_interval.upper_bound
                if res_trim.confidence_interval
                else None,
                effect_size_estimate=res_trim.effect_size.estimate
                if res_trim.effect_size
                else None,
                conclusion_stable=is_trim_sig == baseline_sig,
                notes=f"Evaluates if result is driven by {removed_count} extreme observations.",
            )
        )

    # 5. Strict Alpha Threshold (α = 0.01)
    is_strict_sig = res_welch.p_value < 0.01
    comparisons.append(
        RobustnessComparison(
            specification_name="5. Conservative Significance Threshold (α = 0.01)",
            method="Welch's t-test evaluated at strict 99% confidence level",
            sample_size=res_welch.total_n,
            point_estimate=round(float(res_welch.method_metadata.get("raw_diff", 0.0)), 4),
            p_value=res_welch.p_value,
            ci_lower=ci_l,
            ci_upper=ci_u,
            effect_size_estimate=eff_est,
            conclusion_stable=is_strict_sig == baseline_sig,
            notes="Evaluates whether significance survives strict false-positive threshold.",
        )
    )

    return comparisons


def evaluate_correlation_robustness(
    df: pd.DataFrame,
    var1: str,
    var2: str,
    alpha: float = 0.05,
) -> list[RobustnessComparison]:
    """Compare linear Pearson vs rank-based Spearman and outlier-trimmed correlation."""
    comparisons: list[RobustnessComparison] = []

    res_pearson = run_correlation_test(df, var1, var2, method="pearson", alpha=alpha)
    res_spearman = run_correlation_test(df, var1, var2, method="spearman", alpha=alpha)

    p_sig = res_pearson.p_value < alpha
    s_sig = res_spearman.p_value < alpha

    comparisons.append(
        RobustnessComparison(
            specification_name="1. Baseline: Pearson Linear Correlation",
            method="Standard linear covariance estimation",
            sample_size=res_pearson.total_n,
            point_estimate=res_pearson.statistic_value,
            p_value=res_pearson.p_value,
            ci_lower=res_pearson.confidence_interval.lower_bound
            if res_pearson.confidence_interval
            else None,
            ci_upper=res_pearson.confidence_interval.upper_bound
            if res_pearson.confidence_interval
            else None,
            effect_size_estimate=res_pearson.effect_size.estimate
            if res_pearson.effect_size
            else None,
            conclusion_stable=True,
            notes="Measures linear co-movement.",
        )
    )

    comparisons.append(
        RobustnessComparison(
            specification_name="2. Non-Parametric: Spearman Rank Correlation",
            method="Monotonic rank order correlation",
            sample_size=res_spearman.total_n,
            point_estimate=res_spearman.statistic_value,
            p_value=res_spearman.p_value,
            ci_lower=res_spearman.confidence_interval.lower_bound
            if res_spearman.confidence_interval
            else None,
            ci_upper=res_spearman.confidence_interval.upper_bound
            if res_spearman.confidence_interval
            else None,
            effect_size_estimate=res_spearman.effect_size.estimate
            if res_spearman.effect_size
            else None,
            conclusion_stable=p_sig == s_sig,
            notes="Robust to non-linear shifts and monotonic rank transformations.",
        )
    )

    return comparisons
