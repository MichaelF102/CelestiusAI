"""Unit and integration tests for Prompt 4 Statistical Intelligence Engine."""

import numpy as np
import pandas as pd
import pytest
from src.analytics.categorical_tests import run_chi_square_test
from src.analytics.confidence_intervals import (
    calculate_bootstrap_ci,
    calculate_mean_ci,
    calculate_proportion_ci,
)
from src.analytics.correlation_inference import run_correlation_test
from src.analytics.group_comparisons import (
    run_multi_group_anova,
    run_paired_group_test,
    run_two_group_independent_test,
)
from src.analytics.multiple_testing import (
    adjust_p_values,
    apply_multiple_testing_correction,
)
from src.analytics.regression_diagnostics import (
    calculate_durbin_watson,
    calculate_vif,
)
from src.analytics.regression_inference import fit_ols_regression
from src.analytics.robustness import (
    evaluate_correlation_robustness,
    evaluate_two_group_robustness,
)
from src.analytics.statistical_models import (
    RegressionModelResult,
    StatisticalTestResult,
)
from src.analytics.statistical_service import StatisticalService


@pytest.fixture
def synthetic_stat_df() -> pd.DataFrame:
    """Create a synthetic dataset with known statistical relationships."""
    np.random.seed(42)
    n = 100
    # Group A: Mean = 100, Std = 15
    g1_vals = np.random.normal(loc=100.0, scale=15.0, size=n // 2)
    # Group B: Mean = 120, Std = 25 (Unequal variances, Cohen's d ~ 1.0)
    g2_vals = np.random.normal(loc=120.0, scale=25.0, size=n // 2)

    group_labels = ["Control"] * (n // 2) + ["Treatment"] * (n // 2)
    scores = np.concatenate([g1_vals, g2_vals])

    # Paired pre and post scores with known positive shift (+10)
    pre = np.random.normal(50.0, 10.0, size=n)
    post = pre + 10.0 + np.random.normal(0.0, 3.0, size=n)

    # 3-group ANOVA factor
    multi_groups = ["Segment_1"] * 30 + ["Segment_2"] * 30 + ["Segment_3"] * 40
    multi_scores = np.concatenate(
        [
            np.random.normal(20.0, 5.0, size=30),
            np.random.normal(40.0, 5.0, size=30),
            np.random.normal(60.0, 5.0, size=40),
        ]
    )

    # Correlated variables (r ~ 0.85)
    var_x = np.linspace(10, 100, n)
    var_y = 2.0 * var_x + np.random.normal(0, 15, size=n)

    # Categorical contingency data
    cat_a = ["High", "Low"] * (n // 2)
    cat_b = ["Yes", "No", "Yes", "Yes"] * (n // 4)

    return pd.DataFrame(
        {
            "group": group_labels,
            "score": scores,
            "pre_score": pre,
            "post_score": post,
            "multi_segment": multi_groups,
            "multi_metric": multi_scores,
            "var_x": var_x,
            "var_y": var_y,
            "cat_a": cat_a,
            "cat_b": cat_b,
        }
    )


# -----------------------------------------------------------------------------
# 1. GROUP COMPARISONS & EFFECT SIZES
# -----------------------------------------------------------------------------
def test_two_group_welch_t_test(synthetic_stat_df):
    res = run_two_group_independent_test(
        df=synthetic_stat_df,
        group_col="group",
        measure_col="score",
        group1_name="Control",
        group2_name="Treatment",
        method="welch",
    )
    assert isinstance(res, StatisticalTestResult)
    assert res.test_name == "Welch's Heteroscedastic t-test"
    assert res.p_value < 0.05
    assert res.effect_size is not None
    assert res.effect_size.magnitude in ("medium", "large", "very_large")
    assert res.confidence_interval is not None
    assert res.confidence_interval.lower_bound < res.confidence_interval.upper_bound


def test_two_group_mann_whitney_u(synthetic_stat_df):
    res = run_two_group_independent_test(
        df=synthetic_stat_df,
        group_col="group",
        measure_col="score",
        method="mann_whitney",
    )
    assert res.statistic_name == "Mann-Whitney U"
    assert res.p_value < 0.05
    assert res.effect_size.name == "Cliff's delta"


def test_two_group_insufficient_data():
    df_small = pd.DataFrame({"grp": ["A"], "val": [10.0]})
    res = run_two_group_independent_test(df_small, "grp", "val")
    assert res.p_value == 1.0


def test_paired_group_test(synthetic_stat_df):
    res = run_paired_group_test(
        df=synthetic_stat_df,
        pre_col="pre_score",
        post_col="post_score",
        method="paired_t",
    )
    assert res.test_name == "Paired Student's t-test"
    assert res.p_value < 0.001
    assert res.effect_size.estimate > 0.8
    assert res.confidence_interval.point_estimate > 8.0


def test_multi_group_anova(synthetic_stat_df):
    res = run_multi_group_anova(
        df=synthetic_stat_df,
        group_col="multi_segment",
        measure_col="multi_metric",
    )
    assert res.test_name == "One-Way ANOVA (Omnibus)"
    assert res.p_value < 0.001
    assert res.statistic_value > 10.0
    assert res.effect_size.name in ("Eta-squared (η²)", "Omega-squared (ω²)")
    assert res.effect_size.estimate > 0.5


# -----------------------------------------------------------------------------
# 2. CATEGORICAL INFERENCE TESTS
# -----------------------------------------------------------------------------
def test_chi_square_test(synthetic_stat_df):
    res = run_chi_square_test(
        df=synthetic_stat_df,
        var1="cat_a",
        var2="cat_b",
    )
    assert isinstance(res, StatisticalTestResult)
    assert "Chi-Square" in res.test_name or "Fisher" in res.test_name
    assert res.degrees_of_freedom is not None
    assert res.effect_size.name == "Cramér's V"


def test_chi_square_degenerate():
    df_deg = pd.DataFrame({"v1": ["A", "A", "A"], "v2": ["B", "B", "B"]})
    res = run_chi_square_test(df_deg, "v1", "v2")
    assert res.p_value == 1.0


# -----------------------------------------------------------------------------
# 3. CORRELATION INFERENCE TESTS
# -----------------------------------------------------------------------------
def test_correlation_inference_pearson(synthetic_stat_df):
    res = run_correlation_test(
        df=synthetic_stat_df,
        var1="var_x",
        var2="var_y",
        method="pearson",
    )
    assert res.p_value < 0.001
    assert res.statistic_value > 0.80
    assert res.confidence_interval is not None
    assert res.confidence_interval.lower_bound > 0.70


def test_correlation_inference_spearman(synthetic_stat_df):
    res = run_correlation_test(
        df=synthetic_stat_df,
        var1="var_x",
        var2="var_y",
        method="spearman",
    )
    assert res.p_value < 0.001
    assert res.statistic_value > 0.80


# -----------------------------------------------------------------------------
# 4. CONFIDENCE INTERVALS TESTS
# -----------------------------------------------------------------------------
def test_mean_ci():
    data = [10.0, 12.0, 14.0, 16.0, 18.0]
    ci = calculate_mean_ci(data, confidence_level=0.95)
    assert ci.point_estimate == 14.0
    assert ci.lower_bound < 14.0 < ci.upper_bound


def test_proportion_ci_wilson():
    ci = calculate_proportion_ci(successes=80, trials=100, confidence_level=0.95)
    assert 0.70 < ci.point_estimate == 0.80
    assert 0.70 < ci.lower_bound < 0.80 < ci.upper_bound < 0.90


def test_bootstrap_ci_reproducible():
    data = np.arange(10, 50)
    ci1 = calculate_bootstrap_ci(data, random_seed=42)
    ci2 = calculate_bootstrap_ci(data, random_seed=42)
    assert ci1.point_estimate == ci2.point_estimate
    assert ci1.lower_bound == ci2.lower_bound
    assert ci1.upper_bound == ci2.upper_bound


# -----------------------------------------------------------------------------
# 5. MULTIPLE-TESTING CORRECTION TESTS
# -----------------------------------------------------------------------------
def test_adjust_p_values_fdr_and_bonferroni():
    raw_p = [0.01, 0.04, 0.05, 0.20, 0.50]
    # Bonferroni: p * m
    bonf = adjust_p_values(raw_p, method="bonferroni")
    assert bonf[0] == 0.05
    assert bonf[1] == 0.20
    assert bonf[-1] == 1.0

    # FDR BH
    fdr = adjust_p_values(raw_p, method="fdr_bh")
    assert fdr[0] <= fdr[1] <= fdr[2]
    assert all(f <= b for f, b in zip(fdr, bonf, strict=True))


def test_apply_multiple_testing_correction(synthetic_stat_df):
    res1 = run_two_group_independent_test(synthetic_stat_df, "group", "score")
    res2 = run_correlation_test(synthetic_stat_df, "var_x", "var_y")
    adjusted_family = apply_multiple_testing_correction([res1, res2], method="fdr_bh")
    assert len(adjusted_family) == 2
    assert adjusted_family[0].adjusted_p_value is not None


# -----------------------------------------------------------------------------
# 6. REGRESSION INFERENCE & DIAGNOSTICS TESTS
# -----------------------------------------------------------------------------
def test_fit_ols_regression(synthetic_stat_df):
    reg = fit_ols_regression(
        df=synthetic_stat_df,
        outcome_col="var_y",
        predictor_cols=["var_x"],
    )
    assert isinstance(reg, RegressionModelResult)
    assert reg.r_squared > 0.70
    assert reg.f_pvalue < 0.001
    assert len(reg.coefficients) == 2  # Intercept and var_x
    slope_coeff = next(c for c in reg.coefficients if c.variable == "var_x")
    assert 1.5 < slope_coeff.estimate < 2.5
    assert slope_coeff.p_value < 0.001
    assert len(reg.diagnostics) >= 2


def test_regression_diagnostics():
    df_x = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [2, 4, 6, 8, 10]})
    vif = calculate_vif(df_x)
    assert "a" in vif and "b" in vif
    dw = calculate_durbin_watson(np.array([0.1, -0.1, 0.2, -0.2]))
    assert 0.0 <= dw <= 4.0


# -----------------------------------------------------------------------------
# 7. ROBUSTNESS & SENSITIVITY TESTS
# -----------------------------------------------------------------------------
def test_evaluate_two_group_robustness(synthetic_stat_df):
    comparisons = evaluate_two_group_robustness(
        df=synthetic_stat_df,
        group_col="group",
        measure_col="score",
        group1_name="Control",
        group2_name="Treatment",
    )
    assert len(comparisons) >= 4
    for c in comparisons:
        assert c.specification_name
        assert c.sample_size > 0
        assert c.p_value is not None


def test_evaluate_correlation_robustness(synthetic_stat_df):
    comps = evaluate_correlation_robustness(synthetic_stat_df, "var_x", "var_y")
    assert len(comps) == 2
    assert comps[0].point_estimate > 0.8
    assert comps[1].point_estimate > 0.8


# -----------------------------------------------------------------------------
# 8. STATISTICAL SERVICE & RECOMMENDATION TESTS
# -----------------------------------------------------------------------------
def test_statistical_service_recommendation():
    rec = StatisticalService.recommend_test("two_group", sample_size=100)
    assert "Welch" in rec.recommended_test
    assert len(rec.required_assumptions) > 0
    assert len(rec.alternative_tests) > 0

    rec_cat = StatisticalService.recommend_test("categorical", sample_size=200)
    assert "Chi-Square" in rec_cat.recommended_test
