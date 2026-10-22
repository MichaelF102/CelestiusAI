"""Analytics package providing descriptive stats, distributions, correlations, and statistical inference."""

from src.analytics.analysis_models import (
    CategoricalSummary,
    ChartRecommendation,
    CorrelationPair,
    EDAFinding,
    EDASummary,
    GroupAggregationResult,
    NumericSummary,
    TemporalSummary,
)
from src.analytics.assumption_checks import (
    check_homoscedasticity_levene,
    check_normality_dagostino,
    check_normality_shapiro,
    generate_qq_plot_data,
)
from src.analytics.categorical_tests import run_chi_square_test
from src.analytics.chart_recommender import recommend_charts
from src.analytics.confidence_intervals import (
    calculate_bootstrap_ci,
    calculate_correlation_ci,
    calculate_difference_in_means_ci,
    calculate_mean_ci,
    calculate_paired_difference_ci,
    calculate_proportion_ci,
)
from src.analytics.correlation_inference import run_correlation_test
from src.analytics.descriptive import (
    compute_all_numeric_summaries,
    compute_categorical_summary,
    compute_numeric_summary,
)
from src.analytics.distributions import (
    calculate_empirical_quantiles,
    diagnose_distribution_shape,
)
from src.analytics.eda_service import EDAService
from src.analytics.effect_sizes import (
    calculate_cliffs_delta,
    calculate_cohens_d,
    calculate_cramers_v_effect,
    calculate_eta_squared,
    calculate_paired_cohens_d,
)
from src.analytics.group_comparisons import (
    run_multi_group_anova,
    run_paired_group_test,
    run_two_group_independent_test,
)
from src.analytics.grouped_analysis import (
    analyze_grouped_disparity,
    compute_two_way_contingency,
)
from src.analytics.interpretation import (
    generate_category_concentration_finding,
    generate_correlation_finding,
    generate_grouped_disparity_finding,
    generate_skewness_finding,
)
from src.analytics.multiple_testing import (
    adjust_p_values,
    apply_multiple_testing_correction,
)
from src.analytics.regression_diagnostics import (
    calculate_durbin_watson,
    calculate_vif,
    check_breusch_pagan_heteroscedasticity,
    check_residual_normality_jarque_bera,
    identify_influential_points_cooks_d,
)
from src.analytics.regression_inference import fit_ols_regression
from src.analytics.relationships import (
    compute_cramers_v,
    compute_pairwise_correlations,
)
from src.analytics.robustness import (
    evaluate_correlation_robustness,
    evaluate_two_group_robustness,
)
from src.analytics.statistical_interpretation import (
    render_plain_english_takeaway,
    render_regression_interpretation_markdown,
    render_regression_plain_english,
    render_statistical_interpretation_markdown,
)
from src.analytics.statistical_models import (
    AssumptionCheckResult,
    ConfidenceIntervalResult,
    DesignType,
    EffectSizeResult,
    HypothesisDirection,
    MissingPolicy,
    RegressionCoefficient,
    RegressionModelResult,
    RobustnessComparison,
    StatisticalTestResult,
    TestRecommendation,
)
from src.analytics.statistical_service import StatisticalService
from src.analytics.temporal import analyze_temporal_series

__all__ = [
    "AssumptionCheckResult",
    "CategoricalSummary",
    "ChartRecommendation",
    "ConfidenceIntervalResult",
    "CorrelationPair",
    "DesignType",
    "EDAFinding",
    "EDASummary",
    "EDAService",
    "EffectSizeResult",
    "GroupAggregationResult",
    "HypothesisDirection",
    "MissingPolicy",
    "NumericSummary",
    "RegressionCoefficient",
    "RegressionModelResult",
    "RobustnessComparison",
    "StatisticalService",
    "StatisticalTestResult",
    "TemporalSummary",
    "TestRecommendation",
    "adjust_p_values",
    "analyze_grouped_disparity",
    "analyze_temporal_series",
    "apply_multiple_testing_correction",
    "calculate_bootstrap_ci",
    "calculate_cliffs_delta",
    "calculate_cohens_d",
    "calculate_correlation_ci",
    "calculate_cramers_v_effect",
    "calculate_difference_in_means_ci",
    "calculate_durbin_watson",
    "calculate_empirical_quantiles",
    "calculate_eta_squared",
    "calculate_mean_ci",
    "calculate_paired_cohens_d",
    "calculate_paired_difference_ci",
    "calculate_proportion_ci",
    "calculate_vif",
    "check_breusch_pagan_heteroscedasticity",
    "check_homoscedasticity_levene",
    "check_normality_dagostino",
    "check_normality_shapiro",
    "check_residual_normality_jarque_bera",
    "compute_all_numeric_summaries",
    "compute_categorical_summary",
    "compute_cramers_v",
    "compute_numeric_summary",
    "compute_pairwise_correlations",
    "compute_two_way_contingency",
    "diagnose_distribution_shape",
    "evaluate_correlation_robustness",
    "evaluate_two_group_robustness",
    "fit_ols_regression",
    "generate_category_concentration_finding",
    "generate_correlation_finding",
    "generate_grouped_disparity_finding",
    "generate_qq_plot_data",
    "generate_skewness_finding",
    "identify_influential_points_cooks_d",
    "recommend_charts",
    "render_plain_english_takeaway",
    "render_regression_interpretation_markdown",
    "render_regression_plain_english",
    "render_statistical_interpretation_markdown",
    "run_chi_square_test",
    "run_correlation_test",
    "run_multi_group_anova",
    "run_paired_group_test",
    "run_two_group_independent_test",
]
