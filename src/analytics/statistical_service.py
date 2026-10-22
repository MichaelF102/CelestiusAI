"""Statistical Test Selection and Inference Orchestration Service."""

import logging
from typing import Literal

from src.analytics.categorical_tests import run_chi_square_test
from src.analytics.correlation_inference import run_correlation_test
from src.analytics.group_comparisons import (
    run_multi_group_anova,
    run_paired_group_test,
    run_two_group_independent_test,
)
from src.analytics.regression_inference import fit_ols_regression
from src.analytics.statistical_models import (
    DesignType,
    HypothesisDirection,
    RegressionModelResult,
    StatisticalTestResult,
    TestRecommendation,
)
from src.ingestion.loader import ManagedDataset

logger = logging.getLogger(__name__)


class StatisticalService:
    """Orchestrates test selection, execution, diagnostics, and interpretation."""

    @staticmethod
    def recommend_test(
        analysis_type: Literal[
            "two_group", "multi_group", "paired", "correlation", "categorical", "regression"
        ],
        sample_size: int,
        design: DesignType = "independent",
        is_normal: bool = True,
        is_homoscedastic: bool = True,
    ) -> TestRecommendation:
        """Deterministically recommend inferential test from schema, design, and distributional properties."""
        if analysis_type == "two_group":
            if not is_homoscedastic or sample_size >= 30:
                rec = "Welch's Heteroscedastic t-test"
                rat = (
                    "Welch's t-test is the robust standard recommendation for two independent continuous groups. "
                    "It reliably controls Type I error without requiring the fragile assumption of equal population variances."
                )
                assumptions = [
                    "Independent observations",
                    "Normality or sample size N >= 30 (Central Limit Theorem)",
                ]
                alts = [
                    "Mann-Whitney U Test (Rank-based non-parametric)",
                    "Student's t-test (Equal variance pooled)",
                ]
            elif not is_normal and sample_size < 30:
                rec = "Mann-Whitney U Test"
                rat = "Sample size is small (N < 30) and departs from normality; rank-based comparison avoids distributional bias."
                assumptions = ["Independent observations", "Ordinal or continuous measurement"]
                alts = ["Welch's t-test", "Permutation / Bootstrap test"]
            else:
                rec = "Welch's t-test"
                rat = "Welch's t-test is safe across both equal and unequal variance scenarios."
                assumptions = ["Independent observations", "Continuous metric"]
                alts = ["Student's t-test", "Mann-Whitney U"]

            return TestRecommendation(
                recommended_test=rec,
                rationale=rat,
                required_assumptions=assumptions,
                alternative_tests=alts,
                design_notes="Requires independent sampling. If rows belong to the same subjects, switch design to Paired.",
            )

        if analysis_type == "paired":
            rec = (
                "Paired Student's t-test"
                if (is_normal or sample_size >= 30)
                else "Wilcoxon Signed-Rank Test"
            )
            rat = (
                "Paired t-test accounts for within-subject correlation across aligned observations."
                if (is_normal or sample_size >= 30)
                else "Wilcoxon signed-rank test evaluates median paired change without requiring normal differences."
            )
            return TestRecommendation(
                recommended_test=rec,
                rationale=rat,
                required_assumptions=[
                    "Properly paired / aligned observations",
                    "Normal differences (t-test)",
                ],
                alternative_tests=[
                    "Wilcoxon Signed-Rank Test"
                    if rec != "Wilcoxon Signed-Rank Test"
                    else "Paired t-test"
                ],
                design_notes="Each row must represent the exact same entity or subject measured at two conditions/times.",
            )

        if analysis_type == "multi_group":
            rec = "One-Way ANOVA" if is_homoscedastic else "Welch's ANOVA"
            rat = "One-Way ANOVA tests the omnibus equality of means across 3+ independent groups."
            return TestRecommendation(
                recommended_test=rec,
                rationale=rat,
                required_assumptions=[
                    "Independent observations",
                    "Homogeneity of variance (Levene)",
                    "Normality of residuals",
                ],
                alternative_tests=[
                    "Kruskal-Wallis Test (Non-parametric rank ANOVA)",
                    "Welch's ANOVA",
                ],
                design_notes="Follow significant omnibus results with post-hoc pairwise tests adjusted for multiple testing.",
            )

        if analysis_type == "categorical":
            rec = "Pearson Chi-Square Test of Independence"
            rat = "Tests whether joint distribution of two discrete categorical variables differs from independent expectations."
            return TestRecommendation(
                recommended_test=rec,
                rationale=rat,
                required_assumptions=[
                    "Mutually exclusive categories",
                    "Cochran's rule: expected counts >= 5 in >= 80% of cells",
                ],
                alternative_tests=[
                    "Fisher's Exact Test (recommended for 2x2 tables with sparse expected counts)"
                ],
                design_notes="Measures statistical association, NOT causal direction.",
            )

        if analysis_type == "correlation":
            rec = (
                "Pearson Correlation"
                if (is_normal and sample_size >= 30)
                else "Spearman Rank Correlation"
            )
            rat = (
                "Pearson r measures linear co-variance."
                if (is_normal and sample_size >= 30)
                else "Spearman rho evaluates monotonic association robust to outliers and non-linear monotonic shifts."
            )
            return TestRecommendation(
                recommended_test=rec,
                rationale=rat,
                required_assumptions=[
                    "Bivariate normality (Pearson)",
                    "Monotonic relationship (Spearman)",
                ],
                alternative_tests=["Spearman Rank Correlation", "Kendall's tau"],
                design_notes="Correlation does NOT imply causation.",
            )

        # Default: Regression
        return TestRecommendation(
            recommended_test="Ordinary Least Squares (OLS) Linear Regression",
            rationale="Estimates the conditional expectation of a continuous outcome given numeric and encoded categorical predictors.",
            required_assumptions=[
                "Linearity of expectation",
                "Homoscedasticity of residuals (constant variance)",
                "No severe multicollinearity (VIF < 5)",
                "Independent errors (no serial autocorrelation)",
            ],
            alternative_tests=[
                "Robust Regression (Huber / HC3)",
                "Ridge / Lasso Regularized Regression",
            ],
            design_notes="Coefficients represent conditional associations under this model specification, not causal mechanisms.",
        )

    @staticmethod
    def execute_two_group_analysis(
        managed_dataset: ManagedDataset,
        group_col: str,
        measure_col: str,
        group1_name: str | None = None,
        group2_name: str | None = None,
        method: Literal["welch", "student", "mann_whitney"] = "welch",
        direction: HypothesisDirection = "two-sided",
        confidence_level: float = 0.95,
        alpha: float = 0.05,
    ) -> StatisticalTestResult:
        """Run two-group inference using working dataset."""
        df = managed_dataset.working_df
        return run_two_group_independent_test(
            df=df,
            group_col=group_col,
            measure_col=measure_col,
            group1_name=group1_name,
            group2_name=group2_name,
            method=method,
            direction=direction,
            confidence_level=confidence_level,
            alpha=alpha,
        )

    @staticmethod
    def execute_paired_analysis(
        managed_dataset: ManagedDataset,
        pre_col: str,
        post_col: str,
        method: Literal["paired_t", "wilcoxon"] = "paired_t",
        confidence_level: float = 0.95,
        alpha: float = 0.05,
    ) -> StatisticalTestResult:
        """Run paired inference on working dataset."""
        df = managed_dataset.working_df
        return run_paired_group_test(
            df=df,
            pre_col=pre_col,
            post_col=post_col,
            method=method,
            confidence_level=confidence_level,
            alpha=alpha,
        )

    @staticmethod
    def execute_multi_group_analysis(
        managed_dataset: ManagedDataset,
        group_col: str,
        measure_col: str,
        alpha: float = 0.05,
    ) -> StatisticalTestResult:
        """Run multi-group ANOVA on working dataset."""
        df = managed_dataset.working_df
        return run_multi_group_anova(
            df=df,
            group_col=group_col,
            measure_col=measure_col,
            alpha=alpha,
        )

    @staticmethod
    def execute_categorical_analysis(
        managed_dataset: ManagedDataset,
        var1: str,
        var2: str,
        alpha: float = 0.05,
    ) -> StatisticalTestResult:
        """Run categorical independence test on working dataset."""
        df = managed_dataset.working_df
        return run_chi_square_test(
            df=df,
            var1=var1,
            var2=var2,
            alpha=alpha,
        )

    @staticmethod
    def execute_correlation_analysis(
        managed_dataset: ManagedDataset,
        var1: str,
        var2: str,
        method: Literal["pearson", "spearman"] = "pearson",
        confidence_level: float = 0.95,
        alpha: float = 0.05,
    ) -> StatisticalTestResult:
        """Run correlation inference on working dataset."""
        df = managed_dataset.working_df
        return run_correlation_test(
            df=df,
            var1=var1,
            var2=var2,
            method=method,
            confidence_level=confidence_level,
            alpha=alpha,
        )

    @staticmethod
    def execute_regression_analysis(
        managed_dataset: ManagedDataset,
        outcome_col: str,
        predictor_cols: list[str],
        include_intercept: bool = True,
        confidence_level: float = 0.95,
        alpha: float = 0.05,
    ) -> RegressionModelResult:
        """Fit OLS regression on working dataset."""
        df = managed_dataset.working_df
        return fit_ols_regression(
            df=df,
            outcome_col=outcome_col,
            predictor_cols=predictor_cols,
            include_intercept=include_intercept,
            confidence_level=confidence_level,
            alpha=alpha,
        )
