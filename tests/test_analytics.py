"""Unit and integration tests for Prompt 3 Analytics & EDA Engine."""

import numpy as np
import pandas as pd
import pytest
from src.analytics.analysis_models import (
    CategoricalSummary,
    ChartRecommendation,
    CorrelationPair,
    EDAFinding,
    EDASummary,
    GroupAggregationResult,
    NumericSummary,
)
from src.analytics.chart_recommender import recommend_charts
from src.analytics.descriptive import (
    compute_categorical_summary,
    compute_numeric_summary,
)
from src.analytics.distributions import (
    calculate_empirical_quantiles,
    diagnose_distribution_shape,
)
from src.analytics.eda_service import EDAService
from src.analytics.grouped_analysis import (
    analyze_grouped_disparity,
)
from src.analytics.interpretation import (
    generate_category_concentration_finding,
    generate_correlation_finding,
    generate_skewness_finding,
)
from src.analytics.relationships import (
    compute_cramers_v,
    compute_pairwise_correlations,
)
from src.analytics.temporal import analyze_temporal_series
from src.ingestion.loader import ManagedDataset
from src.ingestion.schema import infer_dataset_schema
from src.visualization.charts import (
    build_bivariate_scatter,
    build_boxplot,
    build_categorical_bar,
    build_correlation_heatmap,
    build_histogram,
    build_violinplot,
)


# -----------------------------------------------------------------------------
# FIXTURES
# -----------------------------------------------------------------------------
@pytest.fixture
def synthetic_eda_df() -> pd.DataFrame:
    """Create a controlled DataFrame with known statistical properties."""
    np.random.seed(42)
    n = 200
    # Positively skewed log-normal variable
    skewed_num = np.random.lognormal(mean=2.0, sigma=1.0, size=n)
    # Symmetric normal variable
    normal_num = np.random.normal(loc=50.0, scale=10.0, size=n)
    # Perfectly linearly correlated variable with noise
    correlated_num = normal_num * 2.5 + np.random.normal(0, 2.0, size=n)
    # Dominant categorical
    categories = ["Tier_A"] * 150 + ["Tier_B"] * 40 + ["Tier_C"] * 10
    # Secondary categorical
    segments = ["Enterprise", "SMB"] * 100
    # Date column
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    # Constant column
    const_col = [42.0] * n

    df = pd.DataFrame(
        {
            "id": [f"ID_{i:04d}" for i in range(n)],
            "skewed_feat": skewed_num,
            "normal_feat": normal_num,
            "corr_feat": correlated_num,
            "category": categories,
            "segment": segments,
            "event_date": dates,
            "constant_feat": const_col,
        }
    )
    # Introduce some NaN values
    df.loc[0:4, "skewed_feat"] = np.nan
    df.loc[10:14, "corr_feat"] = np.nan
    return df


# -----------------------------------------------------------------------------
# 1. DESCRIPTIVE ANALYSIS TESTS
# -----------------------------------------------------------------------------
def test_compute_numeric_summary_basic(synthetic_eda_df):
    summary = compute_numeric_summary(synthetic_eda_df["normal_feat"], "normal_feat")
    assert isinstance(summary, NumericSummary)
    assert summary.column_name == "normal_feat"
    assert summary.count == 200
    assert 48.0 < summary.mean < 52.0
    assert 48.0 < summary.median < 52.0
    assert summary.min < summary.q25 < summary.median < summary.q75 < summary.max
    assert summary.iqr > 0


def test_compute_numeric_summary_with_missing(synthetic_eda_df):
    summary = compute_numeric_summary(synthetic_eda_df["skewed_feat"], "skewed_feat")
    assert summary.missing_count == 5
    assert summary.missing_percentage == 2.5
    assert summary.count == 195


def test_compute_numeric_summary_constant(synthetic_eda_df):
    summary = compute_numeric_summary(synthetic_eda_df["constant_feat"], "constant_feat")
    assert summary.std == 0.0
    assert summary.variance == 0.0
    assert summary.min == summary.max == 42.0
    assert summary.iqr == 0.0


def test_compute_numeric_summary_empty_and_all_nan():
    empty_series = pd.Series([], dtype=float)
    s_empty = compute_numeric_summary(empty_series, "empty_col")
    assert s_empty.count == 0
    assert s_empty.mean == 0.0

    nan_series = pd.Series([np.nan, np.nan, np.nan], dtype=float)
    s_nan = compute_numeric_summary(nan_series, "nan_col")
    assert s_nan.count == 0
    assert s_nan.missing_count == 3


def test_compute_categorical_summary(synthetic_eda_df):
    summary = compute_categorical_summary(synthetic_eda_df["category"], "category")
    assert isinstance(summary, CategoricalSummary)
    assert summary.distinct_count == 3
    assert summary.top_categories[0]["label"] == "Tier_A"
    assert summary.top_categories[0]["count"] == 150
    assert summary.top_1_concentration_pct == 75.0
    assert summary.rare_categories_count >= 0


# -----------------------------------------------------------------------------
# 2. DISTRIBUTION DIAGNOSTICS TESTS
# -----------------------------------------------------------------------------
def test_distribution_diagnostics_skewed(synthetic_eda_df):
    diag = diagnose_distribution_shape(synthetic_eda_df["skewed_feat"])
    assert diag["is_skewed"] is True
    assert diag["skewness_category"] in ("moderately_skewed", "highly_skewed")
    assert diag["preferred_central_metric"] == "median"
    assert "mean_median_divergence_pct" in diag


def test_distribution_diagnostics_normal(synthetic_eda_df):
    diag = diagnose_distribution_shape(synthetic_eda_df["normal_feat"])
    assert diag["skewness_category"] == "symmetric"
    assert diag["preferred_central_metric"] == "mean"


def test_calculate_empirical_quantiles(synthetic_eda_df):
    q_df = calculate_empirical_quantiles(synthetic_eda_df["normal_feat"])
    assert isinstance(q_df, pd.DataFrame)
    assert "Percentile" in q_df.columns
    assert "Value" in q_df.columns
    assert len(q_df) == 13  # Default quantiles


# -----------------------------------------------------------------------------
# 3. RELATIONSHIPS & CORRELATIONS TESTS
# -----------------------------------------------------------------------------
def test_pairwise_correlations_known_correlation(synthetic_eda_df):
    pairs = compute_pairwise_correlations(
        synthetic_eda_df,
        ["normal_feat", "corr_feat", "skewed_feat"],
        method="pearson",
    )
    assert len(pairs) > 0
    # Find normal_feat x corr_feat
    target_pair = next(
        p
        for p in pairs
        if (p.var1 == "normal_feat" and p.var2 == "corr_feat")
        or (p.var1 == "corr_feat" and p.var2 == "normal_feat")
    )
    assert target_pair.coefficient > 0.85
    assert target_pair.direction == "positive"
    assert target_pair.strength == "very_strong"
    assert target_pair.sample_size == 195  # 5 rows had NaNs in corr_feat
    assert target_pair.p_value is not None and target_pair.p_value < 0.001


def test_pairwise_correlations_constant_column(synthetic_eda_df):
    # Constant column has zero variance and should be ignored without raising exception
    pairs = compute_pairwise_correlations(
        synthetic_eda_df,
        ["normal_feat", "constant_feat"],
        method="pearson",
    )
    assert len(pairs) == 0


def test_compute_cramers_v(synthetic_eda_df):
    v_score, contingency = compute_cramers_v(synthetic_eda_df, "category", "segment")
    assert 0.0 <= v_score <= 1.0
    assert isinstance(contingency, pd.DataFrame)
    assert not contingency.empty


# -----------------------------------------------------------------------------
# 4. GROUPED DISPARITY TESTS
# -----------------------------------------------------------------------------
def test_analyze_grouped_disparity_basic(synthetic_eda_df):
    res = analyze_grouped_disparity(
        synthetic_eda_df,
        group_col="category",
        measure_col="normal_feat",
        agg="mean",
    )
    assert isinstance(res, GroupAggregationResult)
    assert res.group_column == "category"
    assert res.measure_column == "normal_feat"
    assert len(res.groups) == 3
    assert res.top_group is not None
    assert res.bottom_group is not None
    assert res.max_to_min_ratio is not None


def test_analyze_grouped_disparity_empty_or_invalid():
    df_empty = pd.DataFrame({"cat": [], "num": []})
    res = analyze_grouped_disparity(df_empty, "cat", "num")
    assert len(res.groups) == 0


# -----------------------------------------------------------------------------
# 5. TEMPORAL SERIES TESTS
# -----------------------------------------------------------------------------
def test_analyze_temporal_series(synthetic_eda_df):
    temp = analyze_temporal_series(synthetic_eda_df, "event_date")
    assert temp.total_records == 200
    assert temp.earliest_date is not None
    assert temp.latest_date is not None
    assert temp.span_days == 199


# -----------------------------------------------------------------------------
# 6. CHART RECOMMENDER TESTS
# -----------------------------------------------------------------------------
def test_chart_recommender(synthetic_eda_df):
    schemas = infer_dataset_schema(synthetic_eda_df)
    recs = recommend_charts(synthetic_eda_df, schemas, max_recommendations=5)
    assert len(recs) > 0
    for r in recs:
        assert isinstance(r, ChartRecommendation)
        assert 0.0 <= r.suitability_score <= 100.0
        assert r.why_appropriate
        assert r.limitations
        assert r.suggested_followup

    # Ensure correlation heatmap is recommended when multiple numeric columns exist
    types = [r.chart_type for r in recs]
    assert "correlation_heatmap" in types or "timeseries_line" in types


# -----------------------------------------------------------------------------
# 7. 6-LAYER INTERPRETATION TESTS
# -----------------------------------------------------------------------------
def test_generate_skewness_finding(synthetic_eda_df):
    num_sum = compute_numeric_summary(synthetic_eda_df["skewed_feat"], "skewed_feat")
    shape_diag = diagnose_distribution_shape(synthetic_eda_df["skewed_feat"])
    fnd = generate_skewness_finding(num_sum, shape_diag)

    assert isinstance(fnd, EDAFinding)
    assert fnd.category == "distribution"
    assert fnd.observation
    assert fnd.quantitative_evidence
    assert fnd.statistical_context
    assert fnd.practical_implication
    assert fnd.limitations
    assert fnd.next_investigation
    assert "median" in fnd.statistical_context.lower()


def test_generate_correlation_finding():
    pair = CorrelationPair(
        var1="revenue",
        var2="marketing_spend",
        coefficient=0.88,
        method="pearson",
        sample_size=500,
        p_value=0.0001,
        strength="very_strong",
        direction="positive",
    )
    fnd = generate_correlation_finding(pair)
    assert fnd.category == "correlation"
    # Grounded rule: NEVER claim causation from correlation
    assert "causation" in fnd.statistical_context.lower()
    assert "confounder" in fnd.limitations.lower() or "bias" in fnd.limitations.lower()
    assert "0.88" in fnd.quantitative_evidence


def test_generate_category_concentration_finding(synthetic_eda_df):
    cat_sum = compute_categorical_summary(synthetic_eda_df["category"], "category")
    fnd = generate_category_concentration_finding(cat_sum)
    assert fnd.category == "concentration"
    assert "Tier_A" in fnd.observation
    assert "75.0" in fnd.observation or "75.0%" in fnd.quantitative_evidence


# -----------------------------------------------------------------------------
# 8. EDA SERVICE & INTEGRATION TESTS
# -----------------------------------------------------------------------------
def test_eda_service_generate_summary(synthetic_eda_df):
    schemas = infer_dataset_schema(synthetic_eda_df)
    managed = ManagedDataset(
        dataset_id="test_eda_ds",
        filename="synthetic_test.csv",
        file_format="csv",
        file_size_bytes=1024,
        fingerprint="test_fp",
        raw_df=synthetic_eda_df,
        working_df=synthetic_eda_df,
        schema_info=schemas,
        ingestion_report=None,  # type: ignore[arg-type]
    )
    summary = EDAService.generate_eda_summary(managed)
    assert isinstance(summary, EDASummary)
    assert summary.row_count == 200
    assert summary.column_count == len(synthetic_eda_df.columns)
    assert len(summary.findings) > 0
    assert len(summary.recommendations) > 0
    assert len(summary.top_correlations) > 0


# -----------------------------------------------------------------------------
# 9. VISUALIZATION CHARTS SMOKE TESTS
# -----------------------------------------------------------------------------
def test_visualization_charts_render(synthetic_eda_df):
    # Test each chart builder returns a valid figure without error
    fig_hist = build_histogram(synthetic_eda_df["normal_feat"], "normal_feat")
    assert fig_hist is not None

    fig_box = build_boxplot(synthetic_eda_df["normal_feat"], "normal_feat")
    assert fig_box is not None

    fig_violin = build_violinplot(synthetic_eda_df["normal_feat"], "normal_feat")
    assert fig_violin is not None

    fig_bar = build_categorical_bar([{"label": "A", "count": 10, "percentage": 100.0}], "cat")
    assert fig_bar is not None

    fig_corr = build_correlation_heatmap(synthetic_eda_df[["normal_feat", "corr_feat"]].corr())
    assert fig_corr is not None

    fig_scatter, sampled = build_bivariate_scatter(synthetic_eda_df, "normal_feat", "corr_feat")
    assert fig_scatter is not None
    assert sampled is False
