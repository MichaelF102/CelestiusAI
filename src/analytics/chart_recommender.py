"""Deterministic chart recommender selecting optimal visualizations from schema metadata."""

import pandas as pd

from src.analytics.analysis_models import ChartRecommendation
from src.ingestion.models import ColumnSchema


def recommend_charts(
    df: pd.DataFrame,
    schemas: dict[str, ColumnSchema],
    max_recommendations: int = 8,
) -> list[ChartRecommendation]:
    """Analyze schema types, cardinality, and relationships to recommend optimal visualizations."""
    recommendations: list[ChartRecommendation] = []

    numeric_cols = [
        c
        for c, s in schemas.items()
        if s.inferred_type == "numeric"
        and not s.is_identifier
        and not s.is_constant
        and c in df.columns
    ]
    categorical_cols = [
        c
        for c, s in schemas.items()
        if s.inferred_type == "categorical"
        and not s.is_constant
        and 2 <= s.distinct_count <= 25
        and c in df.columns
    ]
    datetime_cols = [
        c for c, s in schemas.items() if s.inferred_type == "datetime" and c in df.columns
    ]

    # Rule 1: Multi-numeric Correlation Heatmap (Score: 95)
    if len(numeric_cols) >= 3:
        recommendations.append(
            ChartRecommendation(
                chart_type="correlation_heatmap",
                columns=numeric_cols[:10],
                title="Global Correlation Matrix Heatmap",
                purpose="Evaluate pairwise linear and monotonic associations across all quantitative features.",
                suitability_score=95.0,
                why_appropriate="Multiple continuous variables enable compact multi-feature relationship discovery.",
                limitations="Does not detect non-linear associations and does not establish causality.",
                suggested_followup="Examine top correlated pairs in focused bivariate scatter plots.",
            )
        )

    # Rule 2: Datetime + Primary Numeric Metric (Score: 90)
    if datetime_cols and numeric_cols:
        dt_col = datetime_cols[0]
        num_col = numeric_cols[0]
        recommendations.append(
            ChartRecommendation(
                chart_type="timeseries_line",
                columns=[dt_col, num_col],
                title=f"Chronological Trend: '{num_col}' over Time",
                purpose=f"Track seasonal patterns, structural changes, and trend trajectory in '{num_col}'.",
                suitability_score=90.0,
                why_appropriate=f"'{dt_col}' provides chronological ordering for continuous metric '{num_col}'.",
                limitations="Sensitive to missing reporting intervals and irregular timestamps.",
                suggested_followup="Decompose the series into seasonal, trend, and cyclical components.",
            )
        )

    # Rule 3: Categorical Group + Numeric Metric (Score: 88)
    if categorical_cols and numeric_cols:
        cat_col = categorical_cols[0]
        num_col = numeric_cols[0]
        recommendations.append(
            ChartRecommendation(
                chart_type="grouped_bar",
                columns=[cat_col, num_col],
                title=f"Segment Comparison: Mean '{num_col}' by '{cat_col}'",
                purpose=f"Compare how central tendencies of '{num_col}' diverge across '{cat_col}' segments.",
                suitability_score=88.0,
                why_appropriate=f"Discrete groups in '{cat_col}' ({schemas[cat_col].distinct_count} categories) provide clean comparison buckets.",
                limitations="Does not display intra-group variance unless accompanied by standard deviation or box plots.",
                suggested_followup="Switch to a box plot to evaluate intra-group distribution and outliers.",
            )
        )

    # Rule 4: Two Top Numeric Variables Scatter Plot (Score: 85)
    if len(numeric_cols) >= 2:
        n1 = numeric_cols[0]
        n2 = numeric_cols[1]
        recommendations.append(
            ChartRecommendation(
                chart_type="bivariate_scatter",
                columns=[n1, n2],
                title=f"Bivariate Scatter: '{n1}' vs '{n2}'",
                purpose=f"Inspect linear dispersion, clustering, and individual outlier coordinates between '{n1}' and '{n2}'.",
                suitability_score=85.0,
                why_appropriate="Continuous variables allow direct coordinate mapping and trendline fitting.",
                limitations="Points may suffer from overplotting if record count is high.",
                suggested_followup="Add a categorical color overlay to evaluate segment clustering.",
            )
        )

    # Rule 5: Key Continuous Metric Distribution (Score: 82)
    if numeric_cols:
        num_col = numeric_cols[0]
        recommendations.append(
            ChartRecommendation(
                chart_type="histogram_distribution",
                columns=[num_col],
                title=f"Distribution Profile: '{num_col}'",
                purpose=f"Diagnose modality, asymmetry, skewness, and tail thickness in '{num_col}'.",
                suitability_score=82.0,
                why_appropriate="Univariate quantitative exploration is a foundational precursor to statistical modeling.",
                limitations="Bin count selection can alter visual perception of distribution shapes.",
                suggested_followup="Inspect box plot and check whether median represents central tendency better than mean.",
            )
        )

    # Rule 6: Two Categorical Variables Crosstab Heatmap (Score: 78)
    if len(categorical_cols) >= 2:
        c1 = categorical_cols[0]
        c2 = categorical_cols[1]
        recommendations.append(
            ChartRecommendation(
                chart_type="crosstab_heatmap",
                columns=[c1, c2],
                title=f"Cross-Tabulation Matrix: '{c1}' vs '{c2}'",
                purpose=f"Detect interaction frequencies and category intersections between '{c1}' and '{c2}'.",
                suitability_score=78.0,
                why_appropriate="Both variables are discrete with moderate cardinality.",
                limitations="Large matrices may create visual clutter.",
                suggested_followup="Calculate Cramér's V or perform a Chi-Square test of independence.",
            )
        )

    # Rule 7: Primary Categorical Frequency Distribution (Score: 75)
    if categorical_cols:
        cat_col = categorical_cols[0]
        recommendations.append(
            ChartRecommendation(
                chart_type="categorical_bar",
                columns=[cat_col],
                title=f"Frequency Breakdown: '{cat_col}'",
                purpose=f"Evaluate category prevalence, imbalance, and rare groups in '{cat_col}'.",
                suitability_score=75.0,
                why_appropriate="Low-cardinality discrete feature suits categorical bar ranking.",
                limitations="Does not incorporate continuous performance metrics.",
                suggested_followup="Investigate whether dominant category should be analyzed independently.",
            )
        )

    # Sort descending by suitability score
    recommendations.sort(key=lambda r: r.suitability_score, reverse=True)
    return recommendations[:max_recommendations]
