"""Analytics package providing descriptive stats, distributions, correlations, and recommendations."""

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
from src.analytics.chart_recommender import recommend_charts
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
from src.analytics.relationships import (
    compute_cramers_v,
    compute_pairwise_correlations,
)
from src.analytics.temporal import analyze_temporal_series

__all__ = [
    "CategoricalSummary",
    "ChartRecommendation",
    "CorrelationPair",
    "EDAFinding",
    "EDASummary",
    "GroupAggregationResult",
    "NumericSummary",
    "TemporalSummary",
    "analyze_grouped_disparity",
    "analyze_temporal_series",
    "calculate_empirical_quantiles",
    "compute_all_numeric_summaries",
    "compute_categorical_summary",
    "compute_cramers_v",
    "compute_numeric_summary",
    "compute_pairwise_correlations",
    "compute_two_way_contingency",
    "diagnose_distribution_shape",
    "generate_category_concentration_finding",
    "generate_correlation_finding",
    "generate_grouped_disparity_finding",
    "generate_skewness_finding",
    "recommend_charts",
    "EDAService",
]
