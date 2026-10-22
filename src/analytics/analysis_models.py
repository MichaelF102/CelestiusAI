"""Pydantic schemas for exploratory data analysis, statistics, chart recommendations, and evidence-backed findings."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

CorrelationMethod = Literal["pearson", "spearman"]
AggregationFunc = Literal["mean", "median", "sum", "count", "std", "min", "max"]


class NumericSummary(BaseModel):
    """Statistical summary for continuous quantitative features."""

    model_config = ConfigDict(frozen=True)

    column_name: str
    count: int = Field(..., ge=0)
    missing_count: int = Field(..., ge=0)
    missing_percentage: float = Field(..., ge=0.0, le=100.0)
    mean: float
    median: float
    std: float = Field(..., ge=0.0)
    variance: float = Field(..., ge=0.0)
    min_value: float
    max_value: float
    q25: float
    q75: float
    iqr: float = Field(..., ge=0.0)
    skewness: float | None = None
    kurtosis: float | None = None
    zero_count: int = Field(default=0, ge=0)
    negative_count: int = Field(default=0, ge=0)
    outlier_count_iqr: int = Field(default=0, ge=0)

    @property
    def min(self) -> float:
        """Alias for min_value."""
        return self.min_value

    @property
    def max(self) -> float:
        """Alias for max_value."""
        return self.max_value


class TemporalSummary(BaseModel):
    """Statistical summary for temporal / time-series data."""

    model_config = ConfigDict(frozen=True)

    date_column: str
    value_column: str | None = None
    total_records: int = 0
    earliest_date: str | None = None
    latest_date: str | None = None
    span_days: int = 0
    trend_direction: str = "stable"
    timeline_data: list[dict[str, Any]] = Field(default_factory=list)
    day_of_week_distribution: list[dict[str, Any]] = Field(default_factory=list)


class CategoricalSummary(BaseModel):
    """Statistical summary for discrete categorical features."""

    model_config = ConfigDict(frozen=True)

    column_name: str
    total_count: int = Field(..., ge=0)
    missing_count: int = Field(..., ge=0)
    distinct_count: int = Field(..., ge=0)
    top_categories: list[dict[str, Any]] = Field(default_factory=list)
    top_1_concentration_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    top_3_concentration_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    rare_categories_count: int = Field(default=0, ge=0)
    empty_strings_count: int = Field(default=0, ge=0)


class CorrelationPair(BaseModel):
    """Pairwise correlation metric between two quantitative variables."""

    model_config = ConfigDict(frozen=True)

    var1: str
    var2: str
    method: CorrelationMethod
    coefficient: float = Field(..., ge=-1.0, le=1.0)
    sample_size: int = Field(..., ge=2)
    p_value: float | None = Field(default=None, ge=0.0, le=1.0)
    strength: Literal["very_strong", "strong", "moderate", "weak", "negligible"]
    direction: Literal["positive", "negative", "none"]


class GroupAggregationResult(BaseModel):
    """Grouped analysis summary for a categorical group against a numeric metric."""

    group_column: str
    measure_column: str
    aggregation: AggregationFunc
    groups: list[dict[str, Any]] = Field(default_factory=list)
    total_groups_count: int = Field(..., ge=0)
    top_group: str | None = None
    bottom_group: str | None = None
    max_to_min_ratio: float | None = None
    is_imbalanced: bool = False
    notes: str = ""


class ChartRecommendation(BaseModel):
    """Structured recommendation for an exploratory visualization."""

    model_config = ConfigDict(frozen=True)

    chart_type: str
    columns: list[str]
    title: str
    purpose: str
    suitability_score: float = Field(..., ge=0.0, le=100.0)
    why_appropriate: str
    limitations: str
    suggested_followup: str


class EDAFinding(BaseModel):
    """Grounded analytical finding adhering to the 6-layer structured interpretation model."""

    finding_id: str
    title: str
    category: Literal[
        "distribution",
        "correlation",
        "grouped_disparity",
        "concentration",
        "temporal",
        "anomaly",
    ]
    summary: str
    observation: str
    quantitative_evidence: str
    statistical_context: str
    practical_implication: str
    limitations: str
    next_investigation: str
    severity: Literal["low", "medium", "high", "critical"] = "medium"
    relevant_columns: list[str] = Field(default_factory=list)


class EDASummary(BaseModel):
    """Consolidated exploratory summary of an analyzed dataset."""

    dataset_id: str
    row_count: int
    column_count: int
    findings: list[EDAFinding] = Field(default_factory=list)
    recommendations: list[ChartRecommendation] = Field(default_factory=list)
    top_correlations: list[CorrelationPair] = Field(default_factory=list)
    skewed_columns: list[str] = Field(default_factory=list)
    concentrated_categories: list[str] = Field(default_factory=list)
