"""Pydantic data models for formal statistical inference, tests, diagnostics, and effect sizes."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

HypothesisDirection = Literal["two-sided", "greater", "less"]
DesignType = Literal["independent", "paired", "repeated"]
MissingPolicy = Literal["drop_pairwise", "drop_listwise", "error"]
CorrelationMethod = Literal["pearson", "spearman"]


class EffectSizeResult(BaseModel):
    """Standardized effect-size quantification and domain interpretation."""

    model_config = ConfigDict(frozen=True)

    name: str
    estimate: float
    magnitude: Literal["negligible", "small", "medium", "large", "very_large"]
    ci_lower: float | None = None
    ci_upper: float | None = None
    ci_level: float = 0.95
    interpretation: str


class AssumptionCheckResult(BaseModel):
    """Diagnostic verification of mathematical and design assumptions."""

    model_config = ConfigDict(frozen=True)

    name: str
    statistic_name: str
    statistic_value: float | None = None
    p_value: float | None = None
    is_satisfied: bool
    severity: Literal["info", "warning", "violation"] = "info"
    message: str
    recommendation: str


class ConfidenceIntervalResult(BaseModel):
    """Estimated confidence interval around a sample statistic or effect size."""

    model_config = ConfigDict(frozen=True)

    estimand: str
    point_estimate: float
    lower_bound: float
    upper_bound: float
    confidence_level: float = 0.95
    method: str


class TestRecommendation(BaseModel):
    """Deterministic recommendation of an inferential procedure from schema & design."""

    model_config = ConfigDict(frozen=True)

    recommended_test: str
    rationale: str
    required_assumptions: list[str]
    alternative_tests: list[str] = Field(default_factory=list)
    design_notes: str = ""


class StatisticalTestResult(BaseModel):
    """Comprehensive, validated statistical test result adhering to Section 5 schema."""

    model_config = ConfigDict(frozen=True)

    analysis_id: str
    test_name: str
    research_question: str
    null_hypothesis: str
    alternative_hypothesis: str
    selected_variables: list[str]
    groups: list[str] = Field(default_factory=list)
    sample_sizes: dict[str, int] = Field(default_factory=dict)
    total_n: int = Field(..., ge=1)
    missing_policy: MissingPolicy = "drop_pairwise"
    excluded_count: int = 0
    statistic_name: str
    statistic_value: float
    degrees_of_freedom: float | list[float] | None = None
    p_value: float = Field(..., ge=0.0, le=1.0)
    p_value_display: str
    adjusted_p_value: float | None = Field(default=None, ge=0.0, le=1.0)
    adjustment_method: str | None = None
    confidence_interval: ConfidenceIntervalResult | None = None
    effect_size: EffectSizeResult | None = None
    assumption_checks: list[AssumptionCheckResult] = Field(default_factory=list)
    summary: str
    interpretation_layers: dict[str, str] = Field(default_factory=dict)
    practical_significance: str
    limitations: list[str] = Field(default_factory=list)
    recommended_followups: list[str] = Field(default_factory=list)
    method_metadata: dict[str, Any] = Field(default_factory=dict)


class RegressionCoefficient(BaseModel):
    """Estimated coefficient, uncertainty, and test of individual predictor."""

    model_config = ConfigDict(frozen=True)

    variable: str
    estimate: float
    std_error: float
    t_statistic: float
    p_value: float
    ci_lower: float
    ci_upper: float
    vif: float | None = None


class RegressionModelResult(BaseModel):
    """Ordinary Least Squares regression inference and diagnostic report."""

    model_config = ConfigDict(frozen=True)

    analysis_id: str
    outcome_variable: str
    predictor_variables: list[str]
    n_observations: int
    df_model: float
    df_residuals: float
    r_squared: float
    adj_r_squared: float
    f_statistic: float
    f_pvalue: float
    aic: float
    bic: float
    coefficients: list[RegressionCoefficient]
    diagnostics: list[AssumptionCheckResult] = Field(default_factory=list)
    summary: str
    interpretation_layers: dict[str, str] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)


class RobustnessComparison(BaseModel):
    """Comparison of results across alternative analytical specifications."""

    model_config = ConfigDict(frozen=True)

    specification_name: str
    method: str
    sample_size: int
    point_estimate: float
    p_value: float | None
    ci_lower: float | None = None
    ci_upper: float | None = None
    effect_size_estimate: float | None = None
    conclusion_stable: bool = True
    notes: str = ""
