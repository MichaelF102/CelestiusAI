"""Pydantic schemas for data profiling, quality issues, scoring, and cleaning transformations."""

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.ingestion.models import ColumnSchema

SeverityLevel = Literal["critical", "high", "medium", "low"]
IssueCategory = Literal[
    "missingness",
    "duplicates",
    "outliers",
    "consistency",
    "schema",
    "cardinality",
    "validity",
]

TransformationType = Literal[
    "drop_column",
    "drop_duplicates",
    "drop_missing_rows",
    "impute_numeric",
    "impute_categorical",
    "impute_constant",
    "strip_whitespace",
    "standardize_casing",
    "cap_outliers",
    "filter_outliers",
    "convert_type",
]


class QualityIssue(BaseModel):
    """Normalized data quality problem with evidence and cleaning recommendation."""

    model_config = ConfigDict(frozen=True)

    issue_id: str = Field(..., description="Unique stable identifier for this issue.")
    category: IssueCategory = Field(..., description="Domain category of the issue.")
    severity: SeverityLevel = Field(..., description="Severity level for prioritization.")
    title: str = Field(..., description="Headline describing the anomaly.")
    affected_columns: list[str] = Field(default_factory=list, description="Columns impacted.")
    affected_row_count: int = Field(
        default=0, ge=0, description="Estimated count of rows affected."
    )
    affected_cell_count: int = Field(
        default=0, ge=0, description="Estimated count of cells affected."
    )
    measured_evidence: str = Field(..., description="Numerical / measurable evidence detected.")
    explanation: str = Field(..., description="Why this anomaly is a concern.")
    recommended_action: str = Field(..., description="Proposed corrective transformation.")
    potential_consequences: str = Field(..., description="Impact on downstream modeling or stats.")
    can_preview: bool = Field(default=True, description="True if a dry-run preview is supported.")
    is_resolved: bool = Field(default=False, description="True if issue has been addressed.")
    suggested_action_type: TransformationType | None = Field(
        default=None, description="Action type for the cleaning workbench."
    )
    suggested_action_params: dict[str, Any] = Field(
        default_factory=dict, description="Parameters to prefill the transformation workbench."
    )


class QualityScore(BaseModel):
    """Transparent multi-dimensional data quality score."""

    model_config = ConfigDict(frozen=True)

    overall_score: float = Field(..., ge=0.0, le=100.0, description="Composite quality score.")
    completeness_score: float = Field(
        ..., ge=0.0, le=100.0, description="Score based on missingness."
    )
    uniqueness_score: float = Field(
        ..., ge=0.0, le=100.0, description="Score based on row duplication."
    )
    consistency_score: float = Field(
        ..., ge=0.0, le=100.0, description="Score based on formatting consistency."
    )
    validity_score: float = Field(
        ..., ge=0.0, le=100.0, description="Score based on valid domains/outliers."
    )
    is_provisional: bool = Field(
        default=False, description="True if domain constraints are unconfigured."
    )
    score_breakdown: dict[str, Any] = Field(
        default_factory=dict, description="Details of formula calculation and weights."
    )
    explanation: str = Field(..., description="Human-readable summary of score computation.")


class TransformationSpec(BaseModel):
    """Declarative specification for an approved data cleaning operation."""

    model_config = ConfigDict(frozen=True)

    spec_id: str = Field(..., description="Unique specification ID.")
    action_type: TransformationType = Field(..., description="Type of operation to perform.")
    column_name: str | None = Field(default=None, description="Target column, if column-specific.")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Operation parameters.")
    description: str = Field(default="", description="Audit description of the transformation.")


class TransformationRecord(BaseModel):
    """Audit entry recording an executed transformation and its verified impact."""

    model_config = ConfigDict(frozen=True)

    record_id: str = Field(..., description="Unique ID for this executed record.")
    spec: TransformationSpec = Field(..., description="Transformation specification executed.")
    applied_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp when transformation ran.",
    )
    rows_before: int = Field(..., ge=0)
    rows_after: int = Field(..., ge=0)
    cols_before: int = Field(..., ge=0)
    cols_after: int = Field(..., ge=0)
    cells_modified: int = Field(default=0, ge=0)
    summary: str = Field(..., description="Summary of changes applied.")


class TransformationPreview(BaseModel):
    """Before/after dry-run preview of a proposed transformation."""

    action_type: TransformationType
    column_name: str | None
    rows_before: int
    rows_after: int
    cols_before: int
    cols_after: int
    affected_rows_count: int
    affected_cells_count: int
    before_sample: list[dict[str, Any]] = Field(default_factory=list)
    after_sample: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    summary: str


class DatasetProfile(BaseModel):
    """Comprehensive profiling report for an ingested dataset."""

    dataset_id: str
    row_count: int
    column_count: int
    memory_usage_bytes: int
    total_missing_cells: int
    total_cells: int
    missing_cell_percentage: float
    duplicate_row_count: int
    duplicate_row_percentage: float
    constant_columns: list[str] = Field(default_factory=list)
    near_constant_columns: list[str] = Field(default_factory=list)
    identifier_columns: list[str] = Field(default_factory=list)
    column_schemas: dict[str, ColumnSchema] = Field(default_factory=dict)
    numeric_profiles: dict[str, dict[str, Any]] = Field(default_factory=dict)
    categorical_profiles: dict[str, dict[str, Any]] = Field(default_factory=dict)
    datetime_profiles: dict[str, dict[str, Any]] = Field(default_factory=dict)
    quality_issues: list[QualityIssue] = Field(default_factory=list)
    quality_score: QualityScore
