"""Ingestion data models and container specifications."""

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

InferredType = Literal[
    "numeric", "categorical", "datetime", "boolean", "identifier", "text", "unknown"
]


class ColumnSchema(BaseModel):
    """Detailed inferred schema and semantic metadata for a single column."""

    model_config = ConfigDict(frozen=True)

    column_name: str = Field(..., description="Original column name as parsed.")
    position: int = Field(..., ge=0, description="0-indexed position in dataset.")
    normalized_name: str = Field(..., description="Normalized snake_case identifier.")
    inferred_type: InferredType = Field(
        default="unknown", description="Inferred semantic data type."
    )
    detected_pandas_dtype: str = Field(..., description="Underlying Pandas dtype string.")
    non_null_count: int = Field(..., ge=0, description="Count of valid non-null records.")
    null_count: int = Field(..., ge=0, description="Count of missing/null values.")
    null_percentage: float = Field(
        ..., ge=0.0, le=100.0, description="Percentage of missing values."
    )
    distinct_count: int = Field(..., ge=0, description="Number of unique non-null values.")
    unique_ratio: float = Field(
        ..., ge=0.0, le=1.0, description="Distinct count divided by non-null count."
    )
    is_identifier: bool = Field(
        default=False, description="True if column resembles an ID, key, or code."
    )
    is_datetime_candidate: bool = Field(
        default=False, description="True if text can be safely parsed as datetime."
    )
    is_constant: bool = Field(default=False, description="True if only 1 distinct value exists.")
    is_near_constant: bool = Field(
        default=False, description="True if top value represents >95% of rows."
    )
    is_potential_target: bool = Field(
        default=False, description="True if column is a plausible prediction target."
    )
    sample_values: list[Any] = Field(default_factory=list, description="Sanitized preview values.")
    type_confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence score for inferred type."
    )
    rationale: str = Field(default="", description="Explanation for semantic classification.")
    proposed_conversion: str | None = Field(default=None, description="Suggested dtype conversion.")
    conversion_impact: str | None = Field(
        default=None, description="Expected impact of conversion."
    )


class DatasetIngestionReport(BaseModel):
    """Ingestion audit metadata and summary."""

    dataset_id: str
    filename: str
    file_format: str
    file_size_bytes: int
    content_fingerprint: str
    row_count: int
    column_count: int
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    delimiter_detected: str | None = None
    encoding_used: str | None = None
    sheet_name_used: str | None = None
    available_sheets: list[str] = Field(default_factory=list)
    has_duplicate_columns: bool = False
    duplicate_column_names: list[str] = Field(default_factory=list)
