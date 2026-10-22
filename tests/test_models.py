"""Tests for foundational Pydantic models: validation, serialization, and immutability."""

from datetime import datetime

import pytest
from pydantic import ValidationError
from src.models.common import (
    AnalysisFinding,
    AnalysisPlan,
    AnalysisStep,
    DatasetMetadata,
)


def test_dataset_metadata_valid():
    """Verify valid DatasetMetadata creation and serialization."""
    meta = DatasetMetadata(
        filename="ecommerce_sales_2024.csv",
        file_format="csv",
        row_count=15420,
        column_count=12,
        dataset_id="ds_12345",
        file_size_bytes=1048576,
    )

    assert meta.filename == "ecommerce_sales_2024.csv"
    assert meta.file_format == "csv"
    assert meta.row_count == 15420
    assert meta.column_count == 12
    assert meta.dataset_id == "ds_12345"
    assert meta.file_size_bytes == 1048576
    assert isinstance(meta.created_at, datetime)

    data = meta.model_dump()
    assert data["row_count"] == 15420
    assert data["column_count"] == 12

    # Verify JSON round-trip
    json_str = meta.model_dump_json()
    assert "ecommerce_sales_2024.csv" in json_str
    reconstructed = DatasetMetadata.model_validate_json(json_str)
    assert reconstructed.filename == meta.filename


def test_dataset_metadata_validation_errors():
    """Verify invalid row/column counts or empty filename trigger ValidationError."""
    with pytest.raises(ValidationError):
        DatasetMetadata(
            filename="",
            file_format="csv",
            row_count=10,
            column_count=5,
        )

    with pytest.raises(ValidationError):
        DatasetMetadata(
            filename="data.csv",
            file_format="csv",
            row_count=-1,
            column_count=5,
        )


def test_analysis_finding_structure():
    """Verify AnalysisFinding adheres to the 6-layer interpretation model fields."""
    finding = AnalysisFinding(
        title="Revenue declined by 12% in Q3",
        summary="Decline observed across regional retail accounts due to discounting changes.",
        severity_or_importance="high",
        evidence_references=["table_rev_q3", "chart_regional_breakdown"],
        limitations=["Incomplete data for last week of September"],
        suggested_follow_up=["Audit discount rates applied by sales representatives"],
    )

    assert finding.title == "Revenue declined by 12% in Q3"
    assert finding.severity_or_importance == "high"
    assert len(finding.evidence_references) == 2
    assert len(finding.limitations) == 1
    assert len(finding.suggested_follow_up) == 1

    dump = finding.model_dump()
    assert dump["severity_or_importance"] == "high"


def test_analysis_finding_invalid_severity():
    """Verify invalid severity level raises ValidationError."""
    with pytest.raises(ValidationError):
        AnalysisFinding(
            title="Sample finding",
            summary="Sample summary",
            severity_or_importance="extreme",  # type: ignore[arg-type]
        )


def test_analysis_step_and_plan():
    """Verify AnalysisStep and AnalysisPlan integration and validation."""
    step1 = AnalysisStep(
        step_id="step_01",
        tool_name="profile_missingness",
        purpose="Evaluate column completeness before statistical tests",
        status="completed",
        result_reference="res_missingness_audit",
        execution_time_seconds=0.45,
    )

    step2 = AnalysisStep(
        step_id="step_02",
        tool_name="independent_t_test",
        purpose="Compare mean order value between Segment A and Segment B",
        status="pending",
    )

    plan = AnalysisPlan(
        objective="Investigate why conversion dropped across mobile users",
        steps=[step1, step2],
        expected_outputs=["missingness_report", "t_test_results"],
    )

    assert plan.objective == "Investigate why conversion dropped across mobile users"
    assert len(plan.steps) == 2
    assert plan.steps[0].status == "completed"
    assert plan.steps[1].status == "pending"

    # JSON serialization
    serialized = plan.model_dump_json()
    assert "profile_missingness" in serialized
    deserialized = AnalysisPlan.model_validate_json(serialized)
    assert len(deserialized.steps) == 2
    assert deserialized.steps[0].tool_name == "profile_missingness"
