"""Failure path and boundary condition tests (Prompt 8).

Verifies application robustness against malformed inputs, unsupported formats,
constant targets, invalid statistical requests, budget exhaustions, and missing credentials.
"""

from __future__ import annotations

import pandas as pd
import pytest
from src.agent.orchestration import (
    InvestigationOrchestrator,
    InvestigationStatus,
    ResourceBudget,
)
from src.agent.provider_factory import ProviderFactory
from src.analytics.group_comparisons import run_two_group_independent_test
from src.core.exceptions import DataIngestionError
from src.ingestion.loader import ManagedDataset, load_dataset_from_stream
from src.ingestion.models import DatasetIngestionReport
from src.ingestion.schema import infer_dataset_schema
from src.predictive import (
    PredictionTask,
    TargetSelector,
    train_predictive_models,
)
from src.reporting import (
    ExportService,
    ReportBuilder,
    ReportDocument,
    ReportEvidenceCollector,
    ReportType,
)


def test_malformed_empty_file_rejected():
    """Verify empty byte streams or malformed files raise DataIngestionError."""
    with pytest.raises(DataIngestionError, match="empty"):
        load_dataset_from_stream(
            file_stream=b"",
            filename="empty.csv",
        )


def test_unsupported_file_extension_rejected():
    """Verify unsupported extensions raise DataIngestionError."""
    with pytest.raises(DataIngestionError, match="Unsupported file format"):
        load_dataset_from_stream(
            file_stream=b"some content",
            filename="payload.exe",
        )


def test_target_selector_rejects_constant_and_sparse():
    """Verify constant target and all-null targets are rejected."""
    df_const = pd.DataFrame({"y": [1.0] * 25, "x": range(25)})
    rec_const = TargetSelector.evaluate_target(df_const, "y")
    assert rec_const.is_suitable is False
    assert "constant" in rec_const.reason.lower() or "unique" in rec_const.reason.lower()

    df_null = pd.DataFrame({"y": [None] * 25, "x": range(25)})
    rec_null = TargetSelector.evaluate_target(df_null, "y")
    assert rec_null.is_suitable is False


def test_two_group_test_missing_column_rejected():
    """Verify statistical inference fails cleanly when requested column does not exist."""
    df = pd.DataFrame({"col_a": [1, 2, 3], "col_b": ["x", "y", "x"]})
    with pytest.raises((KeyError, ValueError)):
        run_two_group_independent_test(
            df=df,
            group_col="non_existent_col",
            measure_col="col_a",
        )


def test_investigation_terminates_on_budget_exhaustion():
    """Verify orchestrator terminates cleanly when budget is strictly constrained."""
    df = pd.DataFrame({"revenue": [100.0, 200.0, 300.0], "spend": [10.0, 20.0, 30.0]})
    schema = infer_dataset_schema(df)
    report = DatasetIngestionReport(
        dataset_id="ds_budget",
        filename="budget.csv",
        file_format="csv",
        file_size_bytes=100,
        content_fingerprint="fp_budget",
        row_count=3,
        column_count=2,
    )
    managed = ManagedDataset(
        dataset_id="ds_budget",
        filename="budget.csv",
        file_format="csv",
        file_size_bytes=100,
        fingerprint="fp_budget",
        raw_df=df,
        working_df=df,
        schema_info=schema,
        ingestion_report=report,
    )

    provider = ProviderFactory.get_provider("mock")
    orchestrator = InvestigationOrchestrator(llm_provider=provider)
    state = orchestrator.plan_investigation(
        question="Check marketing",
        managed_dataset=managed,
        budget=ResourceBudget(max_steps=1, max_seconds=0.001),
    )
    result, pack = orchestrator.run_investigation(state=state, managed_dataset=managed)
    assert result is not None
    # State should terminate without crashing
    assert state.status in (
        InvestigationStatus.COMPLETED,
        InvestigationStatus.FAILED,
        InvestigationStatus.CANCELLED,
    )


def test_predictive_modeling_on_insufficient_samples():
    """Verify predictive training handles insufficient sample size gracefully."""
    df_tiny = pd.DataFrame({"y": [1.0, 2.0], "x": [10.0, 20.0]})
    with pytest.raises(ValueError, match="too small"):
        train_predictive_models(
            df=df_tiny,
            target_col="y",
            feature_cols=["x"],
            task_type=PredictionTask.REGRESSION,
            dataset_id="tiny_ds",
            dataset_version=1,
        )


def test_unperformed_predictive_analyses_reported_honestly():
    """Verify report builder does not hallucinate predictive models if none were trained."""
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})
    schema = infer_dataset_schema(df)
    report = DatasetIngestionReport(
        dataset_id="ds_unperformed",
        filename="unperformed.csv",
        file_format="csv",
        file_size_bytes=100,
        content_fingerprint="fp_unperformed",
        row_count=5,
        column_count=2,
    )
    managed = ManagedDataset(
        dataset_id="ds_unperformed",
        filename="unperformed.csv",
        file_format="csv",
        file_size_bytes=100,
        fingerprint="fp_unperformed",
        raw_df=df,
        working_df=df,
        schema_info=schema,
        ingestion_report=report,
    )

    collector = ReportEvidenceCollector(managed_dataset=managed, model_registry=None)
    bundle = collector.collect_all_evidence()
    builder = ReportBuilder(bundle)
    doc: ReportDocument = builder.build_report(report_type=ReportType.PREDICTIVE)

    # Predictive section must be marked was_performed=False
    pred_sec = next(s for s in doc.sections if s.section_id == "sec_predictive_modeling")
    assert pred_sec.was_performed is False
    assert (
        "not executed" in pred_sec.omission_reason.lower()
        or "not run" in pred_sec.omission_reason.lower()
    )


def test_export_service_pdf_graceful_error_handling(monkeypatch):
    """Verify ExportService catches PDF rendering errors and returns actionable fallback."""
    from src.reporting import export_service

    def mock_fail_pdf(doc):
        raise RuntimeError("Simulated missing system font error")

    monkeypatch.setattr(export_service, "render_report_to_pdf", mock_fail_pdf)

    df = pd.DataFrame({"x": [1, 2, 3]})
    schema = infer_dataset_schema(df)
    report = DatasetIngestionReport(
        dataset_id="ds_pdf_fail",
        filename="fail.csv",
        file_format="csv",
        file_size_bytes=50,
        content_fingerprint="fp_pdf_fail",
        row_count=3,
        column_count=1,
    )
    managed = ManagedDataset(
        dataset_id="ds_pdf_fail",
        filename="fail.csv",
        file_format="csv",
        file_size_bytes=50,
        fingerprint="fp_pdf_fail",
        raw_df=df,
        working_df=df,
        schema_info=schema,
        ingestion_report=report,
    )

    collector = ReportEvidenceCollector(managed_dataset=managed)
    doc = ReportBuilder(collector.collect_all_evidence()).build_report(ReportType.EXECUTIVE)

    pdf_bytes, err_msg = ExportService.export_pdf(doc)
    assert pdf_bytes is None
    assert err_msg is not None
    assert "PDF export encountered an error" in err_msg
    # HTML must still work!
    html_str = ExportService.export_html(doc)
    assert "<!DOCTYPE html>" in html_str
