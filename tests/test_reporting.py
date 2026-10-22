"""Unit tests for Intelligence Reporting Engine (Prompt 7).

Tests evidence collection, structured report assembly, omission handling,
deterministic synthesis, and multi-format export (HTML & PDF).
"""

from __future__ import annotations

import pandas as pd
import pytest
from src.ingestion.loader import ManagedDataset
from src.ingestion.models import DatasetIngestionReport
from src.ingestion.schema import infer_dataset_schema
from src.predictive import ModelRegistry, PredictionTask, train_predictive_models
from src.reporting import (
    ExportService,
    ReportBuilder,
    ReportDepth,
    ReportDocument,
    ReportEvidenceCollector,
    ReportType,
)


@pytest.fixture
def managed_dataset_fixture() -> ManagedDataset:
    """Fixture providing a managed dataset with verified metadata."""
    df = pd.DataFrame(
        {
            "revenue": [
                100.0,
                150.0,
                200.0,
                250.0,
                300.0,
                350.0,
                400.0,
                450.0,
                500.0,
                550.0,
                600.0,
                650.0,
                700.0,
                750.0,
                800.0,
                850.0,
            ],
            "spend": [
                10.0,
                20.0,
                15.0,
                30.0,
                25.0,
                40.0,
                35.0,
                50.0,
                45.0,
                60.0,
                55.0,
                70.0,
                65.0,
                80.0,
                75.0,
                90.0,
            ],
            "region": [
                "North",
                "South",
                "North",
                "South",
                "North",
                "South",
                "North",
                "South",
                "North",
                "South",
                "North",
                "South",
                "North",
                "South",
                "North",
                "South",
            ],
        }
    )
    schema_info = infer_dataset_schema(df)
    report = DatasetIngestionReport(
        dataset_id="test_reporting_ds",
        filename="sales_data.csv",
        file_format="csv",
        file_size_bytes=len(df.to_csv().encode()),
        content_fingerprint="abc123fingerprint",
        row_count=len(df),
        column_count=len(df.columns),
    )
    ds = ManagedDataset(
        dataset_id="test_reporting_ds",
        filename="sales_data.csv",
        file_format="csv",
        file_size_bytes=len(df.to_csv().encode()),
        fingerprint="abc123fingerprint",
        raw_df=df,
        working_df=df,
        schema_info=schema_info,
        ingestion_report=report,
    )
    ds.version = 1
    return ds


def test_report_evidence_collector(managed_dataset_fixture):
    registry = ModelRegistry()
    records = train_predictive_models(
        df=managed_dataset_fixture.working_df,
        target_col="revenue",
        feature_cols=["spend"],
        task_type=PredictionTask.REGRESSION,
        dataset_id=managed_dataset_fixture.dataset_id,
        dataset_version=managed_dataset_fixture.version,
    )
    registry.register_models(records)

    collector = ReportEvidenceCollector(
        managed_dataset=managed_dataset_fixture, model_registry=registry
    )
    bundle = collector.collect_all_evidence(
        statistical_results=[
            {
                "test_name": "Pearson Correlation",
                "variables": ["revenue", "spend"],
                "statistic": 0.95,
                "p_value": 0.0001,
                "sample_size": 10,
            }
        ]
    )

    assert bundle["dataset_id"] == "test_reporting_ds"
    assert bundle["dataset_version"] == 1
    assert len(bundle["evidence_items"]) > 0
    assert len(bundle["predictive_records"]) >= 1


def test_report_builder_comprehensive(managed_dataset_fixture):
    collector = ReportEvidenceCollector(managed_dataset=managed_dataset_fixture)
    bundle = collector.collect_all_evidence()

    builder = ReportBuilder(bundle)
    doc: ReportDocument = builder.build_report(
        report_type=ReportType.COMPREHENSIVE,
        report_depth=ReportDepth.STANDARD,
    )

    assert doc.metadata.title.startswith("Comprehensive Dataset Intelligence Report")
    assert doc.metadata.dataset_version == 1
    assert len(doc.sections) >= 3
    # Check that unperformed predictive modeling is reported honestly without hallucination
    pred_sec = next(s for s in doc.sections if s.section_id == "sec_predictive_modeling")
    assert pred_sec.was_performed is False
    assert "Predictive Modeling Evaluation" in doc.unperformed_analyses


def test_html_export(managed_dataset_fixture):
    collector = ReportEvidenceCollector(managed_dataset=managed_dataset_fixture)
    bundle = collector.collect_all_evidence()
    builder = ReportBuilder(bundle)
    doc = builder.build_report(ReportType.EXECUTIVE)

    html_out = ExportService.export_html(doc)
    assert "<!DOCTYPE html>" in html_out
    assert "sales_data.csv" in html_out
    assert "Executive Summary" in html_out
    assert doc.metadata.report_id in html_out


def test_pdf_export(managed_dataset_fixture):
    collector = ReportEvidenceCollector(managed_dataset=managed_dataset_fixture)
    bundle = collector.collect_all_evidence()
    builder = ReportBuilder(bundle)
    doc = builder.build_report(ReportType.EXECUTIVE)

    pdf_bytes, err = ExportService.export_pdf(doc)
    assert err is None
    assert pdf_bytes is not None
    # Check PDF magic header
    assert pdf_bytes.startswith(b"%PDF")
