"""End-to-end user journey integration tests.

Verifies the complete analytical workflow as a connected system:
1. Load synthetic dataset
2. Profile and inspect data quality
3. Apply an approved cleaning transformation
4. Calculate descriptive summaries and correlations
5. Perform inferential statistical testing
6. Generate evidence-grounded interpretation
7. Execute multi-step autonomous investigation
8. Train and evaluate predictive models
9. Assemble and export comprehensive intelligence report
10. Replace dataset and verify stale result invalidation
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from src.agent.analyst_service import AnalystService
from src.agent.orchestration import (
    InvestigationOrchestrator,
    ResourceBudget,
)
from src.agent.provider_factory import ProviderFactory
from src.analytics.group_comparisons import run_two_group_independent_test
from src.core.session_state import (
    KEY_ACTIVE_REPORT_DOC,
    KEY_ANALYSIS_RESULTS,
    KEY_CURRENT_DATASET,
    KEY_DATASET_METADATA,
    KEY_MODEL_REGISTRY,
    KEY_PREDICTIVE_RECORDS,
    init_session_state,
    invalidate_downstream_analyses,
    reset_dataset_state,
)
from src.ingestion.loader import ManagedDataset
from src.ingestion.models import DatasetIngestionReport
from src.ingestion.schema import infer_dataset_schema
from src.ingestion.transformation_history import TransformationPipeline
from src.predictive import ModelRegistry, PredictionTask, train_predictive_models
from src.profiling.models import TransformationSpec
from src.profiling.profiler import profile_dataset
from src.reporting import (
    ExportService,
    ReportBuilder,
    ReportDepth,
    ReportEvidenceCollector,
    ReportType,
)


def create_synthetic_dataset(seed: int = 42, n_rows: int = 120) -> ManagedDataset:
    """Helper to generate a realistic synthetic ManagedDataset."""
    np.random.seed(seed)
    categories = np.random.choice(["Tier_1", "Tier_2", "Tier_3"], size=n_rows, p=[0.5, 0.3, 0.2])
    marketing = np.random.uniform(500, 2000, size=n_rows)
    # Revenue influenced by marketing spend and tier
    tier_boost = np.where(categories == "Tier_1", 800, np.where(categories == "Tier_2", 400, 100))
    revenue = 2.4 * marketing + tier_boost + np.random.normal(0, 50, size=n_rows)

    # Introduce a few missing values and leading whitespace to test cleaning
    marketing_list = list(marketing)
    marketing_list[0] = np.nan
    marketing_list[1] = np.nan

    cat_list = [f"  {c}  " if i % 10 == 0 else c for i, c in enumerate(categories)]

    df = pd.DataFrame(
        {
            "revenue": revenue,
            "marketing_spend": marketing_list,
            "tier": cat_list,
        }
    )

    schema_info = infer_dataset_schema(df)
    report = DatasetIngestionReport(
        dataset_id=f"ds_{seed}",
        filename="company_financials.csv",
        file_format="csv",
        file_size_bytes=len(df.to_csv().encode()),
        content_fingerprint=f"fp_{seed}",
        row_count=len(df),
        column_count=len(df.columns),
    )
    ds = ManagedDataset(
        dataset_id=f"ds_{seed}",
        filename="company_financials.csv",
        file_format="csv",
        file_size_bytes=len(df.to_csv().encode()),
        fingerprint=f"fp_{seed}",
        raw_df=df,
        working_df=df,
        schema_info=schema_info,
        ingestion_report=report,
    )
    ds.version = 1
    return ds


def test_complete_end_to_end_analytical_journey():
    """Verify entire workflow across all 10 analytical milestones."""
    # -----------------------------------------------------------------
    # Step 1: Application startup & Session State initialization
    # -----------------------------------------------------------------
    state: dict = {}
    init_session_state(state)
    assert state[KEY_CURRENT_DATASET] is None

    # -----------------------------------------------------------------
    # Step 2 & 3: Ingestion and Quality Profiling
    # -----------------------------------------------------------------
    managed_ds = create_synthetic_dataset(seed=101, n_rows=120)
    state[KEY_CURRENT_DATASET] = managed_ds
    state[KEY_DATASET_METADATA] = {
        "dataset_id": managed_ds.dataset_id,
        "filename": managed_ds.filename,
        "version": managed_ds.version,
    }

    profile = profile_dataset(managed_ds.working_df, dataset_id=managed_ds.dataset_id)
    assert profile.quality_score is not None
    assert profile.quality_score.overall_score > 0
    assert profile.column_count == 3

    # -----------------------------------------------------------------
    # Step 4: Approved Cleaning Transformation
    # -----------------------------------------------------------------
    pipeline = TransformationPipeline()
    initial_version = managed_ds.version
    assert initial_version == 1

    # Impute numeric missing values in marketing_spend
    spec = TransformationSpec(
        spec_id="spec_impute_marketing",
        action_type="impute_numeric",
        column_name="marketing_spend",
        parameters={"strategy": "median"},
        description="Impute missing marketing spend with median",
    )
    rec = pipeline.apply(spec, managed_ds)
    assert rec.cells_modified == 2
    assert managed_ds.version == 2
    assert managed_ds.working_df["marketing_spend"].isna().sum() == 0

    # -----------------------------------------------------------------
    # Step 5: Statistical Inference
    # -----------------------------------------------------------------
    # Test revenue difference between Tier_1 and Tier_2
    df_clean = managed_ds.working_df.copy()
    df_clean["tier"] = df_clean["tier"].str.strip()

    stat_result = run_two_group_independent_test(
        df=df_clean,
        group_col="tier",
        measure_col="revenue",
        group1_name="Tier_1",
        group2_name="Tier_2",
    )
    assert stat_result.p_value < 0.05
    assert stat_result.effect_size.magnitude in ("large", "medium")

    state[KEY_ANALYSIS_RESULTS] = {
        "statistical_tests": [
            {
                "test_name": stat_result.test_name,
                "variables": ["tier", "revenue"],
                "statistic": stat_result.statistic_value,
                "p_value": stat_result.p_value,
                "sample_size": stat_result.total_n,
            }
        ]
    }

    # -----------------------------------------------------------------
    # Step 6 & 7: Evidence-Grounded AI Interpretation & Autonomous Investigation
    # -----------------------------------------------------------------
    provider = ProviderFactory.get_provider("mock")
    analyst_service = AnalystService(provider=provider)

    response, pack = analyst_service.analyze_question(
        question="What is the relationship between marketing spend and revenue?",
        managed_dataset=managed_ds,
    )
    assert response is not None
    assert pack is not None

    orchestrator = InvestigationOrchestrator(llm_provider=provider)
    inv_state = orchestrator.plan_investigation(
        question="Analyze the financial impact of marketing spend on revenue",
        managed_dataset=managed_ds,
        budget=ResourceBudget(max_steps=3, max_seconds=10.0),
    )
    inv_result, inv_pack = orchestrator.run_investigation(
        state=inv_state,
        managed_dataset=managed_ds,
    )
    assert inv_result is not None
    assert len(inv_state.current_plan.steps) > 0

    # -----------------------------------------------------------------
    # Step 8: Predictive Analytics Modeling & Registry
    # -----------------------------------------------------------------
    model_registry = ModelRegistry()
    state[KEY_MODEL_REGISTRY] = model_registry

    models = train_predictive_models(
        df=df_clean,
        target_col="revenue",
        feature_cols=["marketing_spend"],
        task_type=PredictionTask.REGRESSION,
        dataset_id=managed_ds.dataset_id,
        dataset_version=managed_ds.version,
    )
    assert len(models) >= 2
    best_model = models[0]
    assert best_model.regression_metrics is not None
    assert best_model.regression_metrics.rmse > 0
    model_registry.register_models(models)
    state[KEY_PREDICTIVE_RECORDS] = models

    # -----------------------------------------------------------------
    # Step 9: Report Generation & Multi-Format Export
    # -----------------------------------------------------------------
    collector = ReportEvidenceCollector(managed_dataset=managed_ds, model_registry=model_registry)
    bundle = collector.collect_all_evidence(
        investigation_trace=inv_state,
        statistical_results=state[KEY_ANALYSIS_RESULTS]["statistical_tests"],
    )
    builder = ReportBuilder(bundle)
    doc = builder.build_report(
        report_type=ReportType.COMPREHENSIVE, report_depth=ReportDepth.STANDARD
    )
    state[KEY_ACTIVE_REPORT_DOC] = doc

    assert doc.metadata.dataset_version == 2
    assert len(doc.sections) >= 3

    # HTML export
    html_output = ExportService.export_html(doc)
    assert doc.metadata.report_id in html_output
    assert "<!DOCTYPE html>" in html_output

    # PDF export
    pdf_bytes, pdf_err = ExportService.export_pdf(doc)
    assert pdf_err is None
    assert pdf_bytes is not None
    assert pdf_bytes.startswith(b"%PDF")

    # -----------------------------------------------------------------
    # Step 10: Downstream Invalidation & Dataset Replacement
    # -----------------------------------------------------------------
    # Test 10a: A further cleaning transformation invalidates downstream analyses
    spec_strip = TransformationSpec(
        spec_id="spec_strip_tier",
        action_type="strip_whitespace",
        column_name="tier",
        parameters={},
        description="Strip whitespace in tier",
    )
    pipeline.apply(spec_strip, managed_ds)
    assert managed_ds.version == 3

    invalidate_downstream_analyses(state)
    assert state[KEY_ANALYSIS_RESULTS] == {}
    assert state[KEY_ACTIVE_REPORT_DOC] is None
    assert state[KEY_PREDICTIVE_RECORDS] == []
    # Old models marked as stale
    stale_models = model_registry.list_models(dataset_id=managed_ds.dataset_id, active_version=3)
    assert all(m.is_stale for m in stale_models)

    # Test 10b: Replacing dataset completely resets state
    reset_dataset_state(state)
    assert state[KEY_CURRENT_DATASET] is None
    assert state[KEY_DATASET_METADATA] is None
    assert state[KEY_MODEL_REGISTRY] is None
