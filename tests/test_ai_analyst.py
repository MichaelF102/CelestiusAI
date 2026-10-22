"""Comprehensive test suite for Phase 5: Evidence-Grounded AI Interpretation Engine.

Validates:
1. Evidence pack construction and provenance
2. Provider abstraction (Gemini, Mock, missing credentials, error normalization)
3. Numerical claim validation and tolerance matching
4. Contradiction detection (causal fallacies, directional conflicts, significance mismatch)
5. Controlled analytical tool registry execution and audit
6. Question intent classification and conversational analytical memory
7. End-to-end AnalystService workflow with Mock provider
"""

import numpy as np
import pandas as pd
import pytest
from src.agent.analyst_service import AnalystService
from src.agent.context_builder import AnalyticalContextBuilder
from src.agent.llm_provider import GeminiProvider, MockLLMProvider
from src.agent.provider_factory import ProviderFactory
from src.agent.tool_registry import AnalyticalToolRegistry
from src.config.settings import AppSettings
from src.ingestion.loader import ManagedDataset
from src.ingestion.schema import infer_dataset_schema
from src.interpretation.claim_validation import ClaimValidator
from src.interpretation.contradiction_detection import ContradictionDetector
from src.interpretation.evidence_pack import EvidencePackBuilder
from src.interpretation.hypothesis_generator import HypothesisGenerator
from src.interpretation.interpretation_models import (
    AnalystFindingClaim,
    AnalystResponse,
    EvidenceItem,
    EvidencePack,
    TraceMetadata,
)


@pytest.fixture
def sample_managed_dataset() -> ManagedDataset:
    """Fixture providing a synthetic dataset for testing."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame(
        {
            "segment": np.random.choice(["Enterprise", "SMB"], size=n),
            "revenue": np.random.normal(loc=5000, scale=1000, size=n),
            "discount": np.random.uniform(0.05, 0.30, size=n),
            "satisfaction": np.random.randint(1, 10, size=n),
        }
    )
    schemas = infer_dataset_schema(df)
    md = ManagedDataset(
        dataset_id="test_customers_id",
        filename="test_customers.csv",
        file_format="csv",
        file_size_bytes=1024,
        fingerprint="fp_test_123",
        raw_df=df.copy(),
        working_df=df.copy(),
        schema_info=schemas,
        ingestion_report=None,  # type: ignore[arg-type]
    )
    md.version = 1
    return md


# =============================================================================
# 1. Evidence Pack Tests
# =============================================================================
def test_evidence_pack_builder_basic(sample_managed_dataset: ManagedDataset):
    pack = EvidencePackBuilder.build_pack(
        managed_dataset=sample_managed_dataset,
        relevant_columns=["revenue", "segment"],
    )

    assert isinstance(pack, EvidencePack)
    assert pack.dataset_name == "test_customers.csv"
    assert pack.dataset_version == 1
    assert pack.row_count == 100
    assert pack.column_count == 4

    # Check that dimension items exist
    assert pack.get_item("ev_dim_rows") is not None
    assert pack.get_item("ev_dim_rows").actual_value == 100
    assert pack.get_item("ev_dim_cols").actual_value == 4

    # Check descriptive evidence
    rev_mean = pack.get_item("ev_desc_revenue_mean")
    assert rev_mean is not None
    assert rev_mean.evidence_type == "descriptive_statistic"
    assert rev_mean.sample_size == 100
    assert isinstance(rev_mean.actual_value, float)

    # Check categorical evidence
    seg_mode = pack.get_item("ev_cat_segment_mode")
    assert seg_mode is not None
    assert seg_mode.sample_size == 100

    # Summary for prompt should be formatted text
    prompt_str = pack.summary_for_prompt()
    assert "EVIDENCE PACK" in prompt_str
    assert "revenue_mean" in prompt_str


# =============================================================================
# 2. Provider Abstraction Tests
# =============================================================================
def test_mock_llm_provider():
    provider = MockLLMProvider(model_name="mock-test-model")
    assert provider.provider_name == "mock"
    assert provider.is_configured is True

    avail, msg = provider.check_availability()
    assert avail is True

    # Text generation
    res = provider.generate_text("Explain revenue trend.")
    assert res.status == "success"
    assert "Mock Analysis" in res.raw_text
    assert res.execution_time_ms >= 0

    # Structured generation
    obj, res_struct = provider.generate_structured(
        prompt="Analyze dataset",
        response_model=AnalystResponse,
    )
    assert res_struct.status == "success"
    assert isinstance(obj, AnalystResponse)
    assert obj.question != ""


def test_gemini_provider_unconfigured():
    provider = GeminiProvider(api_key=None, model_name="gemini-1.5-pro")
    assert provider.provider_name == "gemini"
    assert provider.is_configured is False

    avail, msg = provider.check_availability()
    assert avail is False
    assert "not configured" in msg

    res = provider.generate_text("Test prompt")
    assert res.status == "not_configured"

    obj, res_struct = provider.generate_structured("Test prompt", AnalystResponse)
    assert obj is None
    assert res_struct.status == "not_configured"


def test_provider_factory():
    settings = AppSettings(gemini_api_key=None)
    prov = ProviderFactory.get_provider(provider_type="mock", settings=settings)
    assert isinstance(prov, MockLLMProvider)

    auto_prov = ProviderFactory.get_provider(provider_type="auto", settings=settings)
    assert isinstance(auto_prov, GeminiProvider)
    assert auto_prov.is_configured is False


# =============================================================================
# 3. Claim Validation Tests
# =============================================================================
def test_claim_validator(sample_managed_dataset: ManagedDataset):
    pack = EvidencePackBuilder.build_pack(sample_managed_dataset, relevant_columns=["revenue"])
    mean_val = float(pack.get_item("ev_desc_revenue_mean").actual_value)

    validator = ClaimValidator()

    # Case A: Valid verified claim
    valid_claim = AnalystFindingClaim(
        claim=f"Average revenue is {mean_val:.2f}.",
        evidence_ids=["ev_desc_revenue_mean"],
        reported_values={"mean": mean_val},
    )
    validated = validator.validate_claim(valid_claim, pack, active_dataset_version=1)
    assert validated.validation_status == "verified"

    # Case B: Nonexistent evidence ID
    invalid_id_claim = AnalystFindingClaim(
        claim="Average profit is 200.",
        evidence_ids=["ev_nonexistent_metric"],
    )
    validated = validator.validate_claim(invalid_id_claim, pack, active_dataset_version=1)
    assert validated.validation_status == "invalid_reference"

    # Case C: Mismatched numerical value
    mismatched_claim = AnalystFindingClaim(
        claim="Average revenue is 9999999.00.",
        evidence_ids=["ev_desc_revenue_mean"],
        reported_values={"mean": 9999999.0},
    )
    validated = validator.validate_claim(mismatched_claim, pack, active_dataset_version=1)
    assert validated.validation_status == "mismatched"

    # Case D: Stale dataset version
    stale_pack = EvidencePack(
        dataset_id="test",
        dataset_name="test",
        dataset_version=1,
        row_count=100,
        column_count=4,
        items=[
            EvidenceItem(
                id="ev_stale_1",
                evidence_type="descriptive_statistic",
                dataset_version=1,
                source_service="test",
                calculation_method="test",
                metric_name="metric",
                actual_value=10.0,
                formatted_value="10.0",
            )
        ],
    )
    claim_stale = AnalystFindingClaim(
        claim="Metric is 10.0.",
        evidence_ids=["ev_stale_1"],
        reported_values={"metric": 10.0},
    )
    # Active version is 2, but evidence is version 1
    validated = validator.validate_claim(claim_stale, stale_pack, active_dataset_version=2)
    assert validated.validation_status == "stale_version"


# =============================================================================
# 4. Contradiction Detection Tests
# =============================================================================
def test_contradiction_detection():
    detector = ContradictionDetector()

    pack = EvidencePack(
        dataset_id="test",
        dataset_name="test",
        dataset_version=1,
        row_count=100,
        column_count=2,
        items=[
            EvidenceItem(
                id="ev_stat_test_pvalue",
                evidence_type="hypothesis_test",
                dataset_version=1,
                source_service="stat",
                calculation_method="welch",
                metric_name="welch_p_value",
                actual_value=0.45,  # Not significant!
                formatted_value="0.4500",
            ),
            EvidenceItem(
                id="ev_corr_rev_disc_r",
                evidence_type="correlation",
                dataset_version=1,
                source_service="corr",
                calculation_method="pearson",
                metric_name="pearson_r",
                actual_value=-0.35,  # Negative!
                formatted_value="-0.3500",
            ),
        ],
    )

    # Conflict 1: Claiming significance when p = 0.45
    response_sig_conflict = AnalystResponse(
        question="Is it significant?",
        intent_category="hypothesis_testing",
        direct_answer="Yes, there is a statistically significant effect.",
        key_findings=[
            AnalystFindingClaim(
                claim="There is a significant difference between groups.",
                evidence_ids=["ev_stat_test_pvalue"],
            )
        ],
        interpretation="The test confirms significance.",
        trace_metadata=TraceMetadata(provider="test", model="test", execution_time_ms=5.0),
        dataset_version=1,
    )
    flags = detector.detect_contradictions(response_sig_conflict, pack, 1)
    assert any(f.severity == "critical" and "p-value" in f.description for f in flags)

    # Conflict 2: Claiming positive increase when correlation is negative
    response_dir_conflict = AnalystResponse(
        question="How does discount affect revenue?",
        intent_category="correlation",
        direct_answer="Higher discounts lead to an increase in revenue.",
        key_findings=[
            AnalystFindingClaim(
                claim="Revenue shows a strong positive increase with discount.",
                evidence_ids=["ev_corr_rev_disc_r"],
            )
        ],
        interpretation="Discounts cause higher revenue.",
        trace_metadata=TraceMetadata(provider="test", model="test", execution_time_ms=5.0),
        dataset_version=1,
    )
    flags_dir = detector.detect_contradictions(response_dir_conflict, pack, 1)
    # Checks directional conflict or causal fallacy
    assert len(flags_dir) >= 1
    assert any(
        "negative" in f.description.lower() or "caus" in f.description.lower() for f in flags_dir
    )


# =============================================================================
# 5. Tool Registry Tests
# =============================================================================
def test_analytical_tool_registry(sample_managed_dataset: ManagedDataset):
    registry = AnalyticalToolRegistry()

    # 1. Overview
    overview = registry.execute_tool("get_dataset_overview", sample_managed_dataset)
    assert overview["row_count"] == 100
    assert overview["column_count"] == 4

    # 2. Descriptive statistics
    desc = registry.execute_tool(
        "calculate_descriptive_statistics",
        sample_managed_dataset,
        {"columns": ["revenue"]},
    )
    assert "revenue" in desc
    assert desc["revenue"]["mean"] > 0

    # 3. Column profile
    prof = registry.execute_tool(
        "get_column_profile",
        sample_managed_dataset,
        {"column": "revenue"},
    )
    assert prof["non_null_count"] == 100

    # 4. Compare groups
    comp = registry.execute_tool(
        "compare_groups",
        sample_managed_dataset,
        {"group_col": "segment", "measure_col": "revenue"},
    )
    assert "statistic_value" in comp
    assert "p_value" in comp

    # 5. Correlation
    corr = registry.execute_tool(
        "calculate_correlation",
        sample_managed_dataset,
        {"var1": "revenue", "var2": "discount"},
    )
    assert "statistic_value" in corr

    # 6. Fit regression
    reg = registry.execute_tool(
        "fit_regression",
        sample_managed_dataset,
        {"outcome_col": "revenue", "predictor_cols": ["discount", "satisfaction"]},
    )
    assert "r_squared" in reg
    assert len(reg["coefficients"]) == 3  # Intercept + 2 features

    # 7. Audit log check
    assert len(registry.audit_log) >= 6
    assert all(audit.success for audit in registry.audit_log)

    # 8. Invalid column rejection with fuzzy suggestion
    with pytest.raises(ValueError, match="does not exist in dataset"):
        registry.execute_tool(
            "get_column_profile",
            sample_managed_dataset,
            {"column": "revenu_typo"},
        )

    # 9. Blocked/Unknown tool
    with pytest.raises(ValueError, match="Unknown tool"):
        registry.execute_tool("arbitrary_python_exec", sample_managed_dataset)


# =============================================================================
# 6. Context Builder & Conversational Memory Tests
# =============================================================================
def test_context_builder_classification_and_memory(sample_managed_dataset: ManagedDataset):
    builder = AnalyticalContextBuilder(max_history_turns=3)

    # Classification
    q1 = "Compare revenue between segment groups"
    c1 = builder.classify_question(q1, sample_managed_dataset)
    assert c1.intent == "group_comparison"
    assert "revenue" in c1.relevant_columns
    assert "segment" in c1.relevant_columns
    assert c1.requires_group_comparison is True

    # Correlation classification
    q2 = "Is there a correlation between revenue and discount?"
    c2 = builder.classify_question(q2, sample_managed_dataset)
    assert c2.intent == "correlation_association"
    assert c2.requires_correlation is True

    # Mock recording of turn
    mock_resp = AnalystResponse(
        question=q1,
        intent_category=c1.intent,
        direct_answer="Revenue differs slightly across segments.",
        trace_metadata=TraceMetadata(provider="mock", model="test", execution_time_ms=1.0),
        dataset_version=1,
    )
    builder.record_turn(q1, c1.intent, 1, mock_resp, c1.relevant_columns)
    assert len(builder.history) == 1

    # Follow-up detection
    q_followup = "Why might this difference exist?"
    c_follow = builder.classify_question(q_followup, sample_managed_dataset)
    assert c_follow.is_followup is True
    # Inherited columns from previous turn
    assert "revenue" in c_follow.relevant_columns


# =============================================================================
# 7. Hypothesis Generation Tests
# =============================================================================
def test_hypothesis_generator(sample_managed_dataset: ManagedDataset):
    pack = EvidencePackBuilder.build_pack(
        sample_managed_dataset, relevant_columns=["revenue", "discount"]
    )
    hyps = HypothesisGenerator.generate_hypotheses_from_evidence(pack)
    assert len(hyps) >= 1
    assert hyps[0].is_exploratory is True
    assert hyps[0].suggested_method != ""
    assert hyps[0].expected_support_evidence != ""


# =============================================================================
# 8. End-to-End AnalystService Workflow Tests
# =============================================================================
def test_analyst_service_deterministic_fallback(sample_managed_dataset: ManagedDataset):
    # Service configured with unconfigured GeminiProvider -> falls back gracefully
    unconfigured_provider = GeminiProvider(api_key=None)
    service = AnalystService(provider=unconfigured_provider)

    resp, pack = service.analyze_question(
        question="Compare revenue across segment groups",
        managed_dataset=sample_managed_dataset,
    )

    assert isinstance(resp, AnalystResponse)
    assert isinstance(pack, EvidencePack)
    assert resp.dataset_version == 1
    assert (
        "Deterministic Analytics mode" in resp.interpretation or "Analysis of" in resp.direct_answer
    )
    assert len(resp.key_findings) > 0
    assert all(f.validation_status == "verified" for f in resp.key_findings)
    assert len(resp.hypotheses) > 0


def test_analyst_service_with_mock_provider(sample_managed_dataset: ManagedDataset):
    mock_provider = MockLLMProvider()
    service = AnalystService(provider=mock_provider)

    resp, pack = service.analyze_question(
        question="What is the distribution of revenue and discount?",
        managed_dataset=sample_managed_dataset,
    )

    assert isinstance(resp, AnalystResponse)
    assert resp.trace_metadata.provider == "mock"
    assert len(pack.items) > 0
    assert pack.get_item("ev_dim_rows") is not None
