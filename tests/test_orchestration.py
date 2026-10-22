"""Comprehensive test suite for Prompt 6: Agentic Multi-Step Analytical Orchestration Engine.

Tests:
1. Planning and plan validation (DAG checks, cycle detection, column safety, injection blocking)
2. Dependency-aware step execution and caching
3. Investigation state machine (valid transitions, illegal transitions, audit logging)
4. Adaptive plan revision and revision history tracking
5. Termination policies, resource budgets, and loop prevention
6. Error recovery classification
7. End-to-end multi-step orchestration (deterministic offline & mock LLM)
"""

import numpy as np
import pandas as pd
import pytest
from src.agent.llm_provider import MockLLMProvider
from src.agent.orchestration import (
    ErrorCategory,
    ErrorRecoveryClassifier,
    InvalidStateTransitionError,
    InvestigationOrchestrator,
    InvestigationPlan,
    InvestigationResult,
    InvestigationState,
    InvestigationStateMachine,
    InvestigationStatus,
    InvestigationStep,
    PlanRevisionEngine,
    PlanValidationError,
    ResourceBudget,
    StepExecutor,
    StepStatus,
    StepType,
    TerminationEvaluator,
    WorkflowGraphValidator,
)
from src.ingestion.loader import ManagedDataset
from src.ingestion.schema import infer_dataset_schema
from src.interpretation.interpretation_models import EvidenceItem


@pytest.fixture
def sample_dataset() -> ManagedDataset:
    """Create test dataset with known properties."""
    np.random.seed(42)
    n = 120
    df = pd.DataFrame(
        {
            "segment": np.random.choice(["Enterprise", "MidMarket", "SMB"], size=n),
            "revenue": np.random.normal(loc=12000, scale=2500, size=n),
            "discount": np.random.uniform(0.05, 0.40, size=n),
            "n_orders": np.random.randint(1, 20, size=n),
        }
    )
    schemas = infer_dataset_schema(df)
    md = ManagedDataset(
        dataset_id="test_ds_001",
        filename="sales_investigation.csv",
        file_format="csv",
        file_size_bytes=2048,
        fingerprint="fp_inv_123",
        raw_df=df.copy(),
        working_df=df.copy(),
        schema_info=schemas,
        ingestion_report=None,  # type: ignore[arg-type]
    )
    md.version = 1
    return md


# =============================================================================
# 1. PLAN VALIDATION & DAG GRAPH TESTS
# =============================================================================
def test_plan_validation_valid(sample_dataset: ManagedDataset):
    validator = WorkflowGraphValidator()

    step1 = InvestigationStep(
        step_id="s1",
        step_name="Inspect Overview",
        objective="Overview",
        step_type=StepType.DATASET_INSPECTION,
        dependencies=[],
        tool_name="get_dataset_overview",
        tool_arguments={},
        reason="Baseline",
    )
    step2 = InvestigationStep(
        step_id="s2",
        step_name="Compare Revenue",
        objective="Group compare",
        step_type=StepType.GROUP_COMPARISON,
        dependencies=["s1"],
        tool_name="compare_groups",
        tool_arguments={"group_col": "segment", "measure_col": "revenue"},
        reason="Check segment differences",
    )

    plan = InvestigationPlan(
        question="Why did revenue vary?",
        normalized_objective="Analyze revenue variation across segments",
        dataset_id=sample_dataset.dataset_id,
        dataset_version=1,
        steps=[step1, step2],
    )

    validator.validate_plan(plan, sample_dataset)
    ordered = validator.get_topological_execution_order(plan)
    assert [s.step_id for s in ordered] == ["s1", "s2"]


def test_plan_validation_detects_dependency_cycle(sample_dataset: ManagedDataset):
    validator = WorkflowGraphValidator()

    # Step A depends on Step B, and Step B depends on Step A
    s1 = InvestigationStep(
        step_id="s1",
        step_name="Step 1",
        objective="Obj 1",
        step_type=StepType.DATASET_INSPECTION,
        dependencies=["s2"],  # Cycle!
        tool_name="get_dataset_overview",
        tool_arguments={},
        reason="R1",
    )
    s2 = InvestigationStep(
        step_id="s2",
        step_name="Step 2",
        objective="Obj 2",
        step_type=StepType.DATASET_INSPECTION,
        dependencies=["s1"],  # Cycle!
        tool_name="get_dataset_overview",
        tool_arguments={},
        reason="R2",
    )

    plan = InvestigationPlan(
        question="Cycle test",
        normalized_objective="Cycle test",
        dataset_id=sample_dataset.dataset_id,
        dataset_version=1,
        steps=[s1, s2],
    )

    with pytest.raises(PlanValidationError, match="Dependency cycle detected"):
        validator.validate_plan(plan, sample_dataset)


def test_plan_validation_rejects_missing_column(sample_dataset: ManagedDataset):
    validator = WorkflowGraphValidator()

    s1 = InvestigationStep(
        step_id="s1",
        step_name="Bad Column Step",
        objective="Check non-existent column",
        step_type=StepType.DESCRIPTIVE_ANALYSIS,
        dependencies=[],
        tool_name="get_column_profile",
        tool_arguments={"column": "non_existent_column_xyz"},
        reason="Test",
    )
    plan = InvestigationPlan(
        question="Test",
        normalized_objective="Test",
        dataset_id=sample_dataset.dataset_id,
        dataset_version=1,
        steps=[s1],
    )

    with pytest.raises(PlanValidationError, match="not in dataset columns"):
        validator.validate_plan(plan, sample_dataset)


def test_plan_validation_rejects_injection_attempts(sample_dataset: ManagedDataset):
    validator = WorkflowGraphValidator()

    s1 = InvestigationStep(
        step_id="s1",
        step_name="Malicious Step",
        objective="Malicious execution",
        step_type=StepType.DATASET_INSPECTION,
        dependencies=[],
        tool_name="get_dataset_overview",
        tool_arguments={"param": "__import__('os').system('ls')"},
        reason="Attack",
    )
    plan = InvestigationPlan(
        question="Test",
        normalized_objective="Test",
        dataset_id=sample_dataset.dataset_id,
        dataset_version=1,
        steps=[s1],
    )

    with pytest.raises(PlanValidationError, match="forbidden injection keyword"):
        validator.validate_plan(plan, sample_dataset)


def test_plan_validation_rejects_unregistered_tool(sample_dataset: ManagedDataset):
    validator = WorkflowGraphValidator()

    s1 = InvestigationStep(
        step_id="s1",
        step_name="Arbitrary Tool Step",
        objective="Execute arbitrary tool",
        step_type=StepType.DATASET_INSPECTION,
        dependencies=[],
        tool_name="arbitrary_unregistered_tool",
        tool_arguments={},
        reason="Test",
    )
    plan = InvestigationPlan(
        question="Test",
        normalized_objective="Test",
        dataset_id=sample_dataset.dataset_id,
        dataset_version=1,
        steps=[s1],
    )

    with pytest.raises(PlanValidationError, match="unregistered tool"):
        validator.validate_plan(plan, sample_dataset)


# =============================================================================
# 2. STEP EXECUTOR & CACHING TESTS
# =============================================================================
def test_step_executor_success_and_cache(sample_dataset: ManagedDataset):
    executor = StepExecutor()

    state = InvestigationState(
        status=InvestigationStatus.RUNNING,
        current_plan=InvestigationPlan(
            question="Q",
            normalized_objective="O",
            dataset_id=sample_dataset.dataset_id,
            dataset_version=1,
            steps=[],
        ),
        dataset_id=sample_dataset.dataset_id,
        dataset_version=1,
    )

    step = InvestigationStep(
        step_id="s1",
        step_name="Overview",
        objective="Get dimensions",
        step_type=StepType.DATASET_INSPECTION,
        dependencies=[],
        tool_name="get_dataset_overview",
        tool_arguments={},
        reason="Overview",
    )

    # 1. First execution (computes and caches)
    out1, ev1 = executor.execute_step(step, state, sample_dataset)
    assert step.status == StepStatus.COMPLETED
    assert "s1" in state.completed_step_ids
    assert out1["row_count"] == 120

    # 2. Second execution with identical args (should hit cache)
    step2 = InvestigationStep(
        step_id="s2",
        step_name="Overview 2",
        objective="Get dimensions again",
        step_type=StepType.DATASET_INSPECTION,
        dependencies=[],
        tool_name="get_dataset_overview",
        tool_arguments={},
        reason="Overview",
    )
    out2, ev2 = executor.execute_step(step2, state, sample_dataset)
    assert step2.status == StepStatus.COMPLETED
    assert "cache" in step2.output_summary.lower()
    assert out2["row_count"] == 120


def test_step_executor_skips_when_dependency_not_completed(sample_dataset: ManagedDataset):
    executor = StepExecutor()

    state = InvestigationState(
        status=InvestigationStatus.RUNNING,
        current_plan=InvestigationPlan(
            question="Q",
            normalized_objective="O",
            dataset_id=sample_dataset.dataset_id,
            dataset_version=1,
            steps=[],
        ),
        completed_step_ids=[],  # Step "s1" NOT completed!
        dataset_id=sample_dataset.dataset_id,
        dataset_version=1,
    )

    step_downstream = InvestigationStep(
        step_id="s2",
        step_name="Downstream",
        objective="Depends on s1",
        step_type=StepType.GROUP_COMPARISON,
        dependencies=["s1"],
        tool_name="compare_groups",
        tool_arguments={"group_col": "segment", "measure_col": "revenue"},
        reason="Test",
    )

    out, ev = executor.execute_step(step_downstream, state, sample_dataset)
    assert step_downstream.status == StepStatus.SKIPPED
    assert "s2" in state.skipped_step_ids
    assert "prerequisite" in step_downstream.error_message.lower()


# =============================================================================
# 3. STATE MACHINE TRANSITION TESTS
# =============================================================================
def test_state_machine_valid_lifecycle(sample_dataset: ManagedDataset):
    state = InvestigationState(
        status=InvestigationStatus.CREATED,
        current_plan=InvestigationPlan(
            question="Q",
            normalized_objective="O",
            dataset_id=sample_dataset.dataset_id,
            dataset_version=1,
            steps=[],
        ),
        dataset_id=sample_dataset.dataset_id,
        dataset_version=1,
    )

    # CREATED -> PLANNING
    InvestigationStateMachine.transition_to(
        state, InvestigationStatus.PLANNING, reason="Start planning"
    )
    assert state.status == InvestigationStatus.PLANNING

    # PLANNING -> PLAN_VALIDATION
    InvestigationStateMachine.transition_to(
        state, InvestigationStatus.PLAN_VALIDATION, reason="Validate plan"
    )
    assert state.status == InvestigationStatus.PLAN_VALIDATION

    # PLAN_VALIDATION -> READY
    InvestigationStateMachine.transition_to(state, InvestigationStatus.READY, reason="Ready")
    assert state.status == InvestigationStatus.READY

    # READY -> RUNNING
    InvestigationStateMachine.transition_to(state, InvestigationStatus.RUNNING, reason="Start")
    assert state.status == InvestigationStatus.RUNNING

    # RUNNING -> REVIEWING
    InvestigationStateMachine.transition_to(
        state, InvestigationStatus.REVIEWING, reason="Review step 1"
    )
    assert state.status == InvestigationStatus.REVIEWING

    # REVIEWING -> SYNTHESIZING
    InvestigationStateMachine.transition_to(
        state, InvestigationStatus.SYNTHESIZING, reason="Synthesizing"
    )
    assert state.status == InvestigationStatus.SYNTHESIZING

    # SYNTHESIZING -> COMPLETED
    InvestigationStateMachine.transition_to(state, InvestigationStatus.COMPLETED, reason="Done")
    assert state.status == InvestigationStatus.COMPLETED

    assert len(state.transition_log) == 7


def test_state_machine_rejects_illegal_transition(sample_dataset: ManagedDataset):
    state = InvestigationState(
        status=InvestigationStatus.COMPLETED,
        current_plan=InvestigationPlan(
            question="Q",
            normalized_objective="O",
            dataset_id=sample_dataset.dataset_id,
            dataset_version=1,
            steps=[],
        ),
        dataset_id=sample_dataset.dataset_id,
        dataset_version=1,
    )

    # Illegal transition: COMPLETED -> RUNNING
    with pytest.raises(InvalidStateTransitionError, match="Illegal transition"):
        InvestigationStateMachine.transition_to(
            state, InvestigationStatus.RUNNING, reason="Illegal resume"
        )


# =============================================================================
# 4. ADAPTIVE PLAN REVISION TESTS
# =============================================================================
def test_plan_revision_appends_robustness_on_significant_disparity(sample_dataset: ManagedDataset):
    revision_engine = PlanRevisionEngine()

    step1 = InvestigationStep(
        step_id="step_1",
        step_name="Group comparison",
        objective="Compare segments",
        step_type=StepType.GROUP_COMPARISON,
        dependencies=[],
        tool_name="compare_groups",
        tool_arguments={"group_col": "segment", "measure_col": "revenue"},
        reason="Compare",
        status=StepStatus.COMPLETED,
    )

    plan = InvestigationPlan(
        question="Why did revenue vary?",
        normalized_objective="Segment variation",
        dataset_id=sample_dataset.dataset_id,
        dataset_version=1,
        steps=[step1],
    )

    state = InvestigationState(
        status=InvestigationStatus.REVIEWING,
        current_plan=plan,
        completed_step_ids=["step_1"],
        dataset_id=sample_dataset.dataset_id,
        dataset_version=1,
    )

    # Simulate significant evidence from step 1 (p = 0.002 < 0.05)
    sig_evidence = [
        EvidenceItem(
            id="ev_stat_segment_revenue_pvalue",
            evidence_type="group_comparison",
            dataset_version=1,
            source_service="analytics",
            calculation_method="welch",
            relevant_variables=["segment", "revenue"],
            metric_name="welch_p_value",
            actual_value=0.002,
            formatted_value="0.0020",
        )
    ]

    needs_rev, reason, revised_plan = revision_engine.evaluate_and_revise(
        state=state,
        managed_dataset=sample_dataset,
        latest_evidence=sig_evidence,
    )

    assert needs_rev is True
    assert "robustness" in reason.lower()
    assert revised_plan is not None
    assert len(revised_plan.steps) == 2
    assert revised_plan.steps[1].step_type == StepType.ROBUSTNESS_ANALYSIS
    assert revised_plan.steps[1].tool_name == "run_robustness_check"


# =============================================================================
# 5. TERMINATION EVALUATOR & LOOP PREVENTION TESTS
# =============================================================================
def test_termination_evaluator_budget_limit(sample_dataset: ManagedDataset):
    evaluator = TerminationEvaluator()

    plan = InvestigationPlan(
        question="Q",
        normalized_objective="O",
        dataset_id=sample_dataset.dataset_id,
        dataset_version=1,
        steps=[
            InvestigationStep(
                step_id="s1",
                step_name="S1",
                objective="O1",
                step_type=StepType.DATASET_INSPECTION,
                dependencies=[],
                tool_name="get_dataset_overview",
                tool_arguments={},
                reason="R",
            )
        ],
    )

    state = InvestigationState(
        status=InvestigationStatus.RUNNING,
        current_plan=plan,
        budget=ResourceBudget(max_steps=2),
        dataset_id=sample_dataset.dataset_id,
        dataset_version=1,
    )
    # Simulate step budget consumption
    state.consumption.steps_executed = 3

    decision = evaluator.evaluate_termination(state)
    assert decision.should_terminate is True
    assert decision.target_status == InvestigationStatus.BUDGET_EXCEEDED
    assert "Maximum step count" in decision.reason


def test_termination_evaluator_loop_detection(sample_dataset: ManagedDataset):
    evaluator = TerminationEvaluator()

    plan = InvestigationPlan(
        question="Q",
        normalized_objective="O",
        dataset_id=sample_dataset.dataset_id,
        dataset_version=1,
        steps=[
            InvestigationStep(
                step_id="s1",
                step_name="S1",
                objective="O1",
                step_type=StepType.DATASET_INSPECTION,
                dependencies=[],
                tool_name="get_dataset_overview",
                tool_arguments={},
                reason="R",
            )
        ],
    )

    state = InvestigationState(
        status=InvestigationStatus.REVIEWING,
        current_plan=plan,
        dataset_id=sample_dataset.dataset_id,
        dataset_version=1,
    )

    # First check registers signature
    d1 = evaluator.evaluate_termination(state)
    assert d1.should_terminate is False

    # Second check with identical signature after revision triggers loop detection
    state.consumption.plan_revisions = 1
    d2 = evaluator.evaluate_termination(state)
    assert d2.should_terminate is True
    assert "loop detected" in d2.reason.lower()


# =============================================================================
# 6. ERROR RECOVERY CLASSIFIER TESTS
# =============================================================================
def test_error_recovery_classifier():
    cat1, ret1, _ = ErrorRecoveryClassifier.classify_error(
        "HTTP 503: Service Unavailable connection reset"
    )
    assert cat1 == ErrorCategory.TRANSIENT_INFRASTRUCTURE
    assert ret1 is True

    cat2, ret2, _ = ErrorRecoveryClassifier.classify_error(
        "Column 'rev_typo' does not exist in dataset"
    )
    assert cat2 == ErrorCategory.INVALID_TOOL_ARGUMENT
    assert ret2 is False

    cat3, ret3, _ = ErrorRecoveryClassifier.classify_error(
        "Security violation: forbidden injection keyword 'eval('"
    )
    assert cat3 == ErrorCategory.SECURITY_VIOLATION
    assert ret3 is False


# =============================================================================
# 7. END-TO-END ORCHESTRATION TESTS
# =============================================================================
def test_orchestrator_deterministic_offline(sample_dataset: ManagedDataset):
    # Orchestrator with no LLM provider (deterministic offline mode)
    orchestrator = InvestigationOrchestrator(llm_provider=None)

    # 1. Plan investigation
    state = orchestrator.plan_investigation(
        question="Why did revenue vary across customer segments and discount levels?",
        managed_dataset=sample_dataset,
        budget=ResourceBudget(max_steps=6),
    )
    assert state.status == InvestigationStatus.READY
    assert len(state.current_plan.steps) >= 3

    # 2. Run investigation
    result, evidence_pack = orchestrator.run_investigation(state, sample_dataset)

    assert isinstance(result, InvestigationResult)
    assert result.status == InvestigationStatus.COMPLETED
    assert len(result.key_findings) > 0
    assert len(result.investigation_path) >= 3
    assert all(f.validation_status == "verified" for f in result.key_findings)
    assert result.total_duration_ms > 0
    assert len(evidence_pack.items) >= 4


def test_orchestrator_with_mock_llm(sample_dataset: ManagedDataset):
    mock_provider = MockLLMProvider()
    orchestrator = InvestigationOrchestrator(llm_provider=mock_provider)

    state = orchestrator.plan_investigation(
        question="Investigate revenue differences between segments",
        managed_dataset=sample_dataset,
    )
    assert state.status == InvestigationStatus.READY

    result, pack = orchestrator.run_investigation(state, sample_dataset)
    assert result.status == InvestigationStatus.COMPLETED
    assert result.executive_answer != ""
    assert len(pack.items) > 0
