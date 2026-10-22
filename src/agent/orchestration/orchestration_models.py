"""Typed Pydantic models for the multi-step Agentic Investigation Engine.

Defines schemas for:
- Investigation lifecycle states and step types
- Investigation plans and steps with DAG dependency declarations
- Runtime investigation state, resource tracking, and state machine audit
- Final synthesis results linked to verified evidence
"""

import hashlib
import json
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from src.interpretation.interpretation_models import AnalystFindingClaim


class InvestigationStatus(StrEnum):
    """Lifecycle statuses for an analytical investigation."""

    CREATED = "CREATED"
    PLANNING = "PLANNING"
    PLAN_VALIDATION = "PLAN_VALIDATION"
    READY = "READY"
    RUNNING = "RUNNING"
    REVIEWING = "REVIEWING"
    REVISING = "REVISING"
    SYNTHESIZING = "SYNTHESIZING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"


class StepType(StrEnum):
    """Supported analytical step categories."""

    DATASET_INSPECTION = "dataset_inspection"
    DESCRIPTIVE_ANALYSIS = "descriptive_analysis"
    DATA_QUALITY_INVESTIGATION = "data_quality_investigation"
    GROUP_COMPARISON = "group_comparison"
    CORRELATION_ANALYSIS = "correlation_analysis"
    HYPOTHESIS_TESTING = "hypothesis_testing"
    REGRESSION_ANALYSIS = "regression_analysis"
    ROBUSTNESS_ANALYSIS = "robustness_analysis"
    EVIDENCE_VALIDATION = "evidence_validation"
    FOLLOW_UP_ANALYSIS = "follow_up_analysis"
    FINAL_SYNTHESIS = "final_synthesis"


class StepStatus(StrEnum):
    """Execution status for an individual investigation step."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    SUPERSEDED = "SUPERSEDED"


class InvestigationStep(BaseModel):
    """Atomic analytical step within an investigation plan."""

    step_id: str = Field(default_factory=lambda: f"step_{uuid.uuid4().hex[:6]}")
    step_name: str
    objective: str
    step_type: StepType
    dependencies: list[str] = Field(
        default_factory=list, description="IDs of steps that must succeed before this step runs"
    )
    required_evidence_ids: list[str] = Field(default_factory=list)
    tool_name: str = Field(description="Name of approved tool in AnalyticalToolRegistry")
    tool_arguments: dict[str, Any] = Field(default_factory=dict)
    expected_output_schema: str | None = None
    reason: str = Field(description="Analytical justification for why this step is necessary")
    status: StepStatus = StepStatus.PENDING
    attempt_count: int = 0
    actual_evidence_ids: list[str] = Field(default_factory=list)
    output_summary: str | None = None
    error_message: str | None = None
    warning_messages: list[str] = Field(default_factory=list)
    execution_time_ms: float = 0.0
    start_time: str | None = None
    end_time: str | None = None


class InvestigationPlan(BaseModel):
    """Structured, DAG-executable analytical investigation plan."""

    plan_id: str = Field(default_factory=lambda: f"plan_{uuid.uuid4().hex[:8]}")
    question: str
    normalized_objective: str
    dataset_id: str
    dataset_version: int
    assumptions: list[str] = Field(default_factory=list)
    known_limitations: list[str] = Field(default_factory=list)
    steps: list[InvestigationStep] = Field(default_factory=list)
    expected_evidence: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)
    max_steps: int = 10
    plan_version: int = 1
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def signature_hash(self) -> str:
        """Deterministic fingerprint of tools, steps, and parameters for loop detection."""
        items = [
            (
                s.tool_name,
                s.step_type.value,
                json.dumps(s.tool_arguments, sort_keys=True),
                sorted(s.dependencies),
            )
            for s in self.steps
        ]
        raw = json.dumps(items, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def get_step(self, step_id: str) -> InvestigationStep | None:
        """Retrieve step by its ID."""
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None


class ResourceBudget(BaseModel):
    """Configurable resource boundaries to prevent runaway agent execution."""

    max_steps: int = 12
    max_plan_revisions: int = 3
    max_llm_calls: int = 6
    max_tool_calls: int = 15
    max_elapsed_seconds: float = 120.0


class ResourceConsumption(BaseModel):
    """Real-time accounting of resource consumption."""

    steps_executed: int = 0
    plan_revisions: int = 0
    llm_calls: int = 0
    tool_calls: int = 0
    elapsed_seconds: float = 0.0

    def is_exceeded(self, budget: ResourceBudget) -> tuple[bool, str]:
        """Check whether any configured threshold has been breached."""
        if self.steps_executed >= budget.max_steps:
            return True, f"Maximum step count reached ({budget.max_steps})"
        if self.plan_revisions >= budget.max_plan_revisions:
            return True, f"Maximum plan revisions reached ({budget.max_plan_revisions})"
        if self.llm_calls >= budget.max_llm_calls:
            return True, f"Maximum LLM calls reached ({budget.max_llm_calls})"
        if self.tool_calls >= budget.max_tool_calls:
            return True, f"Maximum tool calls reached ({budget.max_tool_calls})"
        if self.elapsed_seconds >= budget.max_elapsed_seconds:
            return True, f"Maximum execution duration exceeded ({budget.max_elapsed_seconds:.0f}s)"
        return False, ""


class InvestigationState(BaseModel):
    """Dynamic, auditable state machine snapshot of an active investigation."""

    investigation_id: str = Field(default_factory=lambda: f"inv_{uuid.uuid4().hex[:8]}")
    status: InvestigationStatus = InvestigationStatus.CREATED
    current_plan: InvestigationPlan
    plan_history: list[InvestigationPlan] = Field(default_factory=list)
    current_step_id: str | None = None
    completed_step_ids: list[str] = Field(default_factory=list)
    failed_step_ids: list[str] = Field(default_factory=list)
    skipped_step_ids: list[str] = Field(default_factory=list)
    registered_evidence_ids: list[str] = Field(default_factory=list)
    findings: list[AnalystFindingClaim] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    budget: ResourceBudget = Field(default_factory=ResourceBudget)
    consumption: ResourceConsumption = Field(default_factory=ResourceConsumption)
    dataset_id: str
    dataset_version: int
    termination_reason: str | None = None
    transition_log: list[dict[str, Any]] = Field(default_factory=list)


class InvestigationResult(BaseModel):
    """Comprehensive final report synthesized from completed multi-step investigation."""

    investigation_id: str
    question: str
    summary: str
    executive_answer: str
    investigation_path: list[str] = Field(
        default_factory=list,
        description="Sequence of analytical actions executed and their rationale",
    )
    key_findings: list[AnalystFindingClaim] = Field(default_factory=list)
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    alternative_explanations: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    recommended_next_actions: list[str] = Field(default_factory=list)
    execution_trace: list[dict[str, Any]] = Field(default_factory=list)
    termination_reason: str
    status: InvestigationStatus
    total_duration_ms: float
    dataset_version: int
