"""Central Investigation Orchestrator.

Orchestrates the entire multi-step analytical investigation lifecycle:
1. Translates questions into validated, DAG-ordered investigation plans.
2. Executes analytical steps through approved deterministic tools.
3. Adapts remaining steps dynamically based on intermediate empirical evidence.
4. Enforces strict resource budgets, DAG consistency, and loop prevention.
5. Verifies quantitative claims and synthesizes a comprehensive final investigation report.
"""

import logging
import time
from typing import Any

from src.agent.llm_provider import LLMProvider
from src.agent.orchestration.investigation_state import InvestigationStateMachine
from src.agent.orchestration.orchestration_models import (
    InvestigationPlan,
    InvestigationResult,
    InvestigationState,
    InvestigationStatus,
    ResourceBudget,
    StepStatus,
)
from src.agent.orchestration.plan_revision import PlanRevisionEngine
from src.agent.orchestration.planner import InvestigationPlanner
from src.agent.orchestration.step_executor import StepExecutor
from src.agent.orchestration.termination import TerminationEvaluator
from src.agent.orchestration.workflow_graph import WorkflowGraphValidator
from src.agent.tool_registry import AnalyticalToolRegistry
from src.ingestion.loader import ManagedDataset
from src.interpretation.claim_validation import ClaimValidator
from src.interpretation.contradiction_detection import ContradictionDetector
from src.interpretation.evidence_pack import EvidencePackBuilder
from src.interpretation.interpretation_models import (
    AnalystFindingClaim,
    EvidenceItem,
    EvidencePack,
)

logger = logging.getLogger(__name__)


class InvestigationOrchestrator:
    """End-to-end coordinator for agentic, multi-step analytical investigations."""

    def __init__(
        self,
        llm_provider: LLMProvider | None = None,
        tool_registry: AnalyticalToolRegistry | None = None,
    ) -> None:
        self.tool_registry = tool_registry or AnalyticalToolRegistry()
        self.llm_provider = llm_provider
        self.validator = WorkflowGraphValidator(self.tool_registry)
        self.planner = InvestigationPlanner(self.tool_registry, self.llm_provider)
        self.step_executor = StepExecutor(self.tool_registry)
        self.revision_engine = PlanRevisionEngine(self.validator)
        self.termination_evaluator = TerminationEvaluator()
        self.claim_validator = ClaimValidator()
        self.contradiction_detector = ContradictionDetector()

    def plan_investigation(
        self,
        question: str,
        managed_dataset: ManagedDataset,
        budget: ResourceBudget | None = None,
    ) -> InvestigationState:
        """Create and validate a new investigation plan, returning the READY state."""
        dataset_id = str(getattr(managed_dataset, "dataset_id", "dataset"))
        dataset_version = getattr(managed_dataset, "version", 1)
        active_budget = budget or ResourceBudget()

        # 1. Initialize state in CREATED
        initial_plan = InvestigationPlan(
            question=question,
            normalized_objective="Initializing analytical plan...",
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            steps=[],
        )
        state = InvestigationState(
            status=InvestigationStatus.CREATED,
            current_plan=initial_plan,
            budget=active_budget,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
        )

        # 2. Transition to PLANNING
        InvestigationStateMachine.transition_to(
            state,
            InvestigationStatus.PLANNING,
            reason="Decomposing question into investigation steps.",
        )

        plan = self.planner.create_plan(
            question=question,
            managed_dataset=managed_dataset,
            max_steps=active_budget.max_steps,
        )
        state.current_plan = plan
        state.consumption.llm_calls += 1

        # 3. Transition to PLAN_VALIDATION
        InvestigationStateMachine.transition_to(
            state,
            InvestigationStatus.PLAN_VALIDATION,
            reason="Verifying DAG dependencies and tool safety.",
        )
        self.validator.validate_plan(
            plan, managed_dataset, max_allowed_steps=active_budget.max_steps
        )

        # 4. Transition to READY
        InvestigationStateMachine.transition_to(
            state, InvestigationStatus.READY, reason="Plan validated and awaiting execution."
        )
        return state

    def run_investigation(
        self,
        state: InvestigationState,
        managed_dataset: ManagedDataset,
    ) -> tuple[InvestigationResult, EvidencePack]:
        """Execute the investigation plan through its full lifecycle to completion."""
        start_clock = time.perf_counter()
        dataset_version = getattr(managed_dataset, "version", 1)

        # 1. Transition to RUNNING
        InvestigationStateMachine.transition_to(
            state, InvestigationStatus.RUNNING, reason="Starting analytical step execution."
        )

        all_collected_evidence: list[EvidenceItem] = []
        execution_trace: list[dict[str, Any]] = []

        # Build initial base evidence pack
        evidence_pack = EvidencePackBuilder.build_pack(managed_dataset=managed_dataset)
        all_collected_evidence.extend(evidence_pack.items)

        while True:
            # Check elapsed time
            state.consumption.elapsed_seconds = round(time.perf_counter() - start_clock, 2)

            # Check stopping criteria and resource limits
            decision = self.termination_evaluator.evaluate_termination(state)
            if decision.should_terminate:
                logger.info(
                    "Termination policy triggered: %s (target: %s)",
                    decision.reason,
                    decision.target_status,
                )
                if decision.target_status != InvestigationStatus.SYNTHESIZING:
                    InvestigationStateMachine.transition_to(
                        state, decision.target_status, reason=decision.reason
                    )
                    break
                else:
                    # Proceed to synthesis
                    break

            # Find next pending step in topological order whose dependencies are satisfied
            plan = state.current_plan
            ordered_steps = self.validator.get_topological_execution_order(plan)
            next_step = next((s for s in ordered_steps if s.status == StepStatus.PENDING), None)

            if not next_step:
                break

            # Execute step
            raw_output, new_evidence = self.step_executor.execute_step(
                step=next_step,
                state=state,
                managed_dataset=managed_dataset,
            )

            execution_trace.append(
                {
                    "step_id": next_step.step_id,
                    "step_name": next_step.step_name,
                    "tool_name": next_step.tool_name,
                    "status": next_step.status.value,
                    "execution_time_ms": next_step.execution_time_ms,
                    "evidence_ids": next_step.actual_evidence_ids,
                    "error": next_step.error_message,
                }
            )

            if new_evidence:
                all_collected_evidence.extend(new_evidence)
                state.registered_evidence_ids.extend([e.id for e in new_evidence])

            # Transition to REVIEWING intermediate results
            InvestigationStateMachine.transition_to(
                state,
                InvestigationStatus.REVIEWING,
                reason=f"Completed step '{next_step.step_id}'. Reviewing evidence.",
            )

            # Check if mid-investigation plan adaptation is warranted
            needs_rev, rev_reason, revised_plan = self.revision_engine.evaluate_and_revise(
                state=state,
                managed_dataset=managed_dataset,
                latest_evidence=new_evidence,
            )
            if needs_rev and revised_plan:
                logger.info("Plan revision triggered: %s", rev_reason)
                InvestigationStateMachine.transition_to(
                    state, InvestigationStatus.REVISING, reason=rev_reason
                )
                state.plan_history.append(state.current_plan)
                state.current_plan = revised_plan
                state.consumption.plan_revisions += 1
                InvestigationStateMachine.transition_to(
                    state,
                    InvestigationStatus.PLAN_VALIDATION,
                    reason="Validating revised DAG plan.",
                )
                InvestigationStateMachine.transition_to(
                    state,
                    InvestigationStatus.RUNNING,
                    reason="Resuming execution with revised plan.",
                )
            else:
                InvestigationStateMachine.transition_to(
                    state,
                    InvestigationStatus.RUNNING,
                    reason="Continuing execution of active plan.",
                )

        # 2. Transition to SYNTHESIZING
        InvestigationStateMachine.transition_to(
            state,
            InvestigationStatus.SYNTHESIZING,
            reason="Synthesizing final evidence-grounded report.",
        )

        # Build complete unified EvidencePack
        final_pack = EvidencePackBuilder.build_pack(
            managed_dataset=managed_dataset,
            max_items=60,
        )
        for it in all_collected_evidence:
            if not final_pack.get_item(it.id):
                final_pack.items.append(it)

        # Produce verified synthesis
        result = self._synthesize_final_report(
            state=state,
            evidence_pack=final_pack,
            execution_trace=execution_trace,
            total_duration_ms=round((time.perf_counter() - start_clock) * 1000.0, 2),
            dataset_version=dataset_version,
        )

        # 3. Transition to terminal COMPLETED or FAILED
        final_status = (
            InvestigationStatus.COMPLETED
            if state.completed_step_ids
            else InvestigationStatus.FAILED
        )
        InvestigationStateMachine.transition_to(
            state, final_status, reason="Investigation synthesis complete."
        )
        result.status = final_status

        return result, final_pack

    def _synthesize_final_report(
        self,
        state: InvestigationState,
        evidence_pack: EvidencePack,
        execution_trace: list[dict[str, Any]],
        total_duration_ms: float,
        dataset_version: int,
    ) -> InvestigationResult:
        """Synthesize structured report with verified findings and claim validation."""
        plan = state.current_plan
        completed = [s for s in plan.steps if s.status == StepStatus.COMPLETED]
        failed = [s for s in plan.steps if s.status == StepStatus.FAILED]

        # Generate evidence-backed claims from completed steps
        claims: list[AnalystFindingClaim] = []
        for s in completed:
            if s.actual_evidence_ids:
                for eid in s.actual_evidence_ids:
                    item = evidence_pack.get_item(eid)
                    if item:
                        claims.append(
                            AnalystFindingClaim(
                                claim=f"{s.step_name}: Observed {item.metric_name} = {item.formatted_value}.",
                                evidence_ids=[item.id],
                                reported_values={item.metric_name: item.actual_value}
                                if isinstance(item.actual_value, (int, float, str))
                                else {},
                                relevant_variables=item.relevant_variables,
                                sample_size=item.sample_size,
                                method=item.calculation_method,
                                caveats=item.caveats,
                                validation_status="verified",
                                validation_notes=["Verified by step execution engine."],
                            )
                        )

        # Validate all claims
        validated_claims = self.claim_validator.validate_all(
            claims=claims,
            evidence_pack=evidence_pack,
            active_dataset_version=dataset_version,
        )

        investigation_path = [
            f"{s.step_name} [{s.tool_name}] -> {s.status.value}" for s in plan.steps
        ]

        exec_answer = (
            f"Investigation into '{plan.question}' completed with {len(completed)} successful steps "
            f"and {len(evidence_pack.items)} verified evidence items from dataset v{dataset_version}."
        )
        if failed:
            exec_answer += f" ({len(failed)} step(s) encountered execution warnings/failures)."

        # Limitations & Unresolved Questions
        limitations = list(plan.known_limitations)
        if failed:
            limitations.append(f"Steps failed during analysis: {[f.step_name for f in failed]}")
        limitations.append("Observational dataset cannot establish counterfactual causal claims.")

        unresolved: list[str] = []
        if any(s.status == StepStatus.FAILED for s in plan.steps):
            unresolved.append("Failed analytical steps left secondary hypotheses unexamined.")

        next_actions = [
            "Conduct longitudinal follow-up tracking to evaluate persistence over time.",
            "Gather granular transaction or covariate telemetry to test for latent confounding.",
        ]

        return InvestigationResult(
            investigation_id=state.investigation_id,
            question=plan.question,
            summary=f"Multi-step investigation executed across {len(plan.steps)} scheduled analytical steps.",
            executive_answer=exec_answer,
            investigation_path=investigation_path,
            key_findings=validated_claims,
            supporting_evidence_ids=[e.id for e in evidence_pack.items],
            alternative_explanations=[
                "Observed disparities may be driven by unobserved confounding factors.",
                "Non-random selection bias in customer or product sampling.",
            ],
            limitations=limitations,
            unresolved_questions=unresolved,
            recommended_next_actions=next_actions,
            execution_trace=execution_trace,
            termination_reason=state.termination_reason or "All scheduled steps evaluated.",
            status=state.status,
            total_duration_ms=total_duration_ms,
            dataset_version=dataset_version,
        )
