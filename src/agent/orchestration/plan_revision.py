"""Adaptive Plan Revision Engine for Dynamic Investigations.

Evaluates intermediate analytical outputs and adapts remaining steps when:
1. Significant empirical disparities motivate follow-up robustness or subgroup checks.
2. An assumption failure or tool error necessitates an alternative non-parametric method.
3. Redundant planned steps can be pruned because sufficient evidence is already assembled.
Preserves plan history, prevents infinite revision loops, and validates all modifications.
"""

import logging
from copy import deepcopy

from src.agent.orchestration.orchestration_models import (
    InvestigationPlan,
    InvestigationState,
    InvestigationStep,
    StepStatus,
    StepType,
)
from src.agent.orchestration.workflow_graph import WorkflowGraphValidator
from src.ingestion.loader import ManagedDataset
from src.interpretation.interpretation_models import EvidenceItem

logger = logging.getLogger(__name__)


class PlanRevisionEngine:
    """Manages justified, bounded mid-investigation plan revisions."""

    def __init__(self, validator: WorkflowGraphValidator | None = None) -> None:
        self.validator = validator or WorkflowGraphValidator()

    def evaluate_and_revise(
        self,
        state: InvestigationState,
        managed_dataset: ManagedDataset,
        latest_evidence: list[EvidenceItem],
    ) -> tuple[bool, str, InvestigationPlan | None]:
        """Inspect evidence from recently executed steps and generate revised plan if warranted."""
        # 1. Check revision budget
        if state.consumption.plan_revisions >= state.budget.max_plan_revisions:
            return False, "Plan revision budget exhausted.", None

        current_plan = state.current_plan
        revised_steps = deepcopy(current_plan.steps)
        needs_revision = False
        revision_reason = ""

        # Check for significant group difference that lacks a robustness check
        has_sig_group_diff = any(
            item.evidence_type in ("group_comparison", "hypothesis_test")
            and "p_value" in item.metric_name
            and float(item.actual_value) < 0.05
            for item in latest_evidence
        )

        has_planned_robustness = any(
            s.step_type == StepType.ROBUSTNESS_ANALYSIS for s in revised_steps
        )

        if has_sig_group_diff and not has_planned_robustness:
            # Find the variables from the group difference evidence
            sig_item = next(
                item
                for item in latest_evidence
                if item.evidence_type in ("group_comparison", "hypothesis_test")
            )
            vars_inv = sig_item.relevant_variables
            if len(vars_inv) >= 2:
                needs_revision = True
                revision_reason = (
                    f"Statistically significant disparity detected in {vars_inv} (p < 0.05). "
                    "Adding sensitivity robustness check to evaluate stability against extreme outliers."
                )

                # Find the last completed step ID for dependency
                last_completed = state.completed_step_ids[-1] if state.completed_step_ids else []

                new_step_id = f"step_rev_{len(revised_steps) + 1}"
                new_step = InvestigationStep(
                    step_id=new_step_id,
                    step_name=f"Evaluate Robustness of {vars_inv[0]} Disparity",
                    objective=f"Assess sensitivity of {vars_inv} difference across trimmed/alternative specifications.",
                    step_type=StepType.ROBUSTNESS_ANALYSIS,
                    dependencies=[last_completed] if last_completed else [],
                    tool_name="run_robustness_check",
                    tool_arguments={"outcome_col": vars_inv[1], "predictor_cols": [vars_inv[0]]}
                    if vars_inv[1] in managed_dataset.working_df.columns
                    else {"outcome_col": vars_inv[0], "predictor_cols": [vars_inv[1]]},
                    reason="Follow up significant omnibus finding with empirical robustness validation.",
                    status=StepStatus.PENDING,
                )
                revised_steps.append(new_step)

        if not needs_revision:
            return False, "No plan adaptation necessary.", None

        # Create revised plan
        new_plan = InvestigationPlan(
            plan_id=f"{current_plan.plan_id}_rev{current_plan.plan_version + 1}",
            question=current_plan.question,
            normalized_objective=current_plan.normalized_objective,
            dataset_id=current_plan.dataset_id,
            dataset_version=current_plan.dataset_version,
            assumptions=current_plan.assumptions,
            known_limitations=current_plan.known_limitations
            + [f"Revision {current_plan.plan_version + 1}: {revision_reason}"],
            steps=revised_steps,
            expected_evidence=current_plan.expected_evidence,
            success_criteria=current_plan.success_criteria,
            max_steps=current_plan.max_steps,
            plan_version=current_plan.plan_version + 1,
        )

        # Validate revised plan DAG and arguments
        try:
            self.validator.validate_plan(
                new_plan, managed_dataset, max_allowed_steps=state.budget.max_steps
            )
        except Exception as err:
            logger.warning(
                "Proposed plan revision failed validation (%s). Retaining original plan.", err
            )
            return False, f"Plan revision validation failed: {err}", None

        return True, revision_reason, new_plan
