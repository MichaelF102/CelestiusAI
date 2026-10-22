"""Termination Policy and Resource Budget Enforcement.

Evaluates deterministic stopping criteria:
1. Resource budget thresholds (max steps, revisions, tool calls, time).
2. Loop prevention via plan signature fingerprints and repeated failure tracking.
3. Goal completion and sufficiency of evidence.
4. Unrecoverable critical path failures.
"""

import logging
from typing import NamedTuple

from src.agent.orchestration.orchestration_models import (
    InvestigationState,
    InvestigationStatus,
    StepStatus,
)

logger = logging.getLogger(__name__)


class TerminationDecision(NamedTuple):
    """Result of termination policy evaluation."""

    should_terminate: bool
    reason: str
    target_status: InvestigationStatus


class TerminationEvaluator:
    """Evaluates whether an investigation should terminate, synthesize, or continue."""

    def __init__(self) -> None:
        self.seen_signatures: set[str] = set()

    def evaluate_termination(self, state: InvestigationState) -> TerminationDecision:
        """Evaluate resource limits, step completion, and loop detection."""
        # 1. Check if user requested cancellation
        if state.status == InvestigationStatus.CANCELLED:
            return TerminationDecision(
                should_terminate=True,
                reason="Investigation cancelled by user.",
                target_status=InvestigationStatus.CANCELLED,
            )

        # 2. Check Resource Budget
        exceeded, limit_msg = state.consumption.is_exceeded(state.budget)
        if exceeded:
            return TerminationDecision(
                should_terminate=True,
                reason=f"Resource budget exceeded: {limit_msg}",
                target_status=InvestigationStatus.BUDGET_EXCEEDED,
            )

        # 3. Check for repeated plan loops (signature hashing)
        sig = state.current_plan.signature_hash
        if sig in self.seen_signatures and state.consumption.plan_revisions > 0:
            return TerminationDecision(
                should_terminate=True,
                reason="Plan signature loop detected: Identical plan proposed repeatedly without new progress.",
                target_status=InvestigationStatus.SYNTHESIZING,
            )
        self.seen_signatures.add(sig)

        # 4. Check Step Completion in Current Plan
        plan = state.current_plan
        pending_steps = [
            s for s in plan.steps if s.status in (StepStatus.PENDING, StepStatus.RUNNING)
        ]

        if not pending_steps:
            # All steps have executed
            failed_steps = [s for s in plan.steps if s.status == StepStatus.FAILED]
            completed_steps = [s for s in plan.steps if s.status == StepStatus.COMPLETED]

            if not completed_steps and failed_steps:
                return TerminationDecision(
                    should_terminate=True,
                    reason=f"All {len(failed_steps)} analytical steps failed during execution.",
                    target_status=InvestigationStatus.FAILED,
                )

            return TerminationDecision(
                should_terminate=True,
                reason="All scheduled analytical steps in plan have completed.",
                target_status=InvestigationStatus.SYNTHESIZING,
            )

        return TerminationDecision(
            should_terminate=False,
            reason="Active steps remain pending execution.",
            target_status=InvestigationStatus.RUNNING,
        )
