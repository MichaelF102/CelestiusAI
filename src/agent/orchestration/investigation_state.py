"""Investigation State Machine and Transition Management.

Guarantees explicit, verifiable state transitions across the investigation lifecycle.
Rejects illegal transitions and maintains an immutable transition audit trail.
"""

import logging
from datetime import UTC, datetime
from typing import Any

from src.agent.orchestration.orchestration_models import (
    InvestigationState,
    InvestigationStatus,
)

logger = logging.getLogger(__name__)


class InvalidStateTransitionError(ValueError):
    """Raised when an illegal state transition is attempted in the investigation state machine."""

    pass


# Strict transition graph
ALLOWED_TRANSITIONS: dict[InvestigationStatus, set[InvestigationStatus]] = {
    InvestigationStatus.CREATED: {
        InvestigationStatus.PLANNING,
        InvestigationStatus.CANCELLED,
        InvestigationStatus.FAILED,
    },
    InvestigationStatus.PLANNING: {
        InvestigationStatus.PLAN_VALIDATION,
        InvestigationStatus.FAILED,
        InvestigationStatus.CANCELLED,
    },
    InvestigationStatus.PLAN_VALIDATION: {
        InvestigationStatus.READY,
        InvestigationStatus.REVISING,
        InvestigationStatus.FAILED,
        InvestigationStatus.CANCELLED,
    },
    InvestigationStatus.READY: {
        InvestigationStatus.RUNNING,
        InvestigationStatus.CANCELLED,
        InvestigationStatus.FAILED,
    },
    InvestigationStatus.RUNNING: {
        InvestigationStatus.REVIEWING,
        InvestigationStatus.SYNTHESIZING,
        InvestigationStatus.RUNNING,  # For continuing step loop
        InvestigationStatus.BUDGET_EXCEEDED,
        InvestigationStatus.FAILED,
        InvestigationStatus.CANCELLED,
    },
    InvestigationStatus.REVIEWING: {
        InvestigationStatus.REVISING,
        InvestigationStatus.SYNTHESIZING,
        InvestigationStatus.RUNNING,  # Next step in current plan
        InvestigationStatus.COMPLETED,
        InvestigationStatus.BUDGET_EXCEEDED,
        InvestigationStatus.FAILED,
        InvestigationStatus.CANCELLED,
    },
    InvestigationStatus.REVISING: {
        InvestigationStatus.PLAN_VALIDATION,
        InvestigationStatus.FAILED,
        InvestigationStatus.BUDGET_EXCEEDED,
        InvestigationStatus.CANCELLED,
    },
    InvestigationStatus.SYNTHESIZING: {
        InvestigationStatus.COMPLETED,
        InvestigationStatus.FAILED,
        InvestigationStatus.CANCELLED,
    },
    InvestigationStatus.BUDGET_EXCEEDED: {
        InvestigationStatus.SYNTHESIZING,
        InvestigationStatus.COMPLETED,
        InvestigationStatus.FAILED,
        InvestigationStatus.CANCELLED,
    },
    InvestigationStatus.COMPLETED: set(),
    InvestigationStatus.FAILED: set(),
    InvestigationStatus.CANCELLED: set(),
}


class InvestigationStateMachine:
    """Manages transition integrity and auditing for an active investigation."""

    @staticmethod
    def transition_to(
        state: InvestigationState,
        new_status: InvestigationStatus,
        reason: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Attempt state transition, verifying it against allowed transition rules."""
        current_status = state.status

        # If already in the target terminal state, no-op
        if current_status == new_status:
            return

        allowed = ALLOWED_TRANSITIONS.get(current_status, set())
        if new_status not in allowed:
            raise InvalidStateTransitionError(
                f"Illegal transition: Cannot transition investigation from '{current_status.value}' to '{new_status.value}'. "
                f"Allowed target states: {[s.value for s in allowed]}"
            )

        log_entry = {
            "from_state": current_status.value,
            "to_state": new_status.value,
            "reason": reason,
            "timestamp": datetime.now(UTC).isoformat(),
            "metadata": metadata or {},
        }
        state.transition_log.append(log_entry)
        state.status = new_status

        if new_status in (
            InvestigationStatus.COMPLETED,
            InvestigationStatus.FAILED,
            InvestigationStatus.CANCELLED,
            InvestigationStatus.BUDGET_EXCEEDED,
        ):
            state.termination_reason = reason or f"Terminated in state {new_status.value}"

        logger.info(
            "Investigation [%s] transitioned %s -> %s: %s",
            state.investigation_id,
            current_status.value,
            new_status.value,
            reason,
        )
