"""Error Classification and Bounded Recovery Engine.

Distinguishes between:
1. Transient infrastructure errors (retryable with backoff).
2. Malformed plans or argument errors (repairable via planner).
3. Data constraint violations or assumption failures (triggers plan revision).
4. Non-recoverable security or policy violations (halt immediately).
"""

import logging
from enum import StrEnum

logger = logging.getLogger(__name__)


class ErrorCategory(StrEnum):
    """Categorization of failures during an investigation."""

    TRANSIENT_INFRASTRUCTURE = "transient_infrastructure"
    INVALID_TOOL_ARGUMENT = "invalid_tool_argument"
    DATA_ASSUMPTION_VIOLATION = "data_assumption_violation"
    MISSING_CAPABILITY = "missing_capability"
    SECURITY_VIOLATION = "security_violation"
    UNKNOWN_FATAL = "unknown_fatal"


class ErrorRecoveryClassifier:
    """Classifies errors and recommends actionable recovery strategies."""

    @staticmethod
    def classify_error(err: Exception | str) -> tuple[ErrorCategory, bool, str]:
        """Classify failure and return (Category, is_retryable, recommended_action)."""
        err_msg = str(err).lower()

        # 1. Security Violations
        if any(
            w in err_msg for w in ["forbidden", "injection", "permission denied", "unauthorized"]
        ):
            return (
                ErrorCategory.SECURITY_VIOLATION,
                False,
                "Halt execution immediately due to policy or safety constraint.",
            )

        # 2. Transient Infrastructure
        if any(
            w in err_msg
            for w in ["timeout", "timed out", "connection reset", "503", "rate limit", "429"]
        ):
            return (
                ErrorCategory.TRANSIENT_INFRASTRUCTURE,
                True,
                "Retry operation after exponential backoff delay (max 2 retries).",
            )

        # 3. Invalid tool arguments or column name typos
        if any(
            w in err_msg
            for w in [
                "does not exist in dataset",
                "did you mean",
                "missing required positional argument",
                "unexpected keyword",
            ]
        ):
            return (
                ErrorCategory.INVALID_TOOL_ARGUMENT,
                False,
                "Request constrained plan repair with corrected column identifiers.",
            )

        # 4. Data Assumption / Sample size insufficiency
        if any(
            w in err_msg
            for w in [
                "sample size",
                "degrees of freedom",
                "singular matrix",
                "insufficient data",
                "at least 2",
            ]
        ):
            return (
                ErrorCategory.DATA_ASSUMPTION_VIOLATION,
                False,
                "Revise plan to use non-parametric or alternate statistical specification.",
            )

        # 5. Missing Tool / Unregistered capability
        if any(w in err_msg for w in ["unregistered tool", "unknown tool"]):
            return (
                ErrorCategory.MISSING_CAPABILITY,
                False,
                "Prune step and explain capability boundary to user.",
            )

        return (
            ErrorCategory.UNKNOWN_FATAL,
            False,
            "Record diagnostic trace and proceed with surviving analytical branches.",
        )
