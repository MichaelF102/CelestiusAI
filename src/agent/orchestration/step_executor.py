"""Dependency-Aware Analytical Step Executor.

Executes atomic investigation steps against approved deterministic analytical tools,
enforces pre-execution dependency checks, manages step output caching, and extracts
verifiable evidence items for registry ingestion.
"""

import json
import logging
import time
from datetime import UTC, datetime
from typing import Any

from src.agent.orchestration.orchestration_models import (
    InvestigationState,
    InvestigationStep,
    StepStatus,
)
from src.agent.tool_registry import AnalyticalToolRegistry
from src.ingestion.loader import ManagedDataset
from src.interpretation.evidence_pack import EvidencePackBuilder
from src.interpretation.interpretation_models import EvidenceItem

logger = logging.getLogger(__name__)


class StepExecutionError(Exception):
    """Raised when an analytical tool step fails during execution."""

    pass


class StepExecutor:
    """Executes single investigation steps with dependency gating, caching, and evidence extraction."""

    def __init__(self, tool_registry: AnalyticalToolRegistry | None = None) -> None:
        self.tool_registry = tool_registry or AnalyticalToolRegistry()
        self._cache: dict[str, dict[str, Any]] = {}

    def clear_cache(self) -> None:
        """Clear execution cache."""
        self._cache.clear()

    def execute_step(
        self,
        step: InvestigationStep,
        state: InvestigationState,
        managed_dataset: ManagedDataset,
    ) -> tuple[dict[str, Any], list[EvidenceItem]]:
        """Execute a single step if its dependencies have succeeded.

        Returns:
            Tuple of (raw_tool_output_dict, list_of_new_evidence_items)
        """
        step.attempt_count += 1
        step.start_time = datetime.now(UTC).isoformat()
        start_clock = time.perf_counter()

        # 1. Dependency validation: Check that all prerequisite steps succeeded
        for dep_id in step.dependencies:
            if dep_id not in state.completed_step_ids:
                step.status = StepStatus.SKIPPED
                step.error_message = f"Prerequisite step '{dep_id}' has not succeeded (completed steps: {state.completed_step_ids})."
                step.end_time = datetime.now(UTC).isoformat()
                step.execution_time_ms = round((time.perf_counter() - start_clock) * 1000.0, 2)
                state.skipped_step_ids.append(step.step_id)
                logger.warning(
                    "Step '%s' skipped: prerequisite '%s' not satisfied.", step.step_id, dep_id
                )
                return {}, []

        # 2. Check Execution Cache for deterministic reuse
        cache_key = self._make_cache_key(state.dataset_version, step.tool_name, step.tool_arguments)
        if cache_key in self._cache:
            cached_result = self._cache[cache_key]
            duration_ms = (time.perf_counter() - start_clock) * 1000.0
            step.status = StepStatus.COMPLETED
            step.output_summary = "Loaded from deterministic step cache."
            step.end_time = datetime.now(UTC).isoformat()
            step.execution_time_ms = round(duration_ms, 2)
            state.completed_step_ids.append(step.step_id)
            state.consumption.steps_executed += 1

            new_evidence = self._extract_evidence(cached_result, state.dataset_version)
            step.actual_evidence_ids = [e.id for e in new_evidence]
            return cached_result, new_evidence

        # 3. Mark step running
        step.status = StepStatus.RUNNING
        state.current_step_id = step.step_id

        # 4. Invoke tool through AnalyticalToolRegistry
        try:
            raw_output = self.tool_registry.execute_tool(
                tool_name=step.tool_name,
                managed_dataset=managed_dataset,
                arguments=step.tool_arguments,
            )

            duration_ms = (time.perf_counter() - start_clock) * 1000.0
            step.status = StepStatus.COMPLETED
            step.end_time = datetime.now(UTC).isoformat()
            step.execution_time_ms = round(duration_ms, 2)
            step.output_summary = f"Executed '{step.tool_name}' successfully."

            # Cache the successful output
            self._cache[cache_key] = raw_output

            # 5. Extract Evidence Items
            new_evidence = self._extract_evidence(raw_output, state.dataset_version)
            step.actual_evidence_ids = [e.id for e in new_evidence]

            state.completed_step_ids.append(step.step_id)
            state.consumption.steps_executed += 1
            state.consumption.tool_calls += 1

            return raw_output, new_evidence

        except Exception as err:
            duration_ms = (time.perf_counter() - start_clock) * 1000.0
            step.status = StepStatus.FAILED
            step.error_message = str(err)
            step.end_time = datetime.now(UTC).isoformat()
            step.execution_time_ms = round(duration_ms, 2)

            state.failed_step_ids.append(step.step_id)
            state.consumption.steps_executed += 1
            state.consumption.tool_calls += 1
            logger.error("Step '%s' (%s) failed: %s", step.step_id, step.tool_name, err)
            return {}, []

    def _make_cache_key(self, version: int, tool_name: str, args: dict[str, Any]) -> str:
        """Construct deterministic cache key."""
        serialized_args = json.dumps(args, sort_keys=True)
        return f"v{version}:{tool_name}:{serialized_args}"

    def _extract_evidence(self, output: dict[str, Any], dataset_version: int) -> list[EvidenceItem]:
        """Convert tool output dictionary into typed EvidenceItem instances."""
        items: list[EvidenceItem] = []
        EvidencePackBuilder._absorb_tool_result(items, output, dataset_version)
        return items
