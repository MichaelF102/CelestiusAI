"""DAG Dependency Analysis and Plan Validation Engine.

Deterministically verifies that generated investigation plans:
1. Form a valid Directed Acyclic Graph (DAG) with no circular dependencies.
2. Only reference approved tools from AnalyticalToolRegistry.
3. Validate that all referenced columns exist in the active dataset.
4. Block any attempt to execute arbitrary code, shell commands, or SQL injections.
5. Respect configured maximum step and resource limits.
6. Return topological execution ordering for safe step scheduling.
"""

import logging
from collections import deque
from typing import Any

import pandas as pd

from src.agent.orchestration.orchestration_models import InvestigationPlan, InvestigationStep
from src.agent.tool_registry import AnalyticalToolRegistry
from src.ingestion.loader import ManagedDataset

logger = logging.getLogger(__name__)


class PlanValidationError(ValueError):
    """Raised when an investigation plan fails safety, structural, or semantic validation."""

    pass


class WorkflowGraphValidator:
    """Validates structural integrity, dependency graphs, and tool parameters of plans."""

    def __init__(self, tool_registry: AnalyticalToolRegistry | None = None) -> None:
        self.tool_registry = tool_registry or AnalyticalToolRegistry()

    def validate_plan(
        self,
        plan: InvestigationPlan,
        managed_dataset: ManagedDataset,
        max_allowed_steps: int = 12,
    ) -> None:
        """Validate investigation plan against dataset schema and safety rules.

        Raises:
            PlanValidationError: If plan is malformed, cyclical, exceeds limits, or violates safety.
        """
        if not plan.steps:
            raise PlanValidationError("Investigation plan must contain at least one step.")

        if len(plan.steps) > max_allowed_steps:
            raise PlanValidationError(
                f"Plan contains {len(plan.steps)} steps, exceeding limit of {max_allowed_steps}."
            )

        # Check dataset version match
        active_version = getattr(managed_dataset, "version", 1)
        if plan.dataset_version != active_version:
            raise PlanValidationError(
                f"Plan was generated for dataset v{plan.dataset_version}, but active version is v{active_version}."
            )

        df: pd.DataFrame = managed_dataset.working_df
        dataset_columns = set(df.columns)
        registered_tool_names = {t.name for t in self.tool_registry.get_tool_definitions()}
        step_ids = {s.step_id for s in plan.steps}

        if len(step_ids) != len(plan.steps):
            raise PlanValidationError("Duplicate step_ids detected in investigation plan.")

        for step in plan.steps:
            # 1. Check approved tool existence
            if step.tool_name not in registered_tool_names:
                raise PlanValidationError(
                    f"Step '{step.step_id}' ({step.step_name}) requests unregistered tool '{step.tool_name}'. "
                    f"Allowed tools: {sorted(registered_tool_names)}"
                )

            # 2. Block security injection vectors in tool arguments
            self._validate_argument_safety(step.step_id, step.tool_arguments)

            # 3. Validate column existence for argument references
            self._validate_column_references(step, dataset_columns)

            # 4. Check dependencies refer to valid prior step IDs
            for dep_id in step.dependencies:
                if dep_id not in step_ids:
                    raise PlanValidationError(
                        f"Step '{step.step_id}' has dependency on non-existent step '{dep_id}'."
                    )
                if dep_id == step.step_id:
                    raise PlanValidationError(
                        f"Step '{step.step_id}' cannot depend on itself (self-cycle)."
                    )

        # 5. Check DAG acyclicity and construct topological ordering
        self.get_topological_execution_order(plan)

    def get_topological_execution_order(
        self,
        plan: InvestigationPlan,
    ) -> list[InvestigationStep]:
        """Compute valid topological ordering of steps using Kahn's algorithm.

        Raises:
            PlanValidationError: If circular dependencies exist.
        """
        in_degree: dict[str, int] = {s.step_id: 0 for s in plan.steps}
        adj_list: dict[str, list[str]] = {s.step_id: [] for s in plan.steps}
        step_map = {s.step_id: s for s in plan.steps}

        for step in plan.steps:
            for dep_id in step.dependencies:
                adj_list[dep_id].append(step.step_id)
                in_degree[step.step_id] += 1

        queue: deque[str] = deque([sid for sid, deg in in_degree.items() if deg == 0])
        ordered: list[InvestigationStep] = []

        while queue:
            curr_id = queue.popleft()
            ordered.append(step_map[curr_id])

            for neighbor_id in adj_list[curr_id]:
                in_degree[neighbor_id] -= 1
                if in_degree[neighbor_id] == 0:
                    queue.append(neighbor_id)

        if len(ordered) != len(plan.steps):
            unresolved = [sid for sid, deg in in_degree.items() if deg > 0]
            raise PlanValidationError(
                f"Dependency cycle detected in plan involving steps: {unresolved}"
            )

        return ordered

    def _validate_argument_safety(self, step_id: str, args: dict[str, Any]) -> None:
        """Reject code injection or shell execution patterns."""
        forbidden_keywords = [
            "__import__",
            "eval(",
            "exec(",
            "subprocess",
            "os.system",
            "shutil",
            "select * from",
            "drop table",
            "delete from",
            "bash",
            "sh ",
            "/bin/",
        ]
        serialized = str(args).lower()
        for kw in forbidden_keywords:
            if kw in serialized:
                raise PlanValidationError(
                    f"Step '{step_id}' tool arguments contain forbidden injection keyword: '{kw}'"
                )

    def _validate_column_references(
        self, step: InvestigationStep, dataset_columns: set[str]
    ) -> None:
        """Ensure column arguments exist in dataset."""
        args = step.tool_arguments
        keys_to_check = [
            "column",
            "group_col",
            "measure_col",
            "outcome_col",
            "var1",
            "var2",
        ]
        for k in keys_to_check:
            if k in args and isinstance(args[k], str):
                col_name = args[k]
                if col_name not in dataset_columns:
                    raise PlanValidationError(
                        f"Step '{step.step_id}' argument '{k}={col_name}' is not in dataset columns."
                    )

        list_keys = ["columns", "predictor_cols"]
        for lk in list_keys:
            if lk in args and isinstance(args[lk], list):
                for col_name in args[lk]:
                    if isinstance(col_name, str) and col_name not in dataset_columns:
                        raise PlanValidationError(
                            f"Step '{step.step_id}' argument '{lk}' references missing column '{col_name}'."
                        )
