"""Analytical Planning Engine for Multi-Step Investigations.

Translates complex natural language analytical questions into structured,
DAG-executable InvestigationPlan instances. Supports both LLM-driven structured
planning and deterministic rule-based planning for offline/credential-free modes.
"""

import logging
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from src.agent.context_builder import AnalyticalContextBuilder, QuestionClassification
from src.agent.llm_provider import LLMProvider
from src.agent.orchestration.orchestration_models import (
    InvestigationPlan,
    InvestigationStep,
    StepStatus,
    StepType,
)
from src.agent.orchestration.workflow_graph import PlanValidationError, WorkflowGraphValidator
from src.agent.tool_registry import AnalyticalToolRegistry
from src.ingestion.loader import ManagedDataset
from src.interpretation.interpretation_models import EvidencePack

logger = logging.getLogger(__name__)

PLANNER_SYSTEM_PROMPT = """You are the Senior Analytical Planning Engine.
Your task is to decompose a complex business/data question into an executable sequence of analytical steps.

CRITICAL RULES:
1. ONLY USE REGISTERED TOOLS:
{tool_definitions}
2. ONLY USE COLUMNS PRESENT IN DATASET:
{columns_summary}
3. FORM A DIRECTED ACYCLIC GRAPH (DAG): Steps can specify dependencies (e.g. step 2 depends on step 1). No circular dependencies.
4. BE CONSERVATIVE & SYSTEMATIC: Start with inspection/quality, compute segment/group comparisons, test associations, and run robustness checks.
5. NO CODE/SQL EXECUTION: Tools execute predefined Python calculations with validated arguments.
6. RETURN VALID JSON matching the InvestigationPlan schema.
"""


class PlannedStepSchema(BaseModel):
    """LLM representation for plan step generation."""

    step_name: str
    objective: str
    step_type: str = Field(description="One of StepType values")
    dependencies: list[str] = Field(default_factory=list)
    tool_name: str
    tool_arguments: dict[str, Any] = Field(default_factory=dict)
    reason: str


class PlannedPlanSchema(BaseModel):
    """LLM representation for plan generation."""

    normalized_objective: str
    assumptions: list[str] = Field(default_factory=list)
    known_limitations: list[str] = Field(default_factory=list)
    steps: list[PlannedStepSchema] = Field(default_factory=list)
    expected_evidence: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)


class InvestigationPlanner:
    """Plans multi-step analytical investigations adapted to active dataset schemas."""

    def __init__(
        self,
        tool_registry: AnalyticalToolRegistry | None = None,
        llm_provider: LLMProvider | None = None,
    ) -> None:
        self.tool_registry = tool_registry or AnalyticalToolRegistry()
        self.llm_provider = llm_provider
        self.validator = WorkflowGraphValidator(self.tool_registry)
        self.context_builder = AnalyticalContextBuilder()

    def create_plan(
        self,
        question: str,
        managed_dataset: ManagedDataset,
        existing_evidence: EvidencePack | None = None,
        max_steps: int = 8,
    ) -> InvestigationPlan:
        """Generate, validate, and return an executable InvestigationPlan."""
        # 1. Attempt LLM-assisted planning if provider is configured
        if self.llm_provider and self.llm_provider.is_configured:
            try:
                plan = self._plan_with_llm(
                    question=question,
                    managed_dataset=managed_dataset,
                    max_steps=max_steps,
                )
                self.validator.validate_plan(plan, managed_dataset, max_allowed_steps=max_steps)
                return plan
            except Exception as err:
                logger.warning(
                    "LLM plan generation or validation failed (%s). Falling back to deterministic planner.",
                    err,
                )

        # 2. Deterministic rule-based planning fallback
        deterministic_plan = self._plan_deterministically(
            question=question,
            managed_dataset=managed_dataset,
            existing_evidence=existing_evidence,
            max_steps=max_steps,
        )
        self.validator.validate_plan(
            deterministic_plan, managed_dataset, max_allowed_steps=max_steps
        )
        return deterministic_plan

    def _plan_with_llm(
        self,
        question: str,
        managed_dataset: ManagedDataset,
        max_steps: int,
    ) -> InvestigationPlan:
        """Prompt LLM for structured plan and transform to InvestigationPlan."""
        assert self.llm_provider is not None
        schema = getattr(managed_dataset, "schema_info", {})
        col_summary = ", ".join(
            f"{c} ({getattr(s, 'inferred_type', 'unknown')})" for c, s in schema.items()
        )

        tool_defs_str = "\n".join(
            f"- {t.name}: {t.description} (args: {list(t.parameters.keys())})"
            for t in self.tool_registry.get_tool_definitions()
        )

        sys_prompt = PLANNER_SYSTEM_PROMPT.format(
            tool_definitions=tool_defs_str,
            columns_summary=col_summary,
        )
        user_prompt = f"Create an investigation plan to answer: '{question}' for dataset '{getattr(managed_dataset, 'filename', 'dataset')}'. Max {max_steps} steps."

        parsed_plan, result = self.llm_provider.generate_structured(
            prompt=user_prompt,
            response_model=PlannedPlanSchema,
            system_instruction=sys_prompt,
        )

        if not parsed_plan or not parsed_plan.steps:
            raise PlanValidationError("LLM returned empty or unparseable plan.")

        steps: list[InvestigationStep] = []
        id_mapping: dict[int, str] = {}

        for idx, ps in enumerate(parsed_plan.steps[:max_steps]):
            step_id = f"step_{idx + 1}"
            id_mapping[idx + 1] = step_id

            # Parse StepType cleanly
            st_val = StepType.DESCRIPTIVE_ANALYSIS
            for member in StepType:
                if member.value == ps.step_type:
                    st_val = member
                    break

            mapped_deps = [
                id_mapping[int(d.replace("step_", ""))]
                for d in ps.dependencies
                if d.replace("step_", "").isdigit() and int(d.replace("step_", "")) in id_mapping
            ]

            steps.append(
                InvestigationStep(
                    step_id=step_id,
                    step_name=ps.step_name,
                    objective=ps.objective,
                    step_type=st_val,
                    dependencies=mapped_deps,
                    tool_name=ps.tool_name,
                    tool_arguments=ps.tool_arguments,
                    reason=ps.reason,
                    status=StepStatus.PENDING,
                )
            )

        return InvestigationPlan(
            question=question,
            normalized_objective=parsed_plan.normalized_objective,
            dataset_id=str(getattr(managed_dataset, "dataset_id", "dataset")),
            dataset_version=getattr(managed_dataset, "version", 1),
            assumptions=parsed_plan.assumptions,
            known_limitations=parsed_plan.known_limitations,
            steps=steps,
            expected_evidence=parsed_plan.expected_evidence,
            success_criteria=parsed_plan.success_criteria,
            max_steps=max_steps,
        )

    def _plan_deterministically(
        self,
        question: str,
        managed_dataset: ManagedDataset,
        existing_evidence: EvidencePack | None = None,
        max_steps: int = 8,
    ) -> InvestigationPlan:
        """Construct grounded, deterministic investigation plan from schema and intent."""
        df: pd.DataFrame = managed_dataset.working_df
        classification: QuestionClassification = self.context_builder.classify_question(
            question=question,
            managed_dataset=managed_dataset,
        )

        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        categorical_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]

        # Target variables
        relevant = classification.relevant_columns
        target_num = next(
            (c for c in relevant if c in numeric_cols), numeric_cols[0] if numeric_cols else None
        )
        target_cat = next(
            (c for c in relevant if c in categorical_cols),
            categorical_cols[0] if categorical_cols else None,
        )

        steps: list[InvestigationStep] = []

        # Step 1: Dataset Overview & Quality
        s1 = InvestigationStep(
            step_id="step_1",
            step_name="Inspect Dataset Overview & Memory",
            objective="Confirm dimensions, memory footprint, and quality score baseline.",
            step_type=StepType.DATASET_INSPECTION,
            dependencies=[],
            tool_name="get_dataset_overview",
            tool_arguments={},
            reason="Establish baseline structural dimensions and sanity of active working data.",
        )
        steps.append(s1)

        # Step 2: Missingness inspection for relevant columns
        cols_to_check = [c for c in [target_num, target_cat] if c] or list(df.columns[:4])
        s2 = InvestigationStep(
            step_id="step_2",
            step_name="Assess Missingness and Data Quality",
            objective=f"Evaluate missing rates and quality flags across key columns: {cols_to_check}.",
            step_type=StepType.DATA_QUALITY_INVESTIGATION,
            dependencies=["step_1"],
            tool_name="inspect_missingness",
            tool_arguments={"columns": cols_to_check},
            reason="Verify that subsequent analyses are not compromised by high missingness.",
        )
        steps.append(s2)

        # Step 3: Descriptive Statistics
        if target_num:
            s3 = InvestigationStep(
                step_id="step_3",
                step_name=f"Calculate Descriptive Statistics for {target_num}",
                objective=f"Compute central tendency, dispersion, and skewness for '{target_num}'.",
                step_type=StepType.DESCRIPTIVE_ANALYSIS,
                dependencies=["step_2"],
                tool_name="calculate_descriptive_statistics",
                tool_arguments={"columns": [target_num]},
                reason=f"Quantify baseline magnitude and distribution shape of target metric '{target_num}'.",
            )
            steps.append(s3)

        # Step 4: Group Comparison (if categorical column exists)
        if target_cat and target_num:
            s4 = InvestigationStep(
                step_id="step_4",
                step_name=f"Compare {target_num} across {target_cat} Segments",
                objective=f"Perform formal inferential comparison of '{target_num}' grouped by '{target_cat}'.",
                step_type=StepType.GROUP_COMPARISON,
                dependencies=["step_3"] if target_num in [s.step_id for s in steps] else ["step_2"],
                tool_name="compare_groups",
                tool_arguments={"group_col": target_cat, "measure_col": target_num},
                reason=f"Test whether disparities across '{target_cat}' are statistically significant.",
            )
            steps.append(s4)

        # Step 5: Correlation or Regression
        if len(numeric_cols) >= 2:
            num2 = next((c for c in numeric_cols if c != target_num), None)
            if target_num and num2:
                s5 = InvestigationStep(
                    step_id="step_5",
                    step_name=f"Analyze Correlation between {target_num} and {num2}",
                    objective=f"Quantify correlation and uncertainty between '{target_num}' and '{num2}'.",
                    step_type=StepType.CORRELATION_ANALYSIS,
                    dependencies=["step_3"]
                    if target_num in [s.step_id for s in steps]
                    else ["step_2"],
                    tool_name="calculate_correlation",
                    tool_arguments={"var1": target_num, "var2": num2, "method": "pearson"},
                    reason="Inspect continuous co-movement and check for potential confounding.",
                )
                steps.append(s5)

                # Step 6: Robustness check on correlation
                s6 = InvestigationStep(
                    step_id="step_6",
                    step_name=f"Evaluate Robustness of {target_num}-{num2} Association",
                    objective="Verify stability against outliers and non-parametric rank alternatives.",
                    step_type=StepType.ROBUSTNESS_ANALYSIS,
                    dependencies=["step_5"],
                    tool_name="run_robustness_check",
                    tool_arguments={"outcome_col": target_num, "predictor_cols": [num2]},
                    reason="Ensure empirical finding is robust and not an artifact of outlier leverage.",
                )
                steps.append(s6)

        return InvestigationPlan(
            question=question,
            normalized_objective=f"Investigate '{question}' through systematic data quality, group disparities, and statistical inference.",
            dataset_id=str(getattr(managed_dataset, "dataset_id", "dataset")),
            dataset_version=getattr(managed_dataset, "version", 1),
            assumptions=[
                "Observations represent an independent sample under stable operating conditions.",
                "Dataset column schemas and measurements are valid for quantitative analysis.",
            ],
            known_limitations=[
                "Observational analysis identifies empirical associations, not proven causal mechanisms.",
            ],
            steps=steps[:max_steps],
            expected_evidence=[
                "ev_dim_rows",
                "ev_quality_score",
                f"ev_desc_{target_num}_mean" if target_num else "ev_dim_cols",
            ],
            success_criteria=[
                "Key target metric variations and segment disparities are quantified with statistical evidence.",
                "Analytical robustness and potential confounding factors are evaluated.",
            ],
            max_steps=max_steps,
        )
