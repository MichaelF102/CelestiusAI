"""Controlled, approved analytical tool registry for the CelestiusAI.

Provides deterministic execution wrappers over established analytical, EDA,
and statistical intelligence modules. The LLM can only request execution through
predefined, validated tool schemas. Arbitrary code, SQL, shell, or filesystem
execution is strictly blocked.
"""

import difflib
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from src.analytics.descriptive import compute_numeric_summary
from src.analytics.robustness import (
    evaluate_correlation_robustness,
)
from src.analytics.statistical_service import StatisticalService
from src.ingestion.loader import ManagedDataset
from src.ingestion.schema import infer_dataset_schema
from src.profiling.missing_values import analyze_missing_values
from src.profiling.profiler import profile_categorical_column, profile_numeric_column

logger = logging.getLogger(__name__)


@dataclass
class ToolExecutionAudit:
    """Audit log entry for every analytical tool invocation."""

    tool_name: str
    dataset_version: int
    arguments: dict[str, Any]
    execution_time_ms: float
    success: bool
    error_message: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


class ToolDefinition(BaseModel):
    """Schema descriptor for approved analytical tools."""

    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    required_parameters: list[str] = Field(default_factory=list)


class AnalyticalToolRegistry:
    """Registry managing approved analytical functions with strict validation and audit."""

    def __init__(self) -> None:
        self._tools: dict[str, tuple[ToolDefinition, Callable[..., Any]]] = {}
        self._audit_log: list[ToolExecutionAudit] = []
        self._register_default_tools()

    @property
    def audit_log(self) -> list[ToolExecutionAudit]:
        return list(self._audit_log)

    def register_tool(
        self,
        definition: ToolDefinition,
        fn: Callable[..., Any],
    ) -> None:
        self._tools[definition.name] = (definition, fn)

    def get_tool_definitions(self) -> list[ToolDefinition]:
        return [tool_def for tool_def, _ in self._tools.values()]

    def _resolve_column(self, df: pd.DataFrame, col_name: str) -> str:
        """Validate column existence with case-insensitive and fuzzy suggestion fallback."""
        if col_name in df.columns:
            return col_name
        # Case insensitive match
        lower_map = {c.lower(): c for c in df.columns}
        if col_name.lower() in lower_map:
            return lower_map[col_name.lower()]
        # Fuzzy match suggestions
        close = difflib.get_close_matches(col_name, df.columns, n=3, cutoff=0.6)
        hint = f" Did you mean: {', '.join(close)}?" if close else ""
        raise ValueError(f"Column '{col_name}' does not exist in dataset.{hint}")

    def execute_tool(
        self,
        tool_name: str,
        managed_dataset: ManagedDataset,
        arguments: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Safely execute an approved tool against the active managed dataset."""
        args = arguments or {}
        if tool_name not in self._tools:
            raise ValueError(
                f"Unknown tool '{tool_name}'. Allowed tools: {sorted(self._tools.keys())}"
            )

        tool_def, fn = self._tools[tool_name]
        start_time = time.perf_counter()
        dataset_version = getattr(managed_dataset, "version", 1)

        try:
            result = fn(managed_dataset, **args)
            duration_ms = (time.perf_counter() - start_time) * 1000.0

            # Convert Pydantic or DataFrame objects into dicts
            serialized: dict[str, Any]
            if hasattr(result, "model_dump"):
                serialized = result.model_dump()
            elif isinstance(result, pd.DataFrame):
                serialized = result.to_dict(orient="records")
            elif isinstance(result, dict):
                serialized = result
            else:
                serialized = {"output": str(result)}

            self._audit_log.append(
                ToolExecutionAudit(
                    tool_name=tool_name,
                    dataset_version=dataset_version,
                    arguments=args,
                    execution_time_ms=round(duration_ms, 2),
                    success=True,
                )
            )
            return serialized

        except Exception as err:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            self._audit_log.append(
                ToolExecutionAudit(
                    tool_name=tool_name,
                    dataset_version=dataset_version,
                    arguments=args,
                    execution_time_ms=round(duration_ms, 2),
                    success=False,
                    error_message=str(err),
                )
            )
            logger.error("Error executing analytical tool '%s': %s", tool_name, err)
            raise

    # -------------------------------------------------------------------------
    # Default Registered Tools
    # -------------------------------------------------------------------------
    def _register_default_tools(self) -> None:
        # 1. get_dataset_overview
        self.register_tool(
            ToolDefinition(
                name="get_dataset_overview",
                description="Returns dimensions, memory usage, inferred types, and quality score of the active dataset.",
                parameters={},
                required_parameters=[],
            ),
            self._tool_get_dataset_overview,
        )

        # 2. get_column_profile
        self.register_tool(
            ToolDefinition(
                name="get_column_profile",
                description="Returns detailed univariate summary, missingness, and distribution metrics for a single column.",
                parameters={"column": {"type": "string", "description": "Column name to profile"}},
                required_parameters=["column"],
            ),
            self._tool_get_column_profile,
        )

        # 3. calculate_descriptive_statistics
        self.register_tool(
            ToolDefinition(
                name="calculate_descriptive_statistics",
                description="Calculates mean, median, std, min, max, IQR, and skewness for specified or all numeric columns.",
                parameters={
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of numeric columns",
                    }
                },
                required_parameters=[],
            ),
            self._tool_calculate_descriptive_statistics,
        )

        # 4. compare_groups
        self.register_tool(
            ToolDefinition(
                name="compare_groups",
                description="Compares continuous metric across categories using Welch's t-test or ANOVA with effect size.",
                parameters={
                    "group_col": {"type": "string", "description": "Categorical grouping column"},
                    "measure_col": {"type": "string", "description": "Continuous metric column"},
                    "group1_name": {
                        "type": "string",
                        "description": "Optional first category name for 2-group comparison",
                    },
                    "group2_name": {
                        "type": "string",
                        "description": "Optional second category name for 2-group comparison",
                    },
                },
                required_parameters=["group_col", "measure_col"],
            ),
            self._tool_compare_groups,
        )

        # 5. calculate_correlation
        self.register_tool(
            ToolDefinition(
                name="calculate_correlation",
                description="Calculates Pearson or Spearman correlation estimate, 95% confidence interval, and p-value between two continuous variables.",
                parameters={
                    "var1": {"type": "string", "description": "First numeric column"},
                    "var2": {"type": "string", "description": "Second numeric column"},
                    "method": {
                        "type": "string",
                        "enum": ["pearson", "spearman"],
                        "description": "Correlation method",
                    },
                },
                required_parameters=["var1", "var2"],
            ),
            self._tool_calculate_correlation,
        )

        # 6. run_hypothesis_test
        self.register_tool(
            ToolDefinition(
                name="run_hypothesis_test",
                description="Executes a formal inferential test: two_group, multi_group, paired, categorical, or correlation.",
                parameters={
                    "test_type": {
                        "type": "string",
                        "enum": ["two_group", "multi_group", "categorical", "correlation"],
                    },
                    "var1": {"type": "string"},
                    "var2": {"type": "string"},
                    "method": {"type": "string", "description": "Optional method variant"},
                },
                required_parameters=["test_type", "var1", "var2"],
            ),
            self._tool_run_hypothesis_test,
        )

        # 7. fit_regression
        self.register_tool(
            ToolDefinition(
                name="fit_regression",
                description="Fits an OLS linear regression model estimating conditional associations, R-squared, coefficients, and p-values.",
                parameters={
                    "outcome_col": {"type": "string", "description": "Target continuous column"},
                    "predictor_cols": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Feature column names",
                    },
                },
                required_parameters=["outcome_col", "predictor_cols"],
            ),
            self._tool_fit_regression,
        )

        # 8. get_regression_diagnostics
        self.register_tool(
            ToolDefinition(
                name="get_regression_diagnostics",
                description="Evaluates Gauss-Markov regression assumptions: multicollinearity (VIF), heteroscedasticity, normality of residuals.",
                parameters={
                    "outcome_col": {"type": "string"},
                    "predictor_cols": {"type": "array", "items": {"type": "string"}},
                },
                required_parameters=["outcome_col", "predictor_cols"],
            ),
            self._tool_get_regression_diagnostics,
        )

        # 9. run_robustness_check
        self.register_tool(
            ToolDefinition(
                name="run_robustness_check",
                description="Evaluates stability of regression findings under 5% outlier trimming and bootstrap resampling.",
                parameters={
                    "outcome_col": {"type": "string"},
                    "predictor_cols": {"type": "array", "items": {"type": "string"}},
                },
                required_parameters=["outcome_col", "predictor_cols"],
            ),
            self._tool_run_robustness_check,
        )

        # 10. inspect_missingness
        self.register_tool(
            ToolDefinition(
                name="inspect_missingness",
                description="Quantifies missing values, percentages, and patterns across columns.",
                parameters={
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional subset of columns",
                    }
                },
                required_parameters=[],
            ),
            self._tool_inspect_missingness,
        )

    # -------------------------------------------------------------------------
    # Tool Implementation Functions
    # -------------------------------------------------------------------------
    def _tool_get_dataset_overview(self, md: ManagedDataset) -> dict[str, Any]:
        df = md.working_df
        schema = getattr(md, "schema_info", {})
        type_map = {col: s.inferred_type for col, s in schema.items()}
        quality_score = None
        if hasattr(md, "profile") and md.profile:
            quality_score = md.profile.overall_quality_score

        return {
            "dataset_name": getattr(md, "filename", getattr(md, "dataset_id", "dataset")),
            "dataset_version": getattr(md, "version", 1),
            "row_count": int(len(df)),
            "column_count": int(len(df.columns)),
            "columns": list(df.columns),
            "column_types": type_map,
            "overall_quality_score": quality_score,
            "total_null_cells": int(df.isna().sum().sum()),
        }

    def _tool_get_column_profile(self, md: ManagedDataset, column: str) -> dict[str, Any]:
        df = md.working_df
        actual_col = self._resolve_column(df, column)
        series = df[actual_col]

        if pd.api.types.is_numeric_dtype(series):
            return profile_numeric_column(series)
        return profile_categorical_column(series)

    def _tool_calculate_descriptive_statistics(
        self,
        md: ManagedDataset,
        columns: list[str] | None = None,
    ) -> dict[str, Any]:
        df = md.working_df
        target_cols = columns or [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        stats_map = {}
        for col in target_cols:
            actual_col = self._resolve_column(df, col)
            if pd.api.types.is_numeric_dtype(df[actual_col]):
                num_summary = compute_numeric_summary(df[actual_col], actual_col)
                stats_map[actual_col] = num_summary.model_dump()
        return stats_map

    def _tool_compare_groups(
        self,
        md: ManagedDataset,
        group_col: str,
        measure_col: str,
        group1_name: str | None = None,
        group2_name: str | None = None,
    ) -> dict[str, Any]:
        df = md.working_df
        actual_grp = self._resolve_column(df, group_col)
        actual_meas = self._resolve_column(df, measure_col)

        unique_groups = df[actual_grp].dropna().unique()
        if len(unique_groups) == 2 or (group1_name and group2_name):
            res = StatisticalService.execute_two_group_analysis(
                managed_dataset=md,
                group_col=actual_grp,
                measure_col=actual_meas,
                group1_name=group1_name,
                group2_name=group2_name,
            )
            return res.model_dump()
        elif len(unique_groups) > 2:
            res = StatisticalService.execute_multi_group_analysis(
                managed_dataset=md,
                group_col=actual_grp,
                measure_col=actual_meas,
            )
            return res.model_dump()
        else:
            raise ValueError(f"Grouping column '{actual_grp}' has fewer than 2 distinct values.")

    def _tool_calculate_correlation(
        self,
        md: ManagedDataset,
        var1: str,
        var2: str,
        method: str = "pearson",
    ) -> dict[str, Any]:
        df = md.working_df
        v1 = self._resolve_column(df, var1)
        v2 = self._resolve_column(df, var2)
        res = StatisticalService.execute_correlation_analysis(
            managed_dataset=md,
            var1=v1,
            var2=v2,
            method="spearman" if method.lower() == "spearman" else "pearson",
        )
        return res.model_dump()

    def _tool_run_hypothesis_test(
        self,
        md: ManagedDataset,
        test_type: str,
        var1: str,
        var2: str,
        method: str | None = None,
    ) -> dict[str, Any]:
        df = md.working_df
        v1 = self._resolve_column(df, var1)
        v2 = self._resolve_column(df, var2)

        if test_type in ("two_group", "group_comparison"):
            return self._tool_compare_groups(md, group_col=v1, measure_col=v2)
        elif test_type == "categorical":
            res = StatisticalService.execute_categorical_analysis(
                managed_dataset=md,
                var1=v1,
                var2=v2,
            )
            return res.model_dump()
        elif test_type == "correlation":
            return self._tool_calculate_correlation(
                md, var1=v1, var2=v2, method=method or "pearson"
            )
        else:
            raise ValueError(f"Unsupported test_type '{test_type}'")

    def _tool_fit_regression(
        self,
        md: ManagedDataset,
        outcome_col: str,
        predictor_cols: list[str],
    ) -> dict[str, Any]:
        df = md.working_df
        outcome = self._resolve_column(df, outcome_col)
        predictors = [self._resolve_column(df, p) for p in predictor_cols]
        res = StatisticalService.execute_regression_analysis(
            managed_dataset=md,
            outcome_col=outcome,
            predictor_cols=predictors,
        )
        return res.model_dump()

    def _tool_get_regression_diagnostics(
        self,
        md: ManagedDataset,
        outcome_col: str,
        predictor_cols: list[str],
    ) -> dict[str, Any]:
        df = md.working_df
        outcome = self._resolve_column(df, outcome_col)
        predictors = [self._resolve_column(df, p) for p in predictor_cols]
        res = StatisticalService.execute_regression_analysis(
            managed_dataset=md,
            outcome_col=outcome,
            predictor_cols=predictors,
        )
        return {
            "outcome_variable": res.outcome_variable,
            "n_observations": res.n_observations,
            "r_squared": res.r_squared,
            "diagnostics": [d.model_dump() for d in res.diagnostics],
        }

    def _tool_run_robustness_check(
        self,
        md: ManagedDataset,
        outcome_col: str,
        predictor_cols: list[str],
    ) -> dict[str, Any]:
        df = md.working_df
        outcome = self._resolve_column(df, outcome_col)
        predictors = [self._resolve_column(df, p) for p in predictor_cols]
        if predictors:
            comp = evaluate_correlation_robustness(df, outcome, predictors[0])
            return {"comparisons": [c.model_dump() for c in comp]}
        return {"comparisons": []}

    def _tool_inspect_missingness(
        self,
        md: ManagedDataset,
        columns: list[str] | None = None,
    ) -> dict[str, Any]:
        df = md.working_df
        target_df = df[[self._resolve_column(df, c) for c in columns]] if columns else df
        schemas = getattr(md, "schema_info", None) or infer_dataset_schema(target_df)
        missing_stats, issues = analyze_missing_values(target_df, schemas)
        return {
            "missing_stats": missing_stats,
            "issues": [issue.model_dump() for issue in issues],
        }
