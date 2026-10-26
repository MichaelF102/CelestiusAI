"""Helpers connecting detected quality issues to executable transformations."""

from __future__ import annotations

from collections.abc import Iterable

from src.ingestion.loader import ManagedDataset
from src.ingestion.transformation_history import TransformationPipeline
from src.profiling.models import DatasetProfile, QualityIssue, TransformationSpec
from src.profiling.profiler import profile_dataset

SAFE_ACTIONS = ("drop_duplicates", "strip_whitespace")


def actionable_issues(profile: DatasetProfile) -> list[QualityIssue]:
    return [
        i
        for i in profile.quality_issues
        if i.suggested_action_type is not None and not i.is_resolved
    ]


def is_safe_issue(issue: QualityIssue) -> bool:
    """Non-destructive actions that can be applied without review."""
    return issue.suggested_action_type in SAFE_ACTIONS or (
        issue.suggested_action_type == "drop_column" and "100%" in issue.title
    )


def spec_from_issue(issue: QualityIssue, prefix: str = "spec") -> TransformationSpec:
    return TransformationSpec(
        spec_id=f"{prefix}_{issue.issue_id}",
        action_type=issue.suggested_action_type,  # type: ignore[arg-type]
        column_name=issue.suggested_action_params.get("column_name"),
        parameters=issue.suggested_action_params,
        description=issue.recommended_action,
    )


def apply_issues(
    pipeline: TransformationPipeline, dataset: ManagedDataset, issues: Iterable[QualityIssue]
) -> tuple[int, list[str]]:
    """Apply each issue's suggested transformation; returns (applied_count, error_messages)."""
    applied = 0
    errors: list[str] = []
    for issue in issues:
        try:
            pipeline.apply(spec_from_issue(issue, "auto"), dataset)
            applied += 1
        except Exception as err:
            errors.append(f"{issue.title}: {err}")
    return applied, errors


def reprofile(dataset: ManagedDataset) -> DatasetProfile:
    return profile_dataset(dataset.working_df, dataset_id=dataset.dataset_id)
