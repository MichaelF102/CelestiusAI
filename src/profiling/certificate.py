"""Quality audit certificate for a profiled dataset."""

from __future__ import annotations

from typing import Any

from src.profiling.models import DatasetProfile


def quality_grade(score: float) -> str:
    if score >= 90:
        return "A (Production Ready)"
    if score >= 75:
        return "B (Usable with Warnings)"
    return "C (Needs Cleaning)"


def build_quality_certificate(
    filename: str,
    fingerprint: str,
    profile: DatasetProfile,
    rows: int,
    columns: int,
    transformations: int,
    baseline_score: float | None = None,
) -> dict[str, Any]:
    qs = profile.quality_score
    return {
        "dataset_filename": filename,
        "fingerprint": fingerprint,
        "quality_grade": quality_grade(qs.overall_score),
        "overall_score": round(qs.overall_score, 1),
        "baseline_score": None if baseline_score is None else round(baseline_score, 1),
        "completeness_score": round(qs.completeness_score, 1),
        "uniqueness_score": round(qs.uniqueness_score, 1),
        "consistency_score": round(qs.consistency_score, 1),
        "validity_score": round(qs.validity_score, 1),
        "total_rows": rows,
        "total_columns": columns,
        "active_issues_count": len(profile.quality_issues),
        "applied_transformations": transformations,
    }
