"""Duplicate row and entity key conflict detection."""

import uuid

import pandas as pd

from src.profiling.models import QualityIssue


def analyze_duplicates(
    df: pd.DataFrame,
    key_columns: list[str] | None = None,
) -> tuple[dict[str, float], list[QualityIssue]]:
    """Analyze exact duplicate rows and optional key-based duplicates."""
    total_rows = len(df)
    stats: dict[str, float] = {
        "exact_duplicate_rows": 0,
        "exact_duplicate_percentage": 0.0,
        "key_duplicate_rows": 0,
    }
    issues: list[QualityIssue] = []

    if total_rows == 0:
        return stats, issues

    # Exact duplicate row check
    duplicate_mask = df.duplicated(keep="first")
    exact_dup_count = int(duplicate_mask.sum())
    exact_dup_pct = round((exact_dup_count / total_rows) * 100.0, 2)

    stats["exact_duplicate_rows"] = exact_dup_count
    stats["exact_duplicate_percentage"] = exact_dup_pct

    if exact_dup_count > 0:
        severity = (
            "critical" if exact_dup_pct >= 25.0 else ("high" if exact_dup_pct >= 5.0 else "medium")
        )
        issues.append(
            QualityIssue(
                issue_id=f"iss_dup_exact_{uuid.uuid4().hex[:6]}",
                category="duplicates",
                severity=severity,
                title=f"Detected {exact_dup_count} exact duplicate rows ({exact_dup_pct}%)",
                affected_columns=list(df.columns),
                affected_row_count=exact_dup_count,
                affected_cell_count=exact_dup_count * len(df.columns),
                measured_evidence=f"{exact_dup_count} identical records observed across all columns.",
                explanation="Duplicate rows inflate sample sizes and bias frequency statistics.",
                recommended_action="Remove duplicate rows while retaining the first occurrence.",
                potential_consequences="Over-represents identical records in statistical tests and machine learning.",
                can_preview=True,
                suggested_action_type="drop_duplicates",
                suggested_action_params={"subset": None, "keep": "first"},
            )
        )

    # Key-based duplicate check
    if key_columns:
        valid_keys = [k for k in key_columns if k in df.columns]
        if valid_keys:
            key_dup_mask = df.duplicated(subset=valid_keys, keep=False)
            key_dup_count = int(key_dup_mask.sum())
            stats["key_duplicate_rows"] = key_dup_count
            if key_dup_count > 0:
                issues.append(
                    QualityIssue(
                        issue_id=f"iss_dup_key_{uuid.uuid4().hex[:6]}",
                        category="duplicates",
                        severity="high",
                        title=f"Conflicting duplicate keys on [{', '.join(valid_keys)}] ({key_dup_count} rows)",
                        affected_columns=valid_keys,
                        affected_row_count=key_dup_count,
                        affected_cell_count=key_dup_count * len(valid_keys),
                        measured_evidence=f"{key_dup_count} records share duplicate business key values.",
                        explanation="Entity keys should be unique; duplicates indicate duplicate entity entries.",
                        recommended_action="Inspect key conflicts and deduplicate or merge rows.",
                        potential_consequences="Violates primary key uniqueness assumptions.",
                        can_preview=True,
                    )
                )

    return stats, issues
