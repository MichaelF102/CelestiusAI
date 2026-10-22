"""Data consistency, whitespace, casing, and domain constraint verification."""

import re
import uuid

import pandas as pd

from src.ingestion.models import ColumnSchema
from src.profiling.models import QualityIssue

NON_NEGATIVE_PATTERNS = re.compile(
    r"(^|[_.-])(age|price|cost|revenue|sales|profit|quantity|count|amount|duration|tenure|visits|days|hours)([_.-]|$)",
    re.IGNORECASE,
)


def analyze_consistency(
    df: pd.DataFrame,
    schemas: dict[str, ColumnSchema],
) -> tuple[dict[str, object], list[QualityIssue]]:
    """Verify string whitespace, casing uniformity, and logical bounds."""
    consistency_stats: dict[str, object] = {
        "whitespace_issues": {},
        "casing_inconsistencies": {},
        "negative_value_issues": {},
    }
    issues: list[QualityIssue] = []

    for col in df.columns:
        col_str = str(col)
        schema = schemas.get(col_str)
        if not schema:
            continue

        series = df[col]
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]

        # 1. Whitespace check for string / categorical columns
        if series.dtype == "object" or schema.inferred_type in ("categorical", "text"):
            non_null_strs = series.dropna().astype(str)
            whitespace_mask = non_null_strs.str.startswith(" ") | non_null_strs.str.endswith(" ")
            ws_count = int(whitespace_mask.sum())
            if ws_count > 0:
                consistency_stats["whitespace_issues"][col_str] = ws_count
                issues.append(
                    QualityIssue(
                        issue_id=f"iss_ws_{uuid.uuid4().hex[:6]}",
                        category="consistency",
                        severity="low",
                        title=f"Leading/trailing whitespace in '{col_str}' ({ws_count} rows)",
                        affected_columns=[col_str],
                        affected_row_count=ws_count,
                        affected_cell_count=ws_count,
                        measured_evidence=f"{ws_count} cells have unstripped leading or trailing spaces.",
                        explanation="Unstripped spaces create unintended duplicate categories (e.g. 'USA' vs ' USA').",
                        recommended_action="Trim leading and trailing whitespace.",
                        potential_consequences="Causes group-by operations and exact matches to fail.",
                        can_preview=True,
                        suggested_action_type="strip_whitespace",
                        suggested_action_params={"column_name": col_str},
                    )
                )

            # 2. Mixed casing check for categorical columns
            if schema.inferred_type == "categorical" and schema.distinct_count > 1:
                unique_vals = non_null_strs.unique()
                lower_map: dict[str, list[str]] = {}
                for v in unique_vals:
                    lower_v = v.strip().lower()
                    lower_map.setdefault(lower_v, []).append(v)

                casing_duplicates = {k: vals for k, vals in lower_map.items() if len(vals) > 1}
                if casing_duplicates:
                    total_casing_impact = sum(len(v) for v in casing_duplicates.values())
                    consistency_stats["casing_inconsistencies"][col_str] = casing_duplicates
                    sample_cases = list(casing_duplicates.values())[:3]
                    issues.append(
                        QualityIssue(
                            issue_id=f"iss_case_{uuid.uuid4().hex[:6]}",
                            category="consistency",
                            severity="medium",
                            title=f"Mixed capitalization in categorical column '{col_str}'",
                            affected_columns=[col_str],
                            affected_row_count=total_casing_impact,
                            affected_cell_count=total_casing_impact,
                            measured_evidence=f"Variants detected: {sample_cases}.",
                            explanation="Different casings of the same label artificially fragment category groups.",
                            recommended_action="Standardize casing to lowercase or title case.",
                            potential_consequences="Inaccurate pivot tables and distorted segment counts.",
                            can_preview=True,
                            suggested_action_type="standardize_casing",
                            suggested_action_params={"column_name": col_str, "case": "title"},
                        )
                    )

        # 3. Non-negative bounds check
        if schema.inferred_type == "numeric" and not schema.is_identifier:
            if NON_NEGATIVE_PATTERNS.search(col_str) or NON_NEGATIVE_PATTERNS.search(
                schema.normalized_name
            ):
                clean_num = series.dropna().astype(float)
                neg_count = int((clean_num < 0).sum())
                if neg_count > 0:
                    consistency_stats["negative_value_issues"][col_str] = neg_count
                    issues.append(
                        QualityIssue(
                            issue_id=f"iss_neg_{uuid.uuid4().hex[:6]}",
                            category="consistency",
                            severity="high",
                            title=f"Unexpected negative values in '{col_str}' ({neg_count} rows)",
                            affected_columns=[col_str],
                            affected_row_count=neg_count,
                            affected_cell_count=neg_count,
                            measured_evidence=f"{neg_count} rows contain values < 0 in expected non-negative field.",
                            explanation=f"Field '{col_str}' logically represents a magnitude that should not be negative.",
                            recommended_action="Investigate whether negative values are data entry errors or returns/credits.",
                            potential_consequences="Invalidates logarithmic transforms and distorts totals.",
                            can_preview=True,
                        )
                    )

        # 4. Constant columns
        if schema.is_constant:
            issues.append(
                QualityIssue(
                    issue_id=f"iss_const_{uuid.uuid4().hex[:6]}",
                    category="cardinality",
                    severity="medium",
                    title=f"Constant column '{col_str}' provides zero variance",
                    affected_columns=[col_str],
                    affected_row_count=len(df),
                    affected_cell_count=len(df),
                    measured_evidence=f"Column has only 1 distinct value: {schema.sample_values[:1]}.",
                    explanation="Columns with zero variance provide no predictive or analytical utility.",
                    recommended_action=f"Drop constant column '{col_str}'.",
                    potential_consequences="Zero variance causes division-by-zero errors in standardization and modeling.",
                    can_preview=True,
                    suggested_action_type="drop_column",
                    suggested_action_params={"column_name": col_str},
                )
            )

    return consistency_stats, issues
