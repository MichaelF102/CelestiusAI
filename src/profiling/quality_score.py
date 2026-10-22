"""Transparent multi-dimensional data quality scoring algorithm."""

from src.profiling.models import QualityIssue, QualityScore

# Dimension weights summing to 1.0
WEIGHT_COMPLETENESS = 0.40
WEIGHT_UNIQUENESS = 0.25
WEIGHT_CONSISTENCY = 0.20
WEIGHT_VALIDITY = 0.15


def calculate_quality_score(
    total_cells: int,
    missing_cells: int,
    total_rows: int,
    duplicate_rows: int,
    issues: list[QualityIssue],
) -> QualityScore:
    """Calculate an objective, weighted 0-100 data quality score across four dimensions.

    Dimensions:
    - Completeness (40%): Evaluates non-missing cell coverage.
    - Uniqueness (25%): Evaluates row duplication.
    - Consistency (20%): Evaluates whitespace, casing, and format uniformities.
    - Validity (15%): Evaluates extreme statistical outliers and invalid domain values.
    """
    if total_cells == 0 or total_rows == 0:
        return QualityScore(
            overall_score=0.0,
            completeness_score=0.0,
            uniqueness_score=0.0,
            consistency_score=0.0,
            validity_score=0.0,
            is_provisional=True,
            score_breakdown={},
            explanation="Dataset is empty or has zero rows/columns.",
        )

    # 1. Completeness Dimension
    missing_pct = (missing_cells / total_cells) * 100.0
    # Penalty: missing 10% cells drops score by 15 points
    completeness_score = max(0.0, min(100.0, 100.0 - (missing_pct * 1.5)))

    # 2. Uniqueness Dimension
    dup_pct = (duplicate_rows / total_rows) * 100.0
    # Penalty: 5% duplicate rows drops score by 10 points
    uniqueness_score = max(0.0, min(100.0, 100.0 - (dup_pct * 2.0)))

    # 3. Consistency Dimension
    consistency_issues = [i for i in issues if i.category in ("consistency", "cardinality")]
    consistency_penalty = sum(
        10.0 if i.severity == "high" else (5.0 if i.severity == "medium" else 2.0)
        for i in consistency_issues
    )
    consistency_score = max(0.0, min(100.0, 100.0 - consistency_penalty))

    # 4. Validity Dimension
    validity_issues = [i for i in issues if i.category in ("outliers", "validity")]
    validity_penalty = sum(
        10.0 if i.severity == "high" else (4.0 if i.severity == "medium" else 1.0)
        for i in validity_issues
    )
    validity_score = max(0.0, min(100.0, 100.0 - validity_penalty))

    # Composite Score
    composite = (
        (completeness_score * WEIGHT_COMPLETENESS)
        + (uniqueness_score * WEIGHT_UNIQUENESS)
        + (consistency_score * WEIGHT_CONSISTENCY)
        + (validity_score * WEIGHT_VALIDITY)
    )
    overall_score = round(max(0.0, min(100.0, composite)), 1)

    breakdown = {
        "weights": {
            "completeness": WEIGHT_COMPLETENESS,
            "uniqueness": WEIGHT_UNIQUENESS,
            "consistency": WEIGHT_CONSISTENCY,
            "validity": WEIGHT_VALIDITY,
        },
        "scores": {
            "completeness": round(completeness_score, 1),
            "uniqueness": round(uniqueness_score, 1),
            "consistency": round(consistency_score, 1),
            "validity": round(validity_score, 1),
        },
        "metrics": {
            "missing_cell_percentage": round(missing_pct, 2),
            "duplicate_row_percentage": round(dup_pct, 2),
            "total_issues_count": len(issues),
            "critical_issues_count": len([i for i in issues if i.severity == "critical"]),
            "high_issues_count": len([i for i in issues if i.severity == "high"]),
        },
    }

    explanation = (
        f"Overall quality score is {overall_score}/100. "
        f"Completeness rated at {completeness_score:.1f}% ({missing_pct:.1f}% missing cells), "
        f"Uniqueness at {uniqueness_score:.1f}% ({dup_pct:.1f}% duplicates), "
        f"Consistency at {consistency_score:.1f}%, and Validity at {validity_score:.1f}%. "
        "Score is marked provisional until user domain constraints are specified."
    )

    return QualityScore(
        overall_score=overall_score,
        completeness_score=round(completeness_score, 1),
        uniqueness_score=round(uniqueness_score, 1),
        consistency_score=round(consistency_score, 1),
        validity_score=round(validity_score, 1),
        is_provisional=True,
        score_breakdown=breakdown,
        explanation=explanation,
    )
