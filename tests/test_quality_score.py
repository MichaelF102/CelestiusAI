"""Tests for transparent multi-dimensional quality scoring algorithm."""

from src.profiling.models import QualityIssue
from src.profiling.quality_score import (
    calculate_quality_score,
)


def test_perfect_quality_score():
    """Verify that a dataset with 0 missing, 0 duplicates, and 0 issues achieves 100 score."""
    qs = calculate_quality_score(
        total_cells=1000,
        missing_cells=0,
        total_rows=100,
        duplicate_rows=0,
        issues=[],
    )

    assert qs.overall_score == 100.0
    assert qs.completeness_score == 100.0
    assert qs.uniqueness_score == 100.0
    assert qs.consistency_score == 100.0
    assert qs.validity_score == 100.0
    assert qs.is_provisional is True


def test_deterministic_scoring_penalties():
    """Verify that specific missingness and duplicate percentages produce mathematically exact scores."""
    # 10% missing cells -> completeness = 100 - (10 * 1.5) = 85.0
    # 5% duplicate rows -> uniqueness = 100 - (5 * 2.0) = 90.0
    # No consistency or validity issues -> 100.0 each
    # Overall = (85 * 0.40) + (90 * 0.25) + (100 * 0.20) + (100 * 0.15)
    #         = 34.0 + 22.5 + 20.0 + 15.0 = 91.5
    qs = calculate_quality_score(
        total_cells=1000,
        missing_cells=100,
        total_rows=100,
        duplicate_rows=5,
        issues=[],
    )

    assert qs.completeness_score == 85.0
    assert qs.uniqueness_score == 90.0
    assert qs.consistency_score == 100.0
    assert qs.validity_score == 100.0
    assert qs.overall_score == 91.5


def test_quality_score_with_issues():
    """Verify that consistency and validity issues reduce their respective dimension scores."""
    issue_consistency = QualityIssue(
        issue_id="iss_1",
        category="consistency",
        severity="medium",
        title="Mixed casing",
        affected_columns=["category"],
        measured_evidence="Multiple casing variants",
        explanation="Inconsistent labels",
        recommended_action="Standardize casing",
        potential_consequences="Fragmented groups",
    )

    issue_outlier = QualityIssue(
        issue_id="iss_2",
        category="outliers",
        severity="high",
        title="Extreme outliers",
        affected_columns=["revenue"],
        measured_evidence="Values exceed 3 std",
        explanation="Distorts averages",
        recommended_action="Cap outliers",
        potential_consequences="Inflated variance",
    )

    qs = calculate_quality_score(
        total_cells=1000,
        missing_cells=0,
        total_rows=100,
        duplicate_rows=0,
        issues=[issue_consistency, issue_outlier],
    )

    # Consistency penalty for medium issue is 5.0 -> 95.0
    assert qs.consistency_score == 95.0
    # Validity penalty for high issue is 10.0 -> 90.0
    assert qs.validity_score == 90.0
    assert qs.overall_score < 100.0


def test_quality_score_empty_data():
    """Verify empty dataset returns 0.0 with provisional flag."""
    qs = calculate_quality_score(0, 0, 0, 0, [])
    assert qs.overall_score == 0.0
    assert qs.is_provisional is True
