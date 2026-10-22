"""Tests for profiling engine, outlier detectors, missingness audits, and consistency checks."""

import pandas as pd
from src.ingestion.schema import infer_dataset_schema
from src.profiling.consistency import analyze_consistency
from src.profiling.duplicates import analyze_duplicates
from src.profiling.missing_values import analyze_missing_values
from src.profiling.outliers import (
    detect_outliers_iqr,
    detect_outliers_modified_zscore,
    detect_outliers_zscore,
)
from src.profiling.profiler import profile_dataset


def test_detect_outliers_synthetic():
    """Verify outlier detection on known synthetic vector."""
    # 20 points centered around 10, plus two obvious outliers (100 and -50)
    data = [10.0, 10.1, 9.9, 10.2, 9.8, 10.0, 10.1, 9.9, 10.0, 10.2] * 2 + [100.0, -50.0]
    series = pd.Series(data)

    iqr_res = detect_outliers_iqr(series, k=1.5)
    assert iqr_res["count"] == 2
    assert iqr_res["percentage"] > 0

    z_res = detect_outliers_zscore(series, threshold=2.5)
    assert z_res["count"] >= 1

    mad_res = detect_outliers_modified_zscore(series, threshold=3.5)
    assert mad_res["count"] == 2


def test_analyze_missingness_and_tokens():
    """Verify detection of both explicit nulls and suspicious string placeholders."""
    df = pd.DataFrame(
        {
            "num_col": [1.0, None, 3.0, None, 5.0],  # 2 nulls (40%)
            "text_col": ["active", "N/A", "pending", "unknown", "closed"],  # 2 suspicious tokens
        }
    )
    schemas = infer_dataset_schema(df)
    stats, issues = analyze_missing_values(df, schemas)

    assert stats["num_col"]["null_count"] == 2
    assert stats["num_col"]["null_percentage"] == 40.0
    assert stats["text_col"]["suspicious_token_count"] == 2

    # Issue generated for moderate missingness and suspicious token
    issue_categories = [i.category for i in issues]
    assert "missingness" in issue_categories


def test_analyze_duplicates():
    """Verify duplicate row detection and key-based conflicts."""
    df = pd.DataFrame(
        {
            "id": [1, 2, 2, 3],
            "val": ["A", "B", "B", "C"],  # Rows 1 and 2 are identical
        }
    )
    stats, issues = analyze_duplicates(df, key_columns=["id"])

    assert stats["exact_duplicate_rows"] == 1
    assert stats["key_duplicate_rows"] == 2
    assert len(issues) >= 1


def test_analyze_consistency():
    """Verify detection of unstripped whitespace and mixed casing."""
    df = pd.DataFrame(
        {
            "whitespace_col": ["apple", " banana ", "orange", "grape"],
            "casing_col": ["Male", "male", "FEMALE", "female"],
            "age": [25, 30, -5, 42],  # Unexpected negative value in 'age'
        }
    )
    schemas = infer_dataset_schema(df)
    stats, issues = analyze_consistency(df, schemas)

    assert "whitespace_col" in stats["whitespace_issues"]
    assert stats["whitespace_issues"]["whitespace_col"] == 1

    assert "casing_col" in stats["casing_inconsistencies"]
    assert "age" in stats["negative_value_issues"]

    titles = [i.title for i in issues]
    assert any("whitespace" in t.lower() for t in titles)
    assert any("capitalization" in t.lower() for t in titles)
    assert any("negative" in t.lower() for t in titles)


def test_full_dataset_profile():
    """Verify profile_dataset produces valid DatasetProfile with quality scores."""
    df = pd.DataFrame(
        {
            "id": range(1, 101),
            "amount": [10.0] * 98 + [500.0, 600.0],  # 2 outliers
            "status": ["valid"] * 95 + ["invalid"] * 5,
        }
    )
    profile = profile_dataset(df, dataset_id="ds_test")

    assert profile.row_count == 100
    assert profile.column_count == 3
    assert profile.total_missing_cells == 0
    assert profile.quality_score.overall_score > 0
    assert "amount" in profile.numeric_profiles
    assert "status" in profile.categorical_profiles


def test_find_candidate_primary_keys():
    """Verify primary key finder discovers single and composite keys."""
    from src.profiling import find_candidate_primary_keys

    # Single column 100% unique
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "category": ["a", "a", "b", "b", "c"],
            "seq": [1, 2, 1, 2, 1],
        }
    )
    cand = find_candidate_primary_keys(df)
    assert len(cand) >= 1
    # 'id' should be the top candidate
    top = cand[0]
    assert top["columns"] == ["id"]
    assert top["is_100_percent_unique"] is True

    # Composite pair
    df_comp = pd.DataFrame(
        {
            "dept": ["sales", "sales", "eng", "eng"],
            "level": [1, 2, 1, 2],
            "score": [90, 85, 95, 80],
        }
    )
    cand_comp = find_candidate_primary_keys(df_comp)
    assert any(
        c["is_100_percent_unique"] and set(c["columns"]) == {"dept", "level"} for c in cand_comp
    )


def test_nullity_matrix_and_correlation():
    """Verify nullity indicator matrix and missingness correlation helpers."""
    from src.profiling import compute_missingness_correlation, compute_nullity_matrix

    df = pd.DataFrame(
        {
            "a": [1.0, None, 3.0, None],
            "b": [10.0, None, 30.0, None],  # perfectly correlated missingness with 'a'
            "c": ["x", "y", "z", "w"],  # complete
        }
    )
    matrix = compute_nullity_matrix(df, max_rows=10)
    assert len(matrix) == 4
    assert matrix.loc["Row 0", "a"] == 1
    assert matrix.loc["Row 1", "a"] == 0

    corr = compute_missingness_correlation(df)
    assert corr is not None
    # a and b should have correlation ~ 1.0
    assert abs(corr.loc["a", "b"] - 1.0) < 1e-5
