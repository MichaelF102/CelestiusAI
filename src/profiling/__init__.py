"""Data profiling and quality intelligence package."""

from src.profiling.consistency import analyze_consistency
from src.profiling.duplicates import analyze_duplicates, find_candidate_primary_keys
from src.profiling.missing_values import (
    analyze_missing_values,
    compute_missingness_correlation,
    compute_nullity_matrix,
)
from src.profiling.models import (
    DatasetProfile,
    QualityIssue,
    QualityScore,
    TransformationPreview,
    TransformationRecord,
    TransformationSpec,
)
from src.profiling.outliers import (
    analyze_outliers,
    detect_outliers_iqr,
    detect_outliers_modified_zscore,
    detect_outliers_zscore,
)
from src.profiling.profiler import profile_dataset
from src.profiling.quality_score import calculate_quality_score

__all__ = [
    "DatasetProfile",
    "QualityIssue",
    "QualityScore",
    "TransformationPreview",
    "TransformationRecord",
    "TransformationSpec",
    "analyze_consistency",
    "analyze_duplicates",
    "analyze_missing_values",
    "analyze_outliers",
    "calculate_quality_score",
    "compute_missingness_correlation",
    "compute_nullity_matrix",
    "detect_outliers_iqr",
    "detect_outliers_modified_zscore",
    "detect_outliers_zscore",
    "find_candidate_primary_keys",
    "profile_dataset",
]
