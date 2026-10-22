"""Missing-value profiling and suspicious null token detector."""

import uuid

import pandas as pd

from src.ingestion.models import ColumnSchema
from src.profiling.models import QualityIssue

SUSPICIOUS_NULL_STRINGS = {
    "n/a",
    "na",
    "null",
    "none",
    "unknown",
    "?",
    "-",
    "--",
    "missing",
    "nil",
    "nan",
    "undefined",
}


def analyze_missing_values(
    df: pd.DataFrame,
    schemas: dict[str, ColumnSchema],
    high_missingness_threshold_pct: float = 20.0,
    critical_missingness_threshold_pct: float = 60.0,
) -> tuple[dict[str, dict[str, float]], list[QualityIssue]]:
    """Profile missingness by column and identify high-severity missing value issues."""
    total_rows = len(df)
    missing_stats: dict[str, dict[str, float]] = {}
    issues: list[QualityIssue] = []

    if total_rows == 0:
        return missing_stats, issues

    for col in df.columns:
        col_str = str(col)
        series = df[col]
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]

        null_count = int(series.isna().sum())
        null_pct = round((null_count / total_rows) * 100.0, 2)

        # Detect suspicious string tokens representing missing values in text/categorical columns
        suspicious_count = 0
        if series.dtype == "object":
            non_null_strs = series.dropna().astype(str).str.strip().str.lower()
            suspicious_mask = non_null_strs.isin(SUSPICIOUS_NULL_STRINGS)
            suspicious_count = int(suspicious_mask.sum())

        missing_stats[col_str] = {
            "null_count": null_count,
            "null_percentage": null_pct,
            "suspicious_token_count": suspicious_count,
        }

        # Issue 1: 100% All-null column
        if null_count == total_rows:
            issues.append(
                QualityIssue(
                    issue_id=f"iss_null_all_{uuid.uuid4().hex[:6]}",
                    category="missingness",
                    severity="critical",
                    title=f"Column '{col_str}' contains 100% missing values",
                    affected_columns=[col_str],
                    affected_row_count=total_rows,
                    affected_cell_count=null_count,
                    measured_evidence=f"All {total_rows} rows are null (100.0%).",
                    explanation=f"Column '{col_str}' carries no analytical signal or information.",
                    recommended_action=f"Drop empty column '{col_str}'.",
                    potential_consequences="Keeping empty columns consumes memory and degrades model training.",
                    can_preview=True,
                    suggested_action_type="drop_column",
                    suggested_action_params={"column_name": col_str},
                )
            )

        # Issue 2: Critical missingness (> 60%)
        elif null_pct >= critical_missingness_threshold_pct:
            issues.append(
                QualityIssue(
                    issue_id=f"iss_null_crit_{uuid.uuid4().hex[:6]}",
                    category="missingness",
                    severity="high",
                    title=f"Severe missingness in column '{col_str}' ({null_pct}%)",
                    affected_columns=[col_str],
                    affected_row_count=null_count,
                    affected_cell_count=null_count,
                    measured_evidence=f"{null_count} of {total_rows} records are missing ({null_pct}%).",
                    explanation="High missingness distorts representative distributions.",
                    recommended_action="Drop column or impute with caution if business justification exists.",
                    potential_consequences="Imputing >60% missing data will introduce artificial variance.",
                    can_preview=True,
                    suggested_action_type="drop_column",
                    suggested_action_params={"column_name": col_str},
                )
            )

        # Issue 3: Moderate missingness (> 20%)
        elif null_pct >= high_missingness_threshold_pct:
            col_schema = schemas.get(col_str)
            suggested_action = (
                "impute_numeric"
                if (col_schema and col_schema.inferred_type == "numeric")
                else "impute_categorical"
            )
            strategy = "median" if suggested_action == "impute_numeric" else "mode"

            issues.append(
                QualityIssue(
                    issue_id=f"iss_null_mod_{uuid.uuid4().hex[:6]}",
                    category="missingness",
                    severity="medium",
                    title=f"Moderate missingness in column '{col_str}' ({null_pct}%)",
                    affected_columns=[col_str],
                    affected_row_count=null_count,
                    affected_cell_count=null_count,
                    measured_evidence=f"{null_count} rows missing ({null_pct}%).",
                    explanation="Missing values will cause row drops in listwise deletion analyses.",
                    recommended_action=f"Impute with {strategy} or fill with placeholder category.",
                    potential_consequences="Dropping rows without imputation reduces sample size.",
                    can_preview=True,
                    suggested_action_type=suggested_action,
                    suggested_action_params={"column_name": col_str, "strategy": strategy},
                )
            )

        # Issue 4: Suspicious placeholder tokens
        if suspicious_count > 0:
            suspicious_pct = round((suspicious_count / total_rows) * 100.0, 2)
            issues.append(
                QualityIssue(
                    issue_id=f"iss_susp_null_{uuid.uuid4().hex[:6]}",
                    category="missingness",
                    severity="low",
                    title=f"Suspicious missingness tokens in '{col_str}' ({suspicious_count} cells)",
                    affected_columns=[col_str],
                    affected_row_count=suspicious_count,
                    affected_cell_count=suspicious_count,
                    measured_evidence=f"{suspicious_count} cells ({suspicious_pct}%) contain tokens like 'N/A' or 'unknown'.",
                    explanation="Text placeholders are treated as valid categories unless standardized to nulls.",
                    recommended_action="Convert placeholder strings to nulls or keep as explicit 'Missing' category.",
                    potential_consequences="Could contaminate categorical groupings and mask real missingness.",
                    can_preview=True,
                )
            )

    return missing_stats, issues


def compute_nullity_matrix(
    df: pd.DataFrame,
    max_rows: int = 250,
) -> pd.DataFrame:
    """Sample dataset rows and return a binary indicator matrix (1=present, 0=null)."""
    if df.empty:
        return pd.DataFrame()

    total_rows = len(df)
    if total_rows <= max_rows:
        sample_df = df
    else:
        # Uniformly spaced sampling across entire dataset
        import numpy as np

        indices = np.linspace(0, total_rows - 1, max_rows, dtype=int)
        sample_df = df.iloc[indices]

    # Invert isna: 1 = data present, 0 = missing
    matrix = (~sample_df.isna()).astype(int)
    matrix.index = [f"Row {i}" for i in range(len(matrix))]
    return matrix


def compute_missingness_correlation(
    df: pd.DataFrame,
) -> pd.DataFrame | None:
    """Compute pairwise Pearson correlation between missingness indicators."""
    if df.empty or len(df.columns) < 2:
        return None

    # Filter to columns that have at least some missingness and aren't completely missing
    cols_with_nulls = [c for c in df.columns if 0 < df[c].isna().sum() < len(df)]
    if len(cols_with_nulls) < 2:
        return None

    indicators = df[cols_with_nulls].isna().astype(int)
    # Compute correlation
    corr = indicators.corr()
    return corr
