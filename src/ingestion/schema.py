"""Automatic dataset schema inference, semantic type detection, and column profiling."""

import re

import pandas as pd

from src.ingestion.models import ColumnSchema, InferredType

IDENTIFIER_PATTERNS = re.compile(
    r"(^|[_.-])(id|key|code|uuid|guid|hash|pk|fk|index|num|number|identifier)([_.-]|$)",
    re.IGNORECASE,
)
TARGET_PATTERNS = re.compile(
    r"(^|[_.-])(target|label|churn|fraud|status|outcome|converted|default|price|revenue|sales|profit|rating|class)([_.-]|$)",
    re.IGNORECASE,
)


def normalize_column_name(name: str) -> str:
    """Transform raw column name into a clean, snake_case identifier."""
    replaced = re.sub(r"[\s\-]+", "_", str(name).strip())
    cleaned = re.sub(r"[^\w]", "", replaced)
    snake = re.sub(r"_+", "_", cleaned).lower().strip("_")
    return snake or "col"


def infer_column_schema(series: pd.Series, position: int) -> ColumnSchema:
    """Analyze a single Pandas Series to infer data types, constraints, and semantic hints."""
    col_name = str(series.name)
    normalized = normalize_column_name(col_name)

    total_rows = len(series)
    null_count = int(series.isna().sum())
    non_null_count = total_rows - null_count
    null_pct = round((null_count / total_rows * 100.0) if total_rows > 0 else 0.0, 2)

    pandas_dtype_str = str(series.dtype)

    if non_null_count == 0:
        return ColumnSchema(
            column_name=col_name,
            position=position,
            normalized_name=normalized,
            inferred_type="unknown",
            detected_pandas_dtype=pandas_dtype_str,
            non_null_count=0,
            null_count=null_count,
            null_percentage=100.0,
            distinct_count=0,
            unique_ratio=0.0,
            is_identifier=False,
            is_datetime_candidate=False,
            is_constant=True,
            is_near_constant=True,
            is_potential_target=False,
            sample_values=[],
            type_confidence=1.0,
            rationale="Column contains 100% missing values.",
        )

    # Calculate distinct count on non-null values
    non_null_series = series.dropna()
    distinct_count = int(non_null_series.nunique())
    unique_ratio = round(distinct_count / non_null_count, 4) if non_null_count > 0 else 0.0

    # Distinct sample values (up to 4, safely string-truncated)
    unique_samples = non_null_series.drop_duplicates().head(4).tolist()
    safe_samples = [
        str(v)[:40] if isinstance(v, str) and len(str(v)) > 40 else (v if not pd.isna(v) else None)
        for v in unique_samples
    ]

    is_constant = distinct_count == 1
    # Near constant: top value accounts for >= 95% of non-null records
    top_freq = non_null_series.value_counts().iloc[0] if distinct_count > 0 else 0
    is_near_constant = is_constant or (non_null_count > 10 and (top_freq / non_null_count) >= 0.95)

    is_id_name = bool(
        IDENTIFIER_PATTERNS.search(col_name) or IDENTIFIER_PATTERNS.search(normalized)
    )
    is_identifier = False
    inferred_type: InferredType = "unknown"
    confidence = 0.8
    rationale = ""
    proposed_conversion: str | None = None
    conversion_impact: str | None = None
    is_datetime_candidate = False

    # Check 1: Datetime dtype already recognized
    if pd.api.types.is_datetime64_any_dtype(series):
        inferred_type = "datetime"
        confidence = 1.0
        rationale = "Recognized native timestamp format."

    # Check 2: Boolean dtype or binary category
    elif pd.api.types.is_bool_dtype(series):
        inferred_type = "boolean"
        confidence = 1.0
        rationale = "Native boolean data type."

    # Check 3: Numeric types (int, float)
    elif pd.api.types.is_numeric_dtype(series):
        # Could be an integer ID column (e.g. user_id, order_id)
        if is_id_name and (unique_ratio > 0.8 or distinct_count == non_null_count):
            inferred_type = "identifier"
            is_identifier = True
            confidence = 0.9
            rationale = "Numeric sequence or key matching identifier patterns."
        elif distinct_count == 2 and set(non_null_series.unique()).issubset({0, 1}):
            inferred_type = "boolean"
            confidence = 0.9
            rationale = "Binary 0/1 indicator variable."
        elif distinct_count <= 10 and unique_ratio < 0.05 and pd.api.types.is_integer_dtype(series):
            inferred_type = "categorical"
            confidence = 0.75
            rationale = "Low-cardinality discrete integers representing categories."
        else:
            inferred_type = "numeric"
            confidence = 0.95
            rationale = "Quantitative numerical variable."

    # Check 4: Object / String / Categorical types
    else:
        # Check if text is datetime candidate
        if not is_id_name and distinct_count > 1 and non_null_count > 0:
            sample_for_dt = non_null_series.head(min(100, non_null_count))
            try:
                converted_dt = pd.to_datetime(sample_for_dt, errors="coerce", format="mixed")
                valid_dt_ratio = converted_dt.notna().sum() / len(sample_for_dt)
                if valid_dt_ratio >= 0.85:
                    is_datetime_candidate = True
                    proposed_conversion = "datetime64[ns]"
                    conversion_impact = (
                        f"Parses {valid_dt_ratio * 100:.0f}% of sampled values into dates, "
                        "enabling chronological trend analysis."
                    )
            except Exception:
                is_datetime_candidate = False

        if is_datetime_candidate and is_id_name is False:
            inferred_type = "datetime"
            confidence = 0.8
            rationale = "Text values cleanly convert into timestamps."

        # Check for boolean text representations
        elif distinct_count == 2 and {
            str(v).strip().lower() for v in non_null_series.unique()
        }.issubset({"true", "false", "yes", "no", "t", "f", "y", "n"}):
            inferred_type = "boolean"
            confidence = 0.9
            rationale = "Text values represent binary boolean flags."
            proposed_conversion = "bool"
            conversion_impact = "Normalizes text flags into efficient boolean indicators."

        # Check for identifier
        elif is_id_name or (unique_ratio >= 0.98 and total_rows > 30):
            inferred_type = "identifier"
            is_identifier = True
            confidence = 0.85
            rationale = "High cardinality and naming indicate an entity key or identifier."

        # Check for numeric stored as string
        elif non_null_count > 0:
            sample_numeric = pd.to_numeric(non_null_series.head(100), errors="coerce")
            if sample_numeric.notna().sum() / len(sample_numeric) >= 0.9:
                inferred_type = "numeric"
                confidence = 0.8
                rationale = "Numeric values currently formatted as strings."
                proposed_conversion = "float64"
                conversion_impact = "Enables quantitative statistics and distributions."
            elif distinct_count <= 60 or unique_ratio < 0.15:
                inferred_type = "categorical"
                confidence = 0.85
                rationale = "Discrete category values with moderate cardinality."
            else:
                inferred_type = "text"
                confidence = 0.8
                rationale = "Unstructured free-form text or high-cardinality descriptions."
        else:
            inferred_type = "categorical"
            confidence = 0.7
            rationale = "Default categorical text classification."

    # Potential target candidate detection
    is_potential_target = False
    if not is_identifier and not is_constant:
        if TARGET_PATTERNS.search(col_name) or TARGET_PATTERNS.search(normalized):
            is_potential_target = True
        elif inferred_type == "boolean":
            is_potential_target = True
        elif inferred_type == "categorical" and distinct_count in (2, 3, 4) and total_rows >= 50:
            is_potential_target = True

    return ColumnSchema(
        column_name=col_name,
        position=position,
        normalized_name=normalized,
        inferred_type=inferred_type,
        detected_pandas_dtype=pandas_dtype_str,
        non_null_count=non_null_count,
        null_count=null_count,
        null_percentage=null_pct,
        distinct_count=distinct_count,
        unique_ratio=unique_ratio,
        is_identifier=is_identifier,
        is_datetime_candidate=is_datetime_candidate,
        is_constant=is_constant,
        is_near_constant=is_near_constant,
        is_potential_target=is_potential_target,
        sample_values=safe_samples,
        type_confidence=round(confidence, 2),
        rationale=rationale,
        proposed_conversion=proposed_conversion,
        conversion_impact=conversion_impact,
    )


def infer_dataset_schema(df: pd.DataFrame) -> dict[str, ColumnSchema]:
    """Infer schemas and semantic classifications for all columns in a DataFrame."""
    schemas: dict[str, ColumnSchema] = {}
    for pos, col in enumerate(df.columns):
        series = df[col]
        # In case df has duplicate column names and returns a DataFrame instead of Series
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]
        schemas[str(col)] = infer_column_schema(series, pos)
    return schemas
