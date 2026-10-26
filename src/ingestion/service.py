"""Streamlit-independent ingestion workflow shared by every upload entry point."""

from __future__ import annotations

from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

from src.core.session_state import (
    KEY_BASELINE_QUALITY_SCORE,
    KEY_CURRENT_DATASET,
    KEY_DATA_QUALITY_REPORT,
    KEY_DATASET_METADATA,
    KEY_DATASET_SCHEMA,
    KEY_TRANSFORMATION_PIPELINE,
    reset_dataset_state,
)
from src.ingestion.loader import ManagedDataset, load_dataset_from_stream
from src.ingestion.transformation_history import TransformationPipeline
from src.models.common import DatasetMetadata
from src.profiling.profiler import profile_dataset


def needs_ingestion(
    state: MutableMapping[str, Any], filename: str, sheet_name: str | None = None
) -> bool:
    """Return True if `filename` (and sheet, for Excel) is not already the active dataset."""
    current: ManagedDataset | None = state.get(KEY_CURRENT_DATASET)
    if current is None or current.filename != filename:
        return True
    return bool(sheet_name and current.ingestion_report.sheet_name_used != sheet_name)


def ingest_into_state(
    state: MutableMapping[str, Any],
    file_bytes: bytes,
    filename: str,
    max_size_mb: int,
    sheet_name: str | None = None,
) -> ManagedDataset:
    """Load, profile, and register an uploaded file as the active dataset.

    Raises DataAnalystError (user-facing) on validation/parse failures, leaving the
    previously active dataset untouched.
    """
    managed = load_dataset_from_stream(
        file_stream=file_bytes,
        filename=filename,
        sheet_name=sheet_name,
        max_size_mb=max_size_mb,
    )
    profile = profile_dataset(
        managed.working_df,
        dataset_id=managed.dataset_id,
        schemas=managed.schema_info,
    )

    # Only reset once the new file has loaded successfully.
    reset_dataset_state(state)
    state[KEY_TRANSFORMATION_PIPELINE] = TransformationPipeline()
    state[KEY_CURRENT_DATASET] = managed
    state[KEY_DATA_QUALITY_REPORT] = profile
    state[KEY_BASELINE_QUALITY_SCORE] = profile.quality_score.overall_score
    state[KEY_DATASET_SCHEMA] = managed.schema_info
    state[KEY_DATASET_METADATA] = DatasetMetadata(
        filename=managed.filename,
        file_format=managed.file_format,
        row_count=len(managed.working_df),
        column_count=len(managed.working_df.columns),
        dataset_id=managed.dataset_id,
        file_size_bytes=managed.file_size_bytes,
    )
    return managed


SAMPLE_DATASET_PATH = (
    Path(__file__).resolve().parents[2] / "sample_data" / "ecommerce_customer_churn.csv"
)


def load_sample_dataset(state: MutableMapping[str, Any], max_size_mb: int) -> ManagedDataset:
    """Load the bundled demo dataset as the active dataset."""
    return ingest_into_state(
        state,
        file_bytes=SAMPLE_DATASET_PATH.read_bytes(),
        filename=SAMPLE_DATASET_PATH.name,
        max_size_mb=max_size_mb,
    )


SAMPLE_CATALOG: tuple[dict[str, str], ...] = (
    {
        "key": "hr_attrition",
        "icon": "🧑‍💼",
        "title": "HR attrition",
        "blurb": "Who leaves and why: overtime, income, tenure.",
        "use": "Classification · Survival",
    },
    {
        "key": "house_prices",
        "icon": "🏠",
        "title": "House prices",
        "blurb": "Listings with area, neighborhood and a few extreme prices.",
        "use": "Regression · Outliers",
    },
    {
        "key": "retail_sales_daily",
        "icon": "🛒",
        "title": "Retail sales (daily)",
        "blurb": "2.5 years of trend, weekly and yearly seasonality, promotions.",
        "use": "Time series · Forecasting",
    },
    {
        "key": "marketing_ab_test",
        "icon": "🧬",
        "title": "Marketing A/B test",
        "blurb": "Control vs treatment conversions by device and channel.",
        "use": "A/B testing · Causal",
    },
    {
        "key": "credit_default",
        "icon": "💳",
        "title": "Credit default",
        "blurb": "Imbalanced defaults with missing income and history.",
        "use": "Imbalanced classification",
    },
    {
        "key": "patient_readmission",
        "icon": "🏥",
        "title": "Patient readmission",
        "blurb": "30-day readmission with plenty of missing clinical values.",
        "use": "Data quality · Classification",
    },
    {
        "key": "iot_sensor_readings",
        "icon": "📟",
        "title": "IoT sensor readings",
        "blurb": "15-minute machine data with drift, spikes and gaps.",
        "use": "Anomalies · Time series",
    },
    {
        "key": "product_reviews",
        "icon": "💬",
        "title": "Product reviews",
        "blurb": "Ratings with short review text.",
        "use": "Text · Classification",
    },
    {
        "key": "student_performance",
        "icon": "🎓",
        "title": "Student performance",
        "blurb": "Scores by study hours, attendance and tutoring.",
        "use": "Regression · Group tests",
    },
    {
        "key": "airline_delays",
        "icon": "✈️",
        "title": "Airline delays",
        "blurb": "Delays by carrier, weather and hour, with messy casing.",
        "use": "Regression · Cleaning",
    },
)


def load_catalog_sample(
    state: MutableMapping[str, Any], key: str, max_size_mb: int
) -> ManagedDataset:
    """Load one of the bundled catalog datasets as the active dataset."""
    if key not in {c["key"] for c in SAMPLE_CATALOG}:
        raise ValueError(f"Unknown sample dataset: {key}")
    path = SAMPLE_DATASET_PATH.parent / f"{key}.csv"
    return ingest_into_state(
        state, file_bytes=path.read_bytes(), filename=path.name, max_size_mb=max_size_mb
    )


def ingest_dataframe(
    state: MutableMapping[str, Any], df: Any, name: str, max_size_mb: int
) -> ManagedDataset:
    """Register an in-memory DataFrame (join result, SQL result, ...) as the active dataset."""
    import io

    buffer = io.BytesIO()
    try:
        df.to_parquet(buffer, engine="pyarrow")
    except Exception:
        safe = df.copy()
        for col in safe.select_dtypes("object").columns:
            safe[col] = safe[col].map(lambda v: None if v is None or v != v else str(v))
        buffer = io.BytesIO()
        safe.to_parquet(buffer, engine="pyarrow")
    filename = name if name.lower().endswith(".parquet") else f"{name}.parquet"
    return ingest_into_state(state, buffer.getvalue(), filename, max_size_mb)
