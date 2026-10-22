"""Tests for data cleaning transformations, preview simulation, pipeline undo, and reset."""

import pandas as pd
import pytest
from src.core.exceptions import AnalyticalExecutionError
from src.ingestion.loader import ManagedDataset
from src.ingestion.models import DatasetIngestionReport
from src.ingestion.schema import infer_dataset_schema
from src.ingestion.transformation_history import TransformationPipeline
from src.ingestion.transformations import apply_transformation, preview_transformation
from src.profiling.models import TransformationSpec


def create_sample_managed_dataset() -> ManagedDataset:
    """Helper fixture to create a ManagedDataset for testing transformations."""
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 4],  # duplicate row 4
            "name": [" Alice ", "Bob", "Charlie", "David", "David"],
            "score": [85.0, None, 92.0, 78.0, 78.0],
            "useless_col": [1, 1, 1, 1, 1],
        }
    )
    schema = infer_dataset_schema(df)
    report = DatasetIngestionReport(
        dataset_id="ds_test",
        filename="test.csv",
        file_format="csv",
        file_size_bytes=100,
        content_fingerprint="abc123",
        row_count=5,
        column_count=4,
    )
    return ManagedDataset(
        dataset_id="ds_test",
        filename="test.csv",
        file_format="csv",
        file_size_bytes=100,
        fingerprint="abc123",
        raw_df=df,
        working_df=df,
        schema_info=schema,
        ingestion_report=report,
    )


def test_preview_transformation_does_not_mutate():
    """Verify preview generates statistics without mutating the underlying DataFrame."""
    managed = create_sample_managed_dataset()
    original_working = managed.working_df.copy(deep=True)

    spec = TransformationSpec(
        spec_id="spec_1",
        action_type="strip_whitespace",
        column_name="name",
    )

    preview = preview_transformation(managed.working_df, spec)
    assert preview.affected_cells_count >= 1
    # Verify underlying dataset was not modified
    assert managed.working_df.equals(original_working)


def test_apply_drop_column():
    """Verify drop_column removes target column and logs record."""
    managed = create_sample_managed_dataset()
    spec = TransformationSpec(
        spec_id="spec_drop",
        action_type="drop_column",
        column_name="useless_col",
    )

    new_df, record = apply_transformation(managed.working_df, spec)
    assert "useless_col" not in new_df.columns
    assert len(new_df.columns) == 3
    assert record.cols_before == 4
    assert record.cols_after == 3


def test_apply_drop_duplicates():
    """Verify drop_duplicates removes duplicate rows."""
    managed = create_sample_managed_dataset()
    spec = TransformationSpec(
        spec_id="spec_dup",
        action_type="drop_duplicates",
    )

    new_df, record = apply_transformation(managed.working_df, spec)
    assert len(new_df) == 4  # Reduced from 5 to 4
    assert record.rows_before == 5
    assert record.rows_after == 4


def test_apply_impute_numeric():
    """Verify numeric imputation with median strategy."""
    managed = create_sample_managed_dataset()
    spec = TransformationSpec(
        spec_id="spec_imp",
        action_type="impute_numeric",
        column_name="score",
        parameters={"strategy": "median"},
    )

    new_df, record = apply_transformation(managed.working_df, spec)
    assert new_df["score"].isna().sum() == 0
    assert record.cells_modified == 1


def test_apply_strip_whitespace():
    """Verify stripping whitespace trims strings."""
    managed = create_sample_managed_dataset()
    spec = TransformationSpec(
        spec_id="spec_ws",
        action_type="strip_whitespace",
        column_name="name",
    )

    new_df, record = apply_transformation(managed.working_df, spec)
    assert new_df.loc[0, "name"] == "Alice"
    assert record.cells_modified >= 1


def test_transformation_pipeline_undo_and_reset():
    """Verify pipeline records history, executes undo, and resets to immutable raw_df."""
    managed = create_sample_managed_dataset()
    pipeline = TransformationPipeline()

    spec1 = TransformationSpec(spec_id="s1", action_type="drop_duplicates")
    spec2 = TransformationSpec(spec_id="s2", action_type="drop_column", column_name="useless_col")

    pipeline.apply(spec1, managed)
    assert len(managed.working_df) == 4
    assert len(pipeline.records) == 1

    pipeline.apply(spec2, managed)
    assert "useless_col" not in managed.working_df.columns
    assert len(pipeline.records) == 2

    # Verify original raw_df is strictly unchanged
    assert len(managed.raw_df) == 5
    assert "useless_col" in managed.raw_df.columns

    # Undo the drop_column transformation
    undone = pipeline.undo(managed)
    assert undone is not None
    assert len(pipeline.records) == 1
    assert "useless_col" in managed.working_df.columns
    assert len(managed.working_df) == 4  # Still deduplicated

    # Reset all transformations
    pipeline.reset(managed)
    assert len(pipeline.records) == 0
    assert len(managed.working_df) == 5
    assert managed.working_df.equals(managed.raw_df)


def test_invalid_column_raises_analytical_error():
    """Verify invalid column name raises AnalyticalExecutionError."""
    managed = create_sample_managed_dataset()
    spec = TransformationSpec(
        spec_id="s_bad",
        action_type="drop_column",
        column_name="non_existent_column",
    )

    with pytest.raises(AnalyticalExecutionError):
        apply_transformation(managed.working_df, spec)
