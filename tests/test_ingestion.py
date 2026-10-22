"""Comprehensive tests for data ingestion: CSV, Excel, Parquet, encodings, and size enforcement."""

import io

import pandas as pd
import pytest
from src.core.exceptions import DataIngestionError
from src.ingestion.loader import get_excel_sheet_names, load_dataset_from_stream
from src.ingestion.validators import (
    compute_content_fingerprint,
    validate_file_extension,
    validate_file_size,
)


def test_validate_file_extension():
    """Verify supported and unsupported extensions."""
    assert validate_file_extension("sales.csv") == "csv"
    assert validate_file_extension("data.XLSX") == "xlsx"
    assert validate_file_extension("report.xls") == "xls"
    assert validate_file_extension("metrics.parquet") == "parquet"

    with pytest.raises(DataIngestionError) as exc:
        validate_file_extension("malware.exe")
    assert "Unsupported file format" in str(exc.value)


def test_validate_file_size():
    """Verify empty files and oversized files trigger DataIngestionError."""
    with pytest.raises(DataIngestionError) as exc_empty:
        validate_file_size(0, max_size_mb=100)
    assert "empty" in str(exc_empty.value).lower()

    with pytest.raises(DataIngestionError) as exc_big:
        validate_file_size(150 * 1024 * 1024, max_size_mb=100)
    assert "exceeds" in str(exc_big.value).lower()


def test_fingerprint_deterministic():
    """Verify SHA-256 fingerprint generation is consistent and non-leaking."""
    data = b"col1,col2\n1,2\n3,4"
    fp1 = compute_content_fingerprint(data)
    fp2 = compute_content_fingerprint(data)
    assert fp1 == fp2
    assert len(fp1) == 16


def test_ingest_csv_standard():
    """Verify standard CSV loading and ManagedDataset creation."""
    csv_bytes = b"customer_id,age,revenue,segment\n101,25,120.5,Retail\n102,34,310.0,Enterprise\n"
    managed = load_dataset_from_stream(csv_bytes, filename="customers.csv")

    assert managed.filename == "customers.csv"
    assert managed.file_format == "csv"
    assert len(managed.working_df) == 2
    assert len(managed.working_df.columns) == 4
    assert managed.raw_df.equals(managed.working_df)


def test_ingest_csv_semicolon_and_latin1():
    """Verify CSV delimiter sniffing and encoding fallback."""
    csv_text = "id;name;montant\n1;Hélène;50,5\n2;François;99,9\n"
    csv_bytes = csv_text.encode("latin1")

    managed = load_dataset_from_stream(csv_bytes, filename="french_sales.csv")
    assert len(managed.working_df) == 2
    assert managed.ingestion_report.delimiter_detected == ";"
    assert "Hélène" in managed.working_df["name"].values


def test_ingest_csv_duplicate_column_disambiguation():
    """Verify duplicate column names are safely disambiguated to prevent Pandas bugs."""
    csv_bytes = b"id,val,val\n1,10,20\n2,30,40\n"
    managed = load_dataset_from_stream(csv_bytes, filename="dup_cols.csv")

    assert managed.ingestion_report.has_duplicate_columns is True
    assert "val" in managed.ingestion_report.duplicate_column_names
    # Check that columns were renamed to unique strings
    assert list(managed.working_df.columns) == ["id", "val", "val_1"]


def test_ingest_excel_xlsx():
    """Verify Excel .xlsx ingestion with openpyxl and sheet inspection."""
    df_src = pd.DataFrame({"product": ["Widget", "Gadget"], "price": [19.99, 49.99]})
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df_src.to_excel(writer, sheet_name="Products", index=False)
        df_src.to_excel(writer, sheet_name="Archive", index=False)

    excel_bytes = buf.getvalue()
    sheets = get_excel_sheet_names(excel_bytes)
    assert "Products" in sheets
    assert "Archive" in sheets

    managed = load_dataset_from_stream(excel_bytes, filename="catalog.xlsx", sheet_name="Products")
    assert managed.file_format == "xlsx"
    assert len(managed.working_df) == 2
    assert "product" in managed.working_df.columns


def test_ingest_parquet():
    """Verify Parquet ingestion via PyArrow."""
    df_src = pd.DataFrame({"sensor_id": [1, 2, 3], "temp": [21.5, 22.0, 21.8]})
    buf = io.BytesIO()
    df_src.to_parquet(buf, engine="pyarrow")
    parquet_bytes = buf.getvalue()

    managed = load_dataset_from_stream(parquet_bytes, filename="sensors.parquet")
    assert managed.file_format == "parquet"
    assert len(managed.working_df) == 3
    assert "sensor_id" in managed.working_df.columns


def test_ingest_empty_data_raises():
    """Verify 0-row dataset raises DataIngestionError."""
    csv_bytes = b"col1,col2\n"  # Headers only, 0 data rows
    with pytest.raises(DataIngestionError) as exc:
        load_dataset_from_stream(csv_bytes, filename="empty_table.csv")
    assert "contains 0 rows" in str(exc.value)
