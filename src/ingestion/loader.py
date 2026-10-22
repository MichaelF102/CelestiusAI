"""Dataset loader supporting CSV, Excel, and Parquet with encoding fallbacks and safety controls."""

import io
import uuid
from typing import BinaryIO

import openpyxl
import pandas as pd

from src.config.settings import get_settings
from src.core.exceptions import DataIngestionError
from src.core.logging_config import get_logger
from src.ingestion.models import ColumnSchema, DatasetIngestionReport
from src.ingestion.schema import infer_dataset_schema
from src.ingestion.validators import (
    compute_content_fingerprint,
    validate_file_extension,
    validate_file_size,
)

logger = get_logger("ingestion.loader")

CSV_ENCODING_FALLBACKS = ["utf-8", "utf-8-sig", "latin1", "cp1252", "iso-8859-1"]
CSV_DELIMITERS = [",", ";", "\t", "|"]


class ManagedDataset:
    """Managed container encapsulating raw immutable data, working copy, and schema metadata."""

    def __init__(
        self,
        dataset_id: str,
        filename: str,
        file_format: str,
        file_size_bytes: int,
        fingerprint: str,
        raw_df: pd.DataFrame,
        working_df: pd.DataFrame,
        schema_info: dict[str, ColumnSchema],
        ingestion_report: DatasetIngestionReport,
    ) -> None:
        self.dataset_id = dataset_id
        self.filename = filename
        self.file_format = file_format
        self.file_size_bytes = file_size_bytes
        self.fingerprint = fingerprint
        # Immutable original data
        self._raw_df = raw_df.copy(deep=True)
        # Working copy for approved transformations
        self.working_df = working_df.copy(deep=True)
        self.schema_info = schema_info
        self.ingestion_report = ingestion_report
        self.version: int = 1

    @property
    def raw_df(self) -> pd.DataFrame:
        """Original ingested DataFrame (immutable reference)."""
        return self._raw_df.copy(deep=True)

    def reset_working_df(self) -> None:
        """Reset the working dataset to match the original raw dataset."""
        self.working_df = self._raw_df.copy(deep=True)
        self.schema_info = infer_dataset_schema(self.working_df)
        self.version += 1

    def update_working_df(self, new_df: pd.DataFrame) -> None:
        """Update active working dataset after an approved transformation."""
        self.working_df = new_df.copy(deep=True)
        self.schema_info = infer_dataset_schema(self.working_df)
        self.version += 1


def get_excel_sheet_names(stream: BinaryIO | bytes) -> list[str]:
    """Retrieve available sheet names from an Excel workbook without reading full contents."""
    if isinstance(stream, bytes):
        file_obj = io.BytesIO(stream)
    else:
        stream.seek(0)
        file_obj = io.BytesIO(stream.read())
        stream.seek(0)

    try:
        workbook = openpyxl.load_workbook(file_obj, read_only=True, keep_links=False)
        sheets = workbook.sheetnames
        workbook.close()
        return sheets
    except Exception as e:
        logger.warning(f"Could not read Excel sheets via openpyxl: {e}")
        return ["Sheet1"]


def _sniff_csv_delimiter(sample_bytes: bytes, encoding: str) -> str:
    """Determine the most plausible CSV delimiter by inspecting first few lines."""
    try:
        sample_text = sample_bytes[:8192].decode(encoding, errors="ignore")
        lines = [line.strip() for line in sample_text.splitlines() if line.strip()][:5]
        if not lines:
            return ","

        delimiter_scores: dict[str, int] = {}
        for delim in CSV_DELIMITERS:
            counts = [line.count(delim) for line in lines]
            if counts and counts[0] > 0 and len(set(counts)) == 1:
                delimiter_scores[delim] = counts[0] * 10
            else:
                delimiter_scores[delim] = sum(counts)

        best_delim = max(delimiter_scores, key=lambda k: delimiter_scores[k])
        return best_delim if delimiter_scores[best_delim] > 0 else ","
    except Exception:
        return ","


def _read_csv_with_fallbacks(raw_bytes: bytes) -> tuple[pd.DataFrame, str, str]:
    """Attempt reading CSV with multiple encodings and detected delimiters."""
    last_error: Exception | None = None

    for encoding in CSV_ENCODING_FALLBACKS:
        delimiter = _sniff_csv_delimiter(raw_bytes, encoding)
        try:
            df = pd.read_csv(
                io.BytesIO(raw_bytes),
                encoding=encoding,
                sep=delimiter,
                engine="python",
                on_bad_lines="error",
            )
            if df.empty and len(raw_bytes) > 50:
                continue
            return df, encoding, delimiter
        except Exception as e:
            last_error = e
            continue

    raise DataIngestionError(
        message=f"Failed to parse CSV file with supported encodings. Last error: {last_error}",
        user_message="Could not parse the CSV file. Please ensure it is well-formatted and not corrupted.",
    )


def load_dataset_from_stream(
    file_stream: BinaryIO | bytes,
    filename: str,
    sheet_name: str | None = None,
    max_size_mb: int | None = None,
) -> ManagedDataset:
    """Ingest, validate, and profile a raw data stream into a ManagedDataset container."""
    settings = get_settings()
    configured_max_mb = max_size_mb or settings.max_upload_size_mb

    file_format = validate_file_extension(filename)

    # Read bytes into memory for validation and parsing
    if isinstance(file_stream, bytes):
        raw_bytes = file_stream
    else:
        file_stream.seek(0)
        raw_bytes = file_stream.read()

    file_size_bytes = len(raw_bytes)
    validate_file_size(file_size_bytes, configured_max_mb)

    fingerprint = compute_content_fingerprint(raw_bytes)
    dataset_id = f"ds_{uuid.uuid4().hex[:10]}"

    detected_encoding: str | None = None
    detected_delimiter: str | None = None
    sheet_used: str | None = None
    available_sheets: list[str] = []

    try:
        if file_format == "csv":
            df, detected_encoding, detected_delimiter = _read_csv_with_fallbacks(raw_bytes)

        elif file_format in ("xlsx", "xls"):
            available_sheets = get_excel_sheet_names(raw_bytes)
            sheet_used = sheet_name or (available_sheets[0] if available_sheets else None)
            df = pd.read_excel(
                io.BytesIO(raw_bytes),
                sheet_name=sheet_used,
                engine="openpyxl" if file_format == "xlsx" else None,
            )

        elif file_format == "parquet":
            df = pd.read_parquet(io.BytesIO(raw_bytes), engine="pyarrow")

        else:
            raise DataIngestionError(
                message=f"Unsupported format: {file_format}",
                user_message=f"Unsupported format: {file_format}",
            )

    except DataIngestionError:
        raise
    except Exception as err:
        logger.error(f"Failed to ingest file '{filename}': {err}")
        raise DataIngestionError(
            message=f"Failed reading {file_format.upper()} data from '{filename}': {err}",
            user_message=f"Failed to load dataset '{filename}'. Please verify the file structure.",
        ) from err

    if df.empty:
        raise DataIngestionError(
            message=f"The uploaded file '{filename}' contains 0 rows of data.",
            user_message=f"The uploaded file '{filename}' is empty or contains no data rows.",
        )

    # Check for duplicate column names in source
    has_duplicates = len(df.columns) != len(set(df.columns))
    duplicate_names: list[str] = []

    if file_format == "csv" and detected_encoding and detected_delimiter:
        try:
            first_line = raw_bytes.splitlines()[0].decode(detected_encoding, errors="ignore")
            raw_tokens = [
                t.strip().strip('"').strip("'") for t in first_line.split(detected_delimiter)
            ]
            seen_tokens: set[str] = set()
            for t in raw_tokens:
                if t in seen_tokens and t not in duplicate_names:
                    duplicate_names.append(t)
                seen_tokens.add(t)
            if duplicate_names:
                has_duplicates = True
        except Exception:
            pass

    if has_duplicates:
        # Disambiguate columns to prevent Pandas indexing collision
        new_cols: list[str] = []
        counts: dict[str, int] = {}
        for col in df.columns:
            # Strip Pandas auto .1, .2 suffixes if present for duplicate names
            col_base = (
                str(col).split(".")[0]
                if any(str(col).startswith(f"{d}.") for d in duplicate_names)
                else str(col)
            )
            if col_base in duplicate_names:
                c_cnt = counts.get(col_base, 0)
                counts[col_base] = c_cnt + 1
                new_cols.append(col_base if c_cnt == 0 else f"{col_base}_{c_cnt}")
            else:
                new_cols.append(str(col))
        df.columns = new_cols
        logger.warning(f"Disambiguated duplicate column names in '{filename}': {duplicate_names}")

    # Generate initial schema
    schema_info = infer_dataset_schema(df)

    report = DatasetIngestionReport(
        dataset_id=dataset_id,
        filename=filename,
        file_format=file_format,
        file_size_bytes=file_size_bytes,
        content_fingerprint=fingerprint,
        row_count=len(df),
        column_count=len(df.columns),
        delimiter_detected=detected_delimiter,
        encoding_used=detected_encoding,
        sheet_name_used=sheet_used,
        available_sheets=available_sheets,
        has_duplicate_columns=has_duplicates,
        duplicate_column_names=duplicate_names,
    )

    logger.info(
        f"Successfully ingested dataset '{filename}' (ID: {dataset_id}, {len(df)} rows, {len(df.columns)} cols)"
    )

    return ManagedDataset(
        dataset_id=dataset_id,
        filename=filename,
        file_format=file_format,
        file_size_bytes=file_size_bytes,
        fingerprint=fingerprint,
        raw_df=df,
        working_df=df,
        schema_info=schema_info,
        ingestion_report=report,
    )
