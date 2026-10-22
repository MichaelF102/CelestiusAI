"""Validation helpers for uploaded files, formats, and resource limits."""

import hashlib
from pathlib import Path
from typing import BinaryIO

from src.core.exceptions import DataIngestionError

SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".parquet"}


def validate_file_extension(filename: str) -> str:
    """Validate that the file extension is supported and return the normalized format."""
    path = Path(filename)
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        supported_str = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise DataIngestionError(
            message=f"Unsupported file format '{ext}' for file '{filename}'. Supported: {supported_str}",
            user_message=f"The file '{filename}' has an unsupported extension. Please upload a CSV, Excel (.xlsx/.xls), or Parquet file.",
        )
    return ext.lstrip(".")


def validate_file_size(size_bytes: int, max_size_mb: int) -> None:
    """Check that file size does not exceed the configured maximum upload limit."""
    if size_bytes <= 0:
        raise DataIngestionError(
            message="Uploaded file is empty (0 bytes).",
            user_message="The uploaded file is empty (0 bytes). Please upload a valid dataset.",
        )

    max_bytes = max_size_mb * 1024 * 1024
    if size_bytes > max_bytes:
        size_mb = size_bytes / (1024 * 1024)
        raise DataIngestionError(
            message=f"File size {size_mb:.2f} MB exceeds maximum allowed limit {max_size_mb} MB.",
            user_message=f"The file size ({size_mb:.1f} MB) exceeds the configured upload limit of {max_size_mb} MB.",
        )


def compute_content_fingerprint(
    stream: BinaryIO | bytes, max_bytes_sample: int = 2 * 1024 * 1024
) -> str:
    """Compute a deterministic SHA-256 content fingerprint without logging or leaking data."""
    hasher = hashlib.sha256()
    if isinstance(stream, bytes):
        hasher.update(stream[:max_bytes_sample])
    else:
        current_pos = stream.tell() if hasattr(stream, "tell") else 0
        chunk = stream.read(max_bytes_sample)
        hasher.update(chunk)
        if hasattr(stream, "seek"):
            stream.seek(current_pos)
    return hasher.hexdigest()[:16]
