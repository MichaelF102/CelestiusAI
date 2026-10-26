"""PII detection and redaction for text that may be sent to an external LLM."""

from __future__ import annotations

import re
from collections import Counter

import pandas as pd

_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]?){13,19}\b"),
    # separators are required so plain digit runs (order ids, timestamps) are not treated as phones
    "phone": re.compile(
        r"(?<!\d)(?:\+\d{1,3}[ .-]?)?(?:\(\d{2,4}\)|\d{2,4})[ .-]\d{3,4}[ .-]?\d{3,4}(?!\d)"
    ),
    "ip_address": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}
_NAME_HINTS: dict[str, tuple[str, ...]] = {
    "email": ("email", "e_mail", "mail"),
    "phone": ("phone", "mobile", "tel", "fax"),
    "ssn": ("ssn", "social_security", "national_id", "passport"),
    "credit_card": ("credit_card", "card_number", "cc_num", "iban"),
    "address": ("address", "street", "zip", "postcode", "postal"),
    "name": ("first_name", "last_name", "full_name", "surname"),
    "birth_date": ("dob", "birth", "date_of_birth"),
    "secret": ("password", "passwd", "secret", "token", "api_key"),
}


def _luhn_ok(digits: str) -> bool:
    nums = [int(c) for c in digits if c.isdigit()]
    if not 13 <= len(nums) <= 19:
        return False
    total = 0
    for i, n in enumerate(reversed(nums)):
        if i % 2:
            n = n * 2 - 9 if n * 2 > 9 else n * 2
        total += n
    return total % 10 == 0


def redact_text(text: str) -> tuple[str, dict[str, int]]:
    """Replace emails, SSNs, valid card numbers, phones and IPs with placeholders."""
    counts: Counter[str] = Counter()

    def make(kind: str):  # type: ignore[no-untyped-def]
        def repl(match: re.Match[str]) -> str:
            if kind == "credit_card" and not _luhn_ok(match.group(0)):
                return match.group(0)
            counts[kind] += 1
            return f"[{kind.upper()}]"

        return repl

    for kind in ("email", "ssn", "credit_card", "ip_address", "phone"):
        text = _PATTERNS[kind].sub(make(kind), text)
    return text, dict(counts)


def detect_sensitive_columns(df: pd.DataFrame, sample: int = 500) -> list[dict[str, object]]:
    """Flag columns whose name or values look like personal data."""
    findings: list[dict[str, object]] = []
    for col in df.columns:
        lowered = str(col).lower()
        by_name = next(
            (k for k, hints in _NAME_HINTS.items() if any(h in lowered for h in hints)), None
        )
        by_value, share = None, 0.0
        s = df[col]
        if not pd.api.types.is_numeric_dtype(s) or s.dtype == object:
            values = s.dropna().astype(str).head(sample)
            if len(values):
                for kind in ("email", "ssn", "phone", "ip_address"):
                    hit = values.str.fullmatch(_PATTERNS[kind].pattern).mean()
                    if hit >= 0.2:
                        by_value, share = kind, float(hit)
                        break
        if by_name or by_value:
            findings.append(
                {
                    "column": col,
                    "kind": by_value or by_name,
                    "evidence": f"{share:.0%} of values match" if by_value else "column name",
                }
            )
    return findings


def mask_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Replace the listed columns with irreversible short hashes (keeps joins/groupings usable)."""
    import hashlib

    out = df.copy()
    for col in columns:
        out[col] = out[col].map(
            lambda v: None if pd.isna(v) else hashlib.sha256(str(v).encode()).hexdigest()[:10]
        )
    return out
