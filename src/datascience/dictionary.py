"""Editable data dictionary: per-column business meaning, glossary, and AI context."""

from __future__ import annotations

import io
from collections.abc import Mapping
from typing import Any

import pandas as pd

from src.core.code_snippets import make_snippet

GLOSSARY_KEY = "__glossary__"
ROLES = ("feature", "identifier", "target", "date", "measure", "dimension", "ignore")


def _role_for(schema: Any) -> str:
    if schema.is_identifier:
        return "identifier"
    if getattr(schema, "is_potential_target", False):
        return "target"
    if schema.inferred_type == "datetime":
        return "date"
    if schema.inferred_type == "numeric":
        return "measure"
    if schema.inferred_type in ("categorical", "boolean"):
        return "dimension"
    return "feature"


def _auto_description(col: str, schema: Any) -> str:
    kind = schema.inferred_type
    if schema.is_identifier:
        return "Unique identifier for each record."
    if kind == "datetime":
        return "Date/time of the event or record."
    if kind == "boolean":
        return "Yes/no flag."
    if kind == "numeric":
        return f"Numeric measure ({schema.distinct_count:,} distinct values)."
    if kind == "categorical":
        return f"Category with {schema.distinct_count:,} distinct values."
    return f"{kind.capitalize()} column."


def build_dictionary(
    df: pd.DataFrame, schema: Mapping[str, Any], existing: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Create/refresh entries for every column, preserving anything the user already wrote."""
    existing = dict(existing or {})
    out: dict[str, Any] = {}
    for col in df.columns:
        s = schema.get(col)
        prior = existing.get(col, {})
        series = df[col]
        sample = series.dropna()
        auto = _auto_description(col, s) if s else ""
        out[col] = {
            "description": prior.get("description") or auto,
            "described_by_user": bool(prior.get("described_by_user")),
            "role": prior.get("role") or (_role_for(s) if s else "feature"),
            "unit": prior.get("unit", ""),
            "rename_to": prior.get("rename_to", ""),
            "type": s.inferred_type if s else str(series.dtype),
            "missing_pct": round(float(series.isna().mean() * 100), 2),
            "distinct": int(series.nunique(dropna=True)),
            "example": str(sample.iloc[0])[:60] if len(sample) else "",
        }
    if GLOSSARY_KEY in existing:
        out[GLOSSARY_KEY] = existing[GLOSSARY_KEY]
    return out


def columns_only(dictionary: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in dictionary.items() if k != GLOSSARY_KEY}


def to_frame(dictionary: Mapping[str, Any]) -> pd.DataFrame:
    rows = [{"column": col, **entry} for col, entry in columns_only(dictionary).items()]
    return pd.DataFrame(
        rows,
        columns=[
            "column",
            "description",
            "role",
            "unit",
            "rename_to",
            "type",
            "missing_pct",
            "distinct",
            "example",
        ],
    )


def apply_edits(dictionary: Mapping[str, Any], edited: pd.DataFrame) -> dict[str, Any]:
    """Merge a data_editor result back; changed descriptions are marked as user-written."""
    out = {k: dict(v) if isinstance(v, dict) else v for k, v in dictionary.items()}
    for _, row in edited.iterrows():
        col = row["column"]
        if col not in out or col == GLOSSARY_KEY:
            continue
        entry = out[col]
        new_desc = "" if pd.isna(row["description"]) else str(row["description"]).strip()
        if new_desc != entry.get("description"):
            entry["description"] = new_desc
            entry["described_by_user"] = bool(new_desc)
        role = row.get("role")
        entry["role"] = role if role in ROLES else entry.get("role", "feature")
        entry["unit"] = "" if pd.isna(row.get("unit")) else str(row.get("unit", "")).strip()
        entry["rename_to"] = (
            "" if pd.isna(row.get("rename_to")) else str(row.get("rename_to", "")).strip()
        )
    return out


def get_glossary(dictionary: Mapping[str, Any]) -> list[dict[str, str]]:
    return list(dictionary.get(GLOSSARY_KEY, []))


def set_glossary(dictionary: Mapping[str, Any], entries: list[dict[str, str]]) -> dict[str, Any]:
    cleaned = [
        {"term": e["term"].strip(), "definition": e["definition"].strip()}
        for e in entries
        if e.get("term")
        and str(e["term"]).strip()
        and e.get("definition")
        and str(e["definition"]).strip()
    ]
    out = dict(dictionary)
    out[GLOSSARY_KEY] = cleaned
    return out


def context_for_llm(dictionary: Mapping[str, Any], max_chars: int = 3000) -> str:
    """Only user-written meaning (never sample values) is shared with a language model."""
    lines = []
    for col, entry in columns_only(dictionary).items():
        if entry.get("described_by_user"):
            unit = f" [{entry['unit']}]" if entry.get("unit") else ""
            lines.append(f"- {col}{unit}: {entry['description']}")
    glossary = get_glossary(dictionary)
    if glossary:
        lines.append("Glossary:")
        lines += [f"- {g['term']}: {g['definition']}" for g in glossary]
    return "\n".join(lines)[:max_chars]


def to_markdown(dictionary: Mapping[str, Any], title: str = "Data dictionary") -> str:
    lines = [
        f"# {title}",
        "",
        "| Column | Description | Role | Unit | Type | Missing % | Distinct |",
        "|---|---|---|---|---|---|---|",
    ]
    for col, e in columns_only(dictionary).items():
        desc = str(e.get("description", "")).replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {col} | {desc} | {e.get('role', '')} | {e.get('unit', '')} | {e.get('type', '')} | {e.get('missing_pct', '')} | {e.get('distinct', '')} |"
        )
    glossary = get_glossary(dictionary)
    if glossary:
        lines += ["", "## Glossary", ""] + [
            f"- **{g['term']}**: {g['definition']}" for g in glossary
        ]
    return "\n".join(lines) + "\n"


def to_csv(dictionary: Mapping[str, Any]) -> str:
    buf = io.StringIO()
    to_frame(dictionary).to_csv(buf, index=False)
    return buf.getvalue()


def dictionary_snippet(**load: Any) -> str:
    body = """summary = pd.DataFrame({
    "type": df.dtypes.astype(str),
    "missing_pct": (df.isna().mean() * 100).round(2),
    "distinct": df.nunique(),
    "example": df.apply(lambda s: s.dropna().iloc[0] if s.notna().any() else None),
})
summary["description"] = ""   # fill in the business meaning of each column
summary
"""
    return make_snippet("Data dictionary skeleton", body, **load)


def data_card_markdown(
    filename: str,
    df: pd.DataFrame,
    dictionary: Mapping[str, Any],
    quality_score: float | None = None,
) -> str:
    """README-style data card: overview, columns, glossary, and auto-detected limitations."""
    entries = columns_only(dictionary)
    missing = df.isna().mean() * 100
    limitations = [
        f"- `{c}` is {missing[c]:.0f}% missing" for c in df.columns if missing[c] > 20
    ] + [f"- `{c}` is constant" for c in df.columns if df[c].nunique(dropna=True) <= 1]
    dups = int(df.duplicated().sum())
    if dups:
        limitations.append(f"- {dups:,} exact duplicate rows")
    lines = [
        f"# Data card: {filename}",
        "",
        "## Overview",
        f"- **Rows:** {len(df):,}",
        f"- **Columns:** {len(df.columns)}",
        f"- **Memory:** {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
    ]
    if quality_score is not None:
        lines.append(f"- **Quality score:** {quality_score:.0f}/100")
    lines += [
        "",
        "## Intended use",
        "_Describe what this data is for and what it should not be used for._",
        "",
    ]
    lines += ["## Columns", ""]
    for col, e in entries.items():
        unit = f" ({e['unit']})" if e.get("unit") else ""
        lines.append(
            f"- **{col}**{unit} - {e.get('description', '')} _[{e.get('role', '')}, {e.get('type', '')}]_"
        )
    glossary = get_glossary(dictionary)
    if glossary:
        lines += ["", "## Glossary", ""] + [
            f"- **{g['term']}**: {g['definition']}" for g in glossary
        ]
    lines += ["", "## Known limitations", ""] + (limitations or ["- None detected automatically."])
    return "\n".join(lines) + "\n"
