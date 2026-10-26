"""Workspace table library: the active dataset plus named side tables (joins, SQL results, ...)."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import pandas as pd

from src.core.session_state import KEY_CURRENT_DATASET, KEY_DATASET_LIBRARY

ACTIVE = "active"
MAX_TABLES = 25


def get_tables(state: MutableMapping[str, Any]) -> dict[str, pd.DataFrame]:
    """`active` (the working copy of the loaded dataset) followed by the library tables."""
    tables: dict[str, pd.DataFrame] = {}
    dataset = state.get(KEY_CURRENT_DATASET)
    if dataset is not None and hasattr(dataset, "working_df"):
        tables[ACTIVE] = dataset.working_df
    tables.update(state.get(KEY_DATASET_LIBRARY) or {})
    return tables


def add_table(state: MutableMapping[str, Any], name: str, df: pd.DataFrame) -> str:
    """Store `df` under `name` (must be new and not 'active'); returns the stored name."""
    library = dict(state.get(KEY_DATASET_LIBRARY) or {})
    if name == ACTIVE:
        raise ValueError("'active' is reserved for the loaded dataset.")
    if name not in library and len(library) >= MAX_TABLES:
        raise ValueError(f"The library is full ({MAX_TABLES} tables). Remove one first.")
    library[name] = df.reset_index(drop=True)
    state[KEY_DATASET_LIBRARY] = library
    return name


def remove_table(state: MutableMapping[str, Any], name: str) -> None:
    library = dict(state.get(KEY_DATASET_LIBRARY) or {})
    library.pop(name, None)
    state[KEY_DATASET_LIBRARY] = library


def unique_name(existing: dict[str, Any], wanted: str) -> str:
    base = wanted or "table"
    name, n = base, 2
    while name in existing or name == ACTIVE:
        name = f"{base}_{n}"
        n += 1
    return name
