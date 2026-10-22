"""Shared session state initialization, validation, and lifecycle management.

Defines standard session state keys, default empty values, and pure helper functions
independent of Streamlit so they can be unit-tested in headless environments.
"""

import uuid
from collections.abc import MutableMapping
from typing import Any

# Standard session state keys
KEY_WORKSPACE_ID = "workspace_id"
KEY_CURRENT_DATASET = "current_dataset"  # In-memory DataFrame reference (runtime only)
KEY_DATASET_METADATA = "dataset_metadata"  # Serialized DatasetMetadata or dict
KEY_DATASET_SCHEMA = "dataset_schema"  # Inferred column types and roles
KEY_DATA_QUALITY_REPORT = "data_quality_report"  # Quality score, anomalies, missingness
KEY_ANALYSIS_RESULTS = "analysis_results"  # Structured results from analytical tools
KEY_GENERATED_FINDINGS = "generated_findings"  # Validated insights with 6-layer evidence
KEY_ANALYTICAL_OBJECTIVE = "analytical_objective"  # Current user question or objective
KEY_ANALYSIS_PLAN = "analysis_plan"  # Multi-step execution plan
KEY_EXECUTION_TRACE = "execution_trace"  # List of steps executed with metrics
KEY_CONVERSATION_HISTORY = "conversation_history"  # Conversational Q&A log

# Persistence Classification:
# - TEMPORARY / IN-MEMORY ONLY: current_dataset, conversation_history, analytical_objective
# - EVENTUALLY PERSISTED (SQLite / Parquet / Reports): workspace_id, dataset_metadata,
#   dataset_schema, data_quality_report, analysis_results, generated_findings,
#   analysis_plan, execution_trace


def get_default_session_state() -> dict[str, Any]:
    """Return a dictionary containing clean default initial values for all state keys."""
    return {
        KEY_WORKSPACE_ID: str(uuid.uuid4())[:8],
        KEY_CURRENT_DATASET: None,
        KEY_DATASET_METADATA: None,
        KEY_DATASET_SCHEMA: {},
        KEY_DATA_QUALITY_REPORT: None,
        KEY_ANALYSIS_RESULTS: {},
        KEY_GENERATED_FINDINGS: [],
        KEY_ANALYTICAL_OBJECTIVE: "",
        KEY_ANALYSIS_PLAN: None,
        KEY_EXECUTION_TRACE: [],
        KEY_CONVERSATION_HISTORY: [],
    }


def init_session_state(state: MutableMapping[str, Any] | None = None) -> MutableMapping[str, Any]:
    """Ensure all required session state keys exist without overwriting active values.

    Args:
        state: A dict-like mutable mapping. If None, attempts to use `streamlit.session_state`.

    Returns:
        The initialized mutable mapping.
    """
    if state is None:
        try:
            import streamlit as st

            target = st.session_state
        except ImportError:
            target = {}
    else:
        target = state

    defaults = get_default_session_state()
    for key, val in defaults.items():
        if key not in target:
            target[key] = val

    return target


def reset_dataset_state(state: MutableMapping[str, Any]) -> None:
    """Reset dataset-derived state when a new dataset is loaded or unloaded."""
    defaults = get_default_session_state()
    keys_to_reset = [
        KEY_CURRENT_DATASET,
        KEY_DATASET_METADATA,
        KEY_DATASET_SCHEMA,
        KEY_DATA_QUALITY_REPORT,
        KEY_ANALYSIS_RESULTS,
        KEY_GENERATED_FINDINGS,
        KEY_ANALYTICAL_OBJECTIVE,
        KEY_ANALYSIS_PLAN,
        KEY_EXECUTION_TRACE,
    ]
    for key in keys_to_reset:
        state[key] = defaults[key]


def is_dataset_loaded(state: MutableMapping[str, Any]) -> bool:
    """Return True if a dataset reference and metadata are currently loaded in state."""
    return state.get(KEY_CURRENT_DATASET) is not None or state.get(KEY_DATASET_METADATA) is not None
