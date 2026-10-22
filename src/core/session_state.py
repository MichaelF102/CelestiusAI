"""Shared session state initialization, validation, and lifecycle management.

Defines standard session state keys, default empty values, and pure helper functions
independent of Streamlit so they can be unit-tested in headless environments.
"""

from __future__ import annotations

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
KEY_TRANSFORMATION_PIPELINE = "transformation_pipeline"  # Active cleaning pipeline instance
KEY_MODEL_REGISTRY = "model_registry"  # Predictive model registry
KEY_PREDICTIVE_RECORDS = "predictive_records"  # Latest predictive model records
KEY_ACTIVE_REPORT_DOC = "active_report_doc"  # Generated report document
KEY_LATEST_INVESTIGATION_STATE = "latest_investigation_state"  # Orchestrated investigation trace


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
        KEY_TRANSFORMATION_PIPELINE: None,
        KEY_MODEL_REGISTRY: None,
        KEY_PREDICTIVE_RECORDS: [],
        KEY_ACTIVE_REPORT_DOC: None,
        KEY_LATEST_INVESTIGATION_STATE: None,
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


def invalidate_downstream_analyses(state: MutableMapping[str, Any]) -> None:
    """Invalidate analyses dependent on prior working dataset version after a transformation."""
    state[KEY_ANALYSIS_RESULTS] = {}
    state[KEY_GENERATED_FINDINGS] = []
    state[KEY_ANALYSIS_PLAN] = None
    state[KEY_EXECUTION_TRACE] = []
    state[KEY_ACTIVE_REPORT_DOC] = None
    state[KEY_LATEST_INVESTIGATION_STATE] = None
    state[KEY_PREDICTIVE_RECORDS] = []

    # Flag existing models in registry as stale
    registry = state.get(KEY_MODEL_REGISTRY)
    dataset = state.get(KEY_CURRENT_DATASET)
    active_version = getattr(dataset, "version", None)
    if registry and hasattr(registry, "list_models"):
        registry.list_models(active_version=active_version)


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
        KEY_TRANSFORMATION_PIPELINE,
        KEY_MODEL_REGISTRY,
        KEY_PREDICTIVE_RECORDS,
        KEY_ACTIVE_REPORT_DOC,
        KEY_LATEST_INVESTIGATION_STATE,
    ]
    for key in keys_to_reset:
        state[key] = defaults[key]


def is_dataset_loaded(state: MutableMapping[str, Any]) -> bool:
    """Return True if a dataset reference and metadata are currently loaded in state."""
    return state.get(KEY_CURRENT_DATASET) is not None or state.get(KEY_DATASET_METADATA) is not None
