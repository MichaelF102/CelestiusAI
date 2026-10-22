"""Tests for session state initialization, rerun preservation, and lifecycle helpers."""

from src.core.session_state import (
    KEY_ANALYSIS_PLAN,
    KEY_ANALYSIS_RESULTS,
    KEY_ANALYTICAL_OBJECTIVE,
    KEY_CONVERSATION_HISTORY,
    KEY_CURRENT_DATASET,
    KEY_DATA_QUALITY_REPORT,
    KEY_DATASET_METADATA,
    KEY_DATASET_SCHEMA,
    KEY_EXECUTION_TRACE,
    KEY_GENERATED_FINDINGS,
    KEY_TRANSFORMATION_PIPELINE,
    KEY_WORKSPACE_ID,
    get_default_session_state,
    init_session_state,
    is_dataset_loaded,
    reset_dataset_state,
)


def test_default_session_state_keys():
    """Verify default session state contains all required architectural keys with clean empty values."""
    defaults = get_default_session_state()

    expected_keys = {
        KEY_WORKSPACE_ID,
        KEY_CURRENT_DATASET,
        KEY_DATASET_METADATA,
        KEY_DATASET_SCHEMA,
        KEY_DATA_QUALITY_REPORT,
        KEY_ANALYSIS_RESULTS,
        KEY_GENERATED_FINDINGS,
        KEY_ANALYTICAL_OBJECTIVE,
        KEY_ANALYSIS_PLAN,
        KEY_EXECUTION_TRACE,
        KEY_CONVERSATION_HISTORY,
        KEY_TRANSFORMATION_PIPELINE,
    }
    assert expected_keys.issubset(defaults.keys())
    assert defaults[KEY_CURRENT_DATASET] is None
    assert defaults[KEY_DATASET_METADATA] is None
    assert defaults[KEY_DATASET_SCHEMA] == {}
    assert defaults[KEY_DATA_QUALITY_REPORT] is None
    assert defaults[KEY_ANALYSIS_RESULTS] == {}
    assert defaults[KEY_GENERATED_FINDINGS] == []
    assert defaults[KEY_ANALYTICAL_OBJECTIVE] == ""
    assert defaults[KEY_ANALYSIS_PLAN] is None
    assert defaults[KEY_EXECUTION_TRACE] == []
    assert defaults[KEY_CONVERSATION_HISTORY] == []
    assert defaults[KEY_TRANSFORMATION_PIPELINE] is None
    assert len(defaults[KEY_WORKSPACE_ID]) > 0


def test_init_session_state_preserves_existing_values():
    """Verify reruns do not overwrite already established session values."""
    state = {
        KEY_WORKSPACE_ID: "custom-workspace-42",
        KEY_ANALYTICAL_OBJECTIVE: "Analyze sales decline Q3",
        KEY_GENERATED_FINDINGS: ["Finding A", "Finding B"],
    }

    initialized = init_session_state(state)

    # Existing values preserved
    assert initialized[KEY_WORKSPACE_ID] == "custom-workspace-42"
    assert initialized[KEY_ANALYTICAL_OBJECTIVE] == "Analyze sales decline Q3"
    assert initialized[KEY_GENERATED_FINDINGS] == ["Finding A", "Finding B"]

    # Missing keys populated with defaults
    assert KEY_CURRENT_DATASET in initialized
    assert initialized[KEY_CURRENT_DATASET] is None
    assert KEY_DATA_QUALITY_REPORT in initialized


def test_reset_dataset_state():
    """Verify resetting dataset clears analysis and metadata without clearing workspace or history."""
    state = {
        KEY_WORKSPACE_ID: "keep-workspace-id",
        KEY_CONVERSATION_HISTORY: [{"user": "hello"}],
        KEY_CURRENT_DATASET: "dummy_df",
        KEY_DATASET_METADATA: {"filename": "data.csv"},
        KEY_ANALYSIS_RESULTS: {"metric": 42},
        KEY_GENERATED_FINDINGS: ["Finding 1"],
    }

    reset_dataset_state(state)

    # Dataset-dependent state is reset
    assert state[KEY_CURRENT_DATASET] is None
    assert state[KEY_DATASET_METADATA] is None
    assert state[KEY_ANALYSIS_RESULTS] == {}
    assert state[KEY_GENERATED_FINDINGS] == []

    # Persistent workspace and conversation retained
    assert state[KEY_WORKSPACE_ID] == "keep-workspace-id"
    assert state[KEY_CONVERSATION_HISTORY] == [{"user": "hello"}]


def test_is_dataset_loaded_helper():
    """Verify dataset loaded checks both current_dataset and metadata."""
    state = get_default_session_state()
    assert is_dataset_loaded(state) is False

    state[KEY_DATASET_METADATA] = {"filename": "sales.parquet"}
    assert is_dataset_loaded(state) is True
