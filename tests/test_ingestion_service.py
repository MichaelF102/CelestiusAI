"""Tests for the shared ingestion service and session-state helpers."""

from pathlib import Path

import pytest
from src.core.exceptions import DataAnalystError
from src.core.session_state import (
    KEY_CURRENT_DATASET,
    KEY_DATA_QUALITY_REPORT,
    KEY_DATASET_METADATA,
    KEY_MODEL_REGISTRY,
    KEY_TRANSFORMATION_PIPELINE,
    get_model_registry,
    get_transformation_pipeline,
    init_session_state,
    update_quality_profile,
)
from src.ingestion.service import ingest_into_state, needs_ingestion
from src.predictive.model_registry import ModelRegistry

SAMPLE = Path(__file__).resolve().parent.parent / "sample_data" / "ecommerce_customer_churn.csv"


def test_ingest_populates_state():
    state = init_session_state({})
    managed = ingest_into_state(state, SAMPLE.read_bytes(), SAMPLE.name, max_size_mb=10)
    assert state[KEY_CURRENT_DATASET] is managed
    assert state[KEY_DATA_QUALITY_REPORT] is not None
    assert state[KEY_DATASET_METADATA].row_count == len(managed.working_df)
    assert state[KEY_TRANSFORMATION_PIPELINE] is not None


def test_needs_ingestion_tracks_active_file():
    state = init_session_state({})
    assert needs_ingestion(state, SAMPLE.name)
    ingest_into_state(state, SAMPLE.read_bytes(), SAMPLE.name, max_size_mb=10)
    assert not needs_ingestion(state, SAMPLE.name)
    assert needs_ingestion(state, "other.csv")


def test_failed_ingest_keeps_previous_dataset():
    state = init_session_state({})
    first = ingest_into_state(state, SAMPLE.read_bytes(), SAMPLE.name, max_size_mb=10)
    with pytest.raises(DataAnalystError):
        ingest_into_state(state, b"", "empty.csv", max_size_mb=10)
    assert state[KEY_CURRENT_DATASET] is first


def test_unsupported_extension_rejected():
    state = init_session_state({})
    with pytest.raises(DataAnalystError):
        ingest_into_state(state, b"a,b\n1,2", "data.txt", max_size_mb=10)


def test_get_model_registry_repairs_invalid_value():
    state = init_session_state({})
    assert state[KEY_MODEL_REGISTRY] is None
    reg = get_model_registry(state)
    assert isinstance(reg, ModelRegistry)
    assert get_model_registry(state) is reg
    state[KEY_MODEL_REGISTRY] = "garbage"
    assert isinstance(get_model_registry(state), ModelRegistry)


def test_transformation_pipeline_is_reused():
    state = init_session_state({})
    assert get_transformation_pipeline(state) is get_transformation_pipeline(state)


def test_update_quality_profile_invalidates_findings():
    state = init_session_state({})
    state["generated_findings"] = ["x"]
    update_quality_profile(state, "profile", {"a": 1})
    assert state[KEY_DATA_QUALITY_REPORT] == "profile"
    assert state["generated_findings"] == []
