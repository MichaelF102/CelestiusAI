"""Tests for project persistence (save, versions, restore, safety)."""

from pathlib import Path

import pandas as pd
import pytest
from src.core.session_state import (
    KEY_CURRENT_DATASET,
    KEY_DATA_DICTIONARY,
    KEY_DATASET_LIBRARY,
    get_transformation_pipeline,
    init_session_state,
)
from src.ingestion.service import ingest_into_state
from src.profiling.actions import actionable_issues, apply_issues, is_safe_issue
from src.workspace.projects import ProjectError, ProjectStore, slugify

SAMPLE = Path(__file__).resolve().parent.parent / "sample_data" / "ecommerce_customer_churn.csv"


def _state():
    state = init_session_state({})
    ds = ingest_into_state(state, SAMPLE.read_bytes(), SAMPLE.name, max_size_mb=10)
    return state, ds


def test_slugify_and_traversal_rejected(tmp_path):
    assert slugify("  My Project #1 ") == "my-project-1"
    with pytest.raises(ProjectError):
        slugify("!!!")
    store = ProjectStore(tmp_path)
    for bad in ("..", "///"):
        with pytest.raises(ProjectError):
            store._dir(bad)
    assert store._dir("../../etc").resolve().is_relative_to(tmp_path.resolve())


def test_save_requires_dataset(tmp_path):
    with pytest.raises(ProjectError):
        ProjectStore(tmp_path).save(init_session_state({}), "x")


@pytest.mark.filterwarnings("ignore:Mismatched null-like:FutureWarning")
def test_round_trip_restores_cleaning_dictionary_and_tables(tmp_path):
    store = ProjectStore(tmp_path)
    state, ds = _state()
    pipeline = get_transformation_pipeline(state)
    apply_issues(
        pipeline,
        ds,
        [i for i in actionable_issues(state["data_quality_report"]) if is_safe_issue(i)],
    )
    state[KEY_DATA_DICTIONARY] = {"age": {"description": "Customer age"}}
    state[KEY_DATASET_LIBRARY] = {"Side Table": pd.DataFrame({"a": [1, 2], "b": ["x", None]})}
    expected = ds.working_df.copy()

    assert store.save(state, "Churn Study", note="first") == 1
    assert store.save(state, "Churn Study", note="second") == 2

    fresh = init_session_state({})
    result = store.load(fresh, "churn-study")
    assert result["meta"]["version"] == 2 and not result["replay_errors"]
    restored = fresh[KEY_CURRENT_DATASET]
    assert restored.filename == SAMPLE.name
    pd.testing.assert_frame_equal(
        restored.working_df.reset_index(drop=True),
        expected.reset_index(drop=True),
        check_dtype=False,
    )
    assert fresh[KEY_DATA_DICTIONARY]["age"]["description"] == "Customer age"
    assert list(fresh[KEY_DATASET_LIBRARY]) == ["Side Table"]
    assert len(get_transformation_pipeline(fresh).records) == len(pipeline.records)


def test_versions_events_and_listing(tmp_path):
    store = ProjectStore(tmp_path)
    state, _ = _state()
    store.save(state, "p")
    store.save(state, "p", note="v2")
    store.load(init_session_state({}), "p", version=1)
    info = store.list_projects()
    assert [(i.slug, i.versions) for i in info] == [("p", 2)]
    assert [e["event"] for e in store.events("p")] == ["saved", "saved", "loaded"]
    assert [v["version"] for v in store.versions("p")] == [1, 2]
    with pytest.raises(ProjectError):
        store.load(init_session_state({}), "p", version=9)
    store.delete("p")
    assert store.list_projects() == []
