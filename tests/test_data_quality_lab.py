"""Tests for Data Quality Lab backend helpers and page flows."""

from pathlib import Path

import pytest
from src.core.session_state import KEY_CURRENT_DATASET, KEY_DATA_QUALITY_REPORT, init_session_state
from src.ingestion.recipe import export_recipe, replay_recipe
from src.ingestion.service import ingest_into_state
from src.ingestion.transformation_history import TransformationPipeline
from src.profiling.actions import (
    actionable_issues,
    apply_issues,
    is_safe_issue,
    reprofile,
    spec_from_issue,
)
from src.profiling.certificate import build_quality_certificate, quality_grade
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "sample_data" / "ecommerce_customer_churn.csv"


@pytest.fixture
def loaded():
    state = init_session_state({})
    dataset = ingest_into_state(state, SAMPLE.read_bytes(), SAMPLE.name, max_size_mb=10)
    return state, dataset, state[KEY_DATA_QUALITY_REPORT]


def test_safe_issues_apply_and_improve_or_keep_score(loaded):
    state, dataset, profile = loaded
    pipeline = TransformationPipeline()
    safe = [i for i in actionable_issues(profile) if is_safe_issue(i)]
    assert safe, "sample data should have whitespace/duplicate issues"
    applied, errors = apply_issues(pipeline, dataset, safe)
    assert applied == len(safe) and not errors
    assert len(pipeline.records) == applied
    assert reprofile(dataset).quality_score.overall_score >= profile.quality_score.overall_score


def test_rollback_to_restores_intermediate_state(loaded):
    _, dataset, profile = loaded
    pipeline = TransformationPipeline()
    issues = actionable_issues(profile)[:3]
    apply_issues(pipeline, dataset, issues)
    n = len(pipeline.records)
    assert n >= 2
    rows_after_first = pipeline.records[0].rows_after
    removed = pipeline.rollback_to(dataset, 1)
    assert len(removed) == n - 1 and len(pipeline.records) == 1
    assert len(dataset.working_df) == rows_after_first
    assert pipeline.rollback_to(dataset, 5) == []  # nothing beyond history
    pipeline.rollback_to(dataset, 0)
    assert dataset.working_df.equals(dataset.raw_df)


def test_undo_last_matches_rollback(loaded):
    _, dataset, profile = loaded
    pipeline = TransformationPipeline()
    apply_issues(pipeline, dataset, actionable_issues(profile)[:2])
    assert pipeline.undo(dataset) is not None
    assert len(pipeline.records) == len(pipeline.specs)
    assert TransformationPipeline().undo(dataset) is None


def test_recipe_round_trip_replays_on_fresh_dataset(loaded):
    state, dataset, profile = loaded
    pipeline = TransformationPipeline()
    apply_issues(pipeline, dataset, [i for i in actionable_issues(profile) if is_safe_issue(i)])
    recipe = export_recipe(pipeline, dataset.filename)

    fresh_state = init_session_state({})
    fresh = ingest_into_state(fresh_state, SAMPLE.read_bytes(), SAMPLE.name, max_size_mb=10)
    replay_pipeline = TransformationPipeline()
    applied, errors = replay_recipe(recipe, replay_pipeline, fresh)
    assert applied == len(pipeline.specs) and not errors
    assert fresh.working_df.equals(dataset.working_df)


def test_replay_rejects_malformed_recipe(loaded):
    _, dataset, _ = loaded
    with pytest.raises(ValueError):
        replay_recipe("not json", TransformationPipeline(), dataset)
    with pytest.raises(ValueError):
        replay_recipe('{"steps": [{"bad": 1}]}', TransformationPipeline(), dataset)


def test_certificate_and_grades(loaded):
    _, dataset, profile = loaded
    cert = build_quality_certificate("f.csv", "abc", profile, 10, 2, 0, baseline_score=50.04)
    assert cert["baseline_score"] == 50.0 and cert["dataset_filename"] == "f.csv"
    assert quality_grade(95).startswith("A") and quality_grade(80).startswith("B")
    assert quality_grade(10).startswith("C")


def test_spec_from_issue_uses_issue_params(loaded):
    _, _, profile = loaded
    issue = actionable_issues(profile)[0]
    spec = spec_from_issue(issue)
    assert spec.action_type == issue.suggested_action_type


# ------------------------------------------------------------------ page flows
def _dq_app():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
    ingest_into_state(at.session_state, SAMPLE.read_bytes(), SAMPLE.name, max_size_mb=10)
    at.switch_page("pages/2_Data_Quality.py").run()
    return at


def test_dq_page_without_dataset_shows_banner():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    at.switch_page("pages/2_Data_Quality.py").run()
    assert not at.exception
    assert any("needs a dataset" in i.value for i in at.info)


def test_dq_status_strip_and_every_section_render():
    at = _dq_app()
    assert not at.exception, [e.value for e in at.exception]
    assert {"Quality Score", "Issues", "Missing Cells", "Duplicate Rows", "Transformations"} <= {
        m.label for m in at.metric
    }
    sections = ["🧬 Schema & Columns", "🔎 Issue Explorer", "🛠️ Clean", "📊 Overview"]
    for sec in sections:
        at.session_state["dq_section"] = sec
        at.run()
        assert not at.exception, (sec, [e.value for e in at.exception])
    for kind in ["👥 Duplicates", "🎯 Outliers", "❓ Missingness"]:
        at.session_state["dq_section"] = "🔎 Issue Explorer"
        at.session_state["dq_issue_kind"] = kind
        at.run()
        assert not at.exception, (kind, [e.value for e in at.exception])


def test_fix_this_jumps_to_clean_with_issue_selected():
    at = _dq_app()
    fix = next(b for b in at.button if b.label == "🛠️ Fix this")
    issue_id = fix.key.removeprefix("fix_")
    fix.click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["dq_section"] == "🛠️ Clean"
    assert at.session_state["dq_single_issue"] == issue_id


def test_auto_clean_then_rollback_via_page():
    at = _dq_app()
    at.session_state["dq_section"] = "🛠️ Clean"
    at.run()
    next(b for b in at.button if b.key == "btn_auto_clean_safe").click().run()
    assert not at.exception, [e.value for e in at.exception]
    ds = at.session_state[KEY_CURRENT_DATASET]
    pipeline = at.session_state["transformation_pipeline"]
    assert pipeline.records
    assert any(m.label == "History" for m in at.metric)
    next(b for b in at.button if b.key == "btn_reset_all").click().run()
    assert not at.exception
    assert not at.session_state["transformation_pipeline"].records
    assert ds.working_df.equals(ds.raw_df)
