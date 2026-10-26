"""Tests for Executive Overview view logic and page flows."""

from pathlib import Path
from types import SimpleNamespace

from src.core.overview import (
    build_summary_markdown,
    column_type_counts,
    group_findings,
    recommend_next_steps,
)
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent


def F(title, category, severity="medium"):
    return SimpleNamespace(title=title, category=category, severity=severity, summary=f"{title}!")


def test_group_findings_sorts_by_severity_and_collapses_correlations():
    findings = [
        F("skew a", "distribution", "low"),
        F("corr 1", "correlation", "medium"),
        F("corr 2", "correlation", "high"),
        F("corr 3", "correlation", "medium"),
        F("anomaly", "anomaly", "critical"),
    ]
    items = group_findings(findings)
    assert [i.severity for i in items] == ["critical", "high", "low"]
    group = next(i for i in items if i.category == "correlation")
    assert group.title == "3 notable relationships"
    assert group.summary == "corr 2" and len(group.details) == 3
    assert next(i for i in items if i.category == "anomaly").page == "pages/2_Data_Quality.py"


def test_small_groups_stay_individual():
    items = group_findings([F("c1", "correlation"), F("c2", "correlation")])
    assert len(items) == 2 and not items[0].details


def test_pydantic_style_severity_field():
    f = SimpleNamespace(title="t", summary="s", severity_or_importance="high", category="x")
    assert group_findings([f])[0].severity == "high"


def test_column_type_counts():
    schema = {
        "a": SimpleNamespace(inferred_type="numeric"),
        "b": SimpleNamespace(inferred_type="numeric"),
        "c": SimpleNamespace(inferred_type="categorical"),
    }
    assert column_type_counts(schema) == {"numeric": 2, "categorical": 1}
    assert column_type_counts(None) == {}


def test_next_steps_prioritize_quality_then_exploration():
    issue = SimpleNamespace(severity="high", title="Bad nulls")
    profile = SimpleNamespace(
        quality_issues=[issue], missing_cell_percentage=3.0, duplicate_row_count=4
    )
    steps = recommend_next_steps(profile, [], {})
    assert "high-severity" in steps[0].title
    assert any("duplicate" in s.title for s in steps)
    assert len(steps) <= 4
    assert recommend_next_steps(None, [], {})[0].page == "pages/3_Exploration_Studio.py"


def test_summary_markdown_lists_findings():
    md = build_summary_markdown("x.csv", 10, 2, None, group_findings([F("t", "anomaly", "high")]))
    assert "# Executive Summary — x.csv" in md and "[HIGH] t" in md


def _app():
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()


def test_hero_shown_without_dataset_and_sample_loads():
    at = _app()
    at.switch_page("pages/1_Executive_Overview.py").run()
    assert any("What would you like to analyze" in m.value for m in at.markdown)
    sample = next(b for b in at.button if "sample dataset" in b.label)
    sample.click().run()
    assert not at.exception
    assert any(m.label == "Rows" for m in at.metric)
    assert at.session_state["current_dataset"].filename == "ecommerce_customer_churn.csv"


def test_example_question_prefills_ai_analyst():
    at = _app()
    at.switch_page("pages/1_Executive_Overview.py").run()
    next(b for b in at.button if b.key == "example_q_0").click().run()
    assert not at.exception
    assert at.text_input(key="ai_question_input").value.startswith("Which customers")


def test_loaded_overview_has_findings_next_steps_and_clear():
    at = _app()
    at.switch_page("pages/1_Executive_Overview.py").run()
    next(b for b in at.button if "sample dataset" in b.label).click().run()
    assert any("Recommended Next Steps" in s.value for s in at.subheader)
    next(b for b in at.button if b.label == "Clear dataset").click().run()
    assert at.session_state["current_dataset"] is None
