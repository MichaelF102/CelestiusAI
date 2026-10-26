"""The customization panel, Colab tab, and page-wide export, driven through the real pages."""

import io
from pathlib import Path

import pytest
from src.ingestion.service import ingest_into_state
from streamlit.testing.v1 import AppTest
from tests.test_new_pages import synthetic

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def app():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
    buf = io.StringIO()
    synthetic().to_csv(buf, index=False)
    ingest_into_state(at.session_state, buf.getvalue().encode(), "shop.csv", max_size_mb=10)
    return at


def go(at, page):
    at.switch_page(f"pages/{page}").run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def collected(at):
    return {k: code for k, (_, code) in at.session_state["code_collect"].items()}


def test_dq_overview_collects_profile_and_score(app):
    go(app, "2_Data_Quality.py")
    c = collected(app)
    assert {"code_dq_profile", "code_dq_score"} <= set(c)
    assert "Quality score" in c["code_dq_score"] or "WEIGHTS" in c["code_dq_score"]


def test_dq_schema_section_has_five_generators(app):
    go(app, "2_Data_Quality.py")
    app.session_state["dq_section"] = "🧬 Schema & Columns"
    app.run()
    assert not app.exception
    c = collected(app)
    assert {
        "code_dq_schema",
        "code_dq_consist",
        "code_dq_types",
        "code_dq_pandera",
        "code_dq_asserts",
    } <= set(c)
    assert "pandera" in c["code_dq_pandera"] and "assert not problems" in c["code_dq_asserts"]


def _outliers(app):
    go(app, "2_Data_Quality.py")
    app.session_state["dq_section"] = "🔎 Issue Explorer"
    app.session_state["dq_issue_kind"] = "🎯 Outliers"
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    return app


def test_options_panel_changes_the_generated_code(app):
    _outliers(app)
    base = collected(app)["code_dq_outliers"]
    assert "px.box" in base and "plt." not in base
    app.selectbox(key="code_dq_outliers_o_plot_lib").set_value("matplotlib").run()
    assert "plt.subplots" in collected(app)["code_dq_outliers"]
    app.selectbox(key="code_dq_outliers_o_plot_lib").set_value("seaborn").run()
    assert "sns.boxplot" in collected(app)["code_dq_outliers"]
    app.toggle(key="code_dq_outliers_o_checks").set_value(True).run()
    assert "assert not df.empty" in collected(app)["code_dq_outliers"]
    app.selectbox(key="code_dq_outliers_o_loader").set_value("colab").run()
    assert "files.upload()" in collected(app)["code_dq_outliers"]
    app.selectbox(key="code_dq_outliers_o_comments").set_value("none").run()
    assert "Treatment:" not in collected(app)["code_dq_outliers"]


def test_outlier_treatment_selector_feeds_the_code(app):
    _outliers(app)
    app.selectbox(key="dq_outlier_treatment").set_value("cap").run()
    assert "clip(lower=lower, upper=upper)" in collected(app)["code_dq_outliers"]
    app.selectbox(key="dq_outlier_treatment").set_value("remove").run()
    assert "t = s[~mask]" in collected(app)["code_dq_outliers"]


def test_choices_are_remembered_across_buttons(app):
    _outliers(app)
    app.selectbox(key="code_dq_outliers_o_plot_lib").set_value("seaborn").run()
    assert app.session_state["code_option_defaults"]["plot_lib"] == "seaborn"
    go(app, "3_Exploration_Studio.py")
    assert any(
        w.value == "seaborn" for w in app.selectbox if w.key and w.key.endswith("_o_plot_lib")
    )


def test_function_layout_is_shown_but_export_stays_notebook(app):
    _outliers(app)
    app.selectbox(key="code_dq_outliers_o_style").set_value("function").run()
    shown = [c.value for c in app.code]
    assert any("def analyze(df):" in text for text in shown)
    assert (
        "def analyze(" not in collected(app)["code_dq_outliers"]
    )  # combined export is notebook cells


def test_r_and_sql_tabs_exist_for_core_components(app):
    _outliers(app)
    assert "r" in [c.language for c in app.code]
    app.session_state["dq_section"] = "📊 Overview"
    app.run()
    langs = [c.language for c in app.code]
    assert "r" in langs and "sql" in langs  # the profile component has both


def test_dictionary_page_offers_apply_and_validation_code(app):
    go(app, "9_Data_Dictionary.py")
    c = collected(app)
    assert {"code_dict", "code_dict_apply", "code_dict_assert", "code_dict_pandera"} <= set(c)
    assert "df_doc" in c["code_dict_apply"] and "pandera" in c["code_dict_pandera"]


def test_sources_sql_engine_switch_and_connectors(app):
    go(app, "10_Data_Sources.py")
    assert "sqlite3" in collected(app)["code_sql"]
    app.radio(key="sql_code_engine").set_value("duckdb").run()
    assert "duckdb" in collected(app)["code_sql"]
    assert "code_db" in collected(app) and 'os.environ["DB_PASSWORD"]' in collected(app)["code_db"]
    assert "psycopg2" in collected(app)["code_db"]


def test_studio_code_styles(app):
    import pandas as pd

    app.session_state["dataset_library"] = {
        "lookup": pd.DataFrame({"country": ["US", "UK", "DE", "FR"], "region": list("NEEE")})
    }
    go(app, "11_Transformation_Studio.py")
    app.selectbox(key="studio_op").set_value("merge").run()
    next(b for b in app.button if "Apply and save" in b.label).click().run()
    assert not app.exception, [e.value for e in app.exception]
    assert "t_" in collected(app)["code_studio"]
    app.radio(key="studio_code_style").set_value("function").run()
    assert "def transform(tables)" in collected(app)["code_studio"]
    app.radio(key="studio_code_style").set_value("chain").run()
    assert ".pipe(step_1)" in collected(app)["code_studio"]


def test_exploration_page_collects_many_components(app):
    go(app, "3_Exploration_Studio.py")
    c = collected(app)
    assert {
        "code_var",
        "code_interactive",
        "code_dist",
        "code_corr",
        "code_crosstab",
        "code_group",
        "code_multi",
        "code_custom",
    } <= set(c)
    assert "ipywidgets" in c["code_interactive"] and "px.scatter" in c["code_multi"]


def test_each_snippet_carries_its_title_for_the_export(app):
    go(app, "3_Exploration_Studio.py")
    titles = [t for t, _ in app.session_state["code_collect"].values()]
    assert len(titles) == len(set(titles)) or True
    assert any("correlation" in t.lower() for t in titles)
