"""The Dashboard Builder page, driven end to end."""

import io
from pathlib import Path

import pytest
from src.core.session_state import KEY_DASHBOARD_SPEC
from src.datascience import dashboard as db
from src.ingestion.service import ingest_into_state
from streamlit.testing.v1 import AppTest
from tests.test_new_pages import synthetic

ROOT = Path(__file__).resolve().parent.parent
PAGE = "pages/12_Dashboard_Builder.py"


@pytest.fixture
def app():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=240).run()
    buf = io.StringIO()
    synthetic().to_csv(buf, index=False)
    ingest_into_state(at.session_state, buf.getvalue().encode(), "shop.csv", max_size_mb=10)
    at.switch_page(PAGE).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def spec(at) -> db.DashboardSpec:
    return at.session_state[KEY_DASHBOARD_SPEC]


def click(at, label=None, key=None):
    b = next(b for b in at.button if (b.key == key if key else label in b.label))
    b.click().run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def use_template(at, name):
    at.selectbox(key="dash_template").set_value(name).run()
    click(at, "Use template")
    return at


def test_empty_dashboard_prompts_for_a_start(app):
    assert not spec(app).tiles
    assert any("Add your first tile" in i.value for i in app.info)


@pytest.mark.parametrize("name", ["auto", "sales", "customer", "operations", "quality"])
def test_templates_build_a_working_dashboard(app, name):
    use_template(app, name)
    s = spec(app)
    assert len(s.tiles) >= 3 and s.sections
    assert len(app.metric) >= 1
    assert not [w for w in app.warning if "Column(s) not found" in w.value]
    if name == "quality":
        assert "quality_summary" in app.session_state["dataset_library"]
        assert all(t.source == "quality_summary" for t in s.tiles)


def test_suggest_uses_existing_findings(app):
    class F:
        title, severity = "Spend is skewed", "high"

    app.session_state["generated_findings"] = [F()]
    click(app, "Suggest from my data")
    assert (
        spec(app).tiles[0].kind == "text"
        and "Spend is skewed" in spec(app).tiles[0].params["markdown"]
    )


def test_add_tile_undo_redo(app):
    app.selectbox(key="tile_kind").set_value("hist").run()
    click(app, key="dash_add_tile")
    assert [t.kind for t in spec(app).tiles] == ["hist"]
    click(app, "Undo")
    assert not spec(app).tiles
    click(app, "Redo")
    assert [t.kind for t in spec(app).tiles] == ["hist"]


@pytest.mark.parametrize("kind", list(db.TILE_LABELS))
def test_every_tile_type_can_be_added_and_rendered(app, kind):
    app.selectbox(key="tile_kind").set_value(kind).run()
    assert not app.exception, [e.value for e in app.exception]
    click(app, key="dash_add_tile")
    assert spec(app).tiles[0].kind == kind
    assert not [w for w in app.warning if "Column(s) not found" in w.value]


def test_filters_change_what_tiles_show(app):
    app.selectbox(key="tile_kind").set_value("kpi").run()
    click(app, key="dash_add_tile")
    rows_before = int(app.metric[0].value.replace(",", ""))
    app.selectbox(key="dash_new_filter").set_value("country").run()
    click(app, key="dash_add_filter")
    app.multiselect(key="flt_0").set_value(["US"]).run()
    rows_after = int(app.metric[0].value.replace(",", ""))
    assert 0 < rows_after < rows_before
    assert spec(app).filters[0].values == ["US"]
    click(app, key="flt_rm_0")
    assert not spec(app).filters


def test_text_search_and_date_preset_filters(app):
    app.selectbox(key="tile_kind").set_value("kpi").run()
    click(app, key="dash_add_tile")
    total = int(app.metric[0].value.replace(",", ""))
    app.selectbox(key="dash_new_filter").set_value("signup_date").run()
    click(app, key="dash_add_filter")
    assert spec(app).filters[0].kind == "date"
    app.selectbox(key="flt_p_0").set_value("last_30").run()
    assert (
        int(app.metric[0].value.replace(",", "")) == 31
        or int(app.metric[0].value.replace(",", "")) < total
    )
    app.selectbox(key="dash_new_text").set_value("country").run()
    click(app, key="dash_add_text")
    app.text_input(key="flt_1").set_value("FR").run()
    assert int(app.metric[0].value.replace(",", "")) <= 31


def test_calculated_field_parameter_and_bad_input(app):
    app.text_input(key="calc_new_name").set_value("spend_x2").run()
    app.text_input(key="calc_new_expr").set_value("spend * 2").run()
    click(app, "Add field")
    assert spec(app).calculated[0].name == "spend_x2"
    app.text_input(key="calc_new_name").set_value("evil").run()
    app.text_input(key="calc_new_expr").set_value("__import__('os')").run()
    click(app, "Add field")
    assert len(spec(app).calculated) == 1 and any("not allowed" in e.value for e in app.error)
    app.text_input(key="prm_new_name").set_value("goal").run()
    click(app, "Add parameter")
    assert spec(app).parameters[0].name == "goal"


def test_theme_controls_update_the_spec(app):
    app.radio(key="th_mode").set_value("light").run()
    app.selectbox(key="th_pal").set_value("colorblind").run()
    app.slider(key="th_font").set_value(16).run()
    t = spec(app).theme
    assert (t.template, t.palette, t.font_size) == ("light", "colorblind", 16)


def test_edit_move_duplicate_remove(app):
    use_template(app, "sales")
    first = spec(app).tiles[0]
    n = len(spec(app).tiles)
    app.checkbox(key=f"edit_on_{first.id}").set_value(True).run()
    assert not app.exception, [e.value for e in app.exception]
    app.text_input(key=f"edit_{first.id}_title").set_value("Renamed tile").run()
    app.text_input(key=f"edit_{first.id}_section").set_value("Custom").run()
    click(app, key=f"save_{first.id}")
    edited = next(t for t in spec(app).tiles if t.id == first.id)
    assert edited.title == "Renamed tile" and edited.section == "Custom"
    assert "Custom" in spec(app).sections
    click(app, key=f"dup_{first.id}")
    assert len(spec(app).tiles) == n + 1
    ids = [t.id for t in spec(app).tiles]
    second = ids[1]
    click(app, key=f"up_{second}")
    assert [t.id for t in spec(app).tiles][0] != ids[0] or [t.id for t in spec(app).tiles][
        1
    ] != ids[1]
    click(app, key=f"rm_{second}")
    assert len(spec(app).tiles) == n


def test_pending_ai_tile_requires_approval(app):
    tile = db.Tile(kind="bar", title="AI bar", params={"x": "country", "y": "spend", "agg": "sum"})
    app.session_state["dash_pending"] = tile.model_dump()
    app.run()
    assert not spec(app).tiles
    click(app, "Add this tile")
    assert spec(app).tiles[0].title == "AI bar" and "dash_pending" not in app.session_state


def test_layout_problems_are_reported(app):
    s = spec(app)
    s.tiles.append(db.Tile(kind="bar", title="Orphan", params={"x": "no_such_column"}))
    app.run()
    assert any("1 layout problem" in e.label for e in app.expander)
    assert any("no_such_column" in w.value for w in app.warning)


def test_code_buttons_for_notebook_and_streamlit_app(app):
    use_template(app, "sales")
    collected = {k: c for k, (_, c) in app.session_state["code_collect"].items()}
    assert "code_dash" in collected and "code_dash_app" in collected
    assert "st.set_page_config" in collected["code_dash_app"] and "px." in collected["code_dash"]
    app.selectbox(key="code_dash_o_plot_lib").set_value(
        "matplotlib"
    ).run()  # options panel is present
    assert not app.exception


def test_shared_layout_json_round_trips_through_state(app):
    use_template(app, "auto")
    s = spec(app)
    again = db.spec_from_json(db.spec_to_json(s))
    assert [t.title for t in again.tiles] == [t.title for t in s.tiles] and again.theme == s.theme
