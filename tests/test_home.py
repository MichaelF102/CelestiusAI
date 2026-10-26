"""Tests for Home page content integrity and rendering."""

import re
from pathlib import Path

from src.core.site_content import MODULES, TRACKS, WORKFLOW
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent


def test_every_linked_page_exists_and_is_registered():
    router = (ROOT / "app.py").read_text()
    paths = {m.page for m in MODULES} | {s.page for s in WORKFLOW}
    for path in paths:
        assert (ROOT / path).exists(), path
        assert f'"{path}"' in router, f"{path} missing from navigation"


def test_modules_cover_all_pages_and_known_tracks():
    pages = {f"pages/{p.name}" for p in (ROOT / "pages").glob("[1-9]*_*.py")}
    assert pages == {m.page for m in MODULES}
    assert {m.track for m in MODULES} <= set(TRACKS)
    assert [s.number for s in WORKFLOW] == list(range(1, len(WORKFLOW) + 1))


def test_home_renders_sections_without_dataset():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
    headings = {s.value for s in at.subheader}
    assert {"About the project", "Components", "Workflow", "Quick guide"} <= headings
    assert any("Try sample data" in b.label for b in at.button)


def test_home_try_sample_loads_dataset():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    next(b for b in at.button if "Try sample data" in b.label).click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["current_dataset"] is not None


def test_no_stale_phase_text():
    text = (ROOT / "pages" / "0_Home.py").read_text() + (
        ROOT / "src/core/site_content.py"
    ).read_text()
    assert not re.search(r"Planned for Phase", text)
