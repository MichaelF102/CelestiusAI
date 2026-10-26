"""Headless Streamlit AppTest runs for each page, plus HTML-escaping checks."""

from pathlib import Path

import pytest
from src.core.session_state import KEY_CURRENT_DATASET
from src.core.ui_components import esc
from src.ingestion.service import ingest_into_state
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "sample_data" / "ecommerce_customer_churn.csv"
PAGES = sorted(p.name for p in (ROOT / "pages").glob("*.py"))


def test_esc_neutralizes_markup():
    assert esc("<img src=x onerror=alert(1)>") == "&lt;img src=x onerror=alert(1)&gt;"
    assert esc('a"b') == "a&quot;b"
    assert esc(5) == "5"


def _app(timeout: int = 120) -> AppTest:
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=timeout)


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_dataset(page):
    at = _app().run()
    at.switch_page(f"pages/{page}").run()
    assert not at.exception, [e.value for e in at.exception]


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_with_dataset(page):
    at = _app().run()
    ingest_into_state(at.session_state, SAMPLE.read_bytes(), SAMPLE.name, max_size_mb=10)
    at.switch_page(f"pages/{page}").run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state[KEY_CURRENT_DATASET] is not None


def test_overview_has_upload_searchbar_and_kpis():
    at = _app().run()
    at.switch_page("pages/1_Executive_Overview.py").run()
    assert not at.exception
    assert len(at.chat_input) == 1
    assert at.chat_input[0].proto.accept_file


def test_exploration_page_renders_content_with_dataset():
    at = _app().run()
    ingest_into_state(at.session_state, SAMPLE.read_bytes(), SAMPLE.name, max_size_mb=10)
    at.switch_page("pages/3_Exploration_Studio.py").run()
    assert not at.exception
    assert len(at.tabs) > 0
    assert len(at.markdown) > 5
