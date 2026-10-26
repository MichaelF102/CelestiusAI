"""Home renders its workspace panel with a dataset loaded."""

from pathlib import Path

from src.ingestion.service import load_catalog_sample
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent


def test_home_with_loaded_dataset():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    load_catalog_sample(at.session_state, "house_prices", 50)
    at.switch_page("pages/0_Home.py").run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("house_prices.csv" in m.value for m in at.markdown)
