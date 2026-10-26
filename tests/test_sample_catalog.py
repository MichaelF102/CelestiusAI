"""The bundled sample datasets load, profile, and appear in the Overview."""

from pathlib import Path

import pandas as pd
import pytest
from src.ingestion.service import SAMPLE_CATALOG, load_catalog_sample
from streamlit.testing.v1 import AppTest


def test_ten_samples_exist():
    assert len(SAMPLE_CATALOG) == 10
    assert len({c["key"] for c in SAMPLE_CATALOG}) == 10


@pytest.mark.parametrize("sample", SAMPLE_CATALOG, ids=lambda s: s["key"])
def test_each_sample_loads(sample):
    state: dict = {}
    managed = load_catalog_sample(state, sample["key"], 50)
    df = managed.working_df
    assert isinstance(df, pd.DataFrame) and len(df) >= 500 and df.shape[1] >= 6


def test_unknown_sample_rejected():
    with pytest.raises(ValueError):
        load_catalog_sample({}, "../secrets", 50)


def test_overview_gallery_loads_a_sample():
    at = AppTest.from_file(
        str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=120
    ).run()
    at.switch_page("pages/1_Executive_Overview.py").run()
    assert not at.exception
    next(b for b in at.button if b.key == "sample_house_prices").click().run()
    assert not at.exception
    assert at.session_state["current_dataset"].filename == "house_prices.csv"
