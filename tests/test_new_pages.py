"""End-to-end AppTest flows for the data-scientist / workspace pages on a realistic dataset."""

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from src.core.session_state import KEY_CURRENT_DATASET
from src.ingestion.service import ingest_into_state
from src.workspace import projects
from streamlit.testing.v1 import AppTest
from tests.page_map import open_page

ROOT = Path(__file__).resolve().parent.parent


def synthetic(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    df = pd.DataFrame(
        {
            "customer_id": np.arange(n),
            "signup_date": pd.date_range("2023-01-01", periods=n, freq="D").astype(str),
            "age": rng.integers(18, 75, n),
            "spend": rng.gamma(2, 40, n).round(2),
            "tickets": rng.poisson(1.5, n),
            "country": rng.choice(["US", "UK", "DE", "FR"], n),
            "variant": rng.choice(["A", "B"], n),
        }
    )
    df["converted"] = (rng.random(n) < 0.10 + 0.04 * (df["variant"] == "B")).astype(int)
    df["churn"] = ((df["tickets"] * 0.4 + rng.normal(0, 1, n) - df["spend"] / 100) > 0.6).astype(
        int
    )
    return df


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setattr(projects, "DEFAULT_ROOT", tmp_path / "projects")
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
    buf = io.StringIO()
    synthetic().to_csv(buf, index=False)
    ingest_into_state(at.session_state, buf.getvalue().encode(), "shop.csv", max_size_mb=10)
    return at


def go(at: AppTest, page: str) -> AppTest:
    open_page(at, page)
    assert not at.exception, [e.value for e in at.exception]
    return at


def click(at: AppTest, label: str, key: str | None = None) -> AppTest:
    button = next(b for b in at.button if (b.key == key if key else label in b.label))
    button.click().run()
    assert not at.exception, [e.value for e in at.exception]
    return at


@pytest.mark.parametrize(
    "page",
    [
        "8_Projects_Sessions.py",
        "9_Data_Dictionary.py",
        "10_Data_Sources.py",
        "11_Transformation_Studio.py",
        "12_Dashboard_Builder.py",
        "13_Feature_Engineering.py",
        "14_Experiments.py",
        "15_Unsupervised_Lab.py",
        "16_Time_Series_Lab.py",
        "17_Model_Tuning.py",
        "18_Model_Deployment.py",
        "19_Settings.py",
    ],
)
def test_page_renders_with_realistic_data(app, page):
    go(app, page)


def test_projects_save_open_roundtrip(app, tmp_path):
    go(app, "8_Projects_Sessions.py")
    click(app, "Save project")
    assert any("Saved" in s.value for s in app.success)
    assert (tmp_path / "projects" / "shop" / "versions" / "v001" / "meta.json").exists()
    app.session_state[KEY_CURRENT_DATASET] = None
    go(app, "8_Projects_Sessions.py")
    click(app, "Open this version")
    assert app.session_state[KEY_CURRENT_DATASET].filename == "shop.csv"


def test_data_sources_sql_run_and_save(app):
    go(app, "10_Data_Sources.py")
    app.text_area(key="sql_text").set_value(
        "SELECT country, COUNT(*) AS n FROM active GROUP BY country"
    ).run()
    click(app, "Run query")
    result = app.session_state["sql_result"]
    assert set(result.columns) == {"country", "n"} and result["n"].sum() == 400
    click(app, "Add to library")
    assert "sql_result" in app.session_state["dataset_library"]


def test_data_sources_blocks_writes(app):
    go(app, "10_Data_Sources.py")
    app.text_area(key="sql_text").set_value("DROP TABLE active").run()
    click(app, "Run query")
    assert any("SELECT" in e.value or "not allowed" in e.value for e in app.error)
    assert len(app.session_state[KEY_CURRENT_DATASET].working_df) == 400


def test_transformation_studio_merge_applies_and_rolls_back(app):
    lookup = pd.DataFrame({"country": ["US", "UK", "DE", "FR"], "region": ["NA", "EU", "EU", "EU"]})
    app.session_state["dataset_library"] = {"lookup": lookup}
    go(app, "11_Transformation_Studio.py")
    app.selectbox(key="studio_op").set_value("merge").run()
    assert not app.exception, [e.value for e in app.exception]
    click(app, "Apply and save")
    ops = app.session_state["transform_ops"]
    assert len(ops) == 1 and ops[0].output in app.session_state["dataset_library"]
    merged = app.session_state["dataset_library"][ops[0].output]
    assert "region" in merged.columns and len(merged) == 400
    click(app, "Undo last step")
    assert not app.session_state["transform_ops"]


def test_feature_engineering_add_step_and_leakage(app):
    go(app, "13_Feature_Engineering.py")
    ms = next(m for m in app.multiselect if m.label == "Numeric columns")
    app.selectbox[0].set_value("log1p").run()
    ms = next(m for m in app.multiselect if m.label == "Numeric columns")
    ms.set_value(["spend"]).run()
    click(app, "Add this step")
    specs = app.session_state["feature_specs"]
    assert specs and specs[0].op == "log1p"
    go(app, "13_Feature_Engineering.py")


def test_dashboard_add_tile(app):
    go(app, "12_Dashboard_Builder.py")
    click(app, "Add tile")
    spec = app.session_state["dashboard_spec"]
    assert len(spec.tiles) == 1
    go(app, "12_Dashboard_Builder.py")
    assert any(m.label for m in app.metric)  # the KPI tile renders


def test_experiments_peeking_simulation(app):
    go(app, "14_Experiments.py")
    click(app, "Run simulation")
    labels = {m.label for m in app.metric}
    assert "False positives when peeking" in labels


def test_unsupervised_lab_renders_tabs_with_data(app):
    go(app, "15_Unsupervised_Lab.py")
    assert {"🧩 Clusters", "📉 PCA", "🚨 Anomalies"} <= {t.label for t in app.tabs}
    assert len(app.dataframe) >= 1


def test_time_series_backtest_flow(app):
    go(app, "16_Time_Series_Lab.py")
    app.selectbox[2].set_value("D").run()  # daily frequency
    click(app, "Run backtest")
    table = app.session_state["ts_backtest"]
    assert len(table) == 5 and table["RMSE"].notna().any()


def test_tuning_then_deployment_flow(app):
    go(app, "17_Model_Tuning.py")
    app.selectbox[0].set_value("churn").run()
    ms = app.multiselect[0]
    ms.set_value(["age", "spend", "tickets", "country"]).run()
    click(app, "Run cross-validation")
    assert len(app.session_state["mt_board"]) == 4
    click(app, "Tune")
    models = app.session_state["fitted_models"]
    assert len(models) == 1
    go(app, "18_Model_Deployment.py")
    click(app, "Predict")
    assert any(m.label.startswith("Predicted") for m in app.metric)


def test_settings_override_and_privacy_scan(app):
    go(app, "19_Settings.py")
    app.selectbox[0].set_value("ollama").run()
    click(app, "Apply for this session")
    assert app.session_state["llm_override"]["provider"] == "ollama"
    click(app, "Reset to defaults")
    assert app.session_state["llm_override"] is None
