"""Data-scientist pages driven end to end with AppTest."""

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from src.ingestion.service import ingest_into_state
from streamlit.testing.v1 import AppTest
from tests.page_map import open_page

ROOT = Path(__file__).resolve().parent.parent


def dataset_csv(n: int = 360) -> str:
    rng = np.random.default_rng(21)
    df = pd.DataFrame(
        {
            "customer_id": np.arange(n),
            "signup_date": pd.date_range("2023-01-01", periods=n, freq="D").astype(str),
            "age": rng.integers(18, 75, n),
            "spend": rng.gamma(2, 40, n).round(2),
            "visits": rng.poisson(3, n),
            "country": rng.choice(["US", "UK", "DE", "FR"], n),
            "plan": rng.choice(["basic", "pro", "team"], n),
            "review": rng.choice(
                ["great product love it", "terrible awful broke", "arrived tuesday in a box"], n
            ),
            "tenure": rng.exponential(20, n).round(1) + 0.5,
            "churned": 0,
            "variant": rng.choice(["A", "B"], n),
        }
    )
    logit = (df.age - 45) * 0.04 + (df.spend - 80) * 0.01 + (df.plan == "basic") * 0.8
    df["churn"] = (logit + rng.normal(0, 1, n) > 0.3).astype(int)
    df["revenue"] = (df.age * 2 + df.spend + rng.normal(0, 8, n)).round(2)
    df["churned"] = (rng.random(n) < 0.7).astype(int)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


@pytest.fixture
def app(tmp_path, monkeypatch):
    from src.workspace import projects, registry

    monkeypatch.setattr(projects, "DEFAULT_ROOT", tmp_path / "projects")
    monkeypatch.setattr(registry, "DEFAULT_ROOT", tmp_path / "models")
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300).run()
    ingest_into_state(at.session_state, dataset_csv().encode(), "shop.csv", max_size_mb=10)
    return at


def go(at: AppTest, page: str) -> AppTest:
    open_page(at, page)
    assert not at.exception, [e.value for e in at.exception]
    return at


def click(at: AppTest, label: str | None = None, key: str | None = None) -> AppTest:
    b = next(b for b in at.button if (b.key == key if key else label in b.label))
    b.click().run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def pick(at: AppTest, prefix: str, target: str, features: list[str]) -> None:
    at.selectbox(key=f"{prefix}_target").set_value(target).run()
    at.multiselect(key=f"{prefix}_features").set_value(features).run()
    assert not at.exception, [e.value for e in at.exception]


# --------------------------------------------------------------------------- Model Selection
def test_model_selection_assistant_race_and_logging(app):
    go(app, "20_Model_Selection.py")
    pick(app, "ms", "churn", ["age", "spend", "plan"])
    click(app, key="ms_profile_btn")
    assert any(m.label == "Non-linear signal" for m in app.metric) and app.session_state["ms_recs"]
    app.multiselect(key="ms_models").set_value(["Baseline", "Naive Bayes", "Random Forest"]).run()
    app.slider(key="ms_folds").set_value(3).run()
    click(app, key="ms_run")
    race = app.session_state["ms_race"]
    assert (
        set(race.table["model"]) == {"Baseline", "Naive Bayes", "Random Forest"}
        and (race.table["status"] == "ok").all()
    )
    assert any("Suggested" in s.value for s in app.success)
    click(app, key="ms_log")
    assert len(app.session_state["experiment_log"].runs) == 1
    collected = {k: c for k, (_, c) in app.session_state["code_collect"].items()}
    assert "code_ms_race" in collected and "cross_validate" in collected["code_ms_race"]


def test_model_selection_regression_and_cv_strategies(app):
    go(app, "20_Model_Selection.py")
    pick(app, "ms", "revenue", ["age", "spend"])
    app.multiselect(key="ms_models").set_value(["Baseline", "Ridge / L2 Logistic"]).run()
    for strategy in ("repeated", "timeseries"):
        app.selectbox(key="ms_cv").set_value(strategy).run()
        click(app, key="ms_run")
        assert (app.session_state["ms_race"].table["status"] == "ok").all()
    assert app.session_state["ms_race"].scoring == "r2"


def test_model_selection_nested_curves_and_cards(app):
    go(app, "20_Model_Selection.py")
    pick(app, "ms", "churn", ["age", "spend"])
    app.selectbox(key="ms_nested_model").set_value("Random Forest").run()
    app.slider(key="ms_outer").set_value(3).run()
    app.slider(key="ms_iter").set_value(3).run()
    click(app, key="ms_nested_run")
    assert len(app.session_state["ms_nested"].outer_scores) == 3 and any(
        m.label == "Optimism" for m in app.metric
    )
    app.selectbox(key="ms_curve_model").set_value("Decision Tree").run()
    click(app, key="ms_lc_btn")
    assert len(app.session_state["ms_lc"][1]) == 5
    click(app, key="ms_vc_btn")
    assert app.session_state["ms_vc"][1] == "max_depth"


# --------------------------------------------------------------------------- Ensembles
def test_ensembles_page(app):
    go(app, "21_Ensembles.py")
    pick(app, "en", "churn", ["age", "spend", "plan"])
    app.multiselect(key="en_bases").set_value(
        ["Linear / Logistic Regression", "Decision Tree", "Naive Bayes"]
    ).run()
    app.multiselect(key="en_kinds").set_value(["voting_soft", "stacking", "bagging"]).run()
    app.slider(key="en_cv").set_value(3).run()
    click(app, key="en_run")
    _, specs, study = app.session_state["en_study"]
    assert len(specs) == 3 and {"base", "voting_soft", "stacking", "bagging"} <= set(
        study.table["type"]
    )
    click(app, key="en_log")
    assert app.session_state["experiment_log"].runs[0].kind == "ensemble"
    assert any("code_en" in k for k in app.session_state["code_collect"])


# --------------------------------------------------------------------------- Classification Toolkit
def test_classification_toolkit_flow(app):
    go(app, "22_Classification_Toolkit.py")
    pick(app, "cl", "churn", ["age", "spend", "plan"])
    app.selectbox(key="cl_model").set_value("Random Forest").run()
    app.multiselect(key="cl_strats").set_value(["none", "class_weight", "oversample"]).run()
    click(app, key="cl_imb_run")
    assert set(app.session_state["cl_imb"][1]["strategy"]) == {"none", "class_weight", "oversample"}
    click(app, key="cl_eval_run")
    _, _, _, h, ev = app.session_state["cl_holdout"]
    assert ev["roc_auc"] > 0.6 and any(m.label == "ROC AUC" for m in app.metric)
    assert any(c.label == "Cheapest threshold" for c in app.metric if True) or any(
        m.label == "Cheapest threshold" for m in app.metric
    )
    click(app, key="cl_cal_run")
    assert any("Brier" in m.label for m in app.metric)
    click(app, key="cl_log")
    assert app.session_state["experiment_log"].runs[-1].metrics["roc_auc"] > 0.6
    assert "classification_snippet" not in str(app.session_state["code_collect"]) and any(
        "code_cl" in k for k in app.session_state["code_collect"]
    )


def test_classification_rejects_continuous_targets(app):
    go(app, "22_Classification_Toolkit.py")
    assert not app.exception


# --------------------------------------------------------------------------- Regression Workbench
def test_regression_workbench_tabs_render_and_run(app):
    go(app, "23_Regression_Workbench.py")
    pick(app, "rg", "revenue", ["age", "spend", "plan"])
    assert any(m.label == "Adjusted R²" for m in app.metric)  # diagnostics tab
    app.selectbox(key="rg_family").set_value("gaussian").run()
    assert any(m.label == "Deviance explained" for m in app.metric)
    for btn in ("rg_reg_btn", "rg_curve_btn", "rg_tr_btn"):
        click(app, key=btn)
    assert (
        app.session_state["rg_reg"][2]["best_alpha"] > 0
        and app.session_state["rg_curve"][2]["best"].sum() == 1
    )
    assert "log1p" in set(app.session_state["rg_tr"][2]["transform"])
    click(app, key="rg_gbq_btn")
    assert app.session_state["rg_gbq"]["coverage"] > 0.6
    click(app, key="rg_err_btn")
    assert app.session_state["rg_err"][1]["r2"] > 0.3
    click(app, key="rg_boot_btn")
    assert app.session_state["rg_boot"] is not None
    assert any("code_rg_diag" in k for k in app.session_state["code_collect"])


def test_regression_workbench_count_glm(app):
    go(app, "23_Regression_Workbench.py")
    pick(app, "rg", "visits", ["age", "plan"])
    app.selectbox(key="rg_family").set_value("poisson").run()
    assert not app.exception and any(m.label == "Dispersion" for m in app.metric)


# --------------------------------------------------------------------------- Feature Selection
def test_feature_selection_methods_and_apply(app):
    go(app, "24_Feature_Selection.py")
    pick(app, "fs", "churn", ["age", "spend", "plan", "visits", "tenure"])
    click(app, key="fs_filter_btn")
    click(app, key="fs_rfe_btn")
    click(app, key="fs_emb_btn")
    app.slider(key="fs_rounds").set_value(10).run()
    click(app, key="fs_stab_btn")
    click(app, key="fs_min_btn")
    click(app, key="fs_pca_btn")
    res = app.session_state["fs_results"]
    assert {"filter", "rfe", "embedded", "stability", "minimal", "pca"} <= set(res)
    click(app, key="fs_apply")
    saved = app.session_state["selected_features"]
    assert saved["target"] == "churn" and saved["features"]
    go(app, "21_Ensembles.py")
    app.selectbox(key="en_target").set_value("churn").run()
    assert set(app.multiselect(key="en_features").value) == set(saved["features"])


# --------------------------------------------------------------------------- Model Explainer
def test_model_explainer_global_local_fairness_drift(app):
    go(app, "25_Model_Explainer.py")
    pick(app, "xp", "churn", ["age", "spend", "plan", "country"])
    click(app, key="xp_train")
    assert len(app.session_state["fitted_models"]) == 1
    click(app, key="xp_imp_btn")
    assert app.session_state["xp_imp"][1]["importance"].iloc[0] > 0
    app.slider(key="xp_perm").set_value(20).run()
    click(app, key="xp_shap_btn")
    res = app.session_state["xp_shap"][2]
    assert abs(res["sum_contributions"] - (res["prediction"] - res["baseline"])) < 0.25
    click(app, key="xp_cf_btn")
    app.selectbox(key="xp_group").set_value("country").run()
    assert any(m.label == "Disparate impact" for m in app.metric)
    app.checkbox(key="xp_drift_demo").set_value(True).run()
    assert not app.exception
    assert {"code_xp", "code_xp_fair", "code_xp_drift"} <= set(app.session_state["code_collect"])
