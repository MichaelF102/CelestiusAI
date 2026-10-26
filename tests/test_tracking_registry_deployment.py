"""Experiment tracking, model registry, scoring checks, API template, and prediction monitoring."""

import ast
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from src.datascience import deployment as dep
from src.datascience import modeling as m
from src.datascience import tracking as tr
from src.workspace.registry import ModelRegistry, RegistryError, model_card_markdown, slugify


@pytest.fixture(scope="module")
def df():
    rng = np.random.default_rng(3)
    n = 400
    out = pd.DataFrame(
        {
            "age": rng.integers(20, 70, n),
            "spend": rng.gamma(2, 50, n),
            "plan": rng.choice(["basic", "pro", "team"], n),
        }
    )
    out["churn"] = (
        (out.age - 45) * 0.04
        + (out.spend - 100) * 0.01
        + (out.plan == "pro") * 0.7
        + rng.normal(0, 1, n)
        > 0.3
    ).astype(int)
    out["revenue"] = out.age * 2 + out.spend + rng.normal(0, 5, n)
    return out


@pytest.fixture(scope="module")
def clf_model(df):
    return m.fit_model(df, "churn", ["age", "spend", "plan"], "Random Forest")


# --------------------------------------------------------------------------- experiment log
def run(name="r", model="Random Forest", **kw):
    base = {
        "name": name,
        "task": "binary_classification",
        "target": "churn",
        "features": ["age", "spend"],
        "model": model,
        "metrics": {"accuracy": 0.8, "f1_macro": 0.7},
    }
    return tr.RunRecord(**{**base, **kw})


def test_log_best_compare_and_roundtrip():
    log = tr.ExperimentLog()
    a = log.add(
        run(
            "a",
            params={"n_estimators": 100, "max_depth": 5},
            metrics={"f1_macro": 0.70},
            data_fingerprint="f1",
        )
    )
    b = log.add(
        run(
            "b",
            params={"n_estimators": 300, "max_depth": 5},
            metrics={"f1_macro": 0.78, "roc_auc": 0.9},
            features=["age", "spend", "plan"],
            data_fingerprint="f1",
        )
    )
    log.add(run("c", target="other", metrics={"f1_macro": 0.99}))
    assert (
        log.best("f1_macro").name == "c"
        and log.best("f1_macro", target="churn").name == "b"
        and log.best("roc_auc").name == "b"
    )
    assert log.best("mae", higher_is_better=False) is None
    cmp = log.compare([a.run_id, b.run_id])
    assert (
        list(cmp["params_diff"].index) == ["n_estimators"]
        and cmp["features_unique"][b.run_id] == ["plan"]
        and cmp["same_data"] is True
    )
    assert cmp["metrics"].loc["f1_macro", b.run_id] == 0.78 and cmp["common_features"] == [
        "age",
        "spend",
    ]
    frame = log.to_frame()
    assert len(frame) == 3 and {"run_id", "model", "f1_macro"} <= set(frame.columns)
    again = tr.ExperimentLog.from_json(log.to_json())
    assert [r.name for r in again.runs] == ["a", "b", "c"] and again.runs[1].params == b.params
    log.remove(a.run_id)
    assert len(log.runs) == 2
    with pytest.raises(tr.TrackingError):
        log.get("nope")
    with pytest.raises(tr.TrackingError, match="at least two"):
        log.compare([b.run_id])
    with pytest.raises(tr.TrackingError):
        tr.ExperimentLog.from_json('[{"bad": 1}]')


@pytest.mark.parametrize(
    "model,task,target,metric",
    [
        ("Random Forest", "binary_classification", "churn", "accuracy"),
        ("Ridge / L2 Logistic", "regression", "revenue", "R2"),
        ("XGBoost", "binary_classification", "churn", "accuracy"),
        ("Polynomial (degree 2) + Ridge", "regression", "revenue", "R2"),
    ],
)
def test_reproduce_code_recovers_the_recorded_metrics(model, task, target, metric, df, tmp_path):
    csv = tmp_path / "d.csv"
    df.to_csv(csv, index=False)
    fitted = m.fit_model(df, target, ["age", "spend", "plan"], model, task=task, params=None)
    record = tr.RunRecord(
        name="x",
        task=task,
        target=target,
        features=["age", "spend", "plan"],
        model=model,
        params={},
        metrics=dict(fitted.metrics.items()),
        seed=42,
        test_size=0.2,
    )
    code = tr.reproduce_code(record, filename=str(csv))
    ast.parse(code)
    ns: dict = {"__name__": "x"}
    exec(compile(code, "<run>", "exec"), ns)  # noqa: S102
    assert ns["metrics"][metric] == pytest.approx(fitted.metrics[metric], abs=1e-9)


def test_reproduce_code_with_tuned_parameters(df, tmp_path):
    csv = tmp_path / "d.csv"
    df.to_csv(csv, index=False)
    fitted = m.fit_model(
        df,
        "churn",
        ["age", "spend", "plan"],
        "Random Forest",
        params={"n_estimators": 20, "max_depth": 3},
    )
    record = tr.RunRecord(
        name="t",
        task="binary_classification",
        target="churn",
        features=["age", "spend", "plan"],
        model="Random Forest (tuned)",
        params={"n_estimators": 20, "max_depth": 3},
        metrics=fitted.metrics,
    )
    code = tr.reproduce_code(record, filename=str(csv))
    assert "max_depth=3" in code and "n_estimators=20" in code
    ns: dict = {}
    exec(compile(code, "<run>", "exec"), ns)  # noqa: S102
    assert ns["metrics"]["accuracy"] == pytest.approx(fitted.metrics["accuracy"])


def test_estimator_code_handles_pipelines_and_wrappers():
    from src.datascience import catalog

    imports, expr = tr.estimator_code(catalog.build("Polynomial (degree 2) + Ridge", "regression"))
    assert "from sklearn.pipeline import Pipeline" in imports and "PolynomialFeatures" in expr
    imp, e = tr.estimator_code(catalog.build("XGBoost", "binary_classification"))
    assert "from xgboost import XGBClassifier" in imp and "EncodedClassifier" not in e


# --------------------------------------------------------------------------- registry
def test_registry_versions_stages_and_cards(clf_model, tmp_path):
    reg = ModelRegistry(tmp_path)
    v1 = reg.register(
        clf_model,
        "Churn Model",
        notes="Retention team",
        extras={"data_fingerprint": "abc", "data_version": 3},
    )
    v2 = reg.register(clf_model, "Churn Model", stage="staging")
    assert (v1["version"], v1["stage"], v2["version"]) == (1, "draft", 2)
    reg.set_stage("churn-model", 1, "production")
    reg.set_stage("churn-model", 2, "production")  # only one production version at a time
    stages = {m_["version"]: m_["stage"] for m_ in reg.versions("churn-model")}
    assert stages == {1: "archived", 2: "production"}
    listing = reg.list_models()[0]
    assert (
        listing["production_version"] == 2
        and listing["versions"] == 2
        and listing["name"] == "Churn Model"
    )
    card = reg.card("churn-model", 1)
    assert (
        "Retention team" in card
        and "`abc`" in card
        and "**Stage:** archived" in card
        and "Random Forest" in card
    )
    loaded = reg.load("churn-model")
    assert (
        loaded.features == clf_model.features
        and loaded.classes == clf_model.classes
        and loaded.estimator_name == "Random Forest"
    )
    sample = pd.DataFrame({"age": [30, 60], "spend": [50.0, 250.0], "plan": ["basic", "pro"]})
    np.testing.assert_array_equal(
        m.score_frame(loaded, sample)[0]["predicted_churn"],
        m.score_frame(clf_model, sample)[0]["predicted_churn"],
    )
    with pytest.raises(RegistryError):
        reg.load("churn-model", 9)
    with pytest.raises(RegistryError):
        reg.set_stage("churn-model", 1, "live")
    reg.delete("churn-model")
    assert reg.list_models() == []


def test_registry_rejects_bad_names_and_traversal(clf_model, tmp_path):
    reg = ModelRegistry(tmp_path)
    for bad in ("", "///", ".."):
        with pytest.raises(RegistryError):
            reg.register(clf_model, bad)
    reg.register(clf_model, "../../evil")
    assert (tmp_path / "evil").exists() and not (tmp_path.parent / "evil").exists()
    with pytest.raises(RegistryError):
        reg.register(clf_model, "x", stage="bogus")
    assert slugify(" My Model! ") == "my-model"


def test_model_card_content(clf_model):
    fair = {"warnings": ["Disparate impact ratio 0.55"], "disparate_impact": 0.55}
    card = model_card_markdown(clf_model, "M", 2, "staging", extras={"fairness": fair})
    assert (
        "# Model card: M (v2)" in card
        and "Disparate impact ratio 0.55" in card
        and "| age | numeric |" in card
    )
    assert "Limitations" in card and "only 400 rows" in card and "joblib.load" in card
    assert "No disparities" in model_card_markdown(
        clf_model, "M", 1, "draft", extras={"fairness": {"warnings": []}}
    )


# --------------------------------------------------------------------------- scoring validation
def test_validate_scoring_frame_flags_real_problems(clf_model):
    ok = m.validate_scoring_frame(
        clf_model, pd.DataFrame({"age": [30], "spend": [60.0], "plan": ["pro"]})
    )
    assert ok["ok"] and not [i for i in ok["issues"] if i[0] in ("error", "warning")]
    bad = pd.DataFrame(
        {
            "age": [30, 999, 18],
            "spend": ["x", "10", "20"],
            "plan": ["pro", "gold", None],
            "extra": [1, 2, 3],
        }
    )
    res = m.validate_scoring_frame(clf_model, bad)
    levels = [lvl for lvl, _ in res["issues"]]
    assert not res["ok"] and "error" in levels and "warning" in levels
    assert res["out_of_range"].set_index("feature").loc[
        "age", "share_outside_training_range"
    ] == pytest.approx(2 / 3)
    assert "gold" in res["unseen_categories"].iloc[0]["unseen_values"] and "extra" in " ".join(
        msg for _, msg in res["issues"]
    )
    assert res["missing_values"].iloc[0]["feature"] == "plan"
    missing = m.validate_scoring_frame(clf_model, pd.DataFrame({"age": [30]}))
    assert not missing["ok"] and set(missing["missing_columns"]) == {"spend", "plan"}


# --------------------------------------------------------------------------- API template
def test_fastapi_template_serves_predictions(clf_model, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    (tmp_path / "model.joblib").write_bytes(m.export_model_bytes(clf_model))
    code = dep.fastapi_template(clf_model)
    ast.parse(code)
    (tmp_path / "app.py").write_text(code)
    monkeypatch.chdir(tmp_path)
    ns: dict = {"__name__": "app"}
    exec(compile(code, "app.py", "exec"), ns)  # noqa: S102
    client = TestClient(ns["app"])
    health = client.get("/health").json()
    assert health["status"] == "ok" and health["target"] == "churn"
    rows = [{"age": 30, "spend": 50.0, "plan": "basic"}, {"age": 65, "spend": 300.0, "plan": "pro"}]
    body = client.post("/predict", json=rows).json()
    expected = m.score_frame(clf_model, pd.DataFrame(rows))[0]
    assert body["predictions"] == expected["predicted_churn"].tolist()
    assert (
        len(body["probabilities"]) == 2 and abs(sum(body["probabilities"][0].values()) - 1) < 1e-9
    )
    assert client.post("/predict", json=[{"age": "not a number"}]).status_code == 422
    assert "fastapi" in dep.requirements_text(clf_model) and "xgboost" not in dep.requirements_text(
        clf_model
    )


def test_exported_bundle_roundtrips_through_joblib(clf_model, tmp_path):
    path = tmp_path / "m.joblib"
    path.write_bytes(m.export_model_bytes(clf_model))
    back = m.model_from_bundle(joblib.load(path))
    assert (
        back.name == clf_model.name
        and back.rows_used == clf_model.rows_used
        and back.estimator_name == "Random Forest"
    )


# --------------------------------------------------------------------------- monitoring
def test_prediction_log_outcomes_and_alerts():
    rng = np.random.default_rng(1)
    rows = pd.DataFrame({"age": rng.integers(20, 70, 60), "spend": rng.gamma(2, 50, 60)})
    preds = rng.integers(0, 2, 60)
    log = tr.log_predictions(None, "m1", rows, preds)
    log = tr.log_predictions(
        log,
        "m1",
        rows.head(5),
        preds[:5],
        ids=pd.Series([f"x{i}" for i in range(5)]),
        proba=np.full(5, 0.7),
    )
    assert (
        len(log) == 65
        and {"prediction_id", "scored_at", "prediction", "age", "score"} <= set(log.columns)
        and log["prediction_id"].is_unique
    )
    outcomes = pd.DataFrame(
        {"prediction_id": log["prediction_id"].iloc[:60], "outcome": preds}
    )  # perfect agreement
    labeled = tr.attach_outcomes(log, outcomes, "prediction_id", "outcome")
    good = tr.live_performance(labeled, "binary_classification", {"f1_macro": 0.9})
    assert good["n_labeled"] == 60 and good["metrics"]["accuracy"] == 1.0 and not good["alerts"]
    wrong = labeled.assign(actual=1 - labeled["actual"])
    bad = tr.live_performance(wrong, "binary_classification", {"f1_macro": 0.9})
    assert (
        bad["alerts"] and "retraining" in bad["alerts"][0] and bad["vs_baseline"]["f1_macro"] < -0.5
    )
    few = tr.live_performance(labeled.head(10), "binary_classification")
    assert "too few" in few["alerts"][0]
    reg = tr.live_performance(
        labeled.assign(
            actual=labeled["prediction"] + 0.5, prediction=labeled["prediction"].astype(float)
        ),
        "regression",
        {"R2": 0.9},
        min_labeled=10,
    )
    assert reg["metrics"]["MAE"] == pytest.approx(0.5)
    with pytest.raises(tr.TrackingError):
        tr.attach_outcomes(log, pd.DataFrame({"x": [1]}), "prediction_id", "outcome")
    assert json.loads(tr.ExperimentLog().to_json()) == []
