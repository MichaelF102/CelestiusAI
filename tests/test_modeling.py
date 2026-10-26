"""Tests for model comparison, tuning, export and scoring."""

import ast

import joblib
import numpy as np
import pandas as pd
import pytest
from src.datascience import modeling as m


@pytest.fixture(scope="module")
def clf_df():
    rng = np.random.default_rng(0)
    n = 240
    df = pd.DataFrame(
        {
            "age": rng.integers(18, 70, n),
            "spend": rng.normal(100, 30, n),
            "plan": rng.choice(["basic", "pro", "team"], n),
        }
    )
    logit = 0.04 * (df["age"] - 40) + 0.03 * (df["spend"] - 100) + (df["plan"] == "pro") * 0.8
    df["churn"] = (logit + rng.normal(0, 1, n) > 0.3).astype(int)
    df.loc[::17, "spend"] = np.nan  # missing values must be imputed, not crash
    return df


@pytest.fixture(scope="module")
def reg_df(clf_df):
    out = clf_df.copy()
    out["revenue"] = (
        out["age"] * 2
        + out["spend"].fillna(100) * 1.5
        + np.random.default_rng(1).normal(0, 5, len(out))
    )
    return out


def test_infer_task():
    assert m.infer_task(pd.Series([0, 1, 0, 1])) == m.BINARY
    assert m.infer_task(pd.Series(["a", "b", "c", "a"])) == m.MULTICLASS
    assert m.infer_task(pd.Series(np.linspace(0, 100, 50))) == m.REGRESSION
    assert m.infer_task(pd.Series([1, 2, 3, 1, 2, 3])) == m.MULTICLASS


def test_leaderboard_beats_baseline_and_is_sorted(clf_df):
    table, task = m.leaderboard(clf_df, "churn", ["age", "spend", "plan"], cv=3)
    assert task == m.BINARY
    assert table["cv_mean"].is_monotonic_decreasing
    assert not table.iloc[0]["model"].startswith("Baseline")
    assert set(table["model"]) >= {"Baseline (most frequent)", "Random Forest"}


def test_regression_leaderboard(reg_df):
    table, task = m.leaderboard(
        reg_df, "revenue", ["age", "spend", "plan"], cv=3, models=["Baseline (mean)", "Ridge"]
    )
    assert task == m.REGRESSION
    assert table.iloc[0]["model"] == "Ridge" and table.iloc[0]["cv_mean"] > 0.9


def test_input_validation(clf_df):
    with pytest.raises(m.ModelingError):
        m.leaderboard(clf_df.head(10), "churn", ["age"])
    with pytest.raises(m.ModelingError):
        m.leaderboard(clf_df, "churn", ["churn"])
    with pytest.raises(m.ModelingError):
        m.leaderboard(clf_df, "churn", [])


def test_tuning_returns_holdout_scored_model(clf_df):
    res = m.tune_model(clf_df, "churn", ["age", "spend", "plan"], "Random Forest", n_iter=4, cv=3)
    assert res.model.task == m.BINARY and res.model.classes == [0, 1]
    assert "accuracy" in res.test_metrics and "roc_auc" in res.test_metrics
    assert res.best_cv_score >= res.results["mean_test_score"].min()
    assert res.results["rank_test_score"].iloc[0] == 1
    with pytest.raises(m.ModelingError):
        m.tune_model(clf_df, "churn", ["age"], "Baseline (most frequent)")


def test_export_roundtrip_and_scoring(clf_df, tmp_path):
    model = m.fit_model(clf_df, "churn", ["age", "spend", "plan"], "Random Forest")
    path = tmp_path / "m.joblib"
    path.write_bytes(m.export_model_bytes(model))
    bundle = joblib.load(path)
    new = clf_df.head(5)
    out, warnings = m.score_frame(model, new)
    assert not warnings and "predicted_churn" in out and "proba_1" in out
    np.testing.assert_array_equal(
        bundle["pipeline"].predict(new[model.features]), out["predicted_churn"]
    )
    # Unknown category + missing column must not crash
    odd = pd.DataFrame({"age": [30], "plan": ["enterprise"]})
    out2, warnings2 = m.score_frame(model, odd)
    assert len(out2) == 1 and warnings2


def test_feature_info_drives_input_widgets(clf_df):
    info = m.describe_features(clf_df[["age", "plan"]])
    assert (
        info["age"]["kind"] == "numeric"
        and info["age"]["min"] <= info["age"]["default"] <= info["age"]["max"]
    )
    assert (
        info["plan"]["kind"] == "categorical" and info["plan"]["default"] in info["plan"]["choices"]
    )


def test_tuning_code_runs(clf_df, tmp_path, monkeypatch):
    csv = tmp_path / "d.csv"
    clf_df.to_csv(csv, index=False)
    monkeypatch.chdir(tmp_path)
    code = m.tuning_code(
        "churn",
        ["age", "spend", "plan"],
        m.BINARY,
        "Random Forest",
        {"model__max_depth": [3, 5]},
        "grid",
        5,
        3,
        "f1_macro",
        filename=str(csv),
    )
    ast.parse(code)
    ns: dict = {}
    exec(compile(code, "<tuning>", "exec"), ns)  # noqa: S102
    assert (tmp_path / "tuned_model.joblib").exists()
    ast.parse(m.scoring_script(m.fit_model(clf_df, "churn", ["age", "spend"], "Random Forest")))


def test_holdout_importance_and_refit(clf_df):
    model = m.fit_model(clf_df, "churn", ["age", "spend", "plan"], "Random Forest")
    imp = m.permutation_importance_table(model)
    assert (
        set(imp["feature"]) == {"age", "spend", "plan"}
        and imp["importance"].is_monotonic_decreasing
    )
    full = m.refit_on_all(model, clf_df)
    assert full.rows_used == len(clf_df) and full.metrics == model.metrics and full.holdout is None
    with pytest.raises(m.ModelingError):
        m.permutation_importance_table(full)


def test_tuning_and_leaderboard_accept_any_catalog_model(clf_df):
    table, task = m.leaderboard(
        clf_df,
        "churn",
        ["age", "spend", "plan"],
        cv=3,
        models=["Baseline (most frequent)", "Naive Bayes", "Extra Trees"],
    )
    assert task == m.BINARY and set(table["model"]) == {
        "Baseline (most frequent)",
        "Naive Bayes",
        "Extra Trees",
    }
    res = m.tune_model(clf_df, "churn", ["age", "spend", "plan"], "Extra Trees", n_iter=3, cv=3)
    assert res.model.estimator_name == "Extra Trees" and set(res.best_params) <= {
        "n_estimators",
        "max_depth",
        "min_samples_leaf",
        "max_features",
    }
    fit = m.fit_model(clf_df, "churn", ["age", "spend", "plan"], "XGBoost")
    assert (
        fit.metrics["accuracy"] > 0.5
        and m.score_frame(fit, clf_df.head(3))[0]["predicted_churn"].notna().all()
    )
    with pytest.raises(m.ModelingError, match="no tunable"):
        m.tune_model(clf_df, "churn", ["age"], "Naive Bayes".replace("Naive Bayes", "Baseline"))
    with pytest.raises(m.ModelingError):
        m.tune_model(clf_df, "churn", ["age"], "Quantum Forest")


@pytest.mark.parametrize("method", ["random", "grid", "halving", "optuna"])
def test_every_search_method_returns_a_scored_model(method, clf_df):
    res = m.tune_model(
        clf_df, "churn", ["age", "spend", "plan"], "Random Forest", method=method, n_iter=6, cv=3
    )
    assert (
        res.model.task == m.BINARY
        and res.best_cv_score > 0.5
        and res.test_metrics["accuracy"] > 0.5
    )
    assert set(res.best_params) <= {"n_estimators", "max_depth", "min_samples_leaf", "max_features"}
    assert res.results["rank_test_score"].iloc[0] == 1 and "mean_test_score" in res.results
    assert (
        res.results["mean_test_score"].iloc[0] == res.results["mean_test_score"].max()
        or method == "halving"
    )


def test_optuna_search_is_reproducible_and_beats_random_on_average(clf_df):
    a = m.tune_model(
        clf_df,
        "churn",
        ["age", "spend", "plan"],
        "Hist Gradient Boosting",
        method="optuna",
        n_iter=8,
        cv=3,
        seed=1,
    )
    b = m.tune_model(
        clf_df,
        "churn",
        ["age", "spend", "plan"],
        "Hist Gradient Boosting",
        method="optuna",
        n_iter=8,
        cv=3,
        seed=1,
    )
    assert a.best_params == b.best_params and a.best_cv_score == pytest.approx(b.best_cv_score)
