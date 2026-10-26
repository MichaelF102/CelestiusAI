"""Ensembles and the classification toolkit."""

import numpy as np
import pandas as pd
import pytest
from src.datascience import classification as cl
from src.datascience import ensembles as en
from src.datascience.modeling import ModelingError


@pytest.fixture(scope="module")
def imb():
    rng = np.random.default_rng(4)
    n = 900
    df = pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "c": rng.normal(size=n),
            "plan": rng.choice(["x", "y"], n),
        }
    )
    logit = 1.8 * df.a - 1.2 * df.b + (df.plan == "x") * 0.6 - 3.2
    df["y"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return df


@pytest.fixture(scope="module")
def reg():
    rng = np.random.default_rng(5)
    n = 400
    df = pd.DataFrame(
        {"a": rng.normal(size=n), "b": rng.normal(size=n), "plan": rng.choice(["x", "y", "z"], n)}
    )
    df["t"] = df.a * 2 + np.sin(df.b * 2) * 3 + (df.plan == "x") + rng.normal(0, 0.4, n)
    return df


# --------------------------------------------------------------------------- ensembles
@pytest.mark.parametrize("kind", ["bagging", "voting_hard", "voting_soft", "stacking"])
def test_classification_ensembles_fit_and_predict(kind, imb):
    X = pd.get_dummies(imb[["a", "b", "c"]])
    ens = en.build_ensemble(
        kind,
        ["Linear / Logistic Regression", "Decision Tree", "Naive Bayes"],
        "binary_classification",
        n_estimators=5,
    )
    ens.fit(X, imb["y"])
    assert set(ens.predict(X)) <= {0, 1}
    if kind != "voting_hard":
        assert np.allclose(ens.predict_proba(X).sum(1), 1)


@pytest.mark.parametrize("kind", ["bagging", "voting_soft", "stacking"])
def test_regression_ensembles(kind, reg):
    X = pd.get_dummies(reg[["a", "b"]])
    ens = en.build_ensemble(
        kind, ["Ridge / L2 Logistic", "Decision Tree"], "regression", n_estimators=4
    )
    assert ens.fit(X, reg["t"]).predict(X).shape == (len(X),)


def test_ensemble_input_validation():
    with pytest.raises(ModelingError, match="Hard voting"):
        en.build_ensemble("voting_hard", ["Ridge / L2 Logistic", "Decision Tree"], "regression")
    with pytest.raises(ModelingError, match="at least two"):
        en.build_ensemble("stacking", ["Decision Tree"], "regression")
    with pytest.raises(ModelingError, match="at least one"):
        en.build_ensemble("bagging", [], "regression")
    with pytest.raises(ModelingError, match="Unknown"):
        en.build_ensemble("magic", ["Decision Tree"], "regression")


def test_study_measures_gain_and_diversity(reg):
    bases = ["Ridge / L2 Logistic", "Decision Tree", "K-Nearest Neighbors"]
    res = en.ensemble_study(
        reg,
        "t",
        ["a", "b", "plan"],
        bases,
        [
            en.EnsembleSpec("Soft vote", "voting_soft", bases),
            en.EnsembleSpec("Stack", "stacking", bases),
            en.EnsembleSpec("Bagged trees", "bagging", ["Decision Tree"], n_estimators=8),
        ],
        cv=3,
    )
    t = res.table.set_index("model")
    assert (
        set(t["type"]) == {"base", "voting_soft", "stacking", "bagging"}
        and res.table["cv_mean"].is_monotonic_decreasing
    )
    best_base = t[t["type"] == "base"]["cv_mean"].max()
    assert res.summary["best_base"] == pytest.approx(best_base)
    assert t.loc["Stack", "gain_vs_best_base"] == pytest.approx(
        t.loc["Stack", "cv_mean"] - best_base
    )
    assert t.loc["Stack", "cv_mean"] > best_base - 0.05  # stacking should not collapse
    assert (
        t.loc["Bagged trees", "cv_mean"] > t.loc["Decision Tree", "cv_mean"] - 0.02
    )  # bagging helps (or at least doesn't hurt) trees
    assert res.diversity.shape == (3, 3) and np.allclose(np.diag(res.diversity), 1)
    assert -1 <= res.summary["mean_error_correlation"] <= 1
    assert "mistakes" in en.diversity_comment(res.summary)
    assert "at least two" in en.diversity_comment({"mean_error_correlation": None})


def test_classification_study_has_disagreement(imb):
    bases = ["Linear / Logistic Regression", "Decision Tree"]
    res = en.ensemble_study(
        imb, "y", ["a", "b", "plan"], bases, [en.EnsembleSpec("Vote", "voting_hard", bases)], cv=3
    )
    d = res.disagreement
    assert (
        d.loc[bases[0], bases[0]] == 0
        and d.loc[bases[0], bases[1]] == pytest.approx(d.loc[bases[1], bases[0]])
        and d.loc[bases[0], bases[1]] > 0
    )


# --------------------------------------------------------------------------- resampling
def test_resampling_strategies_balance_classes():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(110, 3))
    y = np.array([0] * 100 + [1] * 10)
    for strategy, expect in (
        ("oversample", (100, 100)),
        ("undersample", (10, 10)),
        ("smote", (100, 100)),
    ):
        Xr, yr = cl.resample(X, y, strategy)
        assert tuple(np.bincount(yr)) == expect and Xr.shape[1] == 3, strategy
    Xs, ys = cl.resample(X, y, "smote")
    synthetic = Xs[len(X) :]
    minority = X[y == 1]
    assert (
        len(synthetic) == 90
        and synthetic.min() >= minority.min() - 1e-9
        and synthetic.max() <= minority.max() + 1e-9
    )  # inside the minority's range
    assert cl.resample(X, y, "none")[0] is X and cl.resample(X, y, "class_weight")[1] is y
    with pytest.raises(ModelingError):
        cl.resample(X, y, "magic")
    tiny = cl.resample(np.array([[0.0], [1.0], [2.0]]), np.array([0, 0, 1]), "smote")
    assert tuple(np.bincount(tiny[1])) == (2, 2)


def test_resampling_only_touches_training_rows(imb):
    """The resampler lives inside the pipeline, so predict() never sees synthetic rows."""
    X = imb[["a", "b", "c", "plan"]]
    pipe = cl.build_pipeline(X, "Linear / Logistic Regression", "binary_classification", "smote")
    pipe.fit(X.iloc[:600], imb["y"].iloc[:600])
    assert len(pipe.predict(X.iloc[600:700])) == 100
    with pytest.raises(ModelingError, match="class_weight"):
        cl.build_pipeline(X, "K-Nearest Neighbors", "binary_classification", "class_weight")


def test_imbalance_experiment_shows_accuracy_is_misleading(imb):
    assert imb["y"].mean() < 0.2
    table = cl.imbalance_experiment(
        imb, "y", ["a", "b", "c", "plan"], "Random Forest", cv=3
    ).set_index("strategy")
    assert table.loc["none", "accuracy"] > 0.8  # looks great...
    assert (
        table.loc["none", "balanced_accuracy"] < table.loc["none", "accuracy"] - 0.1
    )  # ...but is not
    assert (
        table.loc["class_weight", "balanced_accuracy"]
        >= table.loc["none", "balanced_accuracy"] - 0.02
    )
    assert table.loc["oversample", "f1_minority"] > 0 and not table["roc_auc"].isna().any()
    knn = cl.imbalance_experiment(
        imb, "y", ["a", "b"], "K-Nearest Neighbors", ["none", "class_weight"], cv=3
    ).set_index("strategy")
    assert "class_weight" in knn.loc["class_weight", "note"]


# --------------------------------------------------------------------------- evaluation
@pytest.fixture(scope="module")
def holdout(imb):
    return cl.holdout_fit(imb, "y", ["a", "b", "c", "plan"], "Linear / Logistic Regression")


def test_holdout_evaluation_is_complete_and_consistent(holdout):
    ev = cl.evaluate_holdout(holdout)
    y = holdout.y_test
    assert ev["accuracy"] == pytest.approx((holdout.pred == y).mean())
    assert (
        0.5 < ev["roc_auc"] <= 1
        and 0 < ev["avg_precision"] <= 1
        and 0 <= ev["ks"] <= 1
        and ev["prevalence"] == pytest.approx(y.mean())
    )
    cm = ev["confusion"].to_numpy()
    assert cm.sum() == len(y) and ev["confusion_rates"].sum(axis=1).round(6).eq(1).all()
    rep = ev["report"].set_index("class")
    assert rep["support"].sum() == len(y) and rep.loc["1", "recall"] == pytest.approx(
        cm[1, 1] / cm[1].sum()
    )
    assert ev["roc_curve"]["fpr"].is_monotonic_increasing and ev["roc_curve"].iloc[0].tolist() == [
        0.0,
        0.0,
    ]
    gl = ev["gain_lift"]
    assert (
        gl["cum_gain"].iloc[-1] == pytest.approx(1.0)
        and gl["lift"].iloc[0] > 1 > gl["lift"].iloc[-1]
        and gl["rows"].sum() == len(y)
    )
    assert gl["cum_gain"].is_monotonic_increasing


def test_multiclass_evaluation():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"a": rng.normal(size=300), "b": rng.normal(size=300)})
    df["k"] = pd.cut(df.a + df.b, [-9, -0.7, 0.7, 9], labels=["lo", "mid", "hi"]).astype(str)
    h = cl.holdout_fit(df, "k", ["a", "b"], "Random Forest")
    ev = cl.evaluate_holdout(h)
    assert (
        set(ev["report"]["class"]) == {"lo", "mid", "hi"}
        and ev["macro_f1"] > 0.5
        and ev["roc_auc_ovr"] > 0.7
    )
    assert "roc_auc" not in ev
    with pytest.raises(ModelingError, match="continuous"):
        cl.holdout_fit(df.assign(t=rng.normal(size=300)), "t", ["a"], "Random Forest")


def test_threshold_table_matches_manual_counts(holdout):
    pos = (holdout.y_test == holdout.positive).to_numpy()
    score = holdout.proba[:, holdout.classes.index(holdout.positive)]
    t = cl.threshold_table(pos, score)
    row = t.iloc[(t["threshold"] - 0.3).abs().argmin()]
    pred = score >= row["threshold"]
    assert (
        row["tp"] == (pred & pos).sum()
        and row["fp"] == (pred & ~pos).sum()
        and row["fn"] == (~pred & pos).sum()
    )
    assert row["tp"] + row["fp"] + row["fn"] + row["tn"] == len(pos)
    assert t["recall"].is_monotonic_decreasing and t["threshold"].is_monotonic_increasing


def test_cost_sensitive_threshold_moves_with_costs(holdout):
    pos = (holdout.y_test == holdout.positive).to_numpy()
    score = holdout.proba[:, holdout.classes.index(holdout.positive)]
    cheap_fn = cl.optimal_threshold(pos, score, cost_fp=5, cost_fn=1)
    costly_fn = cl.optimal_threshold(pos, score, cost_fp=1, cost_fn=20)
    assert (
        costly_fn["threshold"] < cheap_fn["threshold"]
    )  # missing a positive is costly -> lower the bar
    assert costly_fn["recall"] >= cheap_fn["recall"] and costly_fn["saving_vs_0.5"] >= 0
    table = costly_fn["table"]
    assert table["cost"].min() == pytest.approx(costly_fn["cost"])
    gain = cl.optimal_threshold(pos, score, cost_fp=1, cost_fn=5, benefit_tp=3)
    assert gain["cost"] < cl.optimal_threshold(pos, score, cost_fp=1, cost_fn=5)["cost"]


def test_calibration_summary_and_recalibration(imb):
    perfect = np.linspace(0.02, 0.98, 500)
    rng = np.random.default_rng(0)
    y = (rng.random(500) < perfect).astype(int)
    summary = cl.calibration_summary(y, perfect)
    assert (
        summary["ece"] < 0.1
        and 0 < summary["brier"] < 0.25
        and list(summary["curve"].columns) == ["mean_predicted", "fraction_positive"]
    )
    overconfident = np.clip((perfect - 0.5) * 3 + 0.5, 0.01, 0.99)
    assert cl.calibration_summary(y, overconfident)["ece"] > summary["ece"]
    h = cl.holdout_fit(imb, "y", ["a", "b", "c"], "Random Forest", test_size=0.5)
    res = cl.calibrate_holdout(h, "sigmoid")
    assert (
        res["method"] == "sigmoid" and res["brier_after"] > 0 and isinstance(res["improved"], bool)
    )
    assert res["brier_after"] <= res["brier_before"] * 1.15
    with pytest.raises(ModelingError):
        cl.calibrate_holdout(
            cl.Holdout(h.pipeline, h.X_test, h.y_test, h.proba, h.pred, [0, 1, 2], None)
        )


def test_multiclass_strategies():
    ovr = cl.multiclass_wrapper("Linear / Logistic Regression", "multiclass_classification", "ovr")
    ovo = cl.multiclass_wrapper("Linear / Logistic Regression", "multiclass_classification", "ovo")
    native = cl.multiclass_wrapper("Random Forest", "multiclass_classification", "native")
    X = np.random.default_rng(0).normal(size=(90, 2))
    y = np.array(list("abc") * 30)
    for est in (ovr, ovo, native):
        assert set(est.fit(X, y).predict(X)) <= set("abc")
