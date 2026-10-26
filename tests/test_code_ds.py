"""Every data-scientist code snippet must run on real data and agree with the app's engine."""

import ast

import numpy as np
import pandas as pd
import pytest
from src.core import code_ds as cd
from src.core.code_options import CodeOptions


@pytest.fixture(autouse=True)
def _no_show(monkeypatch):
    import plotly.graph_objects as go

    monkeypatch.setattr(go.Figure, "show", lambda self, *a, **k: None)


def run(code: str) -> dict:
    ast.parse(code)
    ns: dict = {"__name__": "snippet"}
    exec(compile(code, "<snippet>", "exec"), ns)  # noqa: S102 - generated, trusted
    return ns


@pytest.fixture(scope="module")
def csvs(tmp_path_factory):
    d = tmp_path_factory.mktemp("ds")
    rng = np.random.default_rng(0)
    n = 500
    base = pd.DataFrame(
        {
            "age": rng.integers(20, 70, n),
            "spend": rng.gamma(2, 50, n),
            "plan": rng.choice(["a", "b", "c"], n),
            "region": rng.choice(["N", "S"], n),
        }
    )
    base["churn"] = (
        (
            (base.age - 45) * 0.05
            + (base.spend - 100) * 0.015
            + (base.plan == "a") * 0.8
            + rng.normal(0, 1, n)
        )
        > 0.3
    ).astype(int)
    base["rev"] = base.age * 2 + base.spend + rng.normal(0, 6, n)
    base["visits"] = rng.poisson(np.exp(0.5 + 0.01 * base.age))
    base.to_csv(d / "base.csv", index=False)
    idx = pd.date_range("2022-01-01", periods=300, freq="D")
    ts = pd.DataFrame(
        {
            "day": idx.astype(str),
            "sales": 100
            + 0.1 * np.arange(300)
            + 10 * np.sin(2 * np.pi * np.arange(300) / 7)
            + rng.normal(0, 1.5, 300),
        }
    )
    ts.to_csv(d / "ts.csv", index=False)
    rows = [
        {
            "unit": u,
            "period": p,
            "group": "T" if u < 40 else "C",
            "y": 50 + p * 0.5 + (4 if (u < 40 and p >= 3) else 0) + rng.normal(0, 1),
        }
        for u in range(80)
        for p in range(6)
    ]
    pd.DataFrame(rows).to_csv(d / "did.csv", index=False)
    obs_age = rng.normal(40, 10, 1500)
    t = (rng.random(1500) < 1 / (1 + np.exp(-0.08 * (obs_age - 40)))).astype(int)
    pd.DataFrame(
        {
            "t": t,
            "y": 2 * t + 0.15 * (obs_age - 40) + rng.normal(0, 1, 1500),
            "age": obs_age,
            "z": rng.normal(size=1500),
        }
    ).to_csv(d / "obs.csv", index=False)
    up_age = rng.uniform(18, 70, 3000)
    tt = rng.integers(0, 2, 3000)
    pd.DataFrame(
        {
            "age": up_age,
            "loyal": rng.integers(0, 2, 3000),
            "t": tt,
            "buy": (rng.random(3000) < 0.1 + 0.2 * (up_age < 35) * tt).astype(int),
        }
    ).to_csv(d / "up.csv", index=False)
    x = rng.normal(size=500)
    te, tc = rng.exponential(1 / (0.05 * np.exp(0.7 * x))), rng.uniform(5, 60, 500)
    pd.DataFrame(
        {
            "time": np.minimum(te, tc),
            "event": (te <= tc).astype(int),
            "x": x,
            "plan": rng.choice(["a", "b"], 500),
        }
    ).to_csv(d / "surv.csv", index=False)
    texts = [
        "great product works",
        "love it excellent",
        "terrible broke fast",
        "awful waste money",
        "arrived on tuesday blue box",
    ] * 40
    labels = ["pos", "pos", "neg", "neg", "neu"] * 40
    pd.DataFrame({"text": texts, "label": labels}).to_csv(d / "text.csv", index=False)
    return {k: str(d / f"{k}.csv") for k in ("base", "ts", "did", "obs", "up", "surv", "text")}


FEATS = ["age", "spend", "plan", "region"]


@pytest.mark.parametrize(
    "opts",
    [CodeOptions(), CodeOptions(style="function", checks=True)],
    ids=["notebook", "function+checks"],
)
def test_selection_and_ensemble_snippets(csvs, opts):
    L = {"filename": csvs["base"], "options": opts}
    main = opts.style == "function"
    ns = run(
        cd.race_snippet(
            ["Baseline", "Naive Bayes", "Random Forest"],
            "churn",
            FEATS,
            "binary_classification",
            cv=3,
            **L,
        )
    )
    if not main:
        assert ns["board"].iloc[0]["model"] != "Baseline" and "overfit_gap" in ns["board"]
    run(cd.race_snippet(["Baseline", "Ridge / L2 Logistic"], "rev", FEATS, "regression", cv=3, **L))
    run(
        cd.nested_cv_snippet(
            "Random Forest",
            "churn",
            FEATS,
            "binary_classification",
            outer=3,
            inner=2,
            n_iter=2,
            **L,
        )
    )
    for kind in ("bagging", "voting_hard", "voting_soft", "stacking"):
        run(
            cd.ensemble_snippet(
                kind,
                ["Linear / Logistic Regression", "Decision Tree", "Naive Bayes"],
                "churn",
                FEATS,
                "binary_classification",
                cv=3,
                **L,
            )
        )
    run(
        cd.ensemble_snippet(
            "stacking",
            ["Ridge / L2 Logistic", "Decision Tree"],
            "rev",
            FEATS,
            "regression",
            cv=3,
            **L,
        )
    )


def test_race_snippet_scores_equal_the_engine(csvs):
    from src.datascience import selection as sel

    df = pd.read_csv(csvs["base"])
    ns = run(
        cd.race_snippet(
            ["Random Forest", "Naive Bayes"],
            "churn",
            FEATS,
            "binary_classification",
            cv=4,
            filename=csvs["base"],
        )
    )
    engine = sel.race(
        df, "churn", FEATS, ["Random Forest", "Naive Bayes"], n_splits=4
    ).table.set_index("model")
    code = ns["board"].set_index("model")
    for m in ("Random Forest", "Naive Bayes"):
        assert code.loc[m, "cv_mean"] == pytest.approx(engine.loc[m, "cv_mean"], abs=1e-9)


@pytest.mark.parametrize("strategy", ["none", "class_weight", "oversample", "undersample"])
def test_classification_snippet(strategy, csvs):
    ns = run(
        cd.classification_snippet(
            "Random Forest", "churn", FEATS, strategy, cost_fp=1, cost_fn=8, filename=csvs["base"]
        )
    )
    assert 0.2 < ns["score"].mean() < 0.9 and len(ns["thresholds"]) == 99
    assert ns["best_t"]["threshold"] < 0.5  # missing a churner costs more than a false alarm


def test_classification_snippet_threshold_matches_engine(csvs):
    from src.datascience import classification as cl

    df = pd.read_csv(csvs["base"])
    ns = run(
        cd.classification_snippet(
            "Logistic Regression".replace("Logistic Regression", "Linear / Logistic Regression"),
            "churn",
            FEATS,
            "none",
            cost_fp=1,
            cost_fn=5,
            filename=csvs["base"],
        )
    )
    h = cl.holdout_fit(df, "churn", FEATS, "Linear / Logistic Regression", test_size=0.2)
    pos = (h.y_test == h.positive).to_numpy()
    engine = cl.optimal_threshold(pos, h.proba[:, -1], 1, 5)
    assert ns["best_t"]["threshold"] == pytest.approx(engine["threshold"], abs=0.011)


def test_regression_family_snippets(csvs):
    L = {"filename": csvs["base"]}
    ns = run(cd.regression_diagnostics_snippet("rev", ["age", "spend", "plan"], **L))
    from src.datascience import regression as rg

    d = rg.ols_diagnostics(pd.read_csv(csvs["base"]), "rev", ["age", "spend", "plan"])
    assert ns["model"].rsquared == pytest.approx(d.r2) and ns["diag"][
        "cooks_d"
    ].max() == pytest.approx(d.cooks.max())
    for fam, target in (
        ("poisson", "visits"),
        ("gaussian", "rev"),
        ("gamma", "spend"),
        ("tweedie", "spend"),
        ("binomial", "churn"),
    ):
        glm = run(cd.glm_snippet(target, ["age", "plan"], fam, **L))["glm"]
        engine = rg.glm_fit(pd.read_csv(csvs["base"]), target, ["age", "plan"], fam)
        assert glm.params["age"] == pytest.approx(
            engine.coefficients.loc["age", "coef"], abs=1e-6
        ), fam
    q = run(cd.quantile_snippet("rev", ["age", "spend"], **L))
    assert abs(q["fits"][0.5].params["age"] - 2) < 0.3
    for kind in ("lasso", "ridge"):
        assert (
            run(cd.regularization_snippet("rev", ["age", "spend", "plan"], kind, **L))["coefs"]
            .abs()
            .max()
            > 0
        )


@pytest.mark.parametrize("task,target", [("binary_classification", "churn"), ("regression", "rev")])
def test_feature_selection_and_explainer_snippets(task, target, csvs):
    L = {"filename": csvs["base"]}
    ns = run(cd.feature_selection_snippet(target, FEATS, task, **L))
    assert len(ns["scores"]) > 4
    ex = run(cd.explainer_snippet("Random Forest", target, FEATS, task, feature="age", **L))
    assert "imp" in ex and np.isfinite(ex["imp"].importances_mean).all()


def test_fairness_and_drift_snippets(csvs, tmp_path, monkeypatch):
    ns = run(
        cd.fairness_snippet(
            "region",
            "churn",
            ["age", "spend", "plan", "region"],
            "Random Forest",
            filename=csvs["base"],
        )
    )
    assert set(ns["report"]["group"]) == {"N", "S"} and ns["report"]["n"].sum() == len(ns["X_test"])
    df = pd.read_csv(csvs["base"])
    (tmp_path / "new_data.csv").write_text(df.assign(age=df.age + 30).to_csv(index=False))
    monkeypatch.chdir(tmp_path)
    drift = run(cd.drift_snippet(["age", "spend", "plan"], filename=csvs["base"]))[
        "drift"
    ].set_index("feature")
    from src.datascience import explain as ex

    assert drift.loc["age", "status"] == "major shift" and drift.loc["spend", "status"] == "stable"
    assert drift.loc["age", "psi"] == pytest.approx(ex.psi_numeric(df.age, df.age + 30), abs=1e-9)


@pytest.mark.parametrize("strategy", ["direct", "recursive"])
def test_time_series_ml_snippet(strategy, csvs):
    ns = run(
        cd.tsml_snippet("day", "sales", "Ridge / L2 Logistic", 7, strategy, filename=csvs["ts"])
    )
    pred = np.asarray(ns["preds"])
    assert (
        len(pred) == 7
        and np.abs(ns["actual"] - pred).mean() < np.abs(ns["actual"] - ns["naive"]).mean()
    )


def test_causal_snippets_recover_known_effects(csvs):
    d = run(cd.did_snippet("y", "group", "period", 3, "T", filename=csvs["did"]))
    assert abs(d["fit"].params["treated:post"] - 4.0) < 0.6
    m = run(cd.matching_snippet("t", "y", ["age", "z"], 1, filename=csvs["obs"]))
    assert abs(m["diff"].mean() - 2.0) < 0.5
    u = run(cd.uplift_snippet("t", "buy", ["age", "loyal"], 1, filename=csvs["up"]))
    assert u["qini"][-1] > 0


def test_survival_and_text_snippets(csvs):
    s = run(cd.survival_snippet("time", "event", "plan", ["x", "plan"], filename=csvs["surv"]))
    assert "res" in s and 1.7 < float(np.exp(s["res"].params[0])) < 2.5
    run(cd.survival_snippet("time", "event", filename=csvs["surv"]))
    for model in ("Logistic Regression", "Naive Bayes", "Linear SVM", "SGD (linear)"):
        ns = run(cd.text_snippet("text", "label", model, filename=csvs["text"]))
        assert ns["res"]["test_accuracy"].mean() > 0.9, model
