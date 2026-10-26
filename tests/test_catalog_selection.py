"""Model catalog and model-selection engine."""

import numpy as np
import pandas as pd
import pytest
from src.datascience import catalog as cat
from src.datascience import selection as sel
from src.datascience.modeling import BINARY, MULTICLASS, REGRESSION, ModelingError, infer_task


@pytest.fixture(scope="module")
def clf():
    rng = np.random.default_rng(1)
    n = 320
    df = pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "c": rng.integers(0, 5, n),
            "plan": rng.choice(["x", "y", "z"], n),
        }
    )
    df["y"] = (
        (df.a * 1.5 + (df.b > 0.3) * 1.2 + (df.plan == "x") + rng.normal(0, 0.6, n)) > 0.8
    ).astype(int)
    df.loc[::20, "a"] = np.nan
    return df


@pytest.fixture(scope="module")
def reg(clf):
    out = clf.copy()
    out["t"] = (
        out.a.fillna(0) * 3
        + out.b**2 * 2
        + (out.plan == "y") * 2
        + np.random.default_rng(2).normal(0, 0.5, len(out))
    )
    return out


@pytest.fixture(scope="module")
def multi(clf):
    out = clf.copy()
    out["k"] = pd.cut(
        out.a.fillna(0) + out.b, [-9, -0.5, 0.5, 9], labels=["lo", "mid", "hi"]
    ).astype(str)
    return out


# --------------------------------------------------------------------------- catalog
def test_catalog_size_and_cards():
    assert len(cat.CATALOG) >= 20
    for spec in cat.CATALOG:
        assert spec.card.when_to_use and spec.card.strengths and spec.card.weaknesses
        assert 1 <= spec.card.interpretability <= 5 and 1 <= spec.card.speed <= 5
    assert len({s.name for s in cat.CATALOG}) == len(cat.CATALOG)
    assert cat.get("Ridge").name == "Ridge / L2 Logistic" and cat.get("KNN").family == "neighbors"
    with pytest.raises(KeyError):
        cat.get("Quantum Forest")


@pytest.mark.parametrize("task", [REGRESSION, BINARY, MULTICLASS])
def test_every_available_model_builds_fits_and_predicts(task, clf, reg, multi):
    df, target = {REGRESSION: (reg, "t"), BINARY: (clf, "y"), MULTICLASS: (multi, "k")}[task]
    X = pd.get_dummies(df[["a", "b", "c", "plan"]].fillna(0), dtype=float)
    y = df[target]
    for spec in cat.available_models(task):
        est = cat.build(spec.name, task, seed=3)
        est.fit(X, y)
        pred = est.predict(X)
        assert len(pred) == len(X), spec.name
        if task != REGRESSION and hasattr(est, "predict_proba"):
            proba = est.predict_proba(X)
            assert proba.shape == (len(X), y.nunique()) and np.allclose(
                proba.sum(1), 1, atol=1e-6
            ), spec.name


def test_task_support_and_optional_packages():
    names_reg = {s.name for s in cat.available_models(REGRESSION)}
    names_clf = {s.name for s in cat.available_models(BINARY)}
    assert "ElasticNet" in names_reg and "ElasticNet" not in names_clf
    assert "Naive Bayes" in names_clf and "Naive Bayes" not in names_reg
    assert {"XGBoost", "LightGBM"} <= names_reg  # installed in the dev environment
    assert "XGBoost" not in {
        s.name for s in cat.available_models(REGRESSION, include_optional=False)
    }
    assert cat.optional_status()["CatBoost"] is cat.has_package("catboost")
    if not cat.has_package("catboost"):
        with pytest.raises(ImportError, match="catboost"):
            cat.build("CatBoost", REGRESSION)
    with pytest.raises(ValueError):
        cat.build("Naive Bayes", REGRESSION)


def test_param_spaces_only_contain_valid_parameters():
    for task in (REGRESSION, BINARY):
        for name in cat.tunable(task):
            est = cat.build(name, task)
            space = cat.param_space(name, task)
            assert space and set(space) <= set(est.get_params()), name
    assert cat.param_space("Baseline", REGRESSION) == {}
    assert "l1_ratio" in cat.param_space("ElasticNet", REGRESSION)
    assert "C" in cat.param_space("Linear / Logistic Regression", BINARY)
    assert "C" not in cat.param_space("Linear / Logistic Regression", REGRESSION)


def test_display_names_and_complexity():
    spec = cat.get("Linear / Logistic Regression")
    assert (
        cat.display_name(spec, REGRESSION) == "Linear Regression"
        and cat.display_name(spec, BINARY) == "Logistic Regression"
    )
    assert cat.complexity_rank("Decision Tree") < cat.complexity_rank("Neural Network (MLP)")


def test_xgboost_handles_arbitrary_labels(multi):
    X = pd.get_dummies(multi[["a", "b"]].fillna(0))
    est = cat.build("XGBoost", MULTICLASS).fit(X, multi["k"])
    assert set(est.predict(X)) <= {"lo", "mid", "hi"} and list(est.classes_) == ["hi", "lo", "mid"]
    shifted = cat.build("XGBoost", BINARY).fit(X, np.where(multi["k"] == "hi", 7, 3))
    assert set(shifted.predict(X)) <= {3, 7}


# --------------------------------------------------------------------------- profile & recommendations
def test_profile_captures_the_shape_of_the_data(clf, reg):
    p = sel.profile_for_selection(clf, "y", ["a", "b", "c", "plan"])
    assert (p.n_rows, p.n_features, p.n_numeric, p.n_categorical, p.task) == (320, 4, 3, 1, BINARY)
    assert p.n_classes == 2 and p.imbalance_ratio >= 1 and p.missing_pct > 0
    q = sel.profile_for_selection(reg, "t", ["a", "b", "c", "plan"])
    assert (
        q.task == REGRESSION and q.target_skew is not None and q.nonlinear_gain > 0.0
    )  # b**2 is non-linear


def test_recommendations_explain_themselves():
    small = sel.DataProfile(200, 5, 5, 0, 0, 0, 2, 1.0, None, 0.0, BINARY)
    ranked = sel.recommend_models(small)
    top = {r.name for r in ranked[:5]}
    assert top & {"Linear / Logistic Regression", "Ridge / L2 Logistic", "Naive Bayes"}
    assert all(r.reasons for r in ranked) and ranked == sorted(
        ranked, key=lambda r: r.score, reverse=True
    )
    assert (
        next(r for r in ranked if r.name == "XGBoost").score
        < next(r for r in ranked if r.name == "Ridge / L2 Logistic").score
    )
    big = sel.DataProfile(200_000, 20, 20, 0, 0, 0, 2, 1.0, None, 0.08, BINARY)
    ranked_big = {r.name: r.score for r in sel.recommend_models(big)}
    assert ranked_big["Hist Gradient Boosting"] > ranked_big["Support Vector Machine (RBF)"]
    assert ranked_big["LightGBM"] > ranked_big["K-Nearest Neighbors"]
    imbalanced = sel.DataProfile(5000, 10, 10, 0, 0, 0, 2, 20.0, None, 0.02, BINARY)
    assert any(
        "imbalance" in " ".join(r.reasons)
        for r in sel.recommend_models(imbalanced)
        if r.name == "Random Forest"
    )
    assert len(sel.recommend_models(small, top=3)) == 3
    no_opt = {r.name for r in sel.recommend_models(small, include_optional=False)}
    assert "XGBoost" not in no_opt


def test_nonlinear_signal_moves_trees_above_linear():
    nl = sel.DataProfile(5000, 8, 8, 0, 0, 0, None, None, 0.1, 0.25, REGRESSION)
    scores = {r.name: r.score for r in sel.recommend_models(nl)}
    assert scores["Random Forest"] > scores["Linear / Logistic Regression"]
    lin = sel.DataProfile(5000, 8, 8, 0, 0, 0, None, None, 0.1, 0.0, REGRESSION)
    scores2 = {r.name: r.score for r in sel.recommend_models(lin)}
    assert scores2["Linear / Logistic Regression"] > scores2["Random Forest"]


# --------------------------------------------------------------------------- race
MODELS = [
    "Baseline",
    "Linear / Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "Naive Bayes",
]


def test_race_scores_are_comparable_and_flag_overfitting(clf):
    res = sel.race(clf, "y", ["a", "b", "c", "plan"], MODELS, n_splits=4)
    t = res.table
    assert (
        res.task == BINARY and set(t["status"]) == {"ok"} and t["cv_mean"].is_monotonic_decreasing
    )
    assert (
        t.iloc[0]["model"] != "Baseline"
        and t.set_index("model").loc["Baseline", "cv_mean"] < t.iloc[0]["cv_mean"]
    )
    assert (t["folds"] == 4).all() and set(res.fold_scores) == set(MODELS)
    tree = t.set_index("model").loc["Decision Tree"]
    assert (
        tree["gap"] > t.set_index("model").loc["Linear / Logistic Regression", "gap"] - 0.2
    )  # trees overfit more
    assert set(t["overfit"]) <= {"low", "moderate", "high"}
    assert res.n_train_per_fold > res.n_test_per_fold > 0


@pytest.mark.parametrize("strategy", list(sel.CV_STRATEGIES))
def test_every_cv_strategy_runs(strategy, clf):
    groups = pd.Series(np.arange(len(clf)) % 40, index=clf.index) if strategy == "group" else None
    res = sel.race(
        clf,
        "y",
        ["a", "b"],
        ["Linear / Logistic Regression"],
        n_splits=3,
        n_repeats=2,
        cv_strategy=strategy,
        groups=groups,
    )
    assert res.table.iloc[0]["status"] == "ok"
    if strategy.startswith("repeated"):
        assert res.table.iloc[0]["folds"] == 6


def test_group_cv_never_splits_a_group(clf):
    y = clf["y"]
    groups = pd.Series(np.arange(len(clf)) % 40, index=clf.index)
    for tr, te in sel.make_cv("group", y, 4).split(clf, y, groups):
        assert not set(groups.iloc[tr]) & set(groups.iloc[te])
    with pytest.raises(ModelingError, match="group column"):
        sel.race(clf, "y", ["a"], ["Naive Bayes"], cv_strategy="group")
    for tr, te in sel.make_cv("timeseries", y, 4).split(clf):
        assert tr.max() < te.min()


def test_race_handles_regression_multiclass_and_failures(reg, multi):
    r = sel.race(
        reg,
        "t",
        ["a", "b", "plan"],
        [
            "Baseline",
            "Lasso / L1 Logistic",
            "Polynomial (degree 2) + Ridge",
            "ElasticNet",
            "Huber (robust)",
        ],
        n_splits=3,
    )
    assert r.task == REGRESSION and r.table.iloc[0]["cv_mean"] > 0.5 and r.scoring == "r2"
    m = sel.race(
        multi,
        "k",
        ["a", "b"],
        ["Naive Bayes", "K-Nearest Neighbors", "Support Vector Machine (RBF)"],
        n_splits=3,
    )
    assert m.task == MULTICLASS and (m.table["status"] == "ok").all()
    bad = sel.race(
        reg, "t", ["a", "b"], ["Naive Bayes", "Baseline"], n_splits=3
    )  # NB cannot do regression
    assert any(str(s).startswith("failed") for s in bad.table["status"]) and "ok" in set(
        bad.table["status"]
    )
    with pytest.raises(ModelingError):
        sel.race(reg, "t", ["a"], ["Baseline"], scoring="roc_auc")


def test_time_budget_skips_remaining_models(clf):
    res = sel.race(
        clf,
        "y",
        ["a", "b"],
        ["Support Vector Machine (RBF)", "Random Forest", "Naive Bayes"],
        n_splits=3,
        time_budget_s=0.0001,
    )
    assert (res.table["status"] == "skipped (time budget)").sum() >= 1


def test_one_standard_error_rule_prefers_simpler_models():
    table = pd.DataFrame(
        {
            "model": [
                "Neural Network (MLP)",
                "Linear / Logistic Regression",
                "Decision Tree",
                "Baseline",
            ],
            "cv_mean": [0.850, 0.846, 0.700, 0.5],
            "cv_std": [0.02, 0.02, 0.05, 0.0],
            "folds": [5, 5, 5, 5],
            "status": ["ok"] * 4,
        }
    )
    pick, why = sel.one_se_choice(table)
    assert pick == "Linear / Logistic Regression" and "simpler" in why
    far = table.assign(cv_mean=[0.90, 0.80, 0.70, 0.5], cv_std=[0.005] * 4)
    assert sel.one_se_choice(far)[0] == "Neural Network (MLP)"
    with pytest.raises(ModelingError):
        sel.one_se_choice(
            pd.DataFrame(
                {
                    "model": ["x"],
                    "status": ["failed"],
                    "cv_mean": [np.nan],
                    "cv_std": [np.nan],
                    "folds": [0],
                }
            )
        )


# --------------------------------------------------------------------------- nested CV & comparisons
def test_nested_cv_is_not_more_optimistic_than_the_inner_search(clf):
    res = sel.nested_cv(
        clf, "y", ["a", "b", "c", "plan"], "Random Forest", outer=3, inner=3, n_iter=3
    )
    assert len(res.outer_scores) == 3 and len(res.best_params) == 3 and 0.5 < res.mean <= 1
    assert res.non_nested_best == pytest.approx(res.mean + res.optimism)
    with pytest.raises(ModelingError, match="no tunable"):
        sel.nested_cv(clf, "y", ["a"], "Baseline", outer=3)


def test_model_comparison_statistics():
    rng = np.random.default_rng(0)
    base = rng.normal(0.8, 0.02, 10)
    better = base + 0.03 + rng.normal(0, 0.003, 10)
    clear = sel.compare_models(better, base, n_train=800, n_test=200)
    assert clear["verdict"] == "A is better" and clear["p_value"] < 0.05 and clear["mean_diff"] > 0
    flipped = sel.compare_models(base, better, 800, 200)
    assert flipped["verdict"] == "B is better"
    noise = sel.compare_models(base, base + rng.normal(0, 0.02, 10), 800, 200)
    assert noise["verdict"] == "no reliable difference"
    same = sel.compare_models(base, base, 800, 200)
    assert same["p_value"] == 1.0 and same["wilcoxon_p"] is None
    # the correction makes the test more conservative than a plain paired t-test
    from scipy import stats

    d = better - base + rng.normal(0, 0.02, 10)
    assert (
        sel.compare_models(d + base, base, 800, 200)["p_value"]
        >= stats.ttest_rel(d + base, base).pvalue
    )
    with pytest.raises(ModelingError):
        sel.compare_models(np.array([1.0, 2.0]), np.array([1.0, 2.0]), 10, 5)


def test_pairwise_matrix(clf):
    res = sel.race(
        clf, "y", ["a", "b", "plan"], ["Baseline", "Random Forest", "Naive Bayes"], n_splits=5
    )
    m = sel.pairwise_table(res)
    assert m.shape == (3, 3) and m.isna().to_numpy().diagonal().all()
    assert m.loc["Random Forest", "Baseline"] < 0.2


# --------------------------------------------------------------------------- curves
def test_learning_and_validation_curves(clf):
    lc = sel.learning_curve_table(
        clf, "y", ["a", "b", "plan"], "Decision Tree", sizes=(0.3, 0.6, 1.0), cv=3
    )
    assert (
        len(lc) == 3
        and lc["train_size"].is_monotonic_increasing
        and lc["train_mean"].iloc[-1] >= lc["val_mean"].iloc[-1]
    )
    vc = sel.validation_curve_table(
        clf, "y", ["a", "b", "plan"], "Decision Tree", "max_depth", [1, 3, 8, None], cv=3
    )
    assert list(vc["value"]) == ["1", "3", "8", "None"]
    assert (
        vc["train_mean"].iloc[-1] >= vc["train_mean"].iloc[0]
    )  # deeper trees fit training data better
    assert sel.split_summary(clf, "y")["task"] == infer_task(clf["y"])
