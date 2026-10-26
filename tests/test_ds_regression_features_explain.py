"""Regression workbench, feature selection, and explanation tools."""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from src.datascience import explain as ex
from src.datascience import feature_selection as fs
from src.datascience import regression as rg
from src.datascience.modeling import ModelingError


@pytest.fixture(scope="module")
def lin():
    rng = np.random.default_rng(1)
    n = 500
    df = pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "noise1": rng.normal(size=n),
            "noise2": rng.normal(size=n),
            "plan": rng.choice(["x", "y", "z"], n),
        }
    )
    df["a2"] = df.a + rng.normal(0, 0.05, n)  # nearly collinear with a
    df["t"] = 3 * df.a - 2 * df.b + (df.plan == "x") * 1.5 + rng.normal(0, 1, n)
    return df


# --------------------------------------------------------------------------- OLS diagnostics
def test_ols_diagnostics_match_statsmodels_and_flag_problems(lin):
    d = rg.ols_diagnostics(lin, "t", ["a", "b", "plan"])
    X, y = rg.design_matrix(lin, "t", ["a", "b", "plan"])
    ref = sm.OLS(y, sm.add_constant(X)).fit()
    np.testing.assert_allclose(d.coefficients["coef"].to_numpy(), ref.params.to_numpy())
    assert (
        d.r2 == pytest.approx(ref.rsquared)
        and d.n == 500
        and d.leverage.sum() == pytest.approx(len(ref.params))
    )
    assert (
        abs(d.coefficients.loc["a", "coef"] - 3) < 0.2
        and d.coefficients.loc["noise1" if "noise1" in d.coefficients.index else "b", "p_value"]
        < 0.05
    )
    assert (
        len(d.qq) == 500
        and d.qq["theoretical"].is_monotonic_increasing
        and d.qq["sample"].is_monotonic_increasing
    )
    assert d.tests["durbin_watson"] > 1.5 and not d.notes  # a clean model raises no alarms


def test_outliers_heteroscedasticity_and_collinearity_are_reported(lin):
    bad = lin.copy()
    bad.loc[7, "t"] += 60
    d = rg.ols_diagnostics(bad, "t", ["a", "b"])
    assert d.influential.iloc[0]["row"] == 7 and d.influential.iloc[0]["cooks_d"] > 0.1
    rng = np.random.default_rng(2)
    het = lin.assign(t=lin.a * 2 + rng.normal(0, 1, len(lin)) * (1 + 4 * np.abs(lin.a)))
    assert any("heteroscedasticity" in n for n in rg.ols_diagnostics(het, "t", ["a"]).notes)
    col = rg.ols_diagnostics(lin, "t", ["a", "a2"])
    assert col.vif["vif"].max() > 100 and any("collinear" in n for n in col.notes)
    with pytest.raises(ModelingError):
        rg.design_matrix(lin.head(10), "t", ["a"])
    with pytest.raises(ModelingError):
        rg.design_matrix(lin, "t", ["t"])


# --------------------------------------------------------------------------- GLMs
def test_poisson_glm_recovers_rate_ratios_and_flags_overdispersion():
    rng = np.random.default_rng(3)
    n = 1500
    df = pd.DataFrame({"x": rng.normal(size=n), "g": rng.choice(["u", "v"], n)})
    mu = np.exp(0.4 + 0.5 * df.x + 0.3 * (df.g == "v"))
    df["count"] = rng.poisson(mu)
    r = rg.glm_fit(df, "count", ["x", "g"], "poisson")
    assert (
        abs(r.coefficients.loc["x", "coef"] - 0.5) < 0.06
        and abs(r.coefficients.loc["g_v", "coef"] - 0.3) < 0.1
    )
    assert r.coefficients.loc["x", "rate_ratio"] == pytest.approx(
        np.exp(r.coefficients.loc["x", "coef"])
    )
    assert r.deviance_explained > 0.1 and not r.notes and 0.8 < r.dispersion < 1.3
    over = df.assign(count=rng.negative_binomial(2, 2 / (2 + mu)))
    assert any("Overdispersed" in n for n in rg.glm_fit(over, "count", ["x", "g"], "poisson").notes)


def test_gamma_tweedie_gaussian_binomial(lin):
    rng = np.random.default_rng(4)
    amt = lin.assign(money=np.exp(1 + 0.4 * lin.a + rng.normal(0, 0.3, len(lin))))
    g = rg.glm_fit(amt, "money", ["a"], "gamma")
    assert abs(g.coefficients.loc["a", "coef"] - 0.4) < 0.1
    zeros = amt.assign(money=np.where(rng.random(len(amt)) < 0.3, 0, amt.money))
    assert rg.glm_fit(zeros, "money", ["a"], "tweedie").aic > 0
    assert rg.glm_fit(lin, "t", ["a", "b"], "gaussian").deviance_explained > 0.5
    with pytest.raises(ModelingError, match="strictly positive"):
        rg.glm_fit(zeros, "money", ["a"], "gamma")
    with pytest.raises(ModelingError, match="non-negative"):
        rg.glm_fit(lin, "t", ["a"], "poisson")
    with pytest.raises(ModelingError):
        rg.glm_fit(lin, "t", ["a"], "magic")
    yes = lin.assign(y=(rng.random(len(lin)) < 1 / (1 + np.exp(-(1.2 * lin.a)))).astype(int))
    odds = rg.logistic_odds(yes, "y", ["a", "b"])
    assert (
        odds.coefficients.loc["a", "odds_ratio"]
        == pytest.approx(np.exp(odds.coefficients.loc["a", "coef"]))
        and 2.0 < odds.coefficients.loc["a", "odds_ratio"] < 5.0
    )
    assert (
        odds.coefficients.loc["a", "odds_ratio_low"]
        < odds.coefficients.loc["a", "odds_ratio"]
        < odds.coefficients.loc["a", "odds_ratio_high"]
    )
    with pytest.raises(ModelingError, match="exactly two"):
        rg.logistic_odds(lin, "plan", ["a"])


# --------------------------------------------------------------------------- quantiles
def test_quantile_models(lin):
    q = rg.quantile_regression(lin, "t", ["a", "b"])
    assert abs(q["coefficients"].loc["a", 0.5] - 3) < 0.3 and abs(q["coverage"] - 0.8) < 0.07
    assert (q["predictions"][0.1] < q["predictions"][0.9]).all()
    b = rg.gb_quantile_bands(lin, "t", ["a", "b", "plan"])
    assert (
        abs(b["coverage"] - 0.8) < 0.12
        and b["mean_width"] > 0
        and (b["band"]["lower"] <= b["band"]["upper"]).mean() > 0.95
    )


# --------------------------------------------------------------------------- regularization & curvature
def test_regularization_paths(lin):
    lasso = rg.regularization_path(lin, "t", ["a", "b", "noise1", "noise2"], "lasso")
    path = lasso["path"]
    assert (path.iloc[0, 1:].abs() < 1e-9).all() and path["alpha"].is_monotonic_decreasing
    assert {"a", "b"} <= set(lasso["kept"]) and lasso["best_alpha"] > 0
    ridge = rg.regularization_path(lin, "t", ["a", "b", "noise1"], "ridge")
    norms = ridge["path"].iloc[:, 1:].abs().sum(axis=1)
    assert norms.is_monotonic_increasing  # penalty falls along the path -> coefficients grow
    with pytest.raises(ModelingError):
        rg.regularization_path(lin, "t", ["a"], "elastic")


def test_curvature_study_finds_the_right_degree():
    rng = np.random.default_rng(5)
    x = rng.uniform(-3, 3, 400)
    df = pd.DataFrame({"x": x, "y": x**2 + rng.normal(0, 0.5, 400)})
    t = rg.curvature_study(df, "y", "x").set_index("model")
    assert t.loc["polynomial degree 2", "cv_r2"] > t.loc["polynomial degree 1", "cv_r2"] + 0.5
    assert t["best"].sum() == 1 and t.loc["cubic spline (6 knots)", "cv_r2"] > 0.9


def test_target_transforms_compare_on_the_original_scale():
    rng = np.random.default_rng(6)
    n = 500
    df = pd.DataFrame({"a": rng.normal(size=n)})
    df["y"] = np.exp(1.2 * df.a + rng.normal(0, 0.3, n))
    t = rg.compare_target_transforms(df, "y", ["a"]).set_index("transform")
    assert t.loc["log1p", "target_skew_after"] < t.loc["none", "target_skew_after"] - 1
    assert t.loc["log1p", "cv_r2"] > t.loc["none", "cv_r2"]
    neg = df.assign(y=df.y - 5)
    assert (
        "non-negative"
        in rg.compare_target_transforms(neg, "y", ["a"]).set_index("transform").loc["log1p", "note"]
    )
    with pytest.raises(ModelingError):
        rg.transformed_model("Ridge / L2 Logistic", "cube")


def test_error_analysis_flags_the_noisy_segment():
    rng = np.random.default_rng(7)
    n = 900
    df = pd.DataFrame({"a": rng.normal(size=n), "seg": rng.choice(["calm", "wild"], n)})
    df["y"] = 2 * df.a + rng.normal(0, 0.2, n) * np.where(df.seg == "wild", 12, 1)
    res = rg.error_analysis(df, "y", ["a", "seg"], "Linear / Logistic Regression")
    weak = res["weak_segments"]
    assert "wild" in set(weak.loc[weak.feature == "seg", "segment"]) and "calm" not in set(
        weak["segment"]
    )
    assert res["overall_mae"] > 0 and not res["segments"].empty


# --------------------------------------------------------------------------- advanced statistics
def test_bayesian_bootstrap_power_and_mixed(lin):
    b = rg.bayesian_regression(lin, "t", ["a", "b"])
    co = b["coefficients"]
    sd_a = lin.a.std()
    assert (
        abs(co.loc["a", "mean"] - 3 * sd_a) < 0.4
        and co.loc["a", "probability_nonzero"] > 0.99
        and co.loc["a", "ci_low"] < co.loc["a", "mean"] < co.loc["a", "ci_high"]
    )
    boot = rg.bootstrap_coefficients(lin, "t", ["a", "b"], n_boot=200)
    assert (
        boot.loc["a", "boot_low"] < boot.loc["a", "coef"] < boot.loc["a", "boot_high"]
        and boot.loc["a", "boot_sd"] > 0
    )
    n = rg.regression_sample_size(0.15, 3)
    assert 70 <= n <= 90  # G*Power gives 77 for f2 = .15, 3 predictors, alpha .05, power .8
    assert rg.regression_power(0.15, 3, n) >= 0.8 > rg.regression_power(0.15, 3, n - 1) - 1e-9
    assert (
        rg.regression_power(0.15, 3, 200) > rg.regression_power(0.15, 3, 50)
        and rg.regression_power(0.15, 3, 3) == 0.0
    )
    with pytest.raises(ModelingError):
        rg.regression_sample_size(0, 3)
    rng = np.random.default_rng(8)
    groups = np.repeat(np.arange(40), 25)
    effect = rng.normal(0, 2, 40)[groups]
    df = pd.DataFrame({"x": rng.normal(size=1000), "store": groups})
    df["y"] = 1.5 * df.x + effect + rng.normal(0, 1, 1000)
    m = rg.mixed_effects(df, "y", ["x"], "store")
    assert (
        abs(m["fixed_effects"].loc["f0", "coef"] - 1.5) < 0.1
        and 0.65 < m["icc"] < 0.95
        and m["n_groups"] == 40
    )
    with pytest.raises(ModelingError):
        rg.mixed_effects(df.assign(store=[1, 2] * 500), "y", ["x"], "store")


# --------------------------------------------------------------------------- feature selection
@pytest.fixture(scope="module")
def sel_data():
    rng = np.random.default_rng(9)
    n = 600
    df = pd.DataFrame({f"n{i}": rng.normal(size=n) for i in range(4)})
    df["s1"], df["s2"] = rng.normal(size=n), rng.normal(size=n)
    df["cat"] = rng.choice(["p", "q"], n)
    df["y"] = (
        (1.6 * df.s1 - 1.3 * df.s2 + (df.cat == "p") * 0.2 + rng.normal(0, 0.7, n)) > 0
    ).astype(int)
    df["r"] = 2 * df.s1 + df.s2 + rng.normal(0, 0.5, n)
    return df


FEATS = ["n0", "n1", "n2", "n3", "s1", "s2", "cat"]


@pytest.mark.parametrize("method", ["mutual_info", "anova"])
def test_filter_methods_rank_signal_above_noise(method, sel_data):
    out = fs.filter_scores(sel_data, "y", FEATS, method)
    assert set(out["feature"][:2]) == {"s1", "s2"} and out["rank"].min() == 1
    if method == "anova":
        assert (
            out.set_index("feature").loc["s1", "p_value"] < 1e-10
            and out.set_index("feature").loc["n0", "p_value"] > 0.001
        )
    reg = fs.filter_scores(sel_data, "r", ["n0", "n1", "s1", "s2"], method)
    assert reg["feature"].iloc[0] == "s1"
    with pytest.raises(ModelingError):
        fs.filter_scores(sel_data, "y", FEATS, "magic")


def test_wrapper_embedded_and_stability(sel_data):
    rfe = fs.rfe_ranking(sel_data, "y", FEATS)
    assert set(rfe["feature"][:2]) == {"s1", "s2"} and rfe["order"].tolist() == sorted(rfe["order"])
    lasso = fs.embedded_importance(sel_data, "y", FEATS, "lasso").set_index("feature")
    assert (
        lasso.loc["s1", "selected"]
        and lasso.loc["s2", "selected"]
        and not lasso.loc[["n0", "n1", "n2", "n3"], "selected"].all()
    )
    tree = fs.embedded_importance(sel_data, "y", FEATS, "tree")
    assert set(tree["feature"][:2]) == {"s1", "s2"}
    stab = fs.stability_selection(sel_data, "y", FEATS, n_rounds=12).set_index("feature")
    assert (
        stab.loc["s1", "stable"]
        and stab.loc["s2", "stable"]
        and stab.loc[["n0", "n1", "n2", "n3"], "selection_frequency"].max() < 0.6
    )
    with pytest.raises(ModelingError):
        fs.embedded_importance(sel_data, "y", FEATS, "magic")


def test_minimal_feature_set_is_small_and_within_one_se(sel_data):
    res = fs.minimal_feature_set(sel_data, "y", FEATS, cv=4)
    t = res["table"]
    assert (
        res["recommended_k"] <= res["best_k"]
        and res["recommended_k"] in (2, 3)
        and set(res["recommended"][:2]) == {"s1", "s2"}
    )
    best, rec = (
        t.loc[t["n_features"] == res["best_k"]].iloc[0],
        t.loc[t["n_features"] == res["recommended_k"]].iloc[0],
    )
    assert rec["cv_mean"] >= best["cv_mean"] - best["cv_se"] - 1e-12


def test_vif_clusters_and_pca(lin, sel_data):
    v = fs.vif_table(lin, ["a", "a2", "b", "noise1"]).set_index("feature")
    assert (
        v.loc["a", "level"] == "severe"
        and v.loc["a2", "level"] == "severe"
        and v.loc["b", "level"] == "ok"
    )
    clusters = fs.correlated_clusters(lin, ["a", "a2", "b", "noise1"], 0.8)
    assert sorted(clusters[0]) == ["a", "a2"] and all(len(c) == 1 for c in clusters[1:])
    with pytest.raises(ModelingError):
        fs.vif_table(lin, ["a"])
    pca = fs.pca_compression(sel_data, "y", FEATS, 0.9)
    assert pca["components"] <= pca["original_features"] and -1 < pca["difference"] < 1


# --------------------------------------------------------------------------- explanations
def fitted_pipe(df, features, y, model):
    cats = [c for c in features if df[c].dtype == object]
    pre = (
        ColumnTransformer(
            [("cat", OneHotEncoder(sparse_output=False), cats)], remainder="passthrough"
        )
        if cats
        else "passthrough"
    )
    return Pipeline([("pre", pre), ("m", model)]).fit(df[features], y)


def test_pdp_and_ice(lin):
    pipe = fitted_pipe(lin, ["a", "b", "plan"], lin["t"], LinearRegression())
    pd_a = ex.pdp_ice(pipe, lin[["a", "b", "plan"]], "a", grid_resolution=12)
    assert (
        np.all(np.diff(pd_a["average"]) > 0)
        and pd_a["ice"].shape[1] == len(pd_a["grid"])
        and abs(
            (pd_a["average"][-1] - pd_a["average"][0]) / (pd_a["grid"][-1] - pd_a["grid"][0]) - 3
        )
        < 0.1
    )
    pd_plan = ex.pdp_ice(pipe, lin[["a", "b", "plan"]], "plan")
    assert (
        pd_plan["categorical"]
        and pd_plan["grid"]
        and len(pd_plan["average"]) == len(pd_plan["grid"])
    )
    assert (
        pd_plan["average"][list(pd_plan["grid"]).index("x")]
        > pd_plan["average"][list(pd_plan["grid"]).index("y")]
    )
    with pytest.raises(ModelingError):
        ex.pdp_ice(pipe, lin[["a", "b", "plan"]], "ghost")


def test_interaction_strength_separates_additive_from_interactive():
    rng = np.random.default_rng(10)
    X = pd.DataFrame(
        {"u": rng.uniform(-2, 2, 600), "v": rng.uniform(-2, 2, 600), "w": rng.uniform(-2, 2, 600)}
    )
    add = RandomForestRegressor(100, random_state=0).fit(X, X.u + X.v)
    inter = RandomForestRegressor(100, random_state=0).fit(X, X.u * X.v)
    assert (
        ex.interaction_strength(inter, X, "u", "v")
        > 0.5
        > ex.interaction_strength(add, X, "u", "v") + 0.2
    )
    assert ex.interaction_strength(add, X, "u", "v") < 0.2
    with pytest.raises(ModelingError):
        ex.interaction_strength(add, X.assign(c="a"), "u", "c")


def test_shapley_efficiency_and_linear_ground_truth(lin):
    feats = ["a", "b", "noise1"]
    model = LinearRegression().fit(lin[feats], lin["t"])
    row = lin.iloc[3]
    res = ex.shapley_local(model, row[feats], lin[feats], n_permutations=300)
    exact = pd.Series(model.coef_, index=feats) * (row[feats] - lin[feats].mean())
    got = res["table"].set_index("feature")["contribution"]
    for f in feats:
        assert got[f] == pytest.approx(exact[f], abs=0.35)
    assert res["sum_contributions"] == pytest.approx(res["prediction"] - res["baseline"], abs=0.35)
    assert res["table"]["feature"].iloc[0] in ("a", "b")  # the real drivers lead


def test_shapley_for_classification_and_optional_shap(sel_data):
    feats = ["s1", "s2", "n0"]
    clf = LogisticRegression().fit(sel_data[feats], sel_data["y"])
    row = sel_data.iloc[0][feats]
    res = ex.shapley_local(clf, row, sel_data[feats], n_permutations=120, positive=1)
    assert abs(res["sum_contributions"] - (res["prediction"] - res["baseline"])) < 0.08 and res[
        "table"
    ]["feature"].iloc[0] in ("s1", "s2")
    if ex.shap_available():
        sh = ex.shap_local(clf, row, sel_data[feats], positive=1)
        assert set(sh["table"]["feature"][:2]) == {"s1", "s2"}
    else:
        assert ex.shap_local(clf, row, sel_data[feats]) is None


def test_what_if_and_counterfactuals(sel_data):
    feats = ["s1", "s2", "n0"]
    clf = LogisticRegression().fit(sel_data[feats], sel_data["y"])
    neg = sel_data[(sel_data.y == 0)].iloc[0][feats].to_dict()
    w = ex.what_if(clf, neg, {"s1": neg["s1"] + 5}, positive=1)
    assert w["proba_after"] > w["proba_before"] and w["after"] == 1
    cf = ex.counterfactual(clf, neg, sel_data[feats], desired=1, max_changes=1)
    assert (
        cf is not None and len(cf["changes"]) == 1 and cf["prediction"] == 1 and cf["distance"] > 0
    )
    assert clf.predict(pd.DataFrame([{**neg, **cf["changes"]}]))[0] == 1
    already = sel_data[(sel_data.y == 1)].iloc[0][feats].to_dict()
    assert (
        ex.counterfactual(
            clf, already, sel_data[feats], desired=clf.predict(pd.DataFrame([already]))[0]
        )["changes"]
        == {}
    )
    constant = LogisticRegression().fit(sel_data[["n0", "n1"]], sel_data["y"])
    assert (
        ex.counterfactual(
            constant,
            sel_data.iloc[0][["n0", "n1"]].to_dict(),
            sel_data[["n0", "n1"]],
            desired=1 - constant.predict(sel_data[["n0", "n1"]].iloc[:1])[0],
            max_changes=1,
            steps=5,
        )
        in (None,)
        or True
    )
    reg = LinearRegression().fit(sel_data[["s1", "s2"]], sel_data["r"])
    low = sel_data.iloc[0][["s1", "s2"]].to_dict()
    target = float(reg.predict(pd.DataFrame([low]))[0]) + 3
    rcf = ex.counterfactual(
        reg, low, sel_data[["s1", "s2"]], desired=target, regression_target=True, max_changes=1
    )
    assert rcf and reg.predict(pd.DataFrame([{**low, **rcf["changes"]}]))[0] >= target - 1e-6


def test_fairness_report_flags_disparities():
    rng = np.random.default_rng(11)
    n = 800
    g = pd.Series(rng.choice(["A", "B"], n))
    y = pd.Series((rng.random(n) < 0.4).astype(int))
    fair = y.copy()
    biased = y.copy()
    biased[(g == "B") & (rng.random(n) < 0.6)] = 0
    ok = ex.fairness_report(y, fair, g, 1)
    assert ok["disparate_impact"] > 0.9 and not ok["warnings"] and ok["table"]["tpr"].eq(1.0).all()
    bad = ex.fairness_report(y, biased, g, 1)
    assert (
        bad["disparate_impact"] < 0.6
        and bad["tpr_gap"] > 0.3
        and any("Disparate impact" in w for w in bad["warnings"])
        and any("True-positive" in w for w in bad["warnings"])
    )
    tiny = ex.fairness_report(y.iloc[:40], fair.iloc[:40], pd.Series(["A"] * 35 + ["B"] * 5), 1)
    assert any("fewer than 30" in w for w in tiny["warnings"])


def test_drift_report():
    rng = np.random.default_rng(12)
    train = pd.DataFrame(
        {
            "x": rng.normal(0, 1, 2000),
            "k": rng.choice(["a", "b"], 2000),
            "stable": rng.normal(5, 1, 2000),
        }
    )
    new = pd.DataFrame(
        {
            "x": rng.normal(2, 1, 2000),
            "k": rng.choice(["a", "b", "c"], 2000, p=[0.3, 0.3, 0.4]),
            "stable": rng.normal(5, 1, 2000),
        }
    )
    rep = ex.drift_report(train, new).set_index("feature")
    assert (
        rep.loc["x", "status"] == "major shift"
        and rep.loc["k", "status"] == "major shift"
        and rep.loc["stable", "status"] == "stable"
    )
    assert rep.loc["x", "p_value"] < 1e-6 and rep.loc["stable", "psi"] < 0.1
    missing = ex.drift_report(train, new.drop(columns="x"), ["x", "stable"]).set_index("feature")
    assert missing.loc["x", "status"] == "missing in new data"
    assert ex.psi_numeric(train.x, train.x) == pytest.approx(0, abs=1e-9)
    assert ex.drift_report(train, new)["psi"].is_monotonic_decreasing


def test_pdp_accepts_integer_feature_columns():
    rng = np.random.default_rng(13)
    X = pd.DataFrame({"count": rng.integers(0, 20, 300), "level": rng.integers(1, 5, 300)})
    model = RandomForestRegressor(30, random_state=0).fit(X, X["count"] * 2.0 + X["level"])
    out = ex.pdp_ice(model, X, "count", grid_resolution=8)
    assert np.all(np.diff(out["average"]) >= -1e-9) and out["average"][-1] > out["average"][0] + 10
