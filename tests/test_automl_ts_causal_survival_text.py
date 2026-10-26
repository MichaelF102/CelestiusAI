"""AutoML, ML forecasting, causal/uplift, survival, and text classification."""

import ast
import math

import numpy as np
import pandas as pd
import pytest
from src.datascience import automl as am
from src.datascience import causal as cs
from src.datascience import survival as sv
from src.datascience import text_features as tf
from src.datascience import tsml
from src.datascience.modeling import ModelingError
from src.datascience.timeseries import TimeSeriesError


# --------------------------------------------------------------------------- AutoML
@pytest.fixture(scope="module")
def churn():
    rng = np.random.default_rng(1)
    n = 500
    df = pd.DataFrame(
        {
            "customer_id": np.arange(n),
            "age": rng.integers(20, 70, n),
            "spend": rng.gamma(2, 50, n),
            "plan": rng.choice(["a", "b", "c"], n),
        }
    )
    logit = (df.age - 45) * 0.05 + (df.spend - 100) * 0.015 + (df.plan == "a") * 0.8
    df["churn"] = (logit + rng.normal(0, 1, n) > 0.3).astype(int)
    df["churn_flag"] = df["churn"]  # a leaked copy of the target
    return df


def test_automl_end_to_end_with_explanations(churn):
    res = am.run_automl(churn, "churn", time_budget_s=45, cv=3, n_race=5, top_k=1, tune_iter=3)
    assert res.task == "binary_classification" and res.model is not None and res.total_seconds > 0
    dropped = dict(res.dropped)
    assert (
        "churn_flag" in dropped
        and "customer_id" in dropped
        and "churn_flag" not in res.features_used
        and "customer_id" not in res.features_used
    )
    assert [s.name for s in res.steps][:3] == [
        "Leakage and identifier check",
        "Profile and shortlist",
        "Model race",
    ]
    assert res.chosen and res.chosen_reason and res.model.metrics["accuracy"] > 0.55
    assert res.leaderboard.iloc[0]["cv_mean"] >= res.leaderboard["cv_mean"].dropna().iloc[-1]
    if res.tuned:
        assert res.tuned[0]["tuned_cv"] >= res.tuned[0]["default_cv"] - 0.05
    report = res.report_markdown()
    assert "# AutoML report" in report and "churn_flag" in report and res.chosen in report


def test_automl_code_reproduces_the_held_out_metrics(churn, tmp_path, capsys):
    csv = tmp_path / "d.csv"
    churn.to_csv(csv, index=False)
    res = am.run_automl(
        churn, "churn", time_budget_s=30, cv=3, n_race=4, top_k=1, tune=False, try_ensemble=False
    )
    code = am.automl_code(res, filename=str(csv))
    ast.parse(code)
    exec(compile(code, "<automl>", "exec"), {"__name__": "x"})  # noqa: S102
    out = capsys.readouterr().out
    assert (
        f"accuracy: {res.model.metrics['accuracy']}" in out
        or abs(float(out.split("accuracy:")[1].split()[0]) - res.model.metrics["accuracy"]) < 1e-9
    )


def test_automl_regression_and_validation(churn):
    rng = np.random.default_rng(2)
    df = churn.assign(rev=churn.age * 2 + churn.spend + rng.normal(0, 5, len(churn))).drop(
        columns=["churn", "churn_flag"]
    )
    res = am.run_automl(
        df, "rev", time_budget_s=30, cv=3, n_race=4, top_k=1, tune=False, try_ensemble=False
    )
    assert res.task == "regression" and res.model.metrics["R2"] > 0.8
    with pytest.raises(ModelingError, match="not a column"):
        am.run_automl(df, "ghost")
    only_ids = pd.DataFrame({"id": range(100), "y": np.random.default_rng(0).integers(0, 2, 100)})
    with pytest.raises(ModelingError, match="nothing left"):
        am.run_automl(only_ids, "y", cv=3)
    with pytest.raises(ModelingError):
        am.automl_code(am.AutoMLResult("regression", "y", "r2", [], []))


# --------------------------------------------------------------------------- ML forecasting
@pytest.fixture(scope="module")
def series():
    idx = pd.date_range("2022-01-01", periods=420, freq="D")
    rng = np.random.default_rng(0)
    y = (
        100
        + 0.1 * np.arange(420)
        + 12 * np.sin(2 * np.pi * np.arange(420) / 7)
        + rng.normal(0, 1.5, 420)
    )
    return pd.Series(y, index=idx, name="sales")


def test_supervised_features_never_use_the_future(series):
    X, t = tsml.supervised(series, horizon_step=3, lags=(0, 1, 7), rolling=(7,))
    row = X.index[50]
    pos = series.index.get_loc(row)
    assert (
        X.loc[row, "lag_0"] == series.iloc[pos]
        and X.loc[row, "lag_1"] == series.iloc[pos - 1]
        and X.loc[row, "lag_7"] == series.iloc[pos - 7]
    )
    assert t.loc[row] == series.iloc[pos + 3]
    assert X.loc[row, "roll_mean_7"] == pytest.approx(series.iloc[pos - 6 : pos + 1].mean())
    assert (
        X.loc[row, "dow"] == (row + pd.Timedelta(days=3)).dayofweek
    )  # calendar of the TARGET date
    assert not X.isna().any().any() and len(X) == len(series) - 3 - 7


@pytest.mark.parametrize("strategy", ["direct", "recursive"])
def test_ml_forecasters_beat_naive_on_seasonal_data(strategy, series):
    f = tsml.MLForecaster("Ridge / L2 Logistic", strategy, 14).fit(series.iloc[:-14])
    pred = f.predict()
    assert len(pred) == 14 and pred.index[0] == series.index[-14] and not pred.isna().any()
    actual = series.iloc[-14:]
    naive = np.repeat(series.iloc[-15], 14)
    assert (
        np.mean(np.abs(actual.to_numpy() - pred.to_numpy()))
        < np.mean(np.abs(actual.to_numpy() - naive)) * 0.6
    )


def test_rolling_evaluation_and_forecast(series):
    table = tsml.rolling_evaluation(series, ["Ridge / L2 Logistic", "Random Forest"], 7, folds=2)
    rmse = table.set_index("model")["RMSE"]
    assert {
        "Naive",
        "Seasonal naive",
        "Ridge / L2 Logistic [direct]",
        "Random Forest [recursive]",
    } <= set(rmse.index)
    assert (
        rmse["Ridge / L2 Logistic [direct]"] < rmse["Naive"]
        and table["RMSE"].is_monotonic_increasing
    )
    fc = tsml.forecast_ml(series, "Ridge / L2 Logistic", 7)
    assert (
        list(fc.columns) == ["forecast", "lower", "upper"]
        and (fc["lower"] < fc["forecast"]).all()
        and (fc["forecast"] < fc["upper"]).all()
    )
    with pytest.raises(TimeSeriesError, match="too short"):
        tsml.rolling_evaluation(series.iloc[:50], ["Ridge / L2 Logistic"], 7)
    with pytest.raises(TimeSeriesError):
        tsml.MLForecaster(strategy="magic")
    with pytest.raises(TimeSeriesError, match="regular frequency"):
        tsml.supervised(
            pd.Series(
                np.arange(30.0),
                index=pd.to_datetime(["2024-01-01", "2024-01-03", "2024-01-08"] * 10),
            ),
            1,
        )


# --------------------------------------------------------------------------- causal
@pytest.fixture(scope="module")
def did_data():
    rng = np.random.default_rng(3)
    rows = []
    for unit in range(120):
        treated = unit < 60
        base = rng.normal(50, 5)
        for period in range(8):
            post = period >= 4
            y = (
                base
                + 2 * period * 0
                + 3 * period * 0.5
                + (4.0 if (treated and post) else 0)
                + (2.0 if treated else 0)
                + rng.normal(0, 1)
            )
            rows.append(
                {
                    "unit": unit,
                    "period": period,
                    "group": "treated" if treated else "control",
                    "y": y,
                }
            )
    return pd.DataFrame(rows)


def test_did_recovers_the_effect_and_checks_parallel_trends(did_data):
    r = cs.did(did_data, "y", "group", "period", 4, treated_value="treated", cluster_col="unit")
    assert (
        abs(r["effect"] - 4.0) < 0.5 and r["ci_low"] < 4.0 < r["ci_high"] and r["p_value"] < 0.001
    )
    assert (
        r["manual_effect"] == pytest.approx(r["effect"], abs=1e-6)
        and r["parallel_trends_p"] > 0.05
        and not [n for n in r["notes"] if "violated" in n]
    )
    assert {"control", "treated"} <= set(r["event_study"].columns) and set(r["means"].columns) == {
        "before",
        "after",
    }
    diverging = did_data.assign(
        y=did_data.y + np.where(did_data.group == "treated", did_data.period * 1.5, 0)
    )
    bad = cs.did(diverging, "y", "group", "period", 4, treated_value="treated")
    assert bad["parallel_trends_p"] < 0.05 and any("violated" in n for n in bad["notes"])
    with pytest.raises(cs.CausalError, match="no 'before'"):
        cs.did(did_data, "y", "group", "period", 99, treated_value="treated")
    with pytest.raises(cs.CausalError, match="exactly two"):
        cs.did(did_data.assign(group=did_data.unit % 3), "y", "group", "period", 4)


def test_did_with_dates_and_covariates(did_data):
    dated = did_data.assign(
        day=pd.Timestamp("2024-01-01") + pd.to_timedelta(did_data.period * 30, unit="D"),
        size=np.random.default_rng(1).normal(size=len(did_data)),
    )
    r = cs.did(
        dated, "y", "group", "day", "2024-04-30", treated_value="treated", covariates=["size"]
    )
    assert abs(r["effect"] - 4.0) < 0.6 and "size" in r["coefficients"].index


@pytest.fixture(scope="module")
def observational():
    rng = np.random.default_rng(4)
    n = 2500
    age = rng.normal(40, 10, n)
    income = rng.normal(0, 1, n)
    p = 1 / (1 + np.exp(-(0.08 * (age - 40) + 0.8 * income)))
    t = (rng.random(n) < p).astype(int)
    y = (
        2.0 * t + 0.15 * (age - 40) + 1.5 * income + rng.normal(0, 1, n)
    )  # confounded: true effect = 2
    return pd.DataFrame({"t": t, "y": y, "age": age, "income": income})


def test_matching_removes_confounding_bias(observational):
    r = cs.propensity_matching(observational, "t", "y", ["age", "income"], n_boot=200)
    assert abs(r["naive_difference"] - 2.0) > 0.8  # naive comparison is badly biased
    assert abs(r["att"] - 2.0) < 0.4 and r["ci_low"] < r["att"] < r["ci_high"]
    assert (
        (r["balance"]["smd_before"].abs() > 0.2).all()
        and (r["balance"]["smd_after"].abs() < 0.1).all()
        and r["balance"]["balanced_after"].all()
    )
    assert r["matched_pairs"] > 0.5 * r["treated_total"] and 0.6 < r["ps_auc"] < 0.9
    assert len(r["matched_rows"]) == 2 * r["matched_pairs"]
    with pytest.raises(cs.CausalError, match="at least one covariate"):
        cs.propensity_matching(observational, "t", "y", [])
    tiny = observational.head(15).assign(t=[1] * 3 + [0] * 12)
    with pytest.raises(cs.CausalError, match="at least 10"):
        cs.propensity_matching(tiny, "t", "y", ["age"])


def test_uplift_model_finds_who_responds():
    rng = np.random.default_rng(5)
    n = 12000
    df = pd.DataFrame(
        {"age": rng.uniform(18, 70, n), "loyal": rng.integers(0, 2, n), "t": rng.integers(0, 2, n)}
    )
    base = 0.10 + 0.05 * df.loyal
    lift = np.where(df.age < 35, 0.20, 0.0)  # only young customers respond to the offer
    df["buy"] = (rng.random(n) < base + lift * df.t).astype(int)
    r = cs.uplift_model(df, "t", "buy", ["age", "loyal"], "Random Forest")
    assert (
        r["binary_outcome"]
        and r["auuc"] > 0
        and r["qini"]["qini"].iloc[-1] == pytest.approx(r["qini"]["random"].iloc[-1])
    )
    assert r["top3_observed_uplift"] > r["ate_observed"] + 0.03
    d = r["by_decile"]
    assert d["observed_uplift"].head(3).mean() > d["observed_uplift"].tail(3).mean() + 0.05
    uplift = r["uplift"]
    assert (
        uplift[df.loc[uplift.index, "age"] < 35].mean()
        > uplift[df.loc[uplift.index, "age"] >= 35].mean() + 0.05
    )
    with pytest.raises(cs.CausalError, match="at least 200"):
        cs.uplift_model(df.head(100), "t", "buy", ["age"])


def test_qini_curve_basics():
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    t = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    q = cs.qini_curve(y, t, np.arange(8, 0, -1.0))
    assert (
        len(q) == 8
        and q["fraction"].iloc[-1] == 1.0
        and q["n_targeted"].tolist() == list(range(1, 9))
    )


# --------------------------------------------------------------------------- survival
@pytest.fixture(scope="module")
def surv():
    rng = np.random.default_rng(6)
    n = 800
    x = rng.normal(size=n)
    grp = rng.choice(["basic", "pro"], n)
    rate = 0.05 * np.exp(0.7 * x) * np.where(grp == "pro", 0.5, 1.0)
    t_event = rng.exponential(1 / rate)
    t_cens = rng.uniform(5, 60, n)
    return pd.DataFrame(
        {
            "time": np.minimum(t_event, t_cens),
            "event": (t_event <= t_cens).astype(int),
            "x": x,
            "plan": grp,
        }
    )


def test_kaplan_meier_matches_known_exponential_and_logrank(surv):
    km = sv.kaplan_meier(surv, "time", "event", "plan")
    s = km["summary"].set_index("group")
    assert (
        s.loc["pro", "median_survival"] > s.loc["basic", "median_survival"] * 1.5
        and km["logrank"]["p_value"] < 0.001
    )
    curve = km["curves"][km["curves"].group == "basic"]
    assert (
        curve["survival"].is_monotonic_decreasing
        and (curve["ci_low"] <= curve["survival"]).all()
        and (curve["survival"] <= curve["ci_high"]).all()
    )
    assert curve["survival"].between(0, 1).all() and curve["at_risk"].is_monotonic_decreasing
    one = sv.kaplan_meier(surv, "time", "event")
    assert list(one["summary"]["group"]) == ["all"] and one["logrank"] is None
    with pytest.raises(sv.SurvivalError, match="0/1"):
        sv.kaplan_meier(surv.assign(event=surv.event * 2), "time", "event")
    with pytest.raises(sv.SurvivalError, match="at least 10"):
        sv.kaplan_meier(surv.head(5), "time", "event")


def test_cox_recovers_hazard_ratios_and_concordance(surv):
    c = sv.cox_ph(surv, "time", "event", ["x", "plan"])
    t = c["table"]
    assert (
        1.8 < t.loc["x", "hazard_ratio"] < 2.3 and t.loc["x", "p_value"] < 0.001
    )  # exp(0.7) = 2.01
    assert (
        0.35 < t.loc["plan_pro", "hazard_ratio"] < 0.7
        and t.loc["x", "hr_low"] < t.loc["x", "hazard_ratio"] < t.loc["x", "hr_high"]
    )
    assert (
        0.6 < c["concordance"] < 0.9
        and c["events"] == surv.event.sum()
        and not c["ph_check"]["violated"].any()
    )
    pred = sv.predict_survival(c, {"x": 1.5, "plan_pro": 0.0}, np.array([5.0, 20.0, 50.0]))
    low = sv.predict_survival(c, {"x": -1.5, "plan_pro": 1.0}, np.array([5.0, 20.0, 50.0]))
    assert pred["survival"].is_monotonic_decreasing and (low["survival"] > pred["survival"]).all()
    with pytest.raises(sv.SurvivalError):
        sv.cox_ph(surv, "time", "event", [])


def test_concordance_extremes_and_ph_violation_note():
    t = np.array([1.0, 2.0, 3.0, 4.0])
    e = np.array([1, 1, 1, 1])
    assert sv.concordance(t, e, np.array([4.0, 3.0, 2.0, 1.0])) == 1.0
    assert sv.concordance(t, e, np.array([1.0, 2.0, 3.0, 4.0])) == 0.0
    assert sv.concordance(t, e, np.ones(4)) == 0.5
    rng = np.random.default_rng(7)
    n = 1200
    x = rng.integers(0, 2, n).astype(float)
    # hazard ratio flips over time: early on x=1 is riskier, later it is protective
    t1 = rng.exponential(1 / np.where(x == 1, 0.3, 0.1), n)
    t2 = rng.exponential(1 / np.where(x == 1, 0.02, 0.1), n)
    time = np.where(t1 < 4, t1, 4 + t2)
    df = pd.DataFrame({"time": time, "event": np.ones(n, dtype=int), "x": x})
    c = sv.cox_ph(df, "time", "event", ["x"])
    assert c["ph_check"]["violated"].any() and any("Proportional hazards" in n for n in c["notes"])


# --------------------------------------------------------------------------- text
@pytest.fixture(scope="module")
def reviews():
    rng = np.random.default_rng(8)
    pos = [
        "great product works perfectly",
        "love it excellent quality",
        "fantastic fast delivery happy",
        "amazing value would buy again",
        "perfect fit wonderful",
    ]
    neg = [
        "terrible broke after a day",
        "awful quality waste of money",
        "horrible slow delivery disappointed",
        "worst purchase never again",
        "poor fit cheap material",
    ]
    neu = [
        "arrived on tuesday in a box",
        "the item is blue and medium sized",
        "ordered for my office desk",
    ]
    rows = []
    for _ in range(90):
        rows.append((" ".join(rng.choice(pos, 2)), "positive"))
        rows.append((" ".join(rng.choice(neg, 2)), "negative"))
        rows.append((" ".join(rng.choice(neu, 2)), "neutral"))
    return (
        pd.DataFrame(rows, columns=["text", "label"])
        .sample(frac=1, random_state=0)
        .reset_index(drop=True)
    )


@pytest.mark.parametrize("model", list(tf.TEXT_MODELS))
def test_text_models_classify_and_explain(model, reviews):
    res = tf.text_classification(reviews, "text", "label", model, cv=3)
    assert (
        res["cv_accuracy"] > 0.9 > res["majority_baseline"] + 0.4
        and res["cv_f1_macro"] > 0.9
        and res["vocabulary"] > 10
    )
    assert res["confusion"].to_numpy().sum() == math.ceil(len(reviews) * 0.25) and set(
        res["classes"]
    ) == {"negative", "neutral", "positive"}
    if model != "Linear SVM":
        terms = res["top_terms"]
        pos_terms = set(terms[terms["class"] == "positive"]["term"])
        neg_terms = set(terms[terms["class"] == "negative"]["term"])
        assert pos_terms & {
            "great",
            "excellent",
            "love",
            "perfect",
            "amazing",
            "fantastic",
            "wonderful",
            "happy",
            "quality",
        } and neg_terms & {
            "terrible",
            "awful",
            "horrible",
            "worst",
            "poor",
            "waste",
            "disappointed",
            "broke",
        }
    out = tf.predict_text(res["pipeline"], ["love it perfect fit", "awful horrible waste"])
    assert (
        out["prediction"].tolist() == ["positive", "negative"] and (out["confidence"] > 0.5).all()
    )


def test_text_summary_and_validation(reviews):
    s = tf.text_summary(reviews, "text")
    assert (
        s["documents"] == 270
        and s["vocabulary"] > 20
        and s["top_words"].iloc[0] > 10
        and s["empty"] == 0
    )
    with pytest.raises(tf.TextError, match="at least 30"):
        tf.text_classification(reviews.head(10), "text", "label")
    with pytest.raises(tf.TextError, match="at least 3"):
        tf.text_classification(
            reviews.assign(label=["a"] + ["b"] * (len(reviews) - 1)), "text", "label"
        )
    with pytest.raises(tf.TextError, match="Unknown"):
        tf.build_text_pipeline("Transformer")
