"""Causal and uplift analysis: difference-in-differences, propensity matching, uplift modeling."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.datascience import catalog
from src.datascience.modeling import build_preprocessor


class CausalError(ValueError):
    """Invalid causal-analysis input (message is user-facing)."""


def _binary(series: pd.Series, positive: Any | None, name: str) -> pd.Series:
    levels = sorted(series.dropna().unique(), key=str)
    if len(levels) != 2:
        raise CausalError(f"'{name}' must have exactly two values (found {len(levels)}).")
    pos = positive if positive is not None else levels[-1]
    return (series == pos).astype(int)


# --------------------------------------------------------------------------- difference-in-differences
def did(
    df: pd.DataFrame,
    outcome: str,
    group_col: str,
    time_col: str,
    start: Any,
    treated_value: Any | None = None,
    covariates: list[str] | None = None,
    cluster_col: str | None = None,
) -> dict[str, Any]:
    """Treated-vs-control change before/after `start`, with a parallel-trends check."""
    data = (
        df[
            [
                outcome,
                group_col,
                time_col,
                *(covariates or []),
                *([cluster_col] if cluster_col else []),
            ]
        ]
        .dropna()
        .copy()
    )
    data[outcome] = pd.to_numeric(data[outcome], errors="coerce")
    data = data.dropna(subset=[outcome])
    data["_treated"] = _binary(data[group_col], treated_value, group_col)
    times = data[time_col]
    if (
        pd.api.types.is_numeric_dtype(times)
        and not np.issubdtype(type(start), np.datetime64)
        and not isinstance(start, (str, pd.Timestamp))
    ):
        data["_post"] = (times >= start).astype(int)
        order = times
    else:
        parsed = pd.to_datetime(times, errors="coerce")
        data["_post"] = (parsed >= pd.Timestamp(start)).astype(int)
        order = parsed
    if data["_post"].nunique() < 2:
        raise CausalError("The start point leaves no 'before' or no 'after' observations.")
    cells = data.groupby(["_treated", "_post"])[outcome].agg(["mean", "count"])
    if len(cells) < 4 or (cells["count"] < 3).any():
        raise CausalError("Each treated/control x before/after cell needs at least 3 observations.")
    data["_did"] = data["_treated"] * data["_post"]
    X = pd.get_dummies(
        data[["_treated", "_post", "_did", *(covariates or [])]], drop_first=True, dtype=float
    )
    model = sm.OLS(data[outcome], sm.add_constant(X))
    fit = (
        model.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(data[cluster_col])[0]})
        if cluster_col
        else model.fit(cov_type="HC1")
    )
    ci = fit.conf_int().loc["_did"]
    means = cells["mean"].unstack()
    means.index = ["control", "treated"]
    means.columns = ["before", "after"]
    manual = (means.loc["treated", "after"] - means.loc["treated", "before"]) - (
        means.loc["control", "after"] - means.loc["control", "before"]
    )

    # parallel trends: do treated and control already diverge before the start?
    pre = data[data["_post"] == 0].assign(
        _t=lambda d: pd.factorize(order.loc[d.index], sort=True)[0]
    )
    trend_p = None
    if pre["_t"].nunique() >= 3:
        pre_fit = sm.OLS(
            pre[outcome],
            sm.add_constant(
                pd.DataFrame(
                    {
                        "t": pre["_t"],
                        "treated": pre["_treated"],
                        "t_x_treated": pre["_t"] * pre["_treated"],
                    }
                )
            ),
        ).fit(cov_type="HC1")
        trend_p = float(pre_fit.pvalues["t_x_treated"])
    notes = []
    if trend_p is not None and trend_p < 0.05:
        notes.append(
            f"Pre-treatment trends differ between groups (p = {trend_p:.3f}): the parallel-trends assumption looks violated, so the estimate may be biased."
        )
    if trend_p is None:
        notes.append("Fewer than 3 pre-treatment periods: parallel trends cannot be checked.")
    event = (
        data.assign(_period=order)
        .groupby(["_period", "_treated"])[outcome]
        .mean()
        .unstack()
        .rename(columns={0: "control", 1: "treated"})
        .reset_index()
    )
    return {
        "effect": float(fit.params["_did"]),
        "se": float(fit.bse["_did"]),
        "ci_low": float(ci[0]),
        "ci_high": float(ci[1]),
        "p_value": float(fit.pvalues["_did"]),
        "means": means,
        "manual_effect": float(manual),
        "coefficients": pd.DataFrame({"coef": fit.params, "se": fit.bse, "p_value": fit.pvalues}),
        "parallel_trends_p": trend_p,
        "event_study": event,
        "notes": notes,
        "n": int(fit.nobs),
    }


# --------------------------------------------------------------------------- propensity-score matching
def smd(treated: pd.Series, control: pd.Series) -> float:
    """Standardized mean difference (|SMD| < 0.1 is considered balanced)."""
    pooled = np.sqrt((treated.var(ddof=1) + control.var(ddof=1)) / 2)
    return float((treated.mean() - control.mean()) / pooled) if pooled > 0 else 0.0


def propensity_matching(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    covariates: list[str],
    treated_value: Any | None = None,
    caliper_sd: float = 0.2,
    seed: int = 42,
    n_boot: int = 300,
) -> dict[str, Any]:
    """1:1 nearest-neighbour matching on the logit of the propensity score (no replacement)."""
    if not covariates:
        raise CausalError("Pick at least one covariate to balance.")
    data = df[[treatment, outcome, *covariates]].dropna().copy()
    data["_t"] = _binary(data[treatment], treated_value, treatment)
    data[outcome] = pd.to_numeric(data[outcome], errors="coerce")
    data = data.dropna(subset=[outcome]).reset_index(drop=True)
    if data["_t"].sum() < 10 or (1 - data["_t"]).sum() < 10:
        raise CausalError("Need at least 10 treated and 10 control rows.")
    X = pd.get_dummies(data[covariates], drop_first=True, dtype=float)
    Xs = StandardScaler().fit_transform(X)
    ps_model = LogisticRegression(max_iter=2000).fit(Xs, data["_t"])
    ps = np.clip(ps_model.predict_proba(Xs)[:, 1], 1e-4, 1 - 1e-4)
    logit = np.log(ps / (1 - ps))
    data["_ps"], data["_logit"] = ps, logit
    caliper = caliper_sd * logit.std()
    treated_idx = data.index[data["_t"] == 1].to_numpy()
    control_idx = data.index[data["_t"] == 0].to_numpy()
    rng = np.random.default_rng(seed)
    nn = NearestNeighbors(n_neighbors=min(len(control_idx), 200)).fit(
        logit[control_idx].reshape(-1, 1)
    )
    dist, nbr = nn.kneighbors(logit[treated_idx].reshape(-1, 1))
    used: set[int] = set()
    pairs = []
    for pos in rng.permutation(len(treated_idx)):
        for d, j in zip(dist[pos], nbr[pos], strict=True):
            c = int(control_idx[j])
            if c in used:
                continue
            if d <= caliper:
                used.add(c)
                pairs.append((int(treated_idx[pos]), c))
            break
    if len(pairs) < 5:
        raise CausalError(
            "Fewer than 5 matches within the caliper: the groups barely overlap on these covariates."
        )
    t_i, c_i = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    diff = data.loc[t_i, outcome].to_numpy() - data.loc[c_i, outcome].to_numpy()
    att = float(diff.mean())
    boots = rng.choice(diff, (n_boot, len(diff))).mean(axis=1)
    matched = pd.concat([data.loc[t_i], data.loc[c_i]])
    rows = []
    for col in X.columns:
        rows.append(
            {
                "covariate": col,
                "smd_before": smd(X.loc[data["_t"] == 1, col], X.loc[data["_t"] == 0, col]),
                "smd_after": smd(X.loc[t_i, col], X.loc[c_i, col]),
            }
        )
    balance = pd.DataFrame(rows)
    balance["balanced_after"] = balance["smd_after"].abs() < 0.1
    naive = float(
        data.loc[data["_t"] == 1, outcome].mean() - data.loc[data["_t"] == 0, outcome].mean()
    )
    notes = []
    if len(pairs) < 0.8 * len(treated_idx):
        notes.append(
            f"Only {len(pairs)} of {len(treated_idx)} treated rows found a match within the caliper; the estimate applies to that overlapping subgroup."
        )
    if (balance["smd_after"].abs() >= 0.1).any():
        notes.append(
            "Some covariates remain imbalanced after matching (|SMD| >= 0.1): add covariates or loosen the caliper."
        )
    return {
        "att": att,
        "ci_low": float(np.percentile(boots, 2.5)),
        "ci_high": float(np.percentile(boots, 97.5)),
        "se": float(diff.std(ddof=1) / np.sqrt(len(diff))),
        "naive_difference": naive,
        "matched_pairs": len(pairs),
        "treated_total": int(len(treated_idx)),
        "balance": balance,
        "ps_auc": float(roc_auc_score(data["_t"], ps)),
        "overlap": data.groupby("_t")["_ps"]
        .agg(["min", "max", "mean"])
        .rename(index={0: "control", 1: "treated"}),
        "matched_rows": matched,
        "notes": notes,
    }


# --------------------------------------------------------------------------- uplift
def qini_curve(y: np.ndarray, t: np.ndarray, score: np.ndarray) -> pd.DataFrame:
    """Cumulative incremental outcomes when targeting the highest-scored rows first."""
    order = np.argsort(-score)
    y, t = y[order], t[order]
    n_t, n_c = np.cumsum(t), np.cumsum(1 - t)
    y_t, y_c = np.cumsum(y * t), np.cumsum(y * (1 - t))
    with np.errstate(divide="ignore", invalid="ignore"):
        qini = y_t - np.where(n_c > 0, y_c * n_t / n_c, 0.0)
    out = pd.DataFrame({"n_targeted": np.arange(1, len(y) + 1), "qini": qini})
    out["fraction"] = out["n_targeted"] / len(y)
    out["random"] = out["qini"].iloc[-1] * out["fraction"]
    return out


def uplift_model(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    features: list[str],
    model: str = "Random Forest",
    treated_value: Any | None = None,
    test_size: float = 0.3,
    seed: int = 42,
) -> dict[str, Any]:
    """T-learner: separate outcome models for treated and control; uplift = difference of predictions."""
    data = df[[treatment, outcome, *features]].dropna().copy()
    t = _binary(data[treatment], treated_value, treatment)
    y_raw = data[outcome]
    binary_outcome = y_raw.nunique() == 2
    y = _binary(y_raw, None, outcome) if binary_outcome else pd.to_numeric(y_raw, errors="coerce")
    X = data[features]
    if len(data) < 200:
        raise CausalError("Uplift modeling needs at least 200 rows.")
    strat = t.astype(str) + "_" + (y.astype(str) if binary_outcome else "")
    X_tr, X_te, t_tr, t_te, y_tr, y_te = train_test_split(
        X,
        t,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=strat if strat.value_counts().min() >= 2 else t,
    )
    task = "binary_classification" if binary_outcome else "regression"
    models = {}
    for arm in (1, 0):
        mask = t_tr == arm
        if mask.sum() < 30:
            raise CausalError("Each arm needs at least 30 training rows.")
        pipe = Pipeline(
            [("prep", build_preprocessor(X_tr)), ("m", catalog.build(model, task, seed))]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            models[arm] = pipe.fit(X_tr[mask], y_tr[mask])

    def predict(pipe: Pipeline) -> np.ndarray:
        return (
            pipe.predict_proba(X_te)[:, list(pipe.classes_).index(1)]
            if binary_outcome
            else pipe.predict(X_te)
        )

    uplift = predict(models[1]) - predict(models[0])
    qini = qini_curve(y_te.to_numpy(dtype=float), t_te.to_numpy(), uplift)
    auuc = float(np.trapezoid(qini["qini"] - qini["random"], qini["fraction"]))
    frame = pd.DataFrame({"uplift": uplift, "t": t_te.to_numpy(), "y": y_te.to_numpy(dtype=float)})
    frame["decile"] = pd.qcut(
        frame["uplift"].rank(method="first", ascending=False), 10, labels=range(1, 11)
    )
    rows = []
    for d, g in frame.groupby("decile", observed=True):
        tr, co = g[g.t == 1]["y"], g[g.t == 0]["y"]
        rows.append(
            {
                "decile": int(d),
                "rows": len(g),
                "mean_predicted_uplift": float(g["uplift"].mean()),
                "observed_uplift": float(tr.mean() - co.mean()) if len(tr) and len(co) else np.nan,
            }
        )
    by_decile = pd.DataFrame(rows)
    ate = float(frame[frame.t == 1]["y"].mean() - frame[frame.t == 0]["y"].mean())
    top = by_decile.head(3)
    return {
        "uplift": pd.Series(uplift, index=X_te.index),
        "qini": qini,
        "auuc": auuc,
        "by_decile": by_decile,
        "ate_observed": ate,
        "top3_observed_uplift": float(top["observed_uplift"].mean()),
        "binary_outcome": binary_outcome,
        "models": models,
        "test_index": X_te.index,
    }
