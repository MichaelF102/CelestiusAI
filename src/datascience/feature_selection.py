"""Feature selection: filter, wrapper, embedded, stability, minimal set, VIF, clusters, PCA."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from sklearn.feature_selection import (
    RFE,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.datascience import catalog
from src.datascience.modeling import (
    REGRESSION,
    ModelingError,
    _prepare,
    build_preprocessor,
    default_scoring,
    infer_task,
)
from src.datascience.selection import default_cv_strategy, make_cv


def _matrix(
    df: pd.DataFrame, target: str, features: list[str], seed: int = 42, max_rows: int = 20000
):
    X, y = _prepare(df, target, features, max_rows, seed)
    task = infer_task(y)
    prep = build_preprocessor(X)
    M = prep.fit_transform(X)
    names = list(prep.get_feature_names_out())
    origin = []
    for n in names:
        kind, rest = n.split("__", 1)
        if kind == "num":
            origin.append(rest)
        else:
            origin.append(
                max(
                    (c for c in features if rest == c or rest.startswith(f"{c}_")),
                    key=len,
                    default=rest,
                )
            )
    return X, y, task, np.asarray(M, dtype=float), names, origin


def _aggregate(
    names: list[str], origin: list[str], scores: np.ndarray, features: list[str], how: str = "max"
) -> pd.DataFrame:
    frame = pd.DataFrame({"origin": origin, "score": scores})
    agg = frame.groupby("origin")["score"].agg(how)
    return pd.DataFrame({"feature": features, "score": [float(agg.get(f, 0.0)) for f in features]})


# --------------------------------------------------------------------------- filter
def filter_scores(
    df: pd.DataFrame, target: str, features: list[str], method: str = "mutual_info", seed: int = 42
) -> pd.DataFrame:
    X, y, task, M, names, origin = _matrix(df, target, features, seed)
    reg = task == REGRESSION
    if method == "mutual_info":
        scores = (mutual_info_regression if reg else mutual_info_classif)(M, y, random_state=seed)
        pvals = None
    elif method == "anova":
        stat, pv = (f_regression if reg else f_classif)(M, y)
        scores, pvals = np.nan_to_num(stat), np.nan_to_num(pv, nan=1.0)
    else:
        raise ModelingError("method must be 'mutual_info' or 'anova'.")
    out = _aggregate(names, origin, scores, features)
    if pvals is not None:
        pframe = pd.DataFrame({"origin": origin, "p": pvals}).groupby("origin")["p"].min()
        out["p_value"] = [float(pframe.get(f, 1.0)) for f in features]
    out["rank"] = out["score"].rank(ascending=False, method="min").astype(int)
    return out.sort_values("score", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- wrapper
def rfe_ranking(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    model: str = "Linear / Logistic Regression",
    seed: int = 42,
) -> pd.DataFrame:
    """Recursive feature elimination: rank 1 = last survivor. Works on the expanded (one-hot) columns."""
    X, y, task, M, names, origin = _matrix(df, target, features, seed)
    est = catalog.build(model, task, seed)
    if not (
        hasattr(est, "coef_")
        or model in ("Random Forest", "Extra Trees", "Decision Tree", "Gradient Boosting")
    ):
        # estimators without importances: fall back to a tree model
        est = catalog.build("Random Forest", task, seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rfe = RFE(est, n_features_to_select=1, step=1).fit(StandardScaler().fit_transform(M), y)
    per_col = pd.DataFrame({"origin": origin, "rank": rfe.ranking_}).groupby("origin")["rank"].min()
    out = pd.DataFrame(
        {"feature": features, "rfe_rank": [int(per_col.get(f, len(names))) for f in features]}
    )
    out["order"] = out["rfe_rank"].rank(method="first").astype(int)
    return out.sort_values("order").reset_index(drop=True)


# --------------------------------------------------------------------------- embedded
def embedded_importance(
    df: pd.DataFrame, target: str, features: list[str], method: str = "lasso", seed: int = 42
) -> pd.DataFrame:
    X, y, task, M, names, origin = _matrix(df, target, features, seed)
    Ms = StandardScaler().fit_transform(M)
    if method == "lasso":
        model = _l1_model(task, 5000)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(Ms, y)
        coef = np.abs(model.coef_).max(axis=0) if np.ndim(model.coef_) > 1 else np.abs(model.coef_)
    elif method == "tree":
        model = catalog.build("Random Forest", task, seed).fit(M, y)
        coef = model.feature_importances_
    else:
        raise ModelingError("method must be 'lasso' or 'tree'.")
    out = _aggregate(names, origin, coef, features, "sum" if method == "tree" else "max")
    out["selected"] = out["score"] > (1e-8 if method == "lasso" else out["score"].mean() * 0.5)
    return out.sort_values("score", ascending=False).reset_index(drop=True)


def _l1_model(task: str, max_iter: int, strong: bool = False) -> Any:
    """Lasso (regression) or L1 logistic regression (classification).

    `strong` uses a heavier penalty so only clearly useful features survive (stability selection).
    """
    c = 0.04 if strong else 0.2
    if task == REGRESSION:
        return Lasso(alpha=0.25 if strong else 0.05, max_iter=20000)
    if _new_sk():
        return LogisticRegression(l1_ratio=1.0, solver="saga", C=c, max_iter=max_iter)
    return LogisticRegression(penalty="l1", solver="saga", C=c, max_iter=max_iter)


def _new_sk() -> bool:
    import sklearn

    major, minor = sklearn.__version__.split(".")[:2]
    return (int(major), int(minor)) >= (1, 8)


# --------------------------------------------------------------------------- stability
def stability_selection(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    n_rounds: int = 25,
    subsample: float = 0.7,
    seed: int = 42,
) -> pd.DataFrame:
    """How often does each feature survive Lasso on random subsamples? (>= 0.8 is stable)."""
    X, y, task, M, names, origin = _matrix(df, target, features, seed)
    rng = np.random.default_rng(seed)
    hits = pd.Series(0.0, index=features)
    Ms = StandardScaler().fit_transform(M)
    for _ in range(n_rounds):
        idx = rng.choice(len(y), int(len(y) * subsample), replace=False)
        yy = y.iloc[idx]
        model = _l1_model(task, 3000, strong=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(Ms[idx], yy)
        coef = np.abs(model.coef_).max(axis=0) if np.ndim(model.coef_) > 1 else np.abs(model.coef_)
        chosen = {o for o, c in zip(origin, coef, strict=True) if c > 1e-8}
        for f in chosen:
            hits[f] += 1
    out = pd.DataFrame({"feature": features, "selection_frequency": (hits / n_rounds).to_numpy()})
    out["stable"] = out["selection_frequency"] >= 0.8
    return out.sort_values("selection_frequency", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- minimal set
def minimal_feature_set(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    model: str = "Linear / Logistic Regression",
    cv: int = 5,
    scoring: str | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """CV score as features are added in RFE order; the smallest set within one SE of the best."""
    order = list(rfe_ranking(df, target, features, model, seed)["feature"])
    X, y = _prepare(df, target, features, 20000, seed)
    task = infer_task(y)
    scoring = scoring or default_scoring(task)
    splitter = make_cv(default_cv_strategy(task), y, cv, seed=seed)
    rows = []
    for k in range(1, len(order) + 1):
        cols = order[:k]
        pipe = Pipeline(
            [("prep", build_preprocessor(X[cols])), ("model", catalog.build(model, task, seed))]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = cross_val_score(pipe, X[cols], y, cv=splitter, scoring=scoring)
        rows.append(
            {
                "n_features": k,
                "added": order[k - 1],
                "cv_mean": float(s.mean()),
                "cv_se": float(s.std(ddof=1) / np.sqrt(len(s))),
            }
        )
    table = pd.DataFrame(rows)
    best = table.loc[table["cv_mean"].idxmax()]
    within = table[table["cv_mean"] >= best["cv_mean"] - best["cv_se"]]
    pick = int(within["n_features"].min())
    return {
        "table": table,
        "order": order,
        "best_k": int(best["n_features"]),
        "recommended_k": pick,
        "recommended": order[:pick],
    }


# --------------------------------------------------------------------------- collinearity & compression
def vif_table(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    num = df[features].select_dtypes("number").dropna()
    num = num.loc[:, num.nunique() > 1]
    if num.shape[1] < 2:
        raise ModelingError("Need at least two varying numeric features for VIF.")
    Z = np.column_stack([np.ones(len(num)), num.to_numpy()])
    out = pd.DataFrame(
        {
            "feature": num.columns,
            "vif": [variance_inflation_factor(Z, i + 1) for i in range(num.shape[1])],
        }
    )
    out["level"] = pd.cut(
        out["vif"], [-np.inf, 5, 10, np.inf], labels=["ok", "moderate", "severe"]
    ).astype(str)
    return out.sort_values("vif", ascending=False).reset_index(drop=True)


def correlated_clusters(
    df: pd.DataFrame, features: list[str], threshold: float = 0.8
) -> list[list[str]]:
    """Groups of numeric features whose |correlation| exceeds `threshold` (hierarchical clustering)."""
    num = df[features].select_dtypes("number")
    num = num.loc[:, num.nunique() > 1]
    if num.shape[1] < 2:
        return [[c] for c in num.columns]
    corr = num.corr().abs().fillna(0).to_numpy()
    dist = 1 - corr
    np.fill_diagonal(dist, 0)
    labels = fcluster(
        linkage(squareform(dist, checks=False), "average"), t=1 - threshold, criterion="distance"
    )
    groups: dict[int, list[str]] = {}
    for col, lab in zip(num.columns, labels, strict=True):
        groups.setdefault(int(lab), []).append(col)
    return sorted(groups.values(), key=len, reverse=True)


def pca_compression(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    variance: float = 0.9,
    model: str = "Linear / Logistic Regression",
    cv: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    """Does PCA (keeping `variance` of the signal) help or hurt the model?"""
    X, y = _prepare(df, target, features, 20000, seed)
    task = infer_task(y)
    splitter = make_cv(default_cv_strategy(task), y, cv, seed=seed)
    scoring = default_scoring(task)
    base = Pipeline([("prep", build_preprocessor(X)), ("model", catalog.build(model, task, seed))])
    comp = Pipeline(
        [
            ("prep", build_preprocessor(X)),
            ("pca", PCA(n_components=variance, random_state=seed)),
            ("model", catalog.build(model, task, seed)),
        ]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s0 = cross_val_score(base, X, y, cv=splitter, scoring=scoring)
        s1 = cross_val_score(comp, X, y, cv=splitter, scoring=scoring)
        n_comp = comp.fit(X, y).named_steps["pca"].n_components_
        n_orig = base.named_steps["prep"].fit(X).get_feature_names_out().shape[0]
    return {
        "original_features": int(n_orig),
        "components": int(n_comp),
        "score_original": float(s0.mean()),
        "score_pca": float(s1.mean()),
        "difference": float(s1.mean() - s0.mean()),
    }
