"""Model selection: recommendations, model races, nested CV, statistical comparison, curves."""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    TimeSeriesSplit,
    cross_validate,
    learning_curve,
    train_test_split,
    validation_curve,
)
from sklearn.pipeline import Pipeline

from src.datascience import catalog
from src.datascience.modeling import (
    BINARY,
    MULTICLASS,
    REGRESSION,
    SCORING_CHOICES,
    ModelingError,
    _prepare,
    build_preprocessor,
    default_scoring,
    infer_task,
)

CV_STRATEGIES = {
    "kfold": "K-fold",
    "stratified": "Stratified K-fold",
    "repeated": "Repeated K-fold",
    "repeated_stratified": "Repeated stratified K-fold",
    "group": "Group K-fold (no group in two folds)",
    "timeseries": "Time-series split (rows in time order)",
}


def make_cv(
    strategy: str, y: pd.Series, n_splits: int = 5, n_repeats: int = 2, seed: int = 42
) -> Any:
    """A cross-validation splitter, with fold counts clamped to what the data allows."""
    n = len(y)
    splits = max(2, min(n_splits, n // 5))
    is_clf = not pd.api.types.is_numeric_dtype(y) or y.nunique() <= 10
    if strategy in ("stratified", "repeated_stratified") or (strategy == "kfold" and False):
        smallest = int(y.value_counts().min())
        if smallest < 2:
            raise ModelingError("Every class needs at least 2 rows for stratified folds.")
        splits = max(2, min(splits, smallest))
        if strategy == "stratified":
            return StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
        return RepeatedStratifiedKFold(n_splits=splits, n_repeats=n_repeats, random_state=seed)
    if strategy == "kfold":
        return KFold(n_splits=splits, shuffle=True, random_state=seed)
    if strategy == "repeated":
        return RepeatedKFold(n_splits=splits, n_repeats=n_repeats, random_state=seed)
    if strategy == "group":
        return GroupKFold(n_splits=splits)
    if strategy == "timeseries":
        return TimeSeriesSplit(n_splits=splits)
    del is_clf
    raise ModelingError(f"Unknown validation strategy '{strategy}'.")


def default_cv_strategy(task: str) -> str:
    return "kfold" if task == REGRESSION else "stratified"


# --------------------------------------------------------------------------- data profile & recommendations
@dataclass
class DataProfile:
    n_rows: int
    n_features: int
    n_numeric: int
    n_categorical: int
    missing_pct: float
    high_cardinality: int
    n_classes: int | None
    imbalance_ratio: float | None
    target_skew: float | None
    nonlinear_gain: float  # CV score(HistGB) - CV score(linear), on a small sample
    task: str


def profile_for_selection(
    df: pd.DataFrame, target: str, features: list[str], task: str | None = None, seed: int = 42
) -> DataProfile:
    X, y = _prepare(df, target, features, 5000, seed)
    task = task or infer_task(y)
    num = X.select_dtypes("number").columns
    cat = [c for c in X.columns if c not in num]
    high_card = sum(int(X[c].nunique() > 30) for c in cat)
    counts = y.value_counts() if task != REGRESSION else None
    gain = 0.0
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
        from sklearn.linear_model import LinearRegression, LogisticRegression

        reg = task == REGRESSION
        splitter = make_cv("kfold" if reg else "stratified", y, 3, seed=seed)
        scoring = "r2" if reg else "f1_macro"
        lin = LogisticRegression(max_iter=500) if not reg else LinearRegression()
        gb = (
            HistGradientBoostingRegressor(max_iter=60, random_state=seed)
            if reg
            else HistGradientBoostingClassifier(max_iter=60, random_state=seed)
        )
        s_lin = cross_validate(
            Pipeline([("p", build_preprocessor(X)), ("m", lin)]), X, y, cv=splitter, scoring=scoring
        )["test_score"].mean()
        s_gb = cross_validate(
            Pipeline([("p", build_preprocessor(X)), ("m", gb)]), X, y, cv=splitter, scoring=scoring
        )["test_score"].mean()
        gain = float(s_gb - s_lin)
    except Exception:
        gain = 0.0
    return DataProfile(
        n_rows=int(df[target].notna().sum()),
        n_features=X.shape[1],
        n_numeric=len(num),
        n_categorical=len(cat),
        missing_pct=float(X.isna().mean().mean() * 100),
        high_cardinality=high_card,
        n_classes=int(counts.size) if counts is not None else None,
        imbalance_ratio=float(counts.max() / counts.min())
        if counts is not None and counts.min() > 0
        else None,
        target_skew=float(pd.to_numeric(y, errors="coerce").skew()) if task == REGRESSION else None,
        nonlinear_gain=gain,
        task=task,
    )


class _Tally:
    """Running score with a reason for every adjustment."""

    def __init__(self) -> None:
        self.score = 50.0
        self.why: list[str] = []

    def add(self, points: float, reason: str) -> None:
        self.score += points
        self.why.append(f"{'+' if points >= 0 else ''}{points:g}: {reason}")


@dataclass
class Recommendation:
    name: str
    score: float
    reasons: list[str] = field(default_factory=list)


def recommend_models(
    profile: DataProfile, include_optional: bool = True, top: int | None = None
) -> list[Recommendation]:
    """Rank catalog models for this data. Every point added or removed carries a reason."""
    out: list[Recommendation] = []
    n, p = profile.n_rows, max(profile.n_features, 1)
    for spec in catalog.available_models(profile.task, include_optional, include_baseline=False):
        tally = _Tally()
        add = tally.add
        fam = spec.family

        if n < 500:
            if fam in ("linear", "bayes"):
                add(15, "small data favours simple, regularized models")
            if fam in ("boosting", "neural"):
                add(-15, "small data makes boosting and neural nets overfit")
            if fam in ("neighbors", "svm"):
                add(8, "works well on small data")
        elif n > 50_000:
            if fam in ("svm", "neighbors"):
                add(-20, "slow on this many rows")
            if spec.name in ("Hist Gradient Boosting", "LightGBM", "XGBoost"):
                add(15, "scales to large data")
            if fam == "linear":
                add(5, "scales cheaply")
        if profile.nonlinear_gain > 0.03:
            if fam in ("ensemble", "boosting", "neural", "svm"):
                add(
                    15,
                    f"a quick boosted model beat a linear one by {profile.nonlinear_gain:.2f}: the signal is non-linear",
                )
            if fam == "linear":
                add(-10, "linear models miss the non-linear signal found in a quick test")
        elif profile.nonlinear_gain < 0.005:
            if fam == "linear":
                add(12, "a linear model is already as good as boosting in a quick test")
        if p / n > 0.1 and spec.name.startswith(("Lasso", "ElasticNet", "Ridge")):
            add(12, "many features relative to rows: regularization helps")
        if profile.imbalance_ratio and profile.imbalance_ratio > 4:
            if fam in ("ensemble", "boosting"):
                add(
                    8,
                    f"class imbalance {profile.imbalance_ratio:.0f}:1: tree ensembles cope well (use class weights)",
                )
            if fam == "neighbors":
                add(-8, "nearest neighbours favour the majority class")
        if profile.n_categorical > profile.n_numeric and fam in ("tree", "ensemble", "boosting"):
            add(8, "mostly categorical features suit tree models")
        if profile.high_cardinality and fam in ("linear", "neighbors", "svm"):
            add(-6, "high-cardinality categories inflate one-hot features")
        if profile.missing_pct > 10 and spec.name in (
            "Hist Gradient Boosting",
            "LightGBM",
            "XGBoost",
        ):
            add(6, "handles missing values natively")
        if profile.target_skew and abs(profile.target_skew) > 2 and spec.name.startswith("Huber"):
            add(8, "a skewed target is outlier-prone; Huber is robust")
        if spec.card.interpretability >= 4:
            add(3, "easy to explain")
        out.append(
            Recommendation(
                spec.name,
                round(tally.score, 1),
                tally.why or ["no special signal for or against"],
            )
        )
    out.sort(key=lambda r: r.score, reverse=True)
    return out[:top] if top else out


# --------------------------------------------------------------------------- race
@dataclass
class RaceResult:
    table: pd.DataFrame
    fold_scores: dict[str, np.ndarray]
    task: str
    scoring: str
    n_train_per_fold: int
    n_test_per_fold: int


def _pipe(
    X: pd.DataFrame, model: str, task: str, seed: int, params: dict[str, Any] | None = None
) -> Pipeline:
    return Pipeline(
        [("prep", build_preprocessor(X)), ("model", catalog.build(model, task, seed, params))]
    )


def race(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    models: list[str],
    task: str | None = None,
    cv_strategy: str | None = None,
    n_splits: int = 5,
    n_repeats: int = 2,
    scoring: str | None = None,
    seed: int = 42,
    max_rows: int = 20000,
    time_budget_s: float | None = None,
    groups: pd.Series | None = None,
) -> RaceResult:
    """Cross-validate every model on identical folds; report score, spread, overfit gap and time."""
    X, y = _prepare(df, target, features, max_rows, seed)
    if cv_strategy == "timeseries":
        X, y = df.loc[X.index.sort_values(), features], df.loc[X.index.sort_values(), target]
    task = task or infer_task(y)
    scoring = scoring or default_scoring(task)
    if scoring not in SCORING_CHOICES[task]:
        raise ModelingError(f"'{scoring}' is not available for {task}.")
    strategy = cv_strategy or default_cv_strategy(task)
    splitter = make_cv(strategy, y, n_splits, n_repeats, seed)
    grp = groups.loc[X.index] if groups is not None else None
    if strategy == "group" and grp is None:
        raise ModelingError("Group K-fold needs a group column.")
    folds = list(splitter.split(X, y, grp))
    rows, fold_scores = [], {}
    started = time.perf_counter()
    for name in models:
        if time_budget_s and time.perf_counter() - started > time_budget_s:
            rows.append({"model": name, "status": "skipped (time budget)"})
            continue
        spec = catalog.get(name)
        t0 = time.perf_counter()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = cross_validate(
                    _pipe(X, name, task, seed),
                    X,
                    y,
                    cv=folds,
                    scoring=scoring,
                    return_train_score=True,
                    error_score="raise",
                )
        except Exception as err:
            rows.append(
                {"model": name, "family": spec.family, "status": f"failed: {str(err)[:80]}"}
            )
            continue
        test, train = res["test_score"], res["train_score"]
        fold_scores[name] = test
        rows.append(
            {
                "model": name,
                "family": spec.family,
                "cv_mean": float(test.mean()),
                "cv_std": float(test.std(ddof=1)) if len(test) > 1 else 0.0,
                "train_mean": float(train.mean()),
                "gap": float(train.mean() - test.mean()),
                "seconds": round(time.perf_counter() - t0, 2),
                "folds": len(test),
                "status": "ok",
            }
        )
    table = pd.DataFrame(rows)
    if "cv_mean" in table:
        table = table.sort_values("cv_mean", ascending=False, na_position="last").reset_index(
            drop=True
        )
        table["overfit"] = table["gap"].map(
            lambda g: (
                "" if pd.isna(g) else ("high" if g > 0.15 else ("moderate" if g > 0.05 else "low"))
            )
        )
    n_tr = len(folds[0][0]) if folds else 0
    n_te = len(folds[0][1]) if folds else 0
    return RaceResult(table, fold_scores, task, scoring, n_tr, n_te)


def one_se_choice(table: pd.DataFrame) -> tuple[str, str]:
    """Simplest model within one standard error of the best (the 1-SE rule)."""
    ok = table[table["status"] == "ok"].dropna(subset=["cv_mean"])
    if ok.empty:
        raise ModelingError("No model finished successfully.")
    best = ok.iloc[0]
    se = best["cv_std"] / np.sqrt(max(best["folds"], 1))
    within = ok[ok["cv_mean"] >= best["cv_mean"] - se].copy()
    within["complexity"] = within["model"].map(catalog.complexity_rank)
    pick = within.sort_values(["complexity", "cv_mean"], ascending=[True, False]).iloc[0]
    if pick["model"] == best["model"]:
        why = f"{best['model']} is both the top scorer and the simplest within one standard error ({se:.4f})."
    else:
        why = (
            f"{pick['model']} scores {pick['cv_mean']:.4f}, within one standard error ({se:.4f}) of the best "
            f"({best['model']}: {best['cv_mean']:.4f}), and is simpler or easier to explain."
        )
    return str(pick["model"]), why


# --------------------------------------------------------------------------- nested CV
@dataclass
class NestedResult:
    model: str
    outer_scores: np.ndarray
    mean: float
    std: float
    best_params: list[dict[str, Any]]
    non_nested_best: float
    optimism: float


def nested_cv(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    model: str,
    task: str | None = None,
    outer: int = 5,
    inner: int = 3,
    n_iter: int = 8,
    scoring: str | None = None,
    seed: int = 42,
    max_rows: int = 5000,
) -> NestedResult:
    """Honest performance of 'tune then evaluate': tuning happens inside every outer fold."""
    from sklearn.model_selection import RandomizedSearchCV

    X, y = _prepare(df, target, features, max_rows, seed)
    task = task or infer_task(y)
    scoring = scoring or default_scoring(task)
    space = {f"model__{k}": v for k, v in catalog.param_space(model, task).items()}
    if not space:
        raise ModelingError(f"'{model}' has no tunable hyperparameters.")
    outer_cv = make_cv(default_cv_strategy(task), y, outer, seed=seed)
    inner_cv = make_cv(default_cv_strategy(task), y, inner, seed=seed + 1)
    scores, params, best_inner = [], [], []
    for train_idx, test_idx in outer_cv.split(X, y):
        search = RandomizedSearchCV(
            _pipe(X, model, task, seed),
            space,
            n_iter=n_iter,
            cv=inner_cv,
            scoring=scoring,
            random_state=seed,
            refit=True,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            search.fit(X.iloc[train_idx], y.iloc[train_idx])
        scorer = search.scorer_
        scores.append(float(scorer(search.best_estimator_, X.iloc[test_idx], y.iloc[test_idx])))
        params.append({k.removeprefix("model__"): v for k, v in search.best_params_.items()})
        best_inner.append(float(search.best_score_))
    arr = np.asarray(scores)
    return NestedResult(
        model,
        arr,
        float(arr.mean()),
        float(arr.std(ddof=1)),
        params,
        float(np.mean(best_inner)),
        float(np.mean(best_inner) - arr.mean()),
    )


# --------------------------------------------------------------------------- comparing models
def compare_models(a: np.ndarray, b: np.ndarray, n_train: int, n_test: int) -> dict[str, Any]:
    """Is model A really better than B? Corrected resampled t-test (Nadeau & Bengio) + Wilcoxon."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) != len(b) or len(a) < 3:
        raise ModelingError("Need the same number (>= 3) of fold scores for both models.")
    diff = a - b
    k = len(diff)
    var = diff.var(ddof=1)
    mean = float(diff.mean())
    if var == 0:
        t, p = (0.0, 1.0) if mean == 0 else (np.inf, 0.0)
    else:
        t = mean / np.sqrt((1 / k + n_test / n_train) * var)
        p = float(2 * stats.t.sf(abs(t), k - 1))
    try:
        w = float(stats.wilcoxon(diff).pvalue) if np.any(diff != 0) and k >= 5 else None
    except ValueError:
        w = None
    verdict = "no reliable difference"
    if p < 0.05:
        verdict = "A is better" if mean > 0 else "B is better"
    return {"mean_diff": mean, "t": float(t), "p_value": p, "wilcoxon_p": w, "verdict": verdict}


def pairwise_table(result: RaceResult) -> pd.DataFrame:
    """Matrix of corrected-t p-values between every pair of finished models."""
    names = list(result.fold_scores)
    mat = pd.DataFrame(np.nan, index=names, columns=names)
    for i in names:
        for j in names:
            if i == j:
                continue
            mat.loc[i, j] = compare_models(
                result.fold_scores[i],
                result.fold_scores[j],
                result.n_train_per_fold,
                result.n_test_per_fold,
            )["p_value"]
    return mat


# --------------------------------------------------------------------------- curves
def learning_curve_table(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    model: str,
    task: str | None = None,
    sizes: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0),
    cv: int = 5,
    scoring: str | None = None,
    seed: int = 42,
    max_rows: int = 8000,
) -> pd.DataFrame:
    """Train/validation score by training-set size: is more data worth collecting?"""
    X, y = _prepare(df, target, features, max_rows, seed)
    task = task or infer_task(y)
    splitter = make_cv(default_cv_strategy(task), y, cv, seed=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sz, train, test = learning_curve(
            _pipe(X, model, task, seed),
            X,
            y,
            cv=splitter,
            train_sizes=list(sizes),
            scoring=scoring or default_scoring(task),
        )
    return pd.DataFrame(
        {
            "train_size": sz,
            "train_mean": train.mean(1),
            "val_mean": test.mean(1),
            "val_std": test.std(1, ddof=1),
        }
    )


def validation_curve_table(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    model: str,
    param: str,
    values: list[Any],
    task: str | None = None,
    cv: int = 5,
    scoring: str | None = None,
    seed: int = 42,
    max_rows: int = 8000,
) -> pd.DataFrame:
    """Train/validation score as one hyperparameter changes."""
    X, y = _prepare(df, target, features, max_rows, seed)
    task = task or infer_task(y)
    splitter = make_cv(default_cv_strategy(task), y, cv, seed=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        train, test = validation_curve(
            _pipe(X, model, task, seed),
            X,
            y,
            param_name=f"model__{param}",
            param_range=values,
            cv=splitter,
            scoring=scoring or default_scoring(task),
        )
    return pd.DataFrame(
        {
            "value": [str(v) for v in values],
            "train_mean": train.mean(1),
            "val_mean": test.mean(1),
            "val_std": test.std(1, ddof=1),
        }
    )


def split_summary(df: pd.DataFrame, target: str, seed: int = 42) -> dict[str, Any]:
    """Quick facts the page shows next to the CV settings."""
    y = df[target].dropna()
    task = infer_task(y)
    train, test = train_test_split(y, test_size=0.2, random_state=seed)
    return {"task": task, "rows": len(y), "train_rows": len(train), "test_rows": len(test)}


__all__ = [
    "BINARY",
    "MULTICLASS",
    "REGRESSION",
    "CV_STRATEGIES",
    "DataProfile",
    "NestedResult",
    "RaceResult",
    "Recommendation",
    "compare_models",
    "default_cv_strategy",
    "learning_curve_table",
    "make_cv",
    "nested_cv",
    "one_se_choice",
    "pairwise_table",
    "profile_for_selection",
    "race",
    "recommend_models",
    "split_summary",
    "validation_curve_table",
]
