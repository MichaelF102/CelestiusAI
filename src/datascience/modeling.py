"""Leakage-safe model comparison, hyperparameter tuning, export, and scoring.

All preprocessing lives inside an sklearn ``Pipeline`` so imputation, scaling and encoding
are fitted on training folds only.
"""

from __future__ import annotations

import io
import time
import warnings
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.core.code_snippets import make_snippet

REGRESSION = "regression"
BINARY = "binary_classification"
MULTICLASS = "multiclass_classification"

SCORING_CHOICES = {
    REGRESSION: {
        "r2": "R²",
        "neg_mean_absolute_error": "MAE (negated)",
        "neg_root_mean_squared_error": "RMSE (negated)",
    },
    BINARY: {"f1_macro": "F1 (macro)", "roc_auc": "ROC AUC", "accuracy": "Accuracy"},
    MULTICLASS: {"f1_macro": "F1 (macro)", "accuracy": "Accuracy"},
}
MIN_ROWS = 20


class ModelingError(ValueError):
    """User-facing modeling problem (too little data, bad target, ...)."""


# --------------------------------------------------------------------------- task & data
def infer_task(y: pd.Series) -> str:
    y = y.dropna()
    n_unique = y.nunique()
    if not pd.api.types.is_numeric_dtype(y) or pd.api.types.is_bool_dtype(y):
        return BINARY if n_unique <= 2 else MULTICLASS
    if n_unique <= 2:
        return BINARY
    if pd.api.types.is_integer_dtype(y) and n_unique <= 10:
        return MULTICLASS
    return REGRESSION


def _prepare(
    df: pd.DataFrame, target: str, features: list[str], max_rows: int, seed: int
) -> tuple[pd.DataFrame, pd.Series]:
    if target in features:
        raise ModelingError("The target cannot also be a feature.")
    if not features:
        raise ModelingError("Select at least one feature.")
    data = df[[target, *features]].dropna(subset=[target])
    if len(data) < MIN_ROWS:
        raise ModelingError(
            f"Need at least {MIN_ROWS} rows with a target value (found {len(data)})."
        )
    if len(data) > max_rows:
        data = data.sample(n=max_rows, random_state=seed)
    return data[features], data[target]


def build_preprocessor(X: pd.DataFrame, scale: bool = True) -> ColumnTransformer:
    num_cols = X.select_dtypes("number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    num_steps: list[tuple[str, Any]] = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        [
            ("num", Pipeline(num_steps), num_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="infrequent_if_exist",
                                max_categories=30,
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                cat_cols,
            ),
        ]
    )


def candidate_estimators(task: str, seed: int = 42) -> dict[str, Any]:
    if task == REGRESSION:
        return {
            "Baseline (mean)": DummyRegressor(strategy="mean"),
            "Linear Regression": LinearRegression(),
            "Ridge": Ridge(alpha=1.0),
            "Random Forest": RandomForestRegressor(n_estimators=150, random_state=seed, n_jobs=-1),
            "Hist Gradient Boosting": HistGradientBoostingRegressor(random_state=seed),
        }
    return {
        "Baseline (most frequent)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=150, random_state=seed, n_jobs=-1),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(random_state=seed),
    }


def make_estimator(name: str, task: str, seed: int = 42) -> Any:
    """A core candidate by its familiar name, or any model from the full catalog."""
    core = candidate_estimators(task, seed)
    if name in core:
        return core[name]
    from src.datascience import catalog

    return catalog.build(name, task, seed)


def space_for(name: str, task: str) -> dict[str, list[Any]]:
    """Pipeline-ready ('model__x') hyperparameter space for a core or catalog model."""
    if name in SEARCH_SPACES:
        return SEARCH_SPACES[name]
    from src.datascience import catalog

    try:
        return {f"model__{k}": v for k, v in catalog.param_space(name, task).items()}
    except KeyError:
        return {}


SEARCH_SPACES: dict[str, dict[str, list[Any]]] = {
    "Ridge": {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
    "Logistic Regression": {"model__C": [0.01, 0.1, 1.0, 10.0, 100.0]},
    "Random Forest": {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [None, 5, 10, 20],
        "model__min_samples_leaf": [1, 2, 5],
    },
    "Hist Gradient Boosting": {
        "model__learning_rate": [0.03, 0.05, 0.1, 0.2],
        "model__max_depth": [None, 3, 6, 10],
        "model__max_iter": [100, 200, 300],
        "model__l2_regularization": [0.0, 0.1, 1.0],
    },
}


def _cv_splitter(task: str, y: pd.Series, cv: int, seed: int) -> Any:
    if task == REGRESSION:
        folds = max(2, min(cv, len(y) // 5))
        return KFold(n_splits=folds, shuffle=True, random_state=seed)
    smallest = int(y.value_counts().min())
    if smallest < 2:
        raise ModelingError("Every class needs at least 2 rows for cross-validation.")
    return StratifiedKFold(n_splits=max(2, min(cv, smallest)), shuffle=True, random_state=seed)


def default_scoring(task: str) -> str:
    return next(iter(SCORING_CHOICES[task]))


# --------------------------------------------------------------------------- leaderboard
def leaderboard(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    task: str | None = None,
    cv: int = 5,
    scoring: str | None = None,
    seed: int = 42,
    max_rows: int = 20000,
    models: list[str] | None = None,
) -> tuple[pd.DataFrame, str]:
    """Cross-validate the candidate models; returns (table sorted best-first, task)."""
    X, y = _prepare(df, target, features, max_rows, seed)
    task = task or infer_task(y)
    scoring = scoring or default_scoring(task)
    splitter = _cv_splitter(task, y, cv, seed)
    rows = []
    for name in models or list(candidate_estimators(task, seed)):
        pipe = Pipeline(
            [("prep", build_preprocessor(X)), ("model", make_estimator(name, task, seed))]
        )
        start = time.perf_counter()
        scores = cross_validate(pipe, X, y, cv=splitter, scoring=scoring, error_score="raise")
        rows.append(
            {
                "model": name,
                "cv_mean": float(np.mean(scores["test_score"])),
                "cv_std": float(np.std(scores["test_score"])),
                "folds": len(scores["test_score"]),
                "seconds": round(time.perf_counter() - start, 2),
            }
        )
    table = pd.DataFrame(rows).sort_values("cv_mean", ascending=False).reset_index(drop=True)
    return table, task


# --------------------------------------------------------------------------- tuning
@dataclass
class FittedModel:
    name: str
    task: str
    target: str
    features: list[str]
    estimator_name: str
    pipeline: Pipeline
    metrics: dict[str, float]
    params: dict[str, Any] = field(default_factory=dict)
    feature_info: dict[str, dict[str, Any]] = field(default_factory=dict)
    classes: list[Any] | None = None
    trained_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))
    rows_used: int = 0
    holdout: tuple[pd.DataFrame, pd.Series] | None = None  # (X_test, y_test), never exported


@dataclass
class TuningResult:
    model: FittedModel
    best_params: dict[str, Any]
    best_cv_score: float
    default_cv_score: float
    test_metrics: dict[str, float]
    results: pd.DataFrame
    scoring: str


def describe_features(X: pd.DataFrame) -> dict[str, dict[str, Any]]:
    info: dict[str, dict[str, Any]] = {}
    for col in X.columns:
        s = X[col]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            info[col] = {
                "kind": "numeric",
                "min": float(s.min()),
                "max": float(s.max()),
                "default": float(s.median()),
            }
        else:
            top = s.astype(str).value_counts().head(30)
            info[col] = {"kind": "categorical", "choices": list(top.index), "default": top.index[0]}
    return info


def holdout_metrics(
    task: str, pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, float]:
    pred = pipeline.predict(X_test)
    if task == REGRESSION:
        return {
            "R2": float(r2_score(y_test, pred)),
            "MAE": float(mean_absolute_error(y_test, pred)),
            "RMSE": float(mean_squared_error(y_test, pred) ** 0.5),
        }
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1_macro": float(f1_score(y_test, pred, average="macro")),
    }
    if task == BINARY:
        try:
            proba = pipeline.predict_proba(X_test)[:, 1]
            metrics["roc_auc"] = float(roc_auc_score(y_test, proba))
        except Exception:  # single-class test split etc.
            pass
    return metrics


def optuna_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("optuna") is not None


def _optuna_search(
    pipe: Pipeline,
    space: dict[str, list[Any]],
    splitter: Any,
    scoring: str,
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int,
    seed: int,
) -> tuple[Pipeline, dict[str, Any], float, pd.DataFrame]:
    """Bayesian optimization (TPE) over the list-valued space with Optuna."""
    if not optuna_available():
        raise ModelingError("Bayesian search needs Optuna (pip install optuna).")
    import optuna
    from sklearn.base import clone

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    rows: list[dict[str, Any]] = []

    def objective(trial: Any) -> float:
        params = {
            name: values[trial.suggest_int(f"{name}__idx", 0, len(values) - 1)]
            for name, values in space.items()
        }
        est = clone(pipe).set_params(**params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = cross_validate(est, X, y, cv=splitter, scoring=scoring)["test_score"]
        rows.append(
            {
                **{k.removeprefix("model__"): v for k, v in params.items()},
                "mean_test_score": float(scores.mean()),
                "std_test_score": float(scores.std()),
            }
        )
        return float(scores.mean())

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=max(4, n_trials), show_progress_bar=False)
    best = {name: values[study.best_params[f"{name}__idx"]] for name, values in space.items()}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        best_estimator = clone(pipe).set_params(**best).fit(X, y)
    results = (
        pd.DataFrame(rows).sort_values("mean_test_score", ascending=False).reset_index(drop=True)
    )
    results["rank_test_score"] = (
        results["mean_test_score"].rank(ascending=False, method="first").astype(int)
    )
    return best_estimator, best, float(study.best_value), results


def tune_model(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    model_name: str,
    task: str | None = None,
    method: str = "random",
    n_iter: int = 12,
    cv: int = 5,
    scoring: str | None = None,
    test_size: float = 0.2,
    seed: int = 42,
    max_rows: int = 20000,
    param_space: dict[str, list[Any]] | None = None,
) -> TuningResult:
    """Tune one model with CV on the training split, then score the untouched test split."""
    X, y = _prepare(df, target, features, max_rows, seed)
    task = task or infer_task(y)
    scoring = scoring or default_scoring(task)
    space = param_space or space_for(model_name, task)
    if not space:
        raise ModelingError(f"'{model_name}' has no tunable hyperparameters.")
    try:
        estimator = make_estimator(model_name, task, seed)
    except (KeyError, ImportError, ValueError) as err:
        raise ModelingError(str(err)) from err

    stratify = y if task != REGRESSION and y.value_counts().min() >= 2 else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=stratify
    )
    splitter = _cv_splitter(task, y_tr, cv, seed)
    pipe = Pipeline([("prep", build_preprocessor(X_tr)), ("model", estimator)])

    default_scores = cross_validate(pipe, X_tr, y_tr, cv=splitter, scoring=scoring)["test_score"]

    grid_size = int(np.prod([len(v) for v in space.values()]))
    if method == "optuna":
        best_estimator, best_params, best_score, results = _optuna_search(
            pipe, space, splitter, scoring, X_tr, y_tr, n_iter, seed
        )
    else:
        if method == "halving":
            from sklearn.experimental import enable_halving_search_cv  # noqa: F401
            from sklearn.model_selection import HalvingRandomSearchCV

            search: Any = HalvingRandomSearchCV(
                pipe,
                space,
                n_candidates=max(n_iter, 6),
                factor=3,
                cv=splitter,
                scoring=scoring,
                refit=True,
                random_state=seed,
                min_resources="smallest" if len(y_tr) < 600 else "exhaust",
            )
        elif method == "grid" or grid_size <= n_iter:
            search = GridSearchCV(pipe, space, cv=splitter, scoring=scoring, refit=True)
        else:
            search = RandomizedSearchCV(
                pipe,
                space,
                n_iter=n_iter,
                cv=splitter,
                scoring=scoring,
                refit=True,
                random_state=seed,
            )
        search.fit(X_tr, y_tr)
        best_estimator, best_params, best_score = (
            search.best_estimator_,
            search.best_params_,
            search.best_score_,
        )
        results = (
            pd.DataFrame(search.cv_results_).sort_values("rank_test_score").reset_index(drop=True)
        )
        keep = [c for c in results.columns if c.startswith("param_")] + [
            "mean_test_score",
            "std_test_score",
            "rank_test_score",
        ]
        results = results[keep].rename(columns=lambda c: c.removeprefix("param_model__"))

    test_metrics = holdout_metrics(task, best_estimator, X_te, y_te)
    fitted = FittedModel(
        name=f"{model_name} (tuned)",
        task=task,
        target=target,
        features=list(features),
        estimator_name=model_name,
        pipeline=best_estimator,
        metrics=test_metrics,
        params={k.removeprefix("model__"): v for k, v in best_params.items()},
        feature_info=describe_features(X),
        classes=list(best_estimator.classes_) if task != REGRESSION else None,
        rows_used=len(X),
        holdout=(X_te, y_te),
    )
    return TuningResult(
        model=fitted,
        best_params=fitted.params,
        best_cv_score=float(best_score),
        default_cv_score=float(np.mean(default_scores)),
        test_metrics=test_metrics,
        results=results,
        scoring=scoring,
    )


def fit_model(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    model_name: str,
    task: str | None = None,
    params: dict[str, Any] | None = None,
    test_size: float = 0.2,
    seed: int = 42,
    max_rows: int = 20000,
) -> FittedModel:
    """Fit one named model (optionally with params) and score a holdout split."""
    X, y = _prepare(df, target, features, max_rows, seed)
    task = task or infer_task(y)
    estimator = make_estimator(model_name, task, seed)
    if params:
        estimator.set_params(**params)
    stratify = y if task != REGRESSION and y.value_counts().min() >= 2 else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=stratify
    )
    pipe = Pipeline([("prep", build_preprocessor(X_tr)), ("model", estimator)]).fit(X_tr, y_tr)
    return FittedModel(
        name=model_name,
        task=task,
        target=target,
        features=list(features),
        estimator_name=model_name,
        pipeline=pipe,
        metrics=holdout_metrics(task, pipe, X_te, y_te),
        params=dict(params or {}),
        feature_info=describe_features(X),
        classes=list(pipe.classes_) if task != REGRESSION else None,
        rows_used=len(X),
        holdout=(X_te, y_te),
    )


# --------------------------------------------------------------------------- export & scoring
def export_model_bytes(model: FittedModel) -> bytes:
    """Serialize the fitted pipeline + metadata with joblib (load only from trusted sources)."""
    meta = {
        "name": model.name,
        "task": model.task,
        "target": model.target,
        "features": model.features,
        "feature_info": model.feature_info,
        "classes": model.classes,
        "metrics": model.metrics,
        "params": model.params,
        "trained_at": model.trained_at,
        "estimator_name": model.estimator_name,
        "rows_used": model.rows_used,
    }
    buffer = io.BytesIO()
    joblib.dump({"pipeline": model.pipeline, "meta": meta}, buffer)
    return buffer.getvalue()


def model_from_bundle(bundle: dict[str, Any]) -> FittedModel:
    """Rebuild a FittedModel from an exported joblib bundle (only load files you trust)."""
    meta = bundle["meta"]
    return FittedModel(
        name=meta["name"],
        task=meta["task"],
        target=meta["target"],
        features=list(meta["features"]),
        estimator_name=meta.get("estimator_name", meta["name"]),
        pipeline=bundle["pipeline"],
        metrics=dict(meta.get("metrics", {})),
        params=dict(meta.get("params", {})),
        feature_info=dict(meta.get("feature_info", {})),
        classes=meta.get("classes"),
        trained_at=meta.get("trained_at", ""),
        rows_used=int(meta.get("rows_used", 0)),
    )


def validate_scoring_frame(model: FittedModel, df: pd.DataFrame) -> dict[str, Any]:
    """Pre-scoring checks: schema, out-of-range numbers, unseen categories, and missing values."""
    issues: list[tuple[str, str]] = []
    missing = [c for c in model.features if c not in df.columns]
    if missing:
        issues.append(("error", f"Missing required column(s): {', '.join(missing)}."))
    extra = [c for c in df.columns if c not in model.features and c != model.target]
    if extra:
        issues.append(
            (
                "info",
                f"{len(extra)} column(s) are not used by the model: {', '.join(extra[:6])}{'...' if len(extra) > 6 else ''}.",
            )
        )
    range_rows, unseen_rows, null_rows = [], [], []
    for col in (c for c in model.features if c in df.columns):
        info = model.feature_info.get(col, {})
        series = df[col]
        share_null = float(series.isna().mean())
        if share_null:
            null_rows.append({"feature": col, "missing_share": share_null})
        if info.get("kind") == "numeric":
            nums = pd.to_numeric(series, errors="coerce")
            bad_type = int(series.notna().sum() - nums.notna().sum())
            if bad_type:
                issues.append(("error", f"'{col}' has {bad_type} non-numeric value(s)."))
            outside = float(((nums < info["min"]) | (nums > info["max"])).mean())
            if outside > 0:
                range_rows.append(
                    {
                        "feature": col,
                        "share_outside_training_range": outside,
                        "new_min": float(nums.min()),
                        "new_max": float(nums.max()),
                        "train_min": info["min"],
                        "train_max": info["max"],
                    }
                )
        elif info.get("kind") == "categorical":
            known = set(info.get("choices", []))
            new_vals = sorted(set(series.dropna().astype(str)) - known)
            if new_vals and len(known) < 30:
                unseen_rows.append(
                    {
                        "feature": col,
                        "unseen_values": ", ".join(new_vals[:8]),
                        "rows": int(series.astype(str).isin(new_vals).sum()),
                    }
                )
    out_of_range, unseen, nulls = (
        pd.DataFrame(range_rows),
        pd.DataFrame(unseen_rows),
        pd.DataFrame(null_rows),
    )
    if len(out_of_range):
        worst = out_of_range.sort_values("share_outside_training_range", ascending=False).iloc[0]
        issues.append(
            (
                "warning",
                f"{len(out_of_range)} feature(s) have values outside the training range (worst: '{worst['feature']}', {worst['share_outside_training_range']:.0%} of rows). Predictions there are extrapolations.",
            )
        )
    if len(unseen):
        issues.append(
            (
                "warning",
                f"{len(unseen)} categorical feature(s) contain values never seen in training; they are treated as unknown.",
            )
        )
    if len(nulls):
        issues.append(
            (
                "info",
                f"Missing values in {len(nulls)} feature(s) will be imputed the way training data was.",
            )
        )
    return {
        "ok": not any(level == "error" for level, _ in issues),
        "issues": issues,
        "out_of_range": out_of_range,
        "unseen_categories": unseen,
        "missing_values": nulls,
        "missing_columns": missing,
    }


def score_frame(model: FittedModel, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Predict for every row; returns (frame with predictions, warnings)."""
    warnings: list[str] = []
    missing = [c for c in model.features if c not in df.columns]
    X = df.copy()
    for col in missing:
        X[col] = np.nan
    if missing:
        warnings.append(f"Missing columns filled with blanks (imputed): {', '.join(missing)}")
    X = X[model.features]
    out = df.copy()
    out[f"predicted_{model.target}"] = model.pipeline.predict(X)
    if model.task != REGRESSION and hasattr(model.pipeline, "predict_proba"):
        proba = model.pipeline.predict_proba(X)
        for idx, label in enumerate(model.classes or []):
            out[f"proba_{label}"] = proba[:, idx]
    return out, warnings


# --------------------------------------------------------------------------- code
def tuning_code(
    target: str,
    features: list[str],
    task: str,
    model_name: str,
    space: dict[str, list[Any]],
    method: str,
    n_iter: int,
    cv: int,
    scoring: str,
    seed: int = 42,
    **load: Any,
) -> str:
    kind = "Regressor" if task == REGRESSION else "Classifier"
    estimator = {
        "Ridge": "Ridge()",
        "Logistic Regression": "LogisticRegression(max_iter=1000)",
        "Random Forest": f"RandomForest{kind}(random_state={seed}, n_jobs=-1)",
        "Hist Gradient Boosting": f"HistGradientBoosting{kind}(random_state={seed})",
    }[model_name]
    imports = [
        "import pandas as pd",
        "import numpy as np",
        "from sklearn.compose import ColumnTransformer",
        "from sklearn.impute import SimpleImputer",
        "from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score, GridSearchCV, RandomizedSearchCV",
        "from sklearn.pipeline import Pipeline",
        "from sklearn.preprocessing import OneHotEncoder, StandardScaler",
        "from sklearn.linear_model import Ridge, LogisticRegression",
        "from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, HistGradientBoostingRegressor, HistGradientBoostingClassifier",
    ]
    cv_cls = "KFold" if task == REGRESSION else "StratifiedKFold"
    stratify = "" if task == REGRESSION else ", stratify=y"
    search = (
        f"GridSearchCV(pipe, space, cv=cv, scoring={scoring!r})"
        if method == "grid"
        else f"RandomizedSearchCV(pipe, space, n_iter={n_iter}, cv=cv, scoring={scoring!r}, random_state=SEED)"
    )
    body = f"""SEED = {seed}
target, features = {target!r}, {list(features)!r}
data = df[[target, *features]].dropna(subset=[target])
X, y = data[features], data[target]
num_cols = X.select_dtypes("number").columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

preprocess = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num_cols),
    ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED{stratify})
pipe = Pipeline([("prep", preprocess), ("model", {estimator})])
cv = {cv_cls}(n_splits={cv}, shuffle=True, random_state=SEED)

# %%
space = {space!r}
baseline = cross_val_score(pipe, X_train, y_train, cv=cv, scoring={scoring!r}).mean()
search = {search}
search.fit(X_train, y_train)
print("default CV score:", round(baseline, 4))
print("tuned CV score:  ", round(search.best_score_, 4))
print("best params:", search.best_params_)
pd.DataFrame(search.cv_results_).sort_values("rank_test_score")[["mean_test_score", "std_test_score", "params"]].head(10)

# %%
# Final check on the untouched test split
best = search.best_estimator_
print("test score:", round(best.score(X_test, y_test), 4))

# %%
# Save the tuned model for later scoring
import joblib
joblib.dump(best, "tuned_model.joblib")
"""
    return make_snippet(f"Tune {model_name}", body, imports=imports, **load)


def scoring_script(model: FittedModel) -> str:
    """Standalone code that loads an exported model and scores a new file."""
    meta_features = list(model.features)
    body = f"""import joblib

bundle = joblib.load("model.joblib")        # only load files you trust (joblib uses pickle)
pipeline, meta = bundle["pipeline"], bundle["meta"]
print("Model:", meta["name"], "| target:", meta["target"])

new_data = pd.read_csv("new_data.csv")       # needs these columns: {meta_features}
predictions = pipeline.predict(new_data[meta["features"]])
result = new_data.assign(**{{f"predicted_{{meta['target']}}": predictions}})
result.head()
"""
    return make_snippet(f"Score new data with {model.name}", body, include_load=False)


def permutation_importance_table(
    model: FittedModel, repeats: int = 5, seed: int = 42
) -> pd.DataFrame:
    """Held-out permutation importance (drop in score when a feature is shuffled)."""
    from sklearn.inspection import permutation_importance

    if model.holdout is None:
        raise ModelingError("This model has no stored holdout data.")
    X_te, y_te = model.holdout
    scoring = default_scoring(model.task)
    result = permutation_importance(
        model.pipeline, X_te, y_te, n_repeats=repeats, random_state=seed, scoring=scoring
    )
    return (
        pd.DataFrame(
            {
                "feature": X_te.columns,
                "importance": result.importances_mean,
                "std": result.importances_std,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def refit_on_all(
    model: FittedModel, df: pd.DataFrame, seed: int = 42, max_rows: int = 20000
) -> FittedModel:
    """Refit the same pipeline on every row (for deployment); keeps the honest holdout metrics."""
    from sklearn.base import clone

    X, y = _prepare(df, model.target, model.features, max_rows, seed)
    pipe = clone(model.pipeline).fit(X, y)
    return FittedModel(
        name=f"{model.name} [refit on all data]",
        task=model.task,
        target=model.target,
        features=list(model.features),
        estimator_name=model.estimator_name,
        pipeline=pipe,
        metrics=dict(model.metrics),
        params=dict(model.params),
        feature_info=describe_features(X),
        classes=list(pipe.classes_) if model.task != REGRESSION else None,
        rows_used=len(X),
        holdout=None,
    )
