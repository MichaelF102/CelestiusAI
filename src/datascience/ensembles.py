"""Ensemble methods: bagging, voting (hard/soft), stacking, with diversity analysis."""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    BaggingClassifier,
    BaggingRegressor,
    StackingClassifier,
    StackingRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.model_selection import cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline

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

ENSEMBLE_KINDS = {
    "bagging": "Bagging (many copies of one model on bootstrap samples)",
    "voting_hard": "Voting - majority vote (classification)",
    "voting_soft": "Voting - average of predictions / probabilities",
    "stacking": "Stacking (a meta-model learns how to combine the base models)",
}
DEFAULT_META = {
    "classification": "Linear / Logistic Regression",
    "regression": "Ridge / L2 Logistic",
}


def _slug(name: str, i: int) -> str:
    return f"{i}_{re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')}"


def build_ensemble(
    kind: str,
    base_models: list[str],
    task: str,
    seed: int = 42,
    meta: str | None = None,
    n_estimators: int = 10,
    weights: list[float] | None = None,
) -> Any:
    """An sklearn ensemble over catalog models (to be used as the final pipeline step)."""
    if kind not in ENSEMBLE_KINDS:
        raise ModelingError(f"Unknown ensemble kind '{kind}'.")
    is_reg = task == REGRESSION
    if not base_models:
        raise ModelingError("Pick at least one base model.")
    members = [(_slug(m, i), catalog.build(m, task, seed)) for i, m in enumerate(base_models)]
    if kind == "bagging":
        cls = BaggingRegressor if is_reg else BaggingClassifier
        return cls(estimator=members[0][1], n_estimators=n_estimators, random_state=seed, n_jobs=-1)
    if len(members) < 2:
        raise ModelingError("Voting and stacking need at least two base models.")
    if kind in ("voting_hard", "voting_soft"):
        if is_reg:
            if kind == "voting_hard":
                raise ModelingError("Hard voting only exists for classification; use soft/average.")
            return VotingRegressor(members, weights=weights)
        return VotingClassifier(
            members, voting="hard" if kind == "voting_hard" else "soft", weights=weights
        )
    final = catalog.build(
        meta or DEFAULT_META["regression" if is_reg else "classification"], task, seed
    )
    cls = StackingRegressor if is_reg else StackingClassifier
    return cls(members, final_estimator=final, cv=3, n_jobs=1)


@dataclass
class EnsembleSpec:
    name: str
    kind: str
    base_models: list[str]
    meta: str | None = None
    n_estimators: int = 10


@dataclass
class StudyResult:
    table: pd.DataFrame
    fold_scores: dict[str, np.ndarray]
    diversity: pd.DataFrame | None  # pairwise error correlation between base models
    disagreement: pd.DataFrame | None
    summary: dict[str, Any]


def ensemble_study(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    base_models: list[str],
    ensembles: list[EnsembleSpec],
    task: str | None = None,
    cv: int = 4,
    scoring: str | None = None,
    seed: int = 42,
    max_rows: int = 6000,
) -> StudyResult:
    """Cross-validate base models and ensembles on identical folds; measure the gain and diversity."""
    X, y = _prepare(df, target, features, max_rows, seed)
    task = task or infer_task(y)
    scoring = scoring or default_scoring(task)
    folds = list(make_cv(default_cv_strategy(task), y, cv, seed=seed).split(X, y))

    def pipe(est: Any) -> Pipeline:
        return Pipeline([("prep", build_preprocessor(X)), ("model", est)])

    rows, fold_scores, oof = [], {}, {}
    entries = [(m, "base", catalog.build(m, task, seed)) for m in base_models]
    for spec in ensembles:
        entries.append(
            (
                spec.name,
                spec.kind,
                build_ensemble(
                    spec.kind,
                    spec.base_models or base_models,
                    task,
                    seed,
                    spec.meta,
                    spec.n_estimators,
                ),
            )
        )
    for name, kind, est in entries:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = cross_validate(pipe(est), X, y, cv=folds, scoring=scoring, error_score="raise")
            if kind == "base":
                oof[name] = cross_val_predict(pipe(catalog.build(name, task, seed)), X, y, cv=folds)
        scores = res["test_score"]
        fold_scores[name] = scores
        rows.append(
            {
                "model": name,
                "type": kind,
                "cv_mean": float(scores.mean()),
                "cv_std": float(scores.std(ddof=1)),
                "fit_seconds": float(res["fit_time"].sum()),
            }
        )
    table = pd.DataFrame(rows)
    base_best = table[table["type"] == "base"]["cv_mean"].max()
    table["gain_vs_best_base"] = np.where(
        table["type"] == "base", np.nan, table["cv_mean"] - base_best
    )
    table = table.sort_values("cv_mean", ascending=False).reset_index(drop=True)

    diversity = disagreement = None
    if len(oof) >= 2:
        names = list(oof)
        if task == REGRESSION:
            errs = pd.DataFrame({n: np.asarray(y) - oof[n] for n in names})
            diversity = errs.corr()
            disagreement = pd.DataFrame(np.nan, index=names, columns=names)
        else:
            errs = pd.DataFrame({n: (oof[n] != np.asarray(y)).astype(float) for n in names})
            diversity = errs.corr()
            disagreement = pd.DataFrame(
                {a: {b: float(np.mean(oof[a] != oof[b])) for b in names} for a in names}
            )
    ens_rows = table[table["type"] != "base"]
    summary: dict[str, Any] = {
        "best_base": float(base_best),
        "best_ensemble": float(ens_rows["cv_mean"].max()) if len(ens_rows) else None,
        "mean_error_correlation": float(
            diversity.to_numpy()[np.triu_indices(len(diversity), 1)].mean()
        )
        if diversity is not None
        else None,
    }
    if summary["best_ensemble"] is not None:
        summary["gain"] = summary["best_ensemble"] - base_best
    return StudyResult(table, fold_scores, diversity, disagreement, summary)


def diversity_comment(summary: dict[str, Any]) -> str:
    corr = summary.get("mean_error_correlation")
    gain = summary.get("gain")
    if corr is None:
        return "Add at least two base models to measure diversity."
    tone = (
        "highly similar"
        if corr > 0.7
        else ("moderately similar" if corr > 0.4 else "fairly diverse")
    )
    advice = (
        "Combining them adds little: try models from different families (a linear model, a tree ensemble, a neighbour method)."
        if corr > 0.7
        else "That diversity is what lets ensembles beat their members."
    )
    gain_txt = f" Best ensemble vs best base model: {gain:+.4f}." if gain is not None else ""
    return f"The base models' mistakes are {tone} (average error correlation {corr:.2f}). {advice}{gain_txt}"


def ensemble_code_hint(kind: str) -> str:
    return {
        "bagging": "BaggingClassifier/Regressor(estimator=..., n_estimators=...)",
        "voting_hard": "VotingClassifier(estimators, voting='hard')",
        "voting_soft": "VotingClassifier/VotingRegressor(estimators, voting='soft')",
        "stacking": "StackingClassifier/Regressor(estimators, final_estimator=...)",
    }[kind]
