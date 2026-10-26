"""Classification toolkit: imbalance handling, thresholds, calibration, rich evaluation."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline

from src.datascience import catalog
from src.datascience.modeling import ModelingError, _prepare, build_preprocessor, infer_task

RESAMPLING = {
    "none": "No resampling",
    "class_weight": "Class weights (balanced)",
    "oversample": "Random oversampling of the minority",
    "undersample": "Random undersampling of the majority",
    "smote": "SMOTE (synthetic minority samples)",
}


class ResamplingClassifier(ClassifierMixin, BaseEstimator):
    """Resamples the *training* rows inside fit (so cross-validation folds stay untouched)."""

    def __init__(
        self, estimator: Any = None, strategy: str = "oversample", k: int = 5, seed: int = 42
    ) -> None:
        self.estimator = estimator
        self.strategy = strategy
        self.k = k
        self.seed = seed

    def fit(self, X: Any, y: Any) -> ResamplingClassifier:
        Xr, yr = resample(
            np.asarray(X, dtype=float), np.asarray(y), self.strategy, self.k, self.seed
        )
        self.estimator_ = clone(self.estimator).fit(Xr, yr)
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X: Any) -> Any:
        return self.estimator_.predict(X)

    def predict_proba(self, X: Any) -> Any:
        return self.estimator_.predict_proba(X)


def resample(
    X: np.ndarray, y: np.ndarray, strategy: str, k: int = 5, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Balance classes by over/undersampling or SMOTE interpolation."""
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2 or strategy in ("none", "class_weight"):
        return X, y
    if strategy == "undersample":
        target = counts.min()
        idx = np.concatenate(
            [rng.choice(np.flatnonzero(y == c), target, replace=False) for c in classes]
        )
    elif strategy == "oversample":
        target = counts.max()
        idx = np.concatenate(
            [rng.choice(np.flatnonzero(y == c), target, replace=True) for c in classes]
        )
    elif strategy == "smote":
        target = counts.max()
        parts_X, parts_y = [X], [y]
        for c, n in zip(classes, counts, strict=True):
            need = target - n
            if need <= 0:
                continue
            Xc = X[y == c]
            if len(Xc) < 2:
                extra = Xc[rng.choice(len(Xc), need)]
            else:
                kk = min(k, len(Xc) - 1)
                nn = NearestNeighbors(n_neighbors=kk + 1).fit(Xc)
                neighbours = nn.kneighbors(Xc, return_distance=False)[:, 1:]
                base = rng.integers(0, len(Xc), need)
                partner = neighbours[base, rng.integers(0, kk, need)]
                gap = rng.random((need, 1))
                extra = Xc[base] + gap * (Xc[partner] - Xc[base])
            parts_X.append(extra)
            parts_y.append(np.full(need, c, dtype=y.dtype))
        return np.vstack(parts_X), np.concatenate(parts_y)
    else:
        raise ModelingError(f"Unknown resampling strategy '{strategy}'.")
    return X[idx], y[idx]


def build_pipeline(
    X: pd.DataFrame, model: str, task: str, strategy: str = "none", seed: int = 42
) -> Pipeline:
    est = catalog.build(model, task, seed)
    if strategy == "class_weight":
        if "class_weight" not in est.get_params():
            raise ModelingError(f"{model} has no class_weight option; pick resampling instead.")
        est.set_params(class_weight="balanced")
    elif strategy in ("oversample", "undersample", "smote"):
        est = ResamplingClassifier(est, strategy, seed=seed)
    return Pipeline([("prep", build_preprocessor(X)), ("model", est)])


@dataclass
class Holdout:
    pipeline: Pipeline
    X_test: pd.DataFrame
    y_test: pd.Series
    proba: np.ndarray
    pred: np.ndarray
    classes: list[Any]
    positive: Any | None


def holdout_fit(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    model: str,
    strategy: str = "none",
    test_size: float = 0.25,
    seed: int = 42,
    positive: Any | None = None,
    max_rows: int = 30000,
) -> Holdout:
    X, y = _prepare(df, target, features, max_rows, seed)
    if infer_task(y) == "regression":
        raise ModelingError("The target looks continuous; this toolkit is for classification.")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y if y.value_counts().min() >= 2 else None,
    )
    task = infer_task(y)
    pipe = build_pipeline(X_tr, model, task, strategy, seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(X_tr, y_tr)
    classes = list(pipe.classes_)
    pos = positive if positive is not None else (classes[-1] if len(classes) == 2 else None)
    return Holdout(pipe, X_te, y_te, pipe.predict_proba(X_te), pipe.predict(X_te), classes, pos)


def imbalance_experiment(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    model: str,
    strategies: list[str] | None = None,
    cv: int = 4,
    seed: int = 42,
    max_rows: int = 8000,
) -> pd.DataFrame:
    """Compare resampling strategies with metrics that matter under imbalance (accuracy shown for contrast)."""
    X, y = _prepare(df, target, features, max_rows, seed)
    task = infer_task(y)
    classes = sorted(y.unique(), key=str)
    pos = classes[-1]
    splitter = StratifiedKFold(
        n_splits=max(2, min(cv, int(y.value_counts().min()))), shuffle=True, random_state=seed
    )
    rows = []
    for strat in strategies or list(RESAMPLING):
        accs, bals, f1s, aucs, aps = [], [], [], [], []
        try:
            for tr, te in splitter.split(X, y):
                pipe = build_pipeline(X.iloc[tr], model, task, strat, seed)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pipe.fit(X.iloc[tr], y.iloc[tr])
                pred, proba = pipe.predict(X.iloc[te]), pipe.predict_proba(X.iloc[te])
                yt = y.iloc[te]
                accs.append(float((pred == yt).mean()))
                bals.append(float(balanced_accuracy_score(yt, pred)))
                f1s.append(
                    float(
                        f1_score(yt, pred, pos_label=pos, average="binary")
                        if len(classes) == 2
                        else f1_score(yt, pred, average="macro")
                    )
                )
                if len(classes) == 2:
                    col = list(pipe.classes_).index(pos)
                    aucs.append(float(roc_auc_score(yt == pos, proba[:, col])))
                    aps.append(float(average_precision_score(yt == pos, proba[:, col])))
        except ModelingError as err:
            rows.append({"strategy": strat, "note": str(err)})
            continue
        rows.append(
            {
                "strategy": strat,
                "accuracy": np.mean(accs),
                "balanced_accuracy": np.mean(bals),
                "f1_minority" if len(classes) == 2 else "f1_macro": np.mean(f1s),
                "roc_auc": np.mean(aucs) if aucs else np.nan,
                "avg_precision": np.mean(aps) if aps else np.nan,
                "note": "",
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- evaluation
def gain_lift_table(y_true: np.ndarray, score: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Rows sorted by predicted probability, cut into groups: how concentrated are the positives?"""
    frame = (
        pd.DataFrame({"y": np.asarray(y_true).astype(int), "s": score})
        .sort_values("s", ascending=False)
        .reset_index(drop=True)
    )
    frame["group"] = pd.qcut(frame.index, q=min(bins, len(frame)), labels=False) + 1
    total_pos = frame["y"].sum() or 1
    overall = frame["y"].mean()
    out = (
        frame.groupby("group")
        .agg(rows=("y", "size"), positives=("y", "sum"), min_score=("s", "min"))
        .reset_index()
    )
    out["rate"] = out["positives"] / out["rows"]
    out["cum_gain"] = out["positives"].cumsum() / total_pos
    out["lift"] = out["rate"] / overall if overall else np.nan
    out["cum_lift"] = (
        (out["positives"].cumsum() / out["rows"].cumsum()) / overall if overall else np.nan
    )
    return out


def evaluate_holdout(h: Holdout) -> dict[str, Any]:
    """Everything a classifier report needs, from one held-out split."""
    y, pred, classes = h.y_test, h.pred, h.classes
    cm = confusion_matrix(y, pred, labels=classes)
    prec, rec, f1, sup = precision_recall_fscore_support(y, pred, labels=classes, zero_division=0)
    report = pd.DataFrame(
        {
            "class": [str(c) for c in classes],
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": sup,
        }
    )
    out: dict[str, Any] = {
        "accuracy": float((pred == y).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "confusion": pd.DataFrame(
            cm, index=[f"actual {c}" for c in classes], columns=[f"pred {c}" for c in classes]
        ),
        "confusion_rates": pd.DataFrame(
            cm / cm.sum(axis=1, keepdims=True).clip(min=1),
            index=[f"actual {c}" for c in classes],
            columns=[f"pred {c}" for c in classes],
        ),
        "report": report,
        "log_loss": float(log_loss(y, h.proba, labels=classes)),
    }
    if len(classes) == 2:
        col = classes.index(h.positive)
        pos = (y == h.positive).to_numpy()
        score = h.proba[:, col]
        fpr, tpr, _ = roc_curve(pos, score)
        p, r, _ = precision_recall_curve(pos, score)
        out.update(
            roc_auc=float(roc_auc_score(pos, score)),
            avg_precision=float(average_precision_score(pos, score)),
            roc_curve=pd.DataFrame({"fpr": fpr, "tpr": tpr}),
            pr_curve=pd.DataFrame({"recall": r, "precision": p}),
            ks=float(np.max(tpr - fpr)),
            brier=float(brier_score_loss(pos, score)),
            gain_lift=gain_lift_table(pos, score),
            prevalence=float(pos.mean()),
        )
    else:
        out["macro_f1"] = float(f1.mean())
        out["weighted_f1"] = float(np.average(f1, weights=sup))
        try:
            out["roc_auc_ovr"] = float(roc_auc_score(y, h.proba, multi_class="ovr", labels=classes))
        except ValueError:
            out["roc_auc_ovr"] = None
    return out


# --------------------------------------------------------------------------- thresholds
def threshold_table(y_true: np.ndarray, score: np.ndarray, steps: int = 99) -> pd.DataFrame:
    y = np.asarray(y_true).astype(bool)
    rows = []
    for t in np.linspace(0.01, 0.99, steps):
        pred = score >= t
        tp, fp = int((pred & y).sum()), int((pred & ~y).sum())
        fn, tn = int((~pred & y).sum()), int((~pred & ~y).sum())
        prec = tp / (tp + fp) if tp + fp else np.nan
        rec = tp / (tp + fn) if tp + fn else 0.0
        rows.append(
            {
                "threshold": float(t),
                "precision": prec,
                "recall": rec,
                "f1": (2 * prec * rec / (prec + rec)) if tp and prec == prec else 0.0,
                "fpr": fp / (fp + tn) if fp + tn else 0.0,
                "accuracy": (tp + tn) / len(y),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }
        )
    return pd.DataFrame(rows)


def optimal_threshold(
    y_true: np.ndarray,
    score: np.ndarray,
    cost_fp: float = 1.0,
    cost_fn: float = 5.0,
    benefit_tp: float = 0.0,
    benefit_tn: float = 0.0,
) -> dict[str, Any]:
    """Cutoff minimizing total cost (a false negative usually costs more than a false positive)."""
    table = threshold_table(y_true, score)
    table["cost"] = (
        table["fp"] * cost_fp
        + table["fn"] * cost_fn
        - table["tp"] * benefit_tp
        - table["tn"] * benefit_tn
    )
    best = table.loc[table["cost"].idxmin()]
    at_half = table.iloc[(table["threshold"] - 0.5).abs().argmin()]
    return {
        "threshold": float(best["threshold"]),
        "cost": float(best["cost"]),
        "cost_at_0.5": float(at_half["cost"]),
        "saving_vs_0.5": float(at_half["cost"] - best["cost"]),
        "precision": float(best["precision"]) if best["precision"] == best["precision"] else None,
        "recall": float(best["recall"]),
        "table": table,
    }


# --------------------------------------------------------------------------- calibration
def calibration_summary(y_true: np.ndarray, score: np.ndarray, bins: int = 10) -> dict[str, Any]:
    pos = np.asarray(y_true).astype(int)
    frac, mean_pred = calibration_curve(pos, score, n_bins=bins, strategy="quantile")
    counts = (
        np.histogram(score, bins=np.r_[0, (mean_pred[:-1] + mean_pred[1:]) / 2, 1])[0]
        if len(mean_pred) > 1
        else np.array([len(score)])
    )
    ece = (
        float(np.sum(counts / max(counts.sum(), 1) * np.abs(frac - mean_pred)))
        if len(counts) == len(frac)
        else float(np.mean(np.abs(frac - mean_pred)))
    )
    return {
        "curve": pd.DataFrame({"mean_predicted": mean_pred, "fraction_positive": frac}),
        "brier": float(brier_score_loss(pos, score)),
        "ece": ece,
    }


def calibrate_holdout(h: Holdout, method: str = "sigmoid", seed: int = 42) -> dict[str, Any]:
    """Refit with probability calibration; compare Brier score on the same held-out rows."""
    if len(h.classes) != 2:
        raise ModelingError("Calibration comparison is implemented for binary targets.")
    X_all = h.X_test
    y_all = h.y_test
    X_fit, X_eval, y_fit, y_eval = train_test_split(
        X_all, y_all, test_size=0.5, random_state=seed, stratify=y_all
    )
    col = h.classes.index(h.positive)
    before = brier_score_loss(y_eval == h.positive, h.pipeline.predict_proba(X_eval)[:, col])
    cal = CalibratedClassifierCV(
        h.pipeline, method=method, cv=StratifiedKFold(3, shuffle=True, random_state=seed)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cal.fit(X_fit, y_fit)
    after = brier_score_loss(y_eval == h.positive, cal.predict_proba(X_eval)[:, col])
    return {
        "brier_before": float(before),
        "brier_after": float(after),
        "improved": bool(after < before),
        "model": cal,
        "method": method,
    }


def multiclass_wrapper(model: str, task: str, kind: str = "ovr", seed: int = 42) -> Any:
    est = catalog.build(model, task, seed)
    if kind == "ovr":
        return OneVsRestClassifier(est)
    if kind == "ovo":
        return OneVsOneClassifier(est)
    return est
