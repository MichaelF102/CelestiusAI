"""Model explanations: partial dependence, interactions, Shapley values, what-if, counterfactuals,
fairness, and drift."""

from __future__ import annotations

import itertools
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.inspection import partial_dependence

from src.datascience.modeling import ModelingError


# --------------------------------------------------------------------------- partial dependence
def pdp_ice(
    pipeline: Any,
    X: pd.DataFrame,
    feature: str,
    grid_resolution: int = 20,
    ice_rows: int = 30,
    seed: int = 0,
) -> dict[str, Any]:
    """Average effect (PDP) and per-row curves (ICE) of one feature on the prediction."""
    if feature not in X.columns:
        raise ModelingError(f"'{feature}' is not a feature.")
    categorical = not pd.api.types.is_numeric_dtype(X[feature])
    sample = X.sample(min(len(X), 300), random_state=seed)
    # partial dependence rejects integer columns (implicit rounding), so use floats
    sample = sample.astype(dict.fromkeys(sample.select_dtypes("integer").columns, float))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        avg = partial_dependence(
            pipeline,
            sample,
            [feature],
            kind="average",
            grid_resolution=grid_resolution,
            categorical_features=[feature] if categorical else None,
        )
        ice = partial_dependence(
            pipeline,
            sample.head(ice_rows),
            [feature],
            kind="individual",
            grid_resolution=grid_resolution,
            categorical_features=[feature] if categorical else None,
        )
    values = avg["grid_values"][0]
    avg_vals = np.asarray(avg["average"])
    # multi-class/binary classification returns one curve per class: keep the last class
    average = (
        avg_vals[-1]
        if avg_vals.ndim > 1 and avg_vals.shape[0] > 1 and avg_vals.shape[0] != 1
        else avg_vals[0]
    )
    individual = np.asarray(ice["individual"])
    individual = individual[-1] if individual.ndim == 3 else individual
    return {
        "feature": feature,
        "grid": list(values),
        "average": np.asarray(average),
        "ice": np.asarray(individual),
        "categorical": categorical,
    }


def interaction_strength(
    pipeline: Any, X: pd.DataFrame, f1: str, f2: str, grid: int = 8, seed: int = 0
) -> float:
    """Friedman-style H statistic (0 = additive, 1 = pure interaction) for two numeric features."""
    for f in (f1, f2):
        if not pd.api.types.is_numeric_dtype(X[f]):
            raise ModelingError("Interaction strength is computed for numeric features.")
    sample = X.sample(min(len(X), 150), random_state=seed)
    g1 = np.unique(np.quantile(sample[f1].dropna(), np.linspace(0.05, 0.95, grid)))
    g2 = np.unique(np.quantile(sample[f2].dropna(), np.linspace(0.05, 0.95, grid)))

    def predict(frame: pd.DataFrame) -> np.ndarray:
        if hasattr(pipeline, "predict_proba"):
            return pipeline.predict_proba(frame)[:, -1]
        return pipeline.predict(frame)

    pd2 = np.zeros((len(g1), len(g2)))
    for i, v1 in enumerate(g1):
        for j, v2 in enumerate(g2):
            tmp = sample.copy()
            tmp[f1], tmp[f2] = v1, v2
            pd2[i, j] = predict(tmp).mean()
    p1 = pd2.mean(axis=1, keepdims=True)
    p2 = pd2.mean(axis=0, keepdims=True)
    centered = pd2 - pd2.mean()
    additive = (p1 - pd2.mean()) + (p2 - pd2.mean())
    denom = float(np.sum(centered**2))
    return float(np.sum((centered - additive) ** 2) / denom) if denom > 1e-12 else 0.0


# --------------------------------------------------------------------------- Shapley (model-agnostic)
def _predict_scalar(pipeline: Any, frame: pd.DataFrame, positive: Any | None) -> np.ndarray:
    if hasattr(pipeline, "predict_proba"):
        classes = list(pipeline.classes_)
        col = classes.index(positive) if positive in classes else len(classes) - 1
        return pipeline.predict_proba(frame)[:, col]
    return np.asarray(pipeline.predict(frame), dtype=float)


def shapley_local(
    pipeline: Any,
    row: pd.Series,
    background: pd.DataFrame,
    n_permutations: int = 60,
    positive: Any | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Sampling-based Shapley values for one prediction (works with any model).

    Contributions sum to prediction - baseline (the efficiency property), up to sampling noise
    that shrinks as `n_permutations` grows.
    """
    rng = np.random.default_rng(seed)
    cols = list(background.columns)
    bg = background.sample(min(len(background), 100), random_state=seed).reset_index(drop=True)
    baseline = float(_predict_scalar(pipeline, bg, positive).mean())
    x = row[cols]
    prediction = float(_predict_scalar(pipeline, pd.DataFrame([x]), positive)[0])
    contrib = dict.fromkeys(cols, 0.0)
    for _ in range(n_permutations):
        perm = rng.permutation(cols)
        ref = bg.iloc[rng.integers(0, len(bg))].copy()
        current = ref.copy()
        prev = float(_predict_scalar(pipeline, pd.DataFrame([current]), positive)[0])
        for c in perm:
            current[c] = x[c]
            now = float(_predict_scalar(pipeline, pd.DataFrame([current]), positive)[0])
            contrib[c] += now - prev
            prev = now
    table = pd.DataFrame(
        {
            "feature": cols,
            "value": [x[c] for c in cols],
            "contribution": [contrib[c] / n_permutations for c in cols],
        }
    )
    table["abs"] = table["contribution"].abs()
    table = table.sort_values("abs", ascending=False).drop(columns="abs").reset_index(drop=True)
    return {
        "table": table,
        "baseline": baseline,
        "prediction": prediction,
        "sum_contributions": float(table["contribution"].sum()),
    }


def shap_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("shap") is not None


def shap_local(
    pipeline: Any,
    row: pd.Series,
    background: pd.DataFrame,
    positive: Any | None = None,
    nsamples: int = 100,
    seed: int = 0,
) -> dict[str, Any] | None:
    """SHAP KernelExplainer values when the `shap` package is installed (else None)."""
    if not shap_available():
        return None
    import shap

    cols = list(background.columns)
    bg = background.sample(min(len(background), 30), random_state=seed)

    def fn(arr: np.ndarray) -> np.ndarray:
        return _predict_scalar(
            pipeline, pd.DataFrame(arr, columns=cols).astype(background.dtypes.to_dict()), positive
        )

    explainer = shap.KernelExplainer(fn, bg.to_numpy(dtype=object))
    values = np.asarray(
        explainer.shap_values(
            row[cols].to_numpy(dtype=object).reshape(1, -1), nsamples=nsamples, silent=True
        )
    ).ravel()
    return {
        "table": pd.DataFrame({"feature": cols, "shap": values})
        .sort_values("shap", key=np.abs, ascending=False)
        .reset_index(drop=True),
        "base_value": float(np.ravel(explainer.expected_value)[0]),
    }


# --------------------------------------------------------------------------- what-if & counterfactuals
def what_if(
    pipeline: Any, row: dict[str, Any], changes: dict[str, Any], positive: Any | None = None
) -> dict[str, Any]:
    """Prediction before and after editing some feature values."""
    before_row = pd.DataFrame([row])
    after = {**row, **changes}
    after_row = pd.DataFrame([after])
    out: dict[str, Any] = {
        "before": pipeline.predict(before_row)[0],
        "after": pipeline.predict(after_row)[0],
        "changes": changes,
    }
    if hasattr(pipeline, "predict_proba"):
        classes = list(pipeline.classes_)
        col = classes.index(positive) if positive in classes else len(classes) - 1
        out["proba_before"] = float(pipeline.predict_proba(before_row)[0, col])
        out["proba_after"] = float(pipeline.predict_proba(after_row)[0, col])
    return out


def counterfactual(
    pipeline: Any,
    row: dict[str, Any],
    X_ref: pd.DataFrame,
    desired: Any,
    features: list[str] | None = None,
    max_changes: int = 2,
    steps: int = 15,
    regression_target: bool = False,
    tolerance: float = 0.0,
) -> dict[str, Any] | None:
    """Smallest change (in at most `max_changes` numeric features) that produces `desired`.

    Classification: the predicted class equals `desired`. Regression: prediction >= desired
    (or <= when currently above). Distance is the sum of range-normalized changes.
    """
    num = [c for c in (features or list(X_ref.columns)) if pd.api.types.is_numeric_dtype(X_ref[c])]
    ranges = {c: (float(X_ref[c].min()), float(X_ref[c].max())) for c in num}
    base_pred = pipeline.predict(pd.DataFrame([row]))[0]
    go_up = bool(regression_target and float(base_pred) < float(desired))

    def success(pred: Any) -> bool:
        if not regression_target:
            return pred == desired
        return (
            float(pred) >= float(desired) - tolerance
            if go_up
            else float(pred) <= float(desired) + tolerance
        )

    if success(base_pred):
        return {"changes": {}, "distance": 0.0, "prediction": base_pred}
    best: dict[str, Any] | None = None
    for k in range(1, max_changes + 1):
        for combo in itertools.combinations(num, k):
            grids = [np.linspace(ranges[c][0], ranges[c][1], steps) for c in combo]
            for values in itertools.product(*grids):
                candidate = dict(row)
                candidate.update(dict(zip(combo, values, strict=True)))
                dist = sum(
                    abs(v - float(row[c])) / ((ranges[c][1] - ranges[c][0]) or 1.0)
                    for c, v in zip(combo, values, strict=True)
                )
                if best is not None and dist >= best["distance"]:
                    continue
                pred = pipeline.predict(pd.DataFrame([candidate]))[0]
                if success(pred):
                    best = {
                        "changes": {c: float(v) for c, v in zip(combo, values, strict=True)},
                        "distance": float(dist),
                        "prediction": pred,
                    }
        if best is not None:
            break  # fewer changed features is preferred
    return best


# --------------------------------------------------------------------------- fairness
def fairness_report(
    y_true: pd.Series, y_pred: pd.Series, groups: pd.Series, positive: Any
) -> dict[str, Any]:
    """Per-group selection rate, TPR, FPR, precision, accuracy and the 80% disparate-impact rule."""
    frame = pd.DataFrame(
        {
            "y": (y_true == positive).to_numpy(),
            "p": (y_pred == positive).to_numpy(),
            "g": groups.astype(str).to_numpy(),
        }
    )
    rows = []
    for g, part in frame.groupby("g"):
        tp = int((part.p & part.y).sum())
        fp = int((part.p & ~part.y).sum())
        fn = int((~part.p & part.y).sum())
        tn = int((~part.p & ~part.y).sum())
        rows.append(
            {
                "group": g,
                "n": len(part),
                "base_rate": float(part.y.mean()),
                "selection_rate": float(part.p.mean()),
                "tpr": tp / (tp + fn) if tp + fn else np.nan,
                "fpr": fp / (fp + tn) if fp + tn else np.nan,
                "precision": tp / (tp + fp) if tp + fp else np.nan,
                "accuracy": float((part.p == part.y).mean()),
            }
        )
    table = pd.DataFrame(rows).sort_values("group").reset_index(drop=True)
    sel = table["selection_rate"]
    di = float(sel.min() / sel.max()) if sel.max() > 0 else float("nan")
    tpr_gap = (
        float(table["tpr"].max() - table["tpr"].min())
        if table["tpr"].notna().any()
        else float("nan")
    )
    fpr_gap = (
        float(table["fpr"].max() - table["fpr"].min())
        if table["fpr"].notna().any()
        else float("nan")
    )
    warnings_: list[str] = []
    small = table[table["n"] < 30]
    if len(small):
        warnings_.append(
            f"Groups with fewer than 30 rows have unreliable rates: {', '.join(small['group'])}."
        )
    if di == di and di < 0.8:
        warnings_.append(
            f"Disparate impact ratio {di:.2f} is below the 0.80 rule of thumb: one group is selected far less often."
        )
    if tpr_gap == tpr_gap and tpr_gap > 0.1:
        warnings_.append(
            f"True-positive rates differ by {tpr_gap:.0%} between groups (equal opportunity gap)."
        )
    if fpr_gap == fpr_gap and fpr_gap > 0.1:
        warnings_.append(f"False-positive rates differ by {fpr_gap:.0%} between groups.")
    return {
        "table": table,
        "disparate_impact": di,
        "tpr_gap": tpr_gap,
        "fpr_gap": fpr_gap,
        "warnings": warnings_,
    }


# --------------------------------------------------------------------------- drift
def psi_numeric(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    e, a = (
        pd.to_numeric(expected, errors="coerce").dropna(),
        pd.to_numeric(actual, errors="coerce").dropna(),
    )
    if e.empty or a.empty or e.nunique() < 2:
        return 0.0
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    pe = np.histogram(e, edges)[0] / len(e)
    pa = np.histogram(a, edges)[0] / len(a)
    pe, pa = np.clip(pe, 1e-4, None), np.clip(pa, 1e-4, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def psi_categorical(expected: pd.Series, actual: pd.Series) -> float:
    cats = sorted(set(expected.dropna().astype(str)) | set(actual.dropna().astype(str)))
    pe = (
        expected.astype(str)
        .value_counts(normalize=True)
        .reindex(cats)
        .fillna(0)
        .clip(lower=1e-4)
        .to_numpy()
    )
    pa = (
        actual.astype(str)
        .value_counts(normalize=True)
        .reindex(cats)
        .fillna(0)
        .clip(lower=1e-4)
        .to_numpy()
    )
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def drift_report(
    train: pd.DataFrame, new: pd.DataFrame, features: list[str] | None = None
) -> pd.DataFrame:
    """Compare a new data file with the training data feature by feature (PSI + a significance test)."""
    rows = []
    for col in features or [c for c in train.columns if c in new.columns]:
        if col not in new.columns:
            rows.append(
                {
                    "feature": col,
                    "type": "missing",
                    "psi": np.nan,
                    "p_value": np.nan,
                    "status": "missing in new data",
                }
            )
            continue
        if pd.api.types.is_numeric_dtype(train[col]):
            psi = psi_numeric(train[col], new[col])
            a, b = train[col].dropna(), pd.to_numeric(new[col], errors="coerce").dropna()
            p = float(stats.ks_2samp(a, b).pvalue) if len(a) and len(b) else np.nan
            kind = "numeric"
        else:
            psi = psi_categorical(train[col], new[col])
            ct = pd.concat(
                [train[col].astype(str).value_counts(), new[col].astype(str).value_counts()], axis=1
            ).fillna(0)
            p = float(stats.chi2_contingency(ct.to_numpy())[1]) if ct.shape[0] > 1 else np.nan
            kind = "categorical"
        rows.append(
            {
                "feature": col,
                "type": kind,
                "psi": psi,
                "p_value": p,
                "status": "stable"
                if psi < 0.1
                else ("moderate shift" if psi < 0.25 else "major shift"),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("psi", ascending=False, na_position="last")
        .reset_index(drop=True)
    )
