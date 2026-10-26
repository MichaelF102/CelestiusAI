"""Experiment tracking, reproducible run code, and prediction monitoring."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score

from src.core.code_options import lst as _list
from src.core.code_options import make_snippet
from src.core.code_options import q as _q
from src.datascience import catalog


class TrackingError(ValueError):
    """Invalid tracking input (message is user-facing)."""


class RunRecord(BaseModel):
    run_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    name: str
    kind: str = "manual"  # race | tuning | automl | ensemble | manual
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))
    task: str
    target: str
    features: list[str]
    model: str
    params: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)
    cv_scores: list[float] = Field(default_factory=list)
    seed: int = 42
    data_fingerprint: str = ""
    data_version: int = 1
    rows: int = 0
    duration_s: float = 0.0
    test_size: float = 0.2
    notes: str = ""
    tags: list[str] = Field(default_factory=list)


class ExperimentLog:
    """An ordered collection of runs with comparison and persistence helpers."""

    def __init__(self, runs: list[RunRecord] | None = None) -> None:
        self.runs: list[RunRecord] = list(runs or [])

    def add(self, run: RunRecord) -> RunRecord:
        self.runs.append(run)
        return run

    def get(self, run_id: str) -> RunRecord:
        for r in self.runs:
            if r.run_id == run_id:
                return r
        raise TrackingError(f"Run '{run_id}' not found.")

    def remove(self, run_id: str) -> None:
        self.get(run_id)
        self.runs = [r for r in self.runs if r.run_id != run_id]

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for r in self.runs:
            rows.append(
                {
                    "run_id": r.run_id,
                    "name": r.name,
                    "kind": r.kind,
                    "model": r.model,
                    "target": r.target,
                    "features": len(r.features),
                    "rows": r.rows,
                    "seconds": r.duration_s,
                    "created": r.created_at,
                    "tags": ", ".join(r.tags),
                    **r.metrics,
                }
            )
        return pd.DataFrame(rows)

    def best(
        self, metric: str, higher_is_better: bool = True, target: str | None = None
    ) -> RunRecord | None:
        pool = [
            r for r in self.runs if metric in r.metrics and (target is None or r.target == target)
        ]
        if not pool:
            return None
        return (max if higher_is_better else min)(pool, key=lambda r: r.metrics[metric])

    def compare(self, run_ids: list[str]) -> dict[str, Any]:
        """Side-by-side metrics, parameters that differ, and feature-set differences."""
        runs = [self.get(i) for i in run_ids]
        if len(runs) < 2:
            raise TrackingError("Pick at least two runs to compare.")
        metrics = pd.DataFrame({r.run_id: r.metrics for r in runs})
        all_params = sorted({k for r in runs for k in r.params})
        params = pd.DataFrame(
            {r.run_id: {k: r.params.get(k, "-") for k in all_params} for r in runs},
            index=all_params,
        ).astype(str)
        params = params[params.nunique(axis=1) > 1] if len(params) else params
        common = set.intersection(*(set(r.features) for r in runs))
        feats = {r.run_id: sorted(set(r.features) - common) for r in runs}
        same_data = len({(r.data_fingerprint, r.data_version) for r in runs}) == 1
        return {
            "metrics": metrics,
            "params_diff": params,
            "features_unique": feats,
            "common_features": sorted(common),
            "same_data": same_data,
            "models": {r.run_id: r.model for r in runs},
        }

    def to_json(self) -> str:
        return json.dumps([r.model_dump() for r in self.runs], indent=2)

    @classmethod
    def from_json(cls, text: str | bytes) -> ExperimentLog:
        try:
            return cls([RunRecord.model_validate(item) for item in json.loads(text)])
        except (ValueError, TypeError) as err:
            raise TrackingError(f"Not a valid experiment log: {err}") from err


# --------------------------------------------------------------------------- reproducible code
def _plain(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _same(a: Any, b: Any) -> bool:
    try:
        if a is None or b is None:
            return a is b
        if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
            return True
        return bool(a == b)
    except Exception:
        return False


def _expr(obj: Any) -> str:
    """Constructor expression listing only the parameters that differ from the defaults."""
    if hasattr(obj, "steps"):
        steps = ", ".join(f"({n!r}, {_expr(step)})" for n, step in obj.steps)
        return f"{type(obj).__name__}(steps=[{steps}])"
    params = obj.get_params(deep=False)
    try:
        defaults = type(obj)().get_params(deep=False)
    except Exception:
        defaults = {}
    parts = []
    for key, value in params.items():
        if key in defaults and _same(defaults[key], value):
            continue
        parts.append(f"{key}={_value(value)}")
    return f"{type(obj).__name__}({', '.join(parts)})"


def _value(value: Any) -> str:
    if hasattr(value, "get_params") and not isinstance(value, type):
        return _expr(value)
    if isinstance(value, (list, tuple)):
        inner = ", ".join(_value(v) for v in value)
        return (
            f"[{inner}]"
            if isinstance(value, list)
            else f"({inner}{',' if len(value) == 1 else ''})"
        )
    return repr(_plain(value))


def estimator_code(est: Any) -> tuple[list[str], str]:
    """(import lines, constructor expression) for an sklearn-compatible estimator."""
    if type(est).__name__ == "EncodedClassifier":
        est = est.estimator
    imports: set[str] = set()

    def walk(obj: Any) -> None:
        cls = type(obj)
        root = cls.__module__.split(".")
        mod = ".".join(root[:2]) if root[0] == "sklearn" else root[0]
        imports.add(f"from {mod} import {cls.__name__}")
        if hasattr(obj, "steps"):
            for _, step in obj.steps:
                walk(step)
        for value in obj.get_params(deep=False).values():
            if hasattr(value, "get_params") and not isinstance(value, type):
                walk(value)

    walk(est)
    return sorted(imports), _expr(est)


def reproduce_code(run: RunRecord, **load: Any) -> str:
    """Standalone code that retrains this run's model on the same columns, split and seed."""
    reg = run.task == "regression"
    est = catalog.build(run.model.removesuffix(" (tuned)"), run.task, run.seed, run.params or None)
    est_imports, est_expr = estimator_code(est)
    metric_lines = (
        '    "R2": r2_score(y_test, pred),\n    "MAE": mean_absolute_error(y_test, pred),'
        if reg
        else '    "accuracy": accuracy_score(y_test, pred),\n    "f1_macro": f1_score(y_test, pred, average="macro"),'
    )
    metric_import = (
        "from sklearn.metrics import r2_score, mean_absolute_error"
        if reg
        else "from sklearn.metrics import accuracy_score, f1_score"
    )
    stratify = "" if reg else ", stratify=y"
    imports = [
        "import pandas as pd",
        "import numpy as np",
        "from sklearn.compose import ColumnTransformer",
        "from sklearn.impute import SimpleImputer",
        "from sklearn.model_selection import train_test_split",
        "from sklearn.pipeline import Pipeline",
        "from sklearn.preprocessing import OneHotEncoder, StandardScaler",
        metric_import,
        *est_imports,
    ]
    body = f"""# Run {run.run_id}: {run.name}  ({run.created_at})
SEED = {run.seed}
target, features = {_q(run.target)}, {_list(run.features)}
data = df[[target, *features]].dropna(subset=[target])
X, y = data[features], data[target]

num_cols = X.select_dtypes("number").columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]
preprocess = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num_cols),
    ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", max_categories=30, sparse_output=False))]), cat_cols),
])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size={run.test_size}, random_state=SEED{stratify})

# %%
model = Pipeline([("prep", preprocess), ("model", {est_expr})]).fit(X_train, y_train)
pred = model.predict(X_test)
metrics = {{
{metric_lines}
}}
print("Reproduced metrics:", {{k: round(v, 4) for k, v in metrics.items()}})
print("Recorded metrics:  ", {run.metrics!r})
"""
    return make_snippet(f"Reproduce run {run.run_id}: {run.model}", body, imports=imports, **load)


# --------------------------------------------------------------------------- prediction log & live performance
def log_predictions(
    log: pd.DataFrame | None,
    model_name: str,
    rows: pd.DataFrame,
    predictions: Any,
    ids: pd.Series | None = None,
    proba: Any | None = None,
) -> pd.DataFrame:
    """Append scored rows to a prediction log (one row per prediction, with an id and timestamp)."""
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    n = len(rows)
    entry = pd.DataFrame(
        {
            "prediction_id": ids.to_numpy()
            if ids is not None
            else [uuid.uuid4().hex[:10] for _ in range(n)],
            "model": model_name,
            "scored_at": stamp,
            "prediction": np.asarray(predictions),
        }
    )
    if proba is not None:
        entry["score"] = np.asarray(proba)
    entry = pd.concat([entry.reset_index(drop=True), rows.reset_index(drop=True)], axis=1)
    return entry if log is None or log.empty else pd.concat([log, entry], ignore_index=True)


def attach_outcomes(
    log: pd.DataFrame, outcomes: pd.DataFrame, id_col: str, outcome_col: str
) -> pd.DataFrame:
    """Join real outcomes (by prediction id) onto the log."""
    if id_col not in outcomes.columns or outcome_col not in outcomes.columns:
        raise TrackingError(f"The outcomes file needs '{id_col}' and '{outcome_col}' columns.")
    mapped = outcomes.drop_duplicates(id_col).set_index(id_col)[outcome_col]
    out = log.copy()
    out["actual"] = out["prediction_id"].map(mapped)
    return out


def _label(value: Any) -> str:
    """Canonical text for a class label (so 1, 1.0 and '1' all compare equal)."""
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(_plain(value))


def live_performance(
    log: pd.DataFrame,
    task: str,
    baseline: dict[str, float] | None = None,
    tolerance: float = 0.1,
    min_labeled: int = 30,
) -> dict[str, Any]:
    """Performance on logged predictions that have outcomes, compared with the holdout baseline."""
    labeled = log.dropna(subset=["actual"]) if "actual" in log else log.iloc[0:0]
    result: dict[str, Any] = {
        "n_logged": len(log),
        "n_labeled": len(labeled),
        "metrics": {},
        "alerts": [],
    }
    if len(labeled) < min_labeled:
        result["alerts"].append(
            f"Only {len(labeled)} predictions have outcomes (need {min_labeled}) - too few to judge."
        )
        return result
    if task == "regression":
        actual, pred = pd.to_numeric(labeled["actual"]), pd.to_numeric(labeled["prediction"])
        result["metrics"] = {
            "R2": float(r2_score(actual, pred)),
            "MAE": float(mean_absolute_error(actual, pred)),
        }
        key, higher = "R2", True
    else:
        actual, pred = labeled["actual"].map(_label), labeled["prediction"].map(_label)
        result["metrics"] = {
            "accuracy": float(accuracy_score(actual, pred)),
            "f1_macro": float(f1_score(actual, pred, average="macro")),
        }
        key, higher = "f1_macro", True
    if baseline and key in baseline:
        drop = (
            baseline[key] - result["metrics"][key]
            if higher
            else result["metrics"][key] - baseline[key]
        )
        result["vs_baseline"] = {key: float(-drop)}
        if drop > tolerance:
            result["alerts"].append(
                f"{key} is {drop:.2f} below the holdout value ({baseline[key]:.3f} -> {result['metrics'][key]:.3f}): consider retraining."
            )
    labeled = labeled.assign(scored_at=pd.to_datetime(labeled["scored_at"], errors="coerce"))
    if labeled["scored_at"].nunique() > 1:
        correct = (
            (labeled["actual"].astype(str) == labeled["prediction"].astype(str))
            if task != "regression"
            else (pd.to_numeric(labeled["actual"]) - pd.to_numeric(labeled["prediction"])).abs()
        )
        result["by_day"] = (
            labeled.assign(score=correct.astype(float))
            .groupby(labeled["scored_at"].dt.date)["score"]
            .mean()
            .rename("accuracy" if task != "regression" else "mae")
            .reset_index()
        )
    return result
