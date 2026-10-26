"""AutoML: profile, guard against leakage, race models, tune, ensemble, and explain every step."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.core.code_options import lst as _list
from src.core.code_options import make_snippet
from src.core.code_options import q as _q
from src.datascience import catalog
from src.datascience import ensembles as ens
from src.datascience import features as fe
from src.datascience import selection as sel
from src.datascience.modeling import (
    FittedModel,
    ModelingError,
    _prepare,
    build_preprocessor,
    default_scoring,
    describe_features,
    holdout_metrics,
    infer_task,
    tune_model,
)
from src.datascience.tracking import estimator_code


@dataclass
class AutoMLStep:
    name: str
    detail: str
    seconds: float


@dataclass
class AutoMLResult:
    task: str
    target: str
    scoring: str
    features_used: list[str]
    dropped: list[tuple[str, str]]
    steps: list[AutoMLStep] = field(default_factory=list)
    profile: sel.DataProfile | None = None
    recommendations: list[sel.Recommendation] = field(default_factory=list)
    leaderboard: pd.DataFrame = field(default_factory=pd.DataFrame)
    tuned: list[dict[str, Any]] = field(default_factory=list)
    chosen: str = ""
    chosen_reason: str = ""
    chosen_cv: float = float("nan")
    chosen_params: dict[str, Any] = field(default_factory=dict)
    ensemble_members: list[str] = field(default_factory=list)
    model: FittedModel | None = None
    seed: int = 42
    total_seconds: float = 0.0

    def report_markdown(self) -> str:
        m = self.model.metrics if self.model else {}
        lines = [
            f"# AutoML report: predicting `{self.target}`",
            "",
            f"- **Task:** {self.task.replace('_', ' ')} (metric: {self.scoring})",
            f"- **Features used:** {len(self.features_used)}",
            f"- **Chosen model:** {self.chosen}",
            f"- **Cross-validated score:** {self.chosen_cv:.4f}",
            "- **Held-out metrics:** " + ", ".join(f"{k} = {v:.4f}" for k, v in m.items()),
            f"- **Total time:** {self.total_seconds:.1f}s",
            "",
        ]
        if self.dropped:
            lines += (
                ["## Features removed automatically", ""]
                + [f"- `{f}`: {why}" for f, why in self.dropped]
                + [""]
            )
        lines += ["## What happened", ""] + [
            f"{i}. **{s.name}** ({s.seconds:.1f}s): {s.detail}" for i, s in enumerate(self.steps, 1)
        ]
        lines += ["", "## Why this model", "", self.chosen_reason, ""]
        return "\n".join(lines)


def _log(res: AutoMLResult, name: str, detail: str, t0: float) -> None:
    res.steps.append(AutoMLStep(name, detail, time.perf_counter() - t0))


def _fitted_from_estimator(
    est: Any,
    df: pd.DataFrame,
    target: str,
    features: list[str],
    task: str,
    name: str,
    seed: int,
    params: dict[str, Any] | None = None,
) -> FittedModel:
    X, y = _prepare(df, target, features, 20000, seed)
    stratify = y if task != "regression" and y.value_counts().min() >= 2 else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=stratify
    )
    pipe = Pipeline([("prep", build_preprocessor(X_tr)), ("model", est)]).fit(X_tr, y_tr)
    return FittedModel(
        name=name,
        task=task,
        target=target,
        features=list(features),
        estimator_name=name,
        pipeline=pipe,
        metrics=holdout_metrics(task, pipe, X_te, y_te),
        params=params or {},
        feature_info=describe_features(X),
        classes=list(pipe.classes_) if task != "regression" else None,
        rows_used=len(X),
        holdout=(X_te, y_te),
    )


def run_automl(
    df: pd.DataFrame,
    target: str,
    features: list[str] | None = None,
    task: str | None = None,
    time_budget_s: float = 60.0,
    cv: int = 4,
    seed: int = 42,
    n_race: int = 8,
    top_k: int = 2,
    tune: bool = True,
    tune_iter: int = 6,
    try_ensemble: bool = True,
    exclude_flagged: bool = True,
    scoring: str | None = None,
    include_optional: bool = True,
    max_rows: int = 15000,
) -> AutoMLResult:
    t_start = time.perf_counter()
    if target not in df.columns:
        raise ModelingError(f"Target '{target}' is not a column.")
    feats = [c for c in (features or df.columns) if c != target]
    t0 = time.perf_counter()
    X0, y0 = _prepare(df, target, feats, max_rows, seed)
    task = task or infer_task(y0)
    scoring = scoring or default_scoring(task)
    res = AutoMLResult(
        task=task, target=target, scoring=scoring, features_used=feats, dropped=[], seed=seed
    )

    # 1. leakage and identifier guard
    dropped: list[tuple[str, str]] = []
    for w in fe.leakage_report(df, target, feats):
        if w.level == "high" or (exclude_flagged and w.level == "medium"):
            dropped.append((w.column, w.message))
    drop_cols = {c for c, _ in dropped}
    feats = [c for c in feats if c not in drop_cols]
    if not feats:
        raise ModelingError(
            "Every feature was flagged as leakage or an identifier; nothing left to model."
        )
    res.features_used, res.dropped = feats, dropped
    _log(
        res,
        "Leakage and identifier check",
        f"kept {len(feats)} feature(s); removed {len(dropped)}.",
        t0,
    )

    # 2. profile + recommend
    t0 = time.perf_counter()
    res.profile = sel.profile_for_selection(df, target, feats, task, seed)
    res.recommendations = sel.recommend_models(res.profile, include_optional)
    shortlist = [r.name for r in res.recommendations[:n_race]]
    _log(
        res,
        "Profile and shortlist",
        f"{res.profile.n_rows:,} rows, {res.profile.n_features} features; non-linear signal {res.profile.nonlinear_gain:+.3f}; shortlisted {', '.join(shortlist)}.",
        t0,
    )

    # 3. race
    t0 = time.perf_counter()
    race = sel.race(
        df,
        target,
        feats,
        ["Baseline", *shortlist],
        task,
        cv_strategy=None,
        n_splits=cv,
        scoring=scoring,
        seed=seed,
        max_rows=max_rows,
        time_budget_s=max(5.0, time_budget_s * 0.5),
    )
    res.leaderboard = race.table
    ok = race.table[race.table["status"] == "ok"]
    _log(
        res,
        "Model race",
        f"{len(ok)} of {len(race.table)} models finished; best {ok.iloc[0]['model']} ({ok.iloc[0]['cv_mean']:.4f}).",
        t0,
    )

    # 4. tune the best tunable models
    candidates = [
        {
            "model": r["model"],
            "cv_mean": r["cv_mean"],
            "cv_std": r["cv_std"],
            "folds": r["folds"],
            "status": "ok",
            "params": {},
            "source": "default",
        }
        for _, r in ok.iterrows()
    ]
    if tune:
        t0 = time.perf_counter()
        for name in [m for m in ok["model"] if m != "Baseline" and catalog.param_space(m, task)][
            :top_k
        ]:
            if time.perf_counter() - t_start > time_budget_s * 0.8:
                break
            try:
                tr = tune_model(
                    df,
                    target,
                    feats,
                    name,
                    task,
                    n_iter=tune_iter,
                    cv=cv,
                    scoring=scoring,
                    seed=seed,
                    max_rows=max_rows,
                )
            except ModelingError:
                continue
            row = tr.results.iloc[0]
            res.tuned.append(
                {
                    "model": name,
                    "default_cv": tr.default_cv_score,
                    "tuned_cv": tr.best_cv_score,
                    "params": tr.best_params,
                }
            )
            candidates.append(
                {
                    "model": f"{name} (tuned)",
                    "cv_mean": tr.best_cv_score,
                    "cv_std": float(row["std_test_score"]),
                    "folds": cv,
                    "status": "ok",
                    "params": tr.best_params,
                    "source": name,
                }
            )
        _log(
            res,
            "Hyperparameter tuning",
            "; ".join(
                f"{t['model']}: {t['default_cv']:.4f} -> {t['tuned_cv']:.4f}" for t in res.tuned
            )
            or "no tunable models finished in time.",
            t0,
        )

    # 5. ensemble of the strongest distinct base models
    if try_ensemble and time.perf_counter() - t_start < time_budget_s:
        t0 = time.perf_counter()
        bases = [
            r["model"]
            for r in sorted(
                (c for c in candidates if c["source"] == "default" and c["model"] != "Baseline"),
                key=lambda c: -c["cv_mean"],
            )[:3]
        ]
        if len(bases) >= 3:
            try:
                study = ens.ensemble_study(
                    df,
                    target,
                    feats,
                    bases,
                    [ens.EnsembleSpec("Stacking", "stacking", bases)],
                    task,
                    cv=cv,
                    scoring=scoring,
                    seed=seed,
                    max_rows=min(max_rows, 6000),
                )
                row = study.table[study.table["model"] == "Stacking"].iloc[0]
                candidates.append(
                    {
                        "model": f"Stacking ({', '.join(bases)})",
                        "cv_mean": float(row["cv_mean"]),
                        "cv_std": float(row["cv_std"]),
                        "folds": cv,
                        "status": "ok",
                        "params": {},
                        "source": "ensemble",
                        "bases": bases,
                    }
                )
                res.ensemble_members = bases
                _log(
                    res,
                    "Ensemble",
                    f"stacked {', '.join(bases)}: {row['cv_mean']:.4f} (best base {study.summary['best_base']:.4f}).",
                    t0,
                )
            except Exception as err:
                _log(res, "Ensemble", f"skipped ({str(err)[:60]}).", t0)

    # 6. choose with the one-standard-error rule, then fit the final model on the split
    t0 = time.perf_counter()
    table = pd.DataFrame(candidates).sort_values("cv_mean", ascending=False).reset_index(drop=True)
    pool = table[table["model"] != "Baseline"].copy()
    pick_name, why = sel.one_se_choice(
        pool.assign(model=pool["model"].str.replace(" (tuned)", "", regex=False))
    )
    # prefer the tuned variant of the picked model when it is at least as good
    chosen_row = (
        pool[pool["model"].isin([pick_name, f"{pick_name} (tuned)"])]
        .sort_values("cv_mean", ascending=False)
        .iloc[0]
    )
    if pick_name.startswith("Stacking"):
        chosen_row = pool[pool["model"].str.startswith("Stacking")].iloc[0]
    res.chosen, res.chosen_cv, res.chosen_params = (
        str(chosen_row["model"]),
        float(chosen_row["cv_mean"]),
        dict(chosen_row["params"]),
    )
    res.chosen_reason = why
    if chosen_row["source"] == "ensemble":
        est = ens.build_ensemble("stacking", chosen_row["bases"], task, seed)
        res.model = _fitted_from_estimator(est, df, target, feats, task, res.chosen, seed)
    else:
        base_name = res.chosen.removesuffix(" (tuned)")
        est = catalog.build(base_name, task, seed, res.chosen_params or None)
        res.model = _fitted_from_estimator(
            est, df, target, feats, task, res.chosen, seed, res.chosen_params
        )
        res.model.estimator_name = base_name
    _log(
        res,
        "Final model",
        f"chose {res.chosen} (CV {res.chosen_cv:.4f}); held-out "
        + ", ".join(f"{k}={v:.3f}" for k, v in res.model.metrics.items())
        + ".",
        t0,
    )
    res.total_seconds = time.perf_counter() - t_start
    return res


def automl_code(res: AutoMLResult, **load: Any) -> str:
    """Standalone code that rebuilds the winning pipeline on the cleaned feature set."""
    if res.model is None:
        raise ModelingError("Run AutoML first.")
    est = res.model.pipeline.named_steps["model"]
    imports, expr = estimator_code(est)
    reg = res.task == "regression"
    stratify = "" if reg else ", stratify=y"
    metric_import = (
        "from sklearn.metrics import r2_score, mean_absolute_error"
        if reg
        else "from sklearn.metrics import accuracy_score, f1_score"
    )
    metrics = (
        'print("R2:", r2_score(y_test, pred), "MAE:", mean_absolute_error(y_test, pred))'
        if reg
        else 'print("accuracy:", accuracy_score(y_test, pred), "f1_macro:", f1_score(y_test, pred, average="macro"))'
    )
    dropped_note = (
        "\n".join(f"#   dropped {c}: {why}" for c, why in res.dropped) or "#   (nothing dropped)"
    )
    body = f"""SEED = {res.seed}
# Features removed automatically (leakage / identifiers):
{dropped_note}
target, features = {_q(res.target)}, {_list(res.features_used)}
data = df[[target, *features]].dropna(subset=[target])
X, y = data[features], data[target]
num_cols = X.select_dtypes("number").columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]
preprocess = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num_cols),
    ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", max_categories=30, sparse_output=False))]), cat_cols),
])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED{stratify})

# %%
# AutoML chose: {res.chosen}  (cross-validated {res.scoring} = {res.chosen_cv:.4f})
model = Pipeline([("prep", preprocess), ("model", {expr})]).fit(X_train, y_train)
pred = model.predict(X_test)
{metrics}
"""
    all_imports = [
        "import pandas as pd",
        "import numpy as np",
        "from sklearn.compose import ColumnTransformer",
        "from sklearn.impute import SimpleImputer",
        "from sklearn.model_selection import train_test_split",
        "from sklearn.pipeline import Pipeline",
        "from sklearn.preprocessing import OneHotEncoder, StandardScaler",
        metric_import,
        *imports,
    ]
    return make_snippet(f"AutoML: {res.chosen}", body, imports=all_imports, **load)
