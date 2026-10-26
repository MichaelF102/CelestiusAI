"""Shared Streamlit helpers for the data-scientist pages."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from src.core.session_state import KEY_EXPERIMENT_LOG, KEY_SELECTED_FEATURES
from src.datascience import catalog
from src.datascience.modeling import (
    BINARY,
    MULTICLASS,
    REGRESSION,
    FittedModel,
    infer_task,
)
from src.datascience.tracking import ExperimentLog, RunRecord

TASK_LABELS = {
    REGRESSION: "Regression",
    BINARY: "Binary classification",
    MULTICLASS: "Multiclass classification",
}


def experiment_log() -> ExperimentLog:
    """The session's experiment log (created on first use)."""
    log = st.session_state.get(KEY_EXPERIMENT_LOG)
    if not isinstance(log, ExperimentLog):
        log = ExperimentLog()
        st.session_state[KEY_EXPERIMENT_LOG] = log
    return log


def pick_target_features(
    df: pd.DataFrame,
    key: str,
    task_filter: str | None = None,
    max_default: int = 12,
    show_task: bool = True,
) -> tuple[str, list[str], str]:
    """Target + feature pickers with sensible defaults; returns (target, features, task).

    `task_filter`: 'regression' or 'classification' limits the target choices.
    """
    cols = list(df.columns)
    if task_filter:

        def suits(c: str) -> bool:
            t = infer_task(df[c].dropna()) if df[c].notna().any() else REGRESSION
            return (t == REGRESSION) == (task_filter == "regression")

        cols = [c for c in cols if suits(c)] or cols
    c1, c2 = st.columns([1, 3])
    target = c1.selectbox("Target", cols, key=f"{key}_target")
    pool = [c for c in df.columns if c != target]
    saved = st.session_state.get(KEY_SELECTED_FEATURES) or {}
    default = (
        [c for c in saved.get("features", []) if c in pool] if saved.get("target") == target else []
    )
    if not default:
        default = [
            c
            for c in pool
            if not c.lower().endswith("id")
            and df[c].nunique() > 1
            and (pd.api.types.is_numeric_dtype(df[c]) or df[c].nunique() <= 30)
        ][:max_default]
    # A new target means a new default feature set; drop the stale selection once.
    if st.session_state.get(f"{key}_features_for") != target:
        st.session_state[f"{key}_features_for"] = target
        st.session_state[f"{key}_features"] = default
    features = c2.multiselect("Features", pool, key=f"{key}_features")
    task = infer_task(df[target].dropna())
    if show_task:
        st.caption(
            f"Task: **{TASK_LABELS[task]}**"
            + (
                " · using the feature set chosen on the Feature Selection page"
                if default and saved.get("target") == target
                else ""
            )
        )
    return target, features, task


def model_multiselect(
    task: str,
    key: str,
    default: list[str] | None = None,
    label: str = "Models",
    include_baseline: bool = True,
    only_tunable: bool = False,
) -> list[str]:
    specs = catalog.available_models(task, include_baseline=include_baseline)
    if only_tunable:
        specs = [s for s in specs if catalog.param_space(s.name, task)]
    names = [s.name for s in specs]
    shown = {s.name: catalog.display_name(s, task) for s in specs}
    chosen_default = [d for d in (default or names[:6]) if d in names]
    out = st.multiselect(label, names, default=chosen_default, format_func=shown.get, key=key)
    missing = [n for n, ok in catalog.optional_status().items() if not ok]
    if missing:
        st.caption(
            "Optional boosters not installed: "
            + ", ".join(missing)
            + " (pip install xgboost lightgbm catboost)."
        )
    return out


def model_card_box(name: str, task: str) -> None:
    spec = catalog.get(name)
    card = spec.card
    st.markdown(
        f"**{catalog.display_name(spec, task)}** ({spec.family})  \n"
        f"*When to use:* {card.when_to_use}  \n*Strengths:* {card.strengths}  \n*Weaknesses:* {card.weaknesses}  \n"
        f"Interpretability {'●' * card.interpretability}{'○' * (5 - card.interpretability)} · Speed {'●' * card.speed}{'○' * (5 - card.speed)}"
        + (" · needs scaled inputs" if card.needs_scaling else "")
    )


def record_run(
    name: str,
    kind: str,
    model: str,
    task: str,
    target: str,
    features: list[str],
    metrics: dict[str, float],
    params: dict[str, Any] | None = None,
    cv_scores: list[float] | None = None,
    seed: int = 42,
    duration_s: float = 0.0,
    dataset: Any = None,
    notes: str = "",
    tags: list[str] | None = None,
) -> RunRecord:
    run = RunRecord(
        name=name,
        kind=kind,
        task=task,
        target=target,
        features=list(features),
        model=model,
        params=params or {},
        metrics={k: float(v) for k, v in metrics.items()},
        cv_scores=[float(x) for x in (cv_scores or [])],
        seed=seed,
        data_fingerprint=getattr(dataset, "fingerprint", ""),
        data_version=getattr(dataset, "version", 1),
        rows=len(getattr(dataset, "working_df", [])),
        duration_s=round(duration_s, 2),
        notes=notes,
        tags=tags or [],
    )
    return experiment_log().add(run)


def store_fitted_model(model: FittedModel, key: str) -> str:
    """Save a fitted model in the session so Explainer/Deployment/Tracker can use it."""
    from src.core.session_state import KEY_FITTED_MODELS

    models = dict(st.session_state.get(KEY_FITTED_MODELS) or {})
    unique = key
    n = 2
    while unique in models:
        unique = f"{key} #{n}"
        n += 1
    models[unique] = model
    st.session_state[KEY_FITTED_MODELS] = models
    return unique
