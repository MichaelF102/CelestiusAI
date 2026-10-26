"""Section: Experiment Tracker & Model Registry - compare runs, reproduce them, and promote models."""

import pandas as pd
import plotly.express as px
import streamlit as st

from src.core.code_snippets import load_kwargs
from src.core.ds_ui import experiment_log
from src.core.session_state import KEY_EXPERIMENT_LOG, KEY_FITTED_MODELS
from src.core.ui_components import (
    render_code_button,
    require_dataset,
)
from src.datascience.tracking import ExperimentLog, TrackingError, reproduce_code
from src.workspace.registry import STAGES, ModelRegistry, RegistryError

SUBTITLE = "Every run in one place: compare, reproduce, and promote the best models to a versioned registry."


def render() -> None:
    """Render this section inside its parent page."""

    dataset = require_dataset("Experiment Tracker")
    LOAD = load_kwargs(dataset)
    log = experiment_log()
    registry = ModelRegistry()

    tab_runs, tab_cmp, tab_reg = st.tabs(["📋 Runs", "⚖️ Compare", "🏷️ Model registry"])

    # ------------------------------------------------------------------ runs
    with tab_runs:
        if not log.runs:
            st.info(
                "No runs yet. Use the **Log** buttons on Model Selection, Ensembles, the Classification Toolkit, Model Tuning, or AutoML."
            )
        else:
            frame = log.to_frame()
            f1, f2, f3 = st.columns(3)
            targets = f1.multiselect("Target", sorted(frame["target"].unique()), key="tr_f_target")
            kinds = f2.multiselect("Kind", sorted(frame["kind"].unique()), key="tr_f_kind")
            metric_cols = [
                c
                for c in frame.columns
                if c
                not in (
                    "run_id",
                    "name",
                    "kind",
                    "model",
                    "target",
                    "features",
                    "rows",
                    "seconds",
                    "created",
                    "tags",
                )
            ]
            sort_metric = f3.selectbox("Sort by", metric_cols or ["created"], key="tr_sort")
            mask = pd.Series(True, index=frame.index)
            if targets:
                mask &= frame["target"].isin(targets)
            if kinds:
                mask &= frame["kind"].isin(kinds)
            view = frame[mask]
            if sort_metric in view:
                view = view.sort_values(sort_metric, ascending=False, na_position="last")
            st.dataframe(view, hide_index=True, use_container_width=True)
            if sort_metric in metric_cols and len(view):
                st.plotly_chart(
                    px.bar(
                        view.dropna(subset=[sort_metric]),
                        x="name",
                        y=sort_metric,
                        color="model",
                        template="plotly_dark",
                        title=f"{sort_metric} by run",
                    ),
                    use_container_width=True,
                )
            sel = st.selectbox(
                "Open a run",
                [r.run_id for r in log.runs],
                format_func=lambda i: f"{log.get(i).name} · {log.get(i).created_at}",
                key="tr_open",
            )
            run = log.get(sel)
            c1, c2 = st.columns(2)
            c1.json(
                {
                    "model": run.model,
                    "task": run.task,
                    "target": run.target,
                    "params": run.params,
                    "metrics": run.metrics,
                    "seed": run.seed,
                    "data": {
                        "fingerprint": run.data_fingerprint,
                        "version": run.data_version,
                        "rows": run.rows,
                    },
                }
            )
            run.notes = c2.text_area("Notes", value=run.notes, key=f"tr_notes_{sel}")
            tags = c2.text_input(
                "Tags (comma separated)", value=", ".join(run.tags), key=f"tr_tags_{sel}"
            )
            run.tags = [t.strip() for t in tags.split(",") if t.strip()]
            same_data = (run.data_fingerprint, run.data_version) == (
                getattr(dataset, "fingerprint", ""),
                getattr(dataset, "version", 1),
            )
            if not same_data:
                st.warning(
                    "The loaded dataset differs from the one this run used (different file or cleaning version). Reproduction code uses your column names but may give different numbers."
                )
            try:
                render_code_button(
                    f"Reproduce {run.name}",
                    lambda o: reproduce_code(run, options=o, **LOAD),
                    key="code_tr_run",
                )
            except Exception as err:
                st.caption(f"No reproduction code for this run type: {err}")
            if st.button("Delete this run", key="tr_del"):
                log.remove(sel)
                st.rerun()
            st.download_button(
                "⬇️ Export all runs (JSON)",
                log.to_json(),
                "experiments.json",
                "application/json",
                key="tr_export",
            )
        up = st.file_uploader("Import runs", type=["json"], key="tr_import")
        if up is not None and st.button("Merge imported runs", key="tr_merge"):
            try:
                imported = ExperimentLog.from_json(up.getvalue())
                known = {r.run_id for r in log.runs}
                for r in imported.runs:
                    if r.run_id not in known:
                        log.add(r)
                st.session_state[KEY_EXPERIMENT_LOG] = log
                st.rerun()
            except TrackingError as err:
                st.error(str(err))

    # ------------------------------------------------------------------ compare
    with tab_cmp:
        if len(log.runs) < 2:
            st.info("Log at least two runs to compare them.")
        else:
            ids = st.multiselect(
                "Runs",
                [r.run_id for r in log.runs],
                default=[r.run_id for r in log.runs[-2:]],
                format_func=lambda i: f"{log.get(i).name} ({i})",
                key="tr_cmp_ids",
            )
            if len(ids) >= 2:
                cmp = log.compare(ids)
                if not cmp["same_data"]:
                    st.warning(
                        "These runs used different data versions: scores are not strictly comparable."
                    )
                st.markdown("**Metrics**")
                st.dataframe(
                    cmp["metrics"].style.format("{:.4f}", na_rep="-"), use_container_width=True
                )
                st.markdown("**Parameters that differ**")
                st.dataframe(
                    cmp["params_diff"]
                    if len(cmp["params_diff"])
                    else pd.DataFrame({"": ["identical parameters"]}),
                    use_container_width=True,
                )
                st.markdown("**Features**")
                st.write("Shared:", ", ".join(cmp["common_features"]) or "none")
                for rid, only in cmp["features_unique"].items():
                    if only:
                        st.write(f"Only in `{rid}`:", ", ".join(only))
                m = (
                    cmp["metrics"]
                    .reset_index()
                    .melt(id_vars="index", var_name="run", value_name="value")
                    .dropna()
                )
                st.plotly_chart(
                    px.bar(
                        m,
                        x="index",
                        y="value",
                        color="run",
                        barmode="group",
                        template="plotly_dark",
                        labels={"index": "metric"},
                    ),
                    use_container_width=True,
                )

    # ------------------------------------------------------------------ registry
    with tab_reg:
        models = st.session_state.get(KEY_FITTED_MODELS) or {}
        st.markdown("##### Register a model")
        if not models:
            st.caption("Train a model on Model Tuning, AutoML, or the Explainer first.")
        else:
            r1, r2 = st.columns([2, 2])
            key = r1.selectbox("Model", list(models), key="rg_pick")
            name = r2.text_input("Registry name", value=key.split(" (")[0], key="rg_name")
            notes = st.text_input("Intended use / notes", key="rg_notes")
            stage = st.selectbox("Initial stage", STAGES, key="rg_stage")
            if st.button("Register", type="primary", key="rg_register", disabled=not name.strip()):
                try:
                    meta = registry.register(
                        models[key],
                        name,
                        notes,
                        stage,
                        {
                            "data_fingerprint": getattr(dataset, "fingerprint", ""),
                            "data_version": getattr(dataset, "version", 1),
                        },
                    )
                    st.success(
                        f"Registered **{meta['name']}** version {meta['version']} ({meta['stage']})."
                    )
                except RegistryError as err:
                    st.error(str(err))
        st.markdown("##### Registered models")
        listing = registry.list_models()
        if not listing:
            st.caption("Nothing registered yet.")
        else:
            st.dataframe(pd.DataFrame(listing), hide_index=True, use_container_width=True)
            slug = st.selectbox("Model", [m["slug"] for m in listing], key="rg_view")
            versions = registry.versions(slug)
            ver = st.selectbox("Version", [v["version"] for v in reversed(versions)], key="rg_ver")
            meta = next(v for v in versions if v["version"] == ver)
            st.dataframe(
                pd.DataFrame(versions)[["version", "stage", "registered_at", "estimator", "notes"]],
                hide_index=True,
                use_container_width=True,
            )
            s1, s2 = st.columns(2)
            new_stage = s1.selectbox(
                "Move this version to",
                STAGES,
                index=STAGES.index(meta["stage"]),
                key="rg_new_stage",
            )
            if s2.button("Apply stage", key="rg_apply_stage"):
                registry.set_stage(slug, ver, new_stage)
                st.rerun()
            with st.expander("Model card", expanded=True):
                card = registry.card(slug, ver)
                st.markdown(card)
                st.download_button(
                    "⬇️ Model card (Markdown)",
                    card,
                    f"{slug}_v{ver}_MODEL_CARD.md",
                    "text/markdown",
                    key="rg_card_dl",
                )
            l1, l2 = st.columns(2)
            if l1.button("Load into this session (for Deployment / Explainer)", key="rg_load"):
                loaded = registry.load(slug, ver)
                fitted = dict(st.session_state.get(KEY_FITTED_MODELS) or {})
                fitted[f"{meta['name']} v{ver} (registry)"] = loaded
                st.session_state[KEY_FITTED_MODELS] = fitted
                st.success(
                    "Loaded. Registry models were refit-free exports, so they have no held-out data for the Explainer."
                )
            if l2.button("Delete all versions", key="rg_delete"):
                registry.delete(slug)
                st.rerun()
            st.caption(
                "Registry files are joblib pickles on this machine: only load files you created."
            )
    return
