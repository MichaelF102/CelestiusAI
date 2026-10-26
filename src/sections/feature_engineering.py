"""Section: Feature Engineering - create model-ready columns with leakage checks."""

import streamlit as st

from src.config.settings import get_settings
from src.core.code_snippets import load_kwargs
from src.core.exceptions import DataAnalystError
from src.core.library import add_table, unique_name
from src.core.session_state import KEY_FEATURE_SPECS
from src.core.ui_components import (
    esc,
    render_code_button,
    require_dataset,
    severity_chip,
)
from src.datascience import features as fe
from src.ingestion.service import ingest_dataframe

SUBTITLE = "Build model-ready columns, check for leakage, and save the steps as a reusable recipe."


def render() -> None:
    """Render this section inside its parent page."""

    settings = get_settings()
    dataset = require_dataset("Feature Engineering")
    base = dataset.working_df
    specs: list[fe.FeatureSpec] = st.session_state.get(KEY_FEATURE_SPECS) or []

    try:
        current = fe.apply_specs(base, specs)
    except fe.FeatureError as err:
        st.error(f"A saved step no longer applies: {err}")
        current = base

    num_cols = current.select_dtypes("number").columns.tolist()
    all_cols = list(current.columns)

    st.caption(
        f"Working table: **{len(current):,} rows × {len(current.columns)} columns** "
        f"({len(current.columns) - len(base.columns):+d} vs. the dataset)."
    )

    tab_add, tab_steps, tab_leak = st.tabs(
        ["➕ Add a step", "🧾 Steps & recipe", "🚧 Leakage check"]
    )

    with tab_add:
        op = st.selectbox("Step type", list(fe.OP_LABELS), format_func=fe.OP_LABELS.get)
        params: dict = {}
        if op in ("log1p", "sqrt", "square", "bin_quantile", "bin_width"):
            cols = st.multiselect("Numeric columns", num_cols)
            if op in ("bin_quantile", "bin_width"):
                params["bins"] = st.slider("Number of bins", 2, 20, 4)
            if op in fe.LEARNED_OPS:
                st.caption(
                    "⚠️ Quantile edges are learned from all rows; for strict validation, bin inside the model pipeline."
                )
        elif op in ("onehot", "ordinal", "frequency"):
            cols = st.multiselect("Categorical columns", [c for c in all_cols if c not in num_cols])
            if op == "onehot":
                params["drop_first"] = st.checkbox(
                    "Drop first level (avoids collinearity)", value=True
                )
            if op in fe.LEARNED_OPS:
                st.caption("⚠️ Frequencies are learned from all rows.")
        elif op == "date_parts":
            cols = st.multiselect("Date columns", all_cols)
            params["parts"] = st.multiselect(
                "Parts", list(fe.DATE_PARTS), default=["year", "month", "dayofweek"]
            )
        elif op == "interaction":
            cols = st.multiselect("Two numeric columns", num_cols, max_selections=2)
            params["kind"] = st.selectbox("Combine by", fe.INTERACTIONS)
        elif op == "missing_indicator":
            cols = st.multiselect(
                "Columns with missing values", [c for c in all_cols if current[c].isna().any()]
            )
        else:
            cols = st.multiselect("Columns to drop", all_cols)

        new_spec = fe.FeatureSpec(op=op, columns=cols, params=params)
        if cols:
            try:
                result = fe.apply_spec(current, new_spec)
                added = [c for c in result.columns if c not in current.columns]
                removed = [c for c in current.columns if c not in result.columns]
                st.markdown(
                    f"**Preview** · adds {added or 'no columns'}"
                    + (f" · removes {removed}" if removed else "")
                )
                st.dataframe(
                    result[added or list(result.columns)[:6]].head(8), use_container_width=True
                )
                if st.button("Add this step", type="primary"):
                    st.session_state[KEY_FEATURE_SPECS] = [*specs, new_spec]
                    st.rerun()
            except fe.FeatureError as err:
                st.error(str(err))

    with tab_steps:
        if not specs:
            st.caption("No steps yet.")
        for n, spec in enumerate(specs, start=1):
            s1, s2 = st.columns([6, 1])
            s1.markdown(
                f"**{n}.** {fe.OP_LABELS[spec.op]} → `{esc(', '.join(spec.columns))}`",
                unsafe_allow_html=True,
            )
            if s2.button("Remove", key=f"rm_step_{n}"):
                st.session_state[KEY_FEATURE_SPECS] = [s for i, s in enumerate(specs) if i != n - 1]
                st.rerun()
        if specs:
            o1, o2, o3 = st.columns(3)
            name = unique_name(st.session_state.get("dataset_library") or {}, "features")
            if o1.button("Save as library table"):
                add_table(st.session_state, name, current)
                st.success(f"Saved as `{name}`.")
            if o2.button("Use as the active dataset"):
                try:
                    ingest_dataframe(
                        st.session_state, current, "engineered", settings.max_upload_size_mb
                    )
                    st.switch_page("pages/1_Executive_Overview.py")
                except DataAnalystError as err:
                    st.error(err.user_message)
            with o3:
                render_code_button(
                    "Feature engineering",
                    fe.features_snippet(specs, **load_kwargs(dataset)),
                    key="code_fe",
                )
            st.download_button(
                "⬇️ Download recipe (JSON)",
                fe.specs_to_json(specs),
                "feature_recipe.json",
                "application/json",
            )
        recipe = st.file_uploader("Load a recipe", type=["json"], key="fe_recipe")
        if recipe is not None and st.button("Apply recipe"):
            try:
                loaded = fe.specs_from_json(recipe.getvalue())
                fe.apply_specs(base, loaded)  # validate against this dataset first
                st.session_state[KEY_FEATURE_SPECS] = loaded
                st.rerun()
            except fe.FeatureError as err:
                st.error(str(err))

    with tab_leak:
        st.caption(
            "Pick the model target and candidate features to check for leakage and risky columns."
        )
        target = st.selectbox("Target", all_cols, key="fe_target")
        feats = st.multiselect(
            "Features",
            [c for c in all_cols if c != target],
            default=[c for c in all_cols if c != target][:12],
        )
        warnings_ = fe.leakage_report(current, target, feats)
        if not warnings_:
            st.success("No leakage or risky-feature signals found.")
        for w in sorted(warnings_, key=lambda w: ("high", "medium", "low").index(w.level)):
            st.markdown(
                f"{severity_chip(w.level)} **{esc(w.column)}** - {esc(w.message)}",
                unsafe_allow_html=True,
            )
    return
