"""Page 8: Projects & Sessions - save, version, and restore whole analysis workspaces."""

import pandas as pd
import streamlit as st
from src.config.settings import get_settings
from src.core.session_state import KEY_CURRENT_DATASET
from src.core.ui_components import page_setup
from src.workspace.projects import ProjectError, ProjectStore

page_setup(
    "Projects & Sessions",
    "💾",
    "Save your work as versioned projects and pick it up again after a refresh.",
)
settings = get_settings()
store = ProjectStore()
dataset = st.session_state.get(KEY_CURRENT_DATASET)

save_tab, open_tab, history_tab = st.tabs(["💾 Save", "📂 Open", "🕘 History"])

with save_tab:
    if dataset is None:
        st.info("Load a dataset first (Executive Overview), then save it here.")
    else:
        st.caption(
            "A project stores the raw data once, plus a new **version** each time you save: "
            "cleaning steps, data dictionary, dashboard and side tables."
        )
        existing = store.list_projects()
        default = dataset.filename.rsplit(".", 1)[0]
        c1, c2 = st.columns([2, 3])
        name = c1.text_input("Project name", value=default, key="proj_name")
        note = c2.text_input("Version note (optional)", placeholder="e.g. after cleaning nulls")
        if any(p.slug == (name or "").strip().lower().replace(" ", "-") for p in existing):
            st.caption("A project with this name exists: saving adds a new version.")
        if st.button("Save project", type="primary", disabled=not name.strip()):
            try:
                version = store.save(st.session_state, name, note)
                st.success(f"Saved **{name}** as version {version}.")
            except ProjectError as err:
                st.error(str(err))
            except Exception as err:
                st.error(f"Could not save: {err}")

with open_tab:
    projects = store.list_projects()
    if not projects:
        st.info("No saved projects yet.")
    else:
        labels = {
            p.slug: f"{p.name}  ·  {p.versions} version(s)  ·  {p.updated_at}" for p in projects
        }
        slug = st.selectbox("Project", list(labels), format_func=labels.get)
        versions = store.versions(slug)
        version = st.selectbox(
            "Version",
            [v["version"] for v in reversed(versions)],
            format_func=lambda n: next(
                f"v{n} · {v['saved_at']} · {v.get('note') or 'no note'}"
                for v in versions
                if v["version"] == n
            ),
        )
        meta = next(v for v in versions if v["version"] == version)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Rows", f"{meta['rows']:,}")
        m2.metric("Columns", f"{meta['columns']}")
        m3.metric("Cleaning steps", meta["transformations"])
        q = meta.get("quality_score")
        m4.metric("Quality", f"{q:.0f}/100" if q is not None else "n/a")
        if dataset is not None:
            st.warning("Opening a project replaces the dataset currently loaded in this session.")
        if st.button("Open this version", type="primary"):
            try:
                result = store.load(st.session_state, slug, version, settings.max_upload_size_mb)
            except ProjectError as err:
                st.error(str(err))
            else:
                for err in result["replay_errors"]:
                    st.warning(f"A cleaning step could not be replayed: {err}")
                st.success("Project restored.")
                st.page_link("pages/1_Executive_Overview.py", label="Go to the Overview →")
        with st.expander("Delete this project"):
            confirm = st.checkbox("Yes, permanently delete all versions", key="proj_del_ok")
            if st.button("Delete project", disabled=not confirm):
                store.delete(slug)
                st.rerun()

with history_tab:
    projects = store.list_projects()
    if not projects:
        st.caption("History appears after the first save.")
    else:
        slug = st.selectbox("Project ", [p.slug for p in projects], key="hist_slug")
        st.markdown("##### Versions")
        st.dataframe(
            pd.DataFrame(store.versions(slug))[
                [
                    "version",
                    "saved_at",
                    "note",
                    "rows",
                    "columns",
                    "transformations",
                    "quality_score",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("##### Run history")
        events = store.events(slug)
        st.dataframe(pd.DataFrame(events[::-1]), use_container_width=True, hide_index=True)
