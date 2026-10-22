"""Reusable Streamlit UI components and styling helpers."""

import streamlit as st

from src.config.settings import get_settings
from src.core.session_state import (
    KEY_CURRENT_DATASET,
    KEY_DATASET_METADATA,
    KEY_GENERATED_FINDINGS,
    KEY_WORKSPACE_ID,
    init_session_state,
)


def render_sidebar_status() -> None:
    """Render consistent workspace status, dataset indicator, and version in the sidebar."""
    init_session_state()
    settings = get_settings()

    st.sidebar.markdown(f"### 📊 {settings.app_name}")
    st.sidebar.caption(f"v{settings.app_version} • Environment: `{settings.app_env}`")
    st.sidebar.markdown("---")

    # Workspace details
    workspace_id = st.session_state.get(KEY_WORKSPACE_ID, "default")
    st.sidebar.markdown(f"**Workspace:** `{workspace_id}`")

    # Dataset status
    metadata = st.session_state.get(KEY_DATASET_METADATA)
    if metadata:
        st.sidebar.success(f"**Dataset:** {metadata.get('filename', 'Loaded')}")
        st.sidebar.caption(
            f"{metadata.get('row_count', 0):,} rows | {metadata.get('column_count', 0):,} cols"
        )
    else:
        st.sidebar.info("⚡ **Status:** No dataset loaded")

    st.sidebar.markdown("---")
    st.sidebar.markdown("#### LLM Integration")
    if settings.is_llm_configured:
        st.sidebar.success(f"Configured (`{settings.gemini_model}`)")
    else:
        st.sidebar.caption("Not configured (Phase 1 local mode)")


def render_page_header(title: str, subtitle: str, badge: str = "Foundation") -> None:
    """Render consistent page title header with high-contrast badge."""
    col1, col2 = st.columns([5, 1])
    with col1:
        st.title(title)
        st.caption(subtitle)
    with col2:
        st.markdown(
            f"<div style='text-align: right; margin-top: 1.5rem;'>"
            f"<span style='background: #1e293b; color: #38bdf8; padding: 4px 10px; "
            f"border-radius: 9999px; font-size: 0.8rem; font-weight: 600; "
            f"border: 1px solid #334155;'>{badge}</span></div>",
            unsafe_allow_html=True,
        )
    st.markdown("---")


def render_empty_state_banner(page_name: str, phase_label: str = "Phase 2") -> None:
    """Render an honest, clean empty state when no dataset or model is active."""
    st.info(
        f"**{page_name} Empty State** — No active dataset or analytical results found in the current workspace.\n\n"
        f"Analytical computation and live data processing for this module will be activated in **{phase_label}**.\n\n"
        "👉 Return to the **[Main Workspace](/)** or upload a dataset once ingestion is enabled."
    )


def get_workspace_kpis() -> dict[str, str]:
    """Retrieve verified workspace metrics or explicit empty states."""
    metadata = st.session_state.get(KEY_DATASET_METADATA)
    dataset = st.session_state.get(KEY_CURRENT_DATASET)
    findings = st.session_state.get(KEY_GENERATED_FINDINGS, [])

    if metadata and hasattr(metadata, "filename"):
        dataset_name = metadata.filename
        rows_str = f"{metadata.row_count:,}"
        cols_str = f"{metadata.column_count:,}"
    elif isinstance(metadata, dict):
        dataset_name = str(metadata.get("filename", "Active Dataset"))
        rows_str = f"{metadata.get('row_count', 0):,}"
        cols_str = f"{metadata.get('column_count', 0):,}"
    elif dataset is not None and hasattr(dataset, "shape"):
        dataset_name = "Loaded in memory"
        rows_str = f"{dataset.shape[0]:,}"
        cols_str = f"{dataset.shape[1]:,}"
    else:
        dataset_name = "Not loaded"
        rows_str = "0"
        cols_str = "0"

    insights_str = f"{len(findings):,}" if findings else "0"

    return {
        "dataset": dataset_name,
        "rows": rows_str,
        "cols": cols_str,
        "insights": insights_str,
    }
