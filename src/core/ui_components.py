"""Reusable Streamlit UI components and styling helpers."""

import html
from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import streamlit as st

from src.config.settings import get_settings
from src.core.code_options import (
    COMMENT_LEVELS,
    DEFAULT_OPTIONS,
    LOADERS,
    PALETTES,
    PLOT_LIBS,
    STYLES,
    TEMPLATES,
    CodeOptions,
    build_notebook,
    colab_cell,
    combine_snippets,
)
from src.core.session_state import (
    KEY_CURRENT_DATASET,
    KEY_DATA_QUALITY_REPORT,
    KEY_DATASET_METADATA,
    KEY_GENERATED_FINDINGS,
    KEY_TRANSFORMATION_PIPELINE,
    KEY_UI_COMPACT,
    KEY_WORKSPACE_ID,
    init_session_state,
)

if TYPE_CHECKING:
    from src.core.overview import DisplayFinding


SEVERITY_COLORS = {
    "critical": ("#fecaca", "rgba(239, 68, 68, 0.18)", "#ef4444"),
    "high": ("#fed7aa", "rgba(249, 115, 22, 0.18)", "#f97316"),
    "medium": ("#fde68a", "rgba(234, 179, 8, 0.16)", "#eab308"),
    "low": ("#bae6fd", "rgba(56, 189, 248, 0.16)", "#38bdf8"),
}

_BASE_CSS = """
<style>
.block-container { padding-top: 2.2rem; }
h1 { font-size: 2rem !important; letter-spacing: -0.01em; }
[data-testid="stMetricValue"] { font-size: 1.8rem; font-weight: 650; }
[data-testid="stMetricLabel"] { color: #94a3b8; text-transform: uppercase;
    font-size: 0.72rem; letter-spacing: 0.06em; }
[data-testid="stCaptionContainer"] { color: #94a3b8; }
.chip { display: inline-block; padding: 2px 10px; border-radius: 9999px; font-size: 0.72rem;
    font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; }
.fcard { background: rgba(19, 27, 46, 0.7); border: 1px solid #263248; border-left-width: 4px;
    border-radius: 10px; padding: 12px 16px; margin: 10px 0 4px 0; }
.fcard-title { font-size: 1rem; font-weight: 600; color: #f1f5f9; margin: 4px 0 2px 0; }
.fcard-body { font-size: 0.9rem; color: #b6c2d4; line-height: 1.5; }
.fcard-meta { font-size: 0.72rem; color: #94a3b8; margin-left: 8px; text-transform: uppercase; }
.pill { display: inline-block; padding: 4px 12px; border-radius: 9999px; font-size: 0.82rem;
    background: rgba(2, 132, 199, 0.16); color: #7dd3fc; border: 1px solid #0369a1; }
.sidebar-pill { background: rgba(16, 185, 129, 0.14); border: 1px solid #047857;
    color: #a7f3d0; border-radius: 8px; padding: 8px 12px; font-size: 0.85rem;
    word-break: break-all; }
.hero { text-align: center; padding: 3rem 0 1rem 0; }
.hero h2 { font-size: 2.2rem; font-weight: 700; margin-bottom: 0.4rem; }
.hero p { color: #94a3b8; font-size: 1.05rem; }
/* Only keep a minimum width while the sidebar is open; a forced min-width left an empty
   250px column behind after collapsing it. */
[data-testid="stSidebar"][aria-expanded="true"] { min-width: 250px; }
[data-testid="stSidebar"][aria-expanded="false"] { min-width: 0 !important; }
/* Ambient background: soft colour glows over a deep navy base. */
[data-testid="stApp"] {
    background:
        radial-gradient(900px 500px at 8% -5%, rgba(2, 132, 199, 0.22), transparent 60%),
        radial-gradient(800px 500px at 100% 0%, rgba(124, 58, 237, 0.18), transparent 60%),
        radial-gradient(900px 600px at 50% 110%, rgba(16, 185, 129, 0.10), transparent 60%),
        linear-gradient(180deg, #0b0f17 0%, #0d1424 100%);
    background-attachment: fixed;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] > div:first-child {
    background: linear-gradient(180deg, rgba(19, 27, 46, 0.96) 0%, rgba(11, 15, 23, 0.98) 100%);
    border-right: 1px solid #1f2a40; }
[data-testid="stVerticalBlockBorderWrapper"] { background: rgba(19, 27, 46, 0.45);
    backdrop-filter: blur(2px); }
/* Use the full window width (wide layout), with comfortable side padding. */
.block-container, [data-testid="stMainBlockContainer"] {
    max-width: 100% !important; padding-left: 2.5rem !important; padding-right: 2.5rem !important; }
</style>
"""

_COMPACT_CSS = """
<style>
.block-container { padding-top: 1rem; }
[data-testid="stVerticalBlock"] { gap: 0.6rem; }
[data-testid="stMetricValue"] { font-size: 1.4rem; }
.fcard { padding: 8px 12px; margin-top: 6px; }
</style>
"""


def inject_global_styles() -> None:
    """Apply the shared typography scale, card styles, and optional compact density."""
    st.markdown(_BASE_CSS, unsafe_allow_html=True)
    if st.session_state.get(KEY_UI_COMPACT):
        st.markdown(_COMPACT_CSS, unsafe_allow_html=True)


def severity_chip(severity: str) -> str:
    fg, bg, _ = SEVERITY_COLORS.get(severity, SEVERITY_COLORS["medium"])
    return f'<span class="chip" style="color:{fg};background:{bg};">{esc(severity)}</span>'


def render_finding_card(item: "DisplayFinding", key: str) -> None:
    """Render a severity-coded finding card with optional detail list and a deep-link."""
    _, _, border = SEVERITY_COLORS.get(item.severity, SEVERITY_COLORS["medium"])
    st.markdown(
        f'<div class="fcard" style="border-left-color:{border};">'
        f'{severity_chip(item.severity)}<span class="fcard-meta">{esc(item.category.replace("_", " "))}</span>'
        f'<div class="fcard-title">{esc(item.title)}</div>'
        f'<div class="fcard-body">{esc(item.summary)}</div></div>',
        unsafe_allow_html=True,
    )
    if item.details:
        with st.expander(f"Show all {len(item.details)}"):
            for d in item.details:
                st.markdown(f"- {d}")
    st.page_link(item.page, label="Explore →", icon=None, help="Open the relevant workspace")


def render_sidebar_status() -> None:
    """Render consistent workspace status, dataset indicator, and version in the sidebar."""
    init_session_state()
    reset_code_collection()
    inject_global_styles()
    settings = get_settings()

    st.sidebar.markdown(f"### 📊 {settings.app_name}")
    st.sidebar.caption(f"v{settings.app_version} • Environment: `{settings.app_env}`")
    st.sidebar.markdown("---")

    # Workspace details
    workspace_id = st.session_state.get(KEY_WORKSPACE_ID, "default")
    st.sidebar.markdown(f"**Workspace:** `{workspace_id}`")

    # Dataset status
    dataset = st.session_state.get(KEY_CURRENT_DATASET)
    metadata = st.session_state.get(KEY_DATASET_METADATA)

    if dataset is not None and hasattr(dataset, "working_df"):
        st.sidebar.markdown(
            f'<div class="sidebar-pill"><strong>Dataset</strong><br>{esc(dataset.filename)}</div>',
            unsafe_allow_html=True,
        )
        st.sidebar.caption(
            f"{len(dataset.working_df):,} rows | {len(dataset.working_df.columns):,} cols"
        )
    elif metadata:
        fname = (
            metadata.filename
            if hasattr(metadata, "filename")
            else metadata.get("filename", "Loaded")
        )
        r_cnt = (
            metadata.row_count if hasattr(metadata, "row_count") else metadata.get("row_count", 0)
        )
        c_cnt = (
            metadata.column_count
            if hasattr(metadata, "column_count")
            else metadata.get("column_count", 0)
        )
        st.sidebar.markdown(
            f'<div class="sidebar-pill"><strong>Dataset</strong><br>{esc(fname)}</div>',
            unsafe_allow_html=True,
        )
        st.sidebar.caption(f"{r_cnt:,} rows | {c_cnt:,} cols")
    else:
        st.sidebar.info("⚡ **Status:** No dataset loaded")

    # Quality score badge in sidebar if profiled
    profile = st.session_state.get(KEY_DATA_QUALITY_REPORT)
    if profile is not None and hasattr(profile, "quality_score"):
        score = profile.quality_score.overall_score
        st.sidebar.markdown(f"**Quality Score:** `{score}/100`")

    st.sidebar.markdown("---")
    st.sidebar.markdown("#### LLM Integration")
    if settings.is_llm_configured:
        st.sidebar.success(f"Configured (`{settings.gemini_model}`)")
    else:
        st.sidebar.caption("Not configured (Local mode)")

    st.sidebar.markdown("---")
    st.sidebar.toggle(
        "Compact density",
        key=KEY_UI_COMPACT,
        help="Tighter spacing and smaller metrics for dense screens.",
    )


def esc(value: object) -> str:
    """HTML-escape any dataset- or model-derived value before interpolating into raw HTML."""
    return html.escape(str(value), quote=True)


def render_page_header(title: str, subtitle: str, badge: str = "") -> None:
    """Render consistent page title header; an optional badge is shown top-right."""
    col1, col2 = st.columns([5, 1])
    with col1:
        st.title(title)
        st.caption(subtitle)
    if badge:
        with col2:
            st.markdown(
                f"<div style='text-align: right; margin-top: 1.5rem;'>"
                f"<span class='pill'>{esc(badge)}</span></div>",
                unsafe_allow_html=True,
            )
    st.markdown("---")


def render_empty_state_banner(page_name: str, phase_label: str = "") -> None:
    """Render a clean empty state when no dataset is active (phase_label kept for compatibility)."""
    st.info(
        f"**{page_name}** needs a dataset. Attach a CSV or Excel file on the Executive Overview, "
        "or load the sample dataset to try it out."
    )
    st.page_link("pages/1_Executive_Overview.py", label="Upload a dataset →", icon="📥")


def get_workspace_kpis() -> dict[str, str]:
    """Retrieve verified workspace metrics or explicit empty states."""
    dataset = st.session_state.get(KEY_CURRENT_DATASET)
    metadata = st.session_state.get(KEY_DATASET_METADATA)
    findings = st.session_state.get(KEY_GENERATED_FINDINGS, [])
    profile = st.session_state.get(KEY_DATA_QUALITY_REPORT)
    pipeline = st.session_state.get(KEY_TRANSFORMATION_PIPELINE)

    if dataset is not None and hasattr(dataset, "working_df"):
        dataset_name = dataset.filename
        rows_str = f"{len(dataset.working_df):,}"
        cols_str = f"{len(dataset.working_df.columns):,}"
    elif metadata and hasattr(metadata, "filename"):
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

    quality_score_str = "N/A"
    issues_count_str = "0"
    missing_pct_str = "0.0%"

    if profile is not None and hasattr(profile, "quality_score"):
        quality_score_str = f"{profile.quality_score.overall_score:.0f}/100"
        issues_count_str = f"{len(profile.quality_issues):,}"
        missing_pct_str = f"{profile.missing_cell_percentage:.1f}%"

    transformations_count_str = "0"
    if pipeline is not None and hasattr(pipeline, "records"):
        transformations_count_str = f"{len(pipeline.records):,}"

    return {
        "dataset": dataset_name,
        "rows": rows_str,
        "cols": cols_str,
        "insights": insights_str,
        "quality_score": quality_score_str,
        "issues_count": issues_count_str,
        "missing_pct": missing_pct_str,
        "transformations_count": transformations_count_str,
    }


UPLOAD_SESSION_KEY = (
    "overview_upload_payload"  # (filename, bytes) kept only for Excel sheet switching
)


def render_upload_searchbar(show_help: bool = True) -> None:
    """Render a search-bar style input to attach a CSV/Excel file and load it as the dataset."""
    from src.core.exceptions import DataAnalystError
    from src.core.logging_config import get_logger
    from src.ingestion.loader import get_excel_sheet_names
    from src.ingestion.service import ingest_into_state, needs_ingestion

    logger = get_logger("ui.upload_searchbar")
    settings = get_settings()

    def _ingest(file_bytes: bytes, filename: str, sheet: str | None) -> bool:
        try:
            with st.spinner(f"Ingesting and profiling '{filename}'..."):
                ingest_into_state(
                    st.session_state,
                    file_bytes=file_bytes,
                    filename=filename,
                    max_size_mb=settings.max_upload_size_mb,
                    sheet_name=sheet,
                )
            return True
        except DataAnalystError as err:
            st.error(f"**Ingestion Error:** {err.user_message}")
            logger.error(f"User-facing ingestion failure: {err.message}")
        except Exception as ex:
            st.error(f"**Unexpected Error:** Failed to read file: {ex}")
            logger.exception("Unexpected ingestion error")
        return False

    with st.container():
        submitted = st.chat_input(
            f"Attach a CSV or Excel file (max {settings.max_upload_size_mb} MB) and press Enter",
            accept_file=True,
            file_type=["csv", "xlsx", "xls"],
            max_upload_size=settings.max_upload_size_mb,
            key="overview_upload_bar",
        )
        if show_help:
            st.caption(
                "Click the attachment icon, choose a **CSV / Excel** file, then press Enter."
            )

    if submitted:
        files = getattr(submitted, "files", None) or submitted.get("files", [])
        if not files:
            st.warning("No file attached. Use the attachment icon to pick a CSV or Excel file.")
        else:
            upload = files[0]
            data = upload.getvalue()
            if _ingest(data, upload.name, None):
                if upload.name.lower().endswith((".xlsx", ".xls")):
                    st.session_state[UPLOAD_SESSION_KEY] = (upload.name, data)
                else:
                    st.session_state.pop(UPLOAD_SESSION_KEY, None)
                st.toast(f"Loaded {upload.name}", icon="✅")
                st.rerun()

    # Worksheet switcher for multi-sheet Excel workbooks loaded via the bar.
    payload = st.session_state.get(UPLOAD_SESSION_KEY)
    active = st.session_state.get(KEY_CURRENT_DATASET)
    if payload and active is not None and active.filename == payload[0]:
        try:
            sheets = get_excel_sheet_names(payload[1])
        except Exception:
            sheets = []
        if len(sheets) > 1:
            used = active.ingestion_report.sheet_name_used
            index = sheets.index(used) if used in sheets else 0
            chosen = st.selectbox("Worksheet", sheets, index=index, key="overview_sheet_select")
            if needs_ingestion(st.session_state, payload[0], chosen) and _ingest(
                payload[1], payload[0], chosen
            ):
                st.rerun()


CODE_DEFAULTS_KEY = "code_option_defaults"
CODE_COLLECT_KEY = "code_collect"


def reset_code_collection() -> None:
    """Called at the start of every page run so the export only has this run's snippets."""
    st.session_state[CODE_COLLECT_KEY] = {}


def _collect(key: str, title: str, code: str) -> None:
    st.session_state.setdefault(CODE_COLLECT_KEY, {})[key] = (title, code)


def _code_options_panel(key: str) -> CodeOptions:
    """Widgets for customizing generated code; choices are remembered across buttons."""
    d: dict[str, Any] = st.session_state.get(CODE_DEFAULTS_KEY) or {}

    def pick(label: str, options: list[str], field: str, fmt=None, **kw: Any) -> str:  # type: ignore[no-untyped-def]
        current = d.get(field, getattr(DEFAULT_OPTIONS, field))
        idx = options.index(current) if current in options else 0
        return st.selectbox(
            label, options, index=idx, key=f"{key}_o_{field}", format_func=fmt or str, **kw
        )

    with st.expander("⚙️ Customize the code", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            plot_lib = pick("Charts with", list(PLOT_LIBS), "plot_lib")
            comments = pick("Comments", list(COMMENT_LEVELS), "comments")
        with c2:
            loader = pick("Load data from", list(LOADERS), "loader", fmt=LOADERS.get)
            style = pick(
                "Layout",
                list(STYLES),
                "style",
                fmt={"notebook": "Notebook cells", "function": "Functions + main()"}.get,
            )
        with c3:
            template = pick("Chart theme", list(TEMPLATES), "template")
            palette = pick("Palette", list(PALETTES), "palette")
        checks = st.toggle(
            "Add sanity checks (asserts)", value=bool(d.get("checks", False)), key=f"{key}_o_checks"
        )
        source_url = ""
        if loader in ("url", "gsheet"):
            source_url = st.text_input(
                "Sheet / file URL" if loader == "gsheet" else "File URL",
                value=d.get("source_url", ""),
                key=f"{key}_o_url",
            )
        s1, s2, s3 = st.columns(3)
        title = s1.text_input(
            "Chart title", value=d.get("title", ""), key=f"{key}_o_title", placeholder="(default)"
        )
        x_label = s2.text_input("X label", value=d.get("x_label", ""), key=f"{key}_o_xl")
        y_label = s3.text_input("Y label", value=d.get("y_label", ""), key=f"{key}_o_yl")
        w1, w2 = st.columns(2)
        width = w1.slider("Chart width", 400, 1600, int(d.get("width", 900)), 50, key=f"{key}_o_w")
        height = w2.slider(
            "Chart height", 250, 1000, int(d.get("height", 450)), 50, key=f"{key}_o_h"
        )
    opts = CodeOptions(
        plot_lib=plot_lib,
        comments=comments,
        checks=checks,
        style=style,
        loader=loader,
        source_url=source_url,
        template=template,
        palette=palette,
        width=width,
        height=height,
        title=title,
        x_label=x_label,
        y_label=y_label,
    ).sanitized()
    st.session_state[CODE_DEFAULTS_KEY] = {f: getattr(opts, f) for f in opts.__dataclass_fields__}
    return opts


def render_code_button(
    title: str,
    builder: Any,
    key: str,
    label: str = "</> View code",
    r_builder: Callable[[], str] | None = None,
    sql_builder: Callable[[], str] | None = None,
) -> None:
    """Popover with runnable code. `builder` is a ready string or ``f(CodeOptions) -> str``.

    Callable builders get the customization panel plus Colab/R/SQL tabs; every snippet shown
    on a page is also collected for the page-wide "export all code" notebook.
    """
    slug = "".join(c if c.isalnum() else "_" for c in title.lower()).strip("_")[:40] or "snippet"
    with st.popover(label):
        st.markdown(f"**{esc(title)}**")
        if callable(builder):
            opts = _code_options_panel(key)
            code = builder(opts)
            _collect(key, title, builder(replace(opts, style="notebook")))
        else:
            code = builder
            _collect(key, title, code)
        tabs = (
            ["🐍 Python", "📋 Colab cell"]
            + (["📊 R"] if callable(builder) else [])
            + (["🗄️ SQL"] if callable(builder) else [])
        )
        views = st.tabs(tabs)
        with views[0]:
            st.caption(
                "Standalone code for Google Colab / Jupyter. Replace the file name with your own data path."
            )
            st.code(code, language="python")
            c_py, c_nb = st.columns(2)
            c_py.download_button(
                "⬇️ .py script",
                data=code,
                file_name=f"{slug}.py",
                mime="text/x-python",
                key=f"{key}_dl_py",
                use_container_width=True,
            )
            c_nb.download_button(
                "⬇️ .ipynb notebook",
                data=build_notebook(code),
                file_name=f"{slug}.ipynb",
                mime="application/x-ipynb+json",
                key=f"{key}_dl_nb",
                use_container_width=True,
            )
        with views[1]:
            st.caption("One pasteable cell: copy it with the icon in the corner.")
            st.code(colab_cell(code), language="python")
        if callable(builder):
            for view, kind, fn, lang, ext in (
                (views[2], "R", r_builder, "r", "R"),
                (views[3], "SQL", sql_builder, "sql", "sql"),
            ):
                with view:
                    if fn is None:
                        st.caption(f"No {kind} version of this component yet.")
                    else:
                        text = fn()
                        st.code(text, language=lang)
                        st.download_button(
                            f"⬇️ .{ext}",
                            data=text,
                            file_name=f"{slug}.{ext}",
                            key=f"{key}_dl_{lang}",
                            use_container_width=True,
                        )


def render_code_export() -> None:
    """Sidebar download of every snippet shown on this page, combined into one notebook."""
    items = list((st.session_state.get(CODE_COLLECT_KEY) or {}).values())
    if not items:
        return
    combined = combine_snippets(items, title="Analysis export")
    with st.sidebar:
        st.markdown("---")
        st.markdown("#### </> Code export")
        st.caption(f"{len(items)} snippet(s) on this page")
        st.download_button(
            "⬇️ All code (.ipynb)",
            build_notebook(combined),
            "analysis_export.ipynb",
            "application/x-ipynb+json",
            key="export_all_nb",
            use_container_width=True,
        )
        st.download_button(
            "⬇️ All code (.py)",
            combined,
            "analysis_export.py",
            "text/x-python",
            key="export_all_py",
            use_container_width=True,
        )


def require_dataset(page_name: str) -> Any:
    """Return the active ManagedDataset or show the empty-state banner and stop the page."""
    dataset = st.session_state.get(KEY_CURRENT_DATASET)
    if dataset is None or not hasattr(dataset, "working_df"):
        render_empty_state_banner(page_name)
        st.stop()
    return dataset


def page_setup(title: str, icon: str, subtitle: str) -> None:
    """Common page boilerplate: config, state, sidebar, and header."""
    settings = get_settings()
    st.set_page_config(page_title=f"{settings.app_name} | {title}", page_icon=icon, layout="wide")
    init_session_state()
    render_sidebar_status()
    render_page_header(title=f"{icon} {title}", subtitle=subtitle)
