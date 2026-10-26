"""Page 1: Executive Overview - High-level summary of datasets, metrics, and key findings."""

import streamlit as st
from src.config.settings import get_settings
from src.core.caching import cached_eda_summary
from src.core.code_snippets import load_kwargs, profile_snippet
from src.core.exceptions import DataAnalystError
from src.core.logging_config import get_logger
from src.core.overview import (
    build_summary_markdown,
    column_type_counts,
    format_type_counts,
    group_findings,
    recommend_next_steps,
)
from src.core.session_state import (
    KEY_ANALYSIS_PLAN,
    KEY_ANALYST_PREFILL,
    KEY_CURRENT_DATASET,
    KEY_DATA_QUALITY_REPORT,
    KEY_DATASET_SCHEMA,
    KEY_EXECUTION_TRACE,
    KEY_GENERATED_FINDINGS,
    init_session_state,
    reset_dataset_state,
)
from src.core.ui_components import (
    UPLOAD_SESSION_KEY,
    esc,
    render_code_button,
    render_finding_card,
    render_page_header,
    render_sidebar_status,
    render_upload_searchbar,
    severity_chip,
)
from src.ingestion.loader import ManagedDataset
from src.ingestion.service import SAMPLE_CATALOG, load_catalog_sample, load_sample_dataset

logger = get_logger("ui.overview")
settings = get_settings()
st.set_page_config(
    page_title=f"{settings.app_name} | Executive Overview",
    page_icon="📈",
    layout="wide",
)

init_session_state()
render_sidebar_status()

TOP_FINDINGS = 5
EXAMPLE_QUESTIONS = [
    "Which customers are most likely to churn, and why?",
    "Which numeric features differ most between churned and retained customers?",
    "Could missing values or outliers be distorting the patterns?",
]


def _load_sample_and_go(question: str | None = None) -> None:
    try:
        load_sample_dataset(st.session_state, settings.max_upload_size_mb)
    except DataAnalystError as err:
        st.error(err.user_message)
        return
    if question:
        st.session_state[KEY_ANALYST_PREFILL] = question
        st.switch_page("pages/5_AI_Analyst.py")
    st.rerun()


def _render_sample_gallery() -> None:
    """Ten ready-made datasets, each suited to different parts of the app."""
    for row in range(0, len(SAMPLE_CATALOG), 5):
        cols = st.columns(5)
        for col, sample in zip(cols, SAMPLE_CATALOG[row : row + 5], strict=False):
            with col, st.container(border=True):
                st.markdown(f"##### {sample['icon']} {sample['title']}")
                st.caption(sample["blurb"])
                st.markdown(f"<small>🎯 {sample['use']}</small>", unsafe_allow_html=True)
                if st.button("Load", key=f"sample_{sample['key']}", use_container_width=True):
                    try:
                        load_catalog_sample(
                            st.session_state, sample["key"], settings.max_upload_size_mb
                        )
                    except DataAnalystError as err:
                        st.error(err.user_message)
                        return
                    st.rerun()


dataset = st.session_state.get(KEY_CURRENT_DATASET)

# ------------------------------------------------------------------ empty state / hero
if not isinstance(dataset, ManagedDataset):
    st.markdown(
        '<div class="hero"><h2>What would you like to analyze?</h2>'
        "<p>Attach a CSV or Excel file to profile quality, explore patterns, "
        "run statistics and ask the AI Analyst.</p></div>",
        unsafe_allow_html=True,
    )
    render_upload_searchbar()
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        if st.button("✨ Try the sample dataset", use_container_width=True, type="primary"):
            _load_sample_and_go()
    st.markdown("##### Or explore a sample dataset")
    _render_sample_gallery()
    st.caption("Or start from an example question on the sample customer-churn data:")
    cols = st.columns(len(EXAMPLE_QUESTIONS))
    for i, (col, question) in enumerate(zip(cols, EXAMPLE_QUESTIONS, strict=True)):
        with col:
            if st.button(question, key=f"example_q_{i}", use_container_width=True):
                _load_sample_and_go(question)
    st.stop()

# ------------------------------------------------------------------ loaded state
render_page_header(
    title="📈 Executive Overview",
    subtitle="High-level synthesis of dataset health, analytical progress, and verified findings.",
)
render_upload_searchbar(show_help=False)
with st.expander("🧪 Try a different sample dataset"):
    _render_sample_gallery()

df = dataset.working_df
profile = st.session_state.get(KEY_DATA_QUALITY_REPORT)
schema = st.session_state.get(KEY_DATASET_SCHEMA) or dataset.schema_info

# Make the page useful straight after upload: derive grounded findings once (cached).
if not st.session_state.get(KEY_GENERATED_FINDINGS):
    try:
        summary = cached_eda_summary(
            dataset, dataset.dataset_id, dataset.version, dataset.fingerprint
        )
        st.session_state[KEY_GENERATED_FINDINGS] = summary.findings
    except Exception:
        logger.exception("Automatic EDA findings failed")
findings = st.session_state.get(KEY_GENERATED_FINDINGS, [])
display = group_findings(findings)

info_col, clear_col = st.columns([6, 1])
with info_col:
    st.markdown(
        f'<span class="pill">📄 {esc(dataset.filename)}</span> '
        f'<span class="pill">{dataset.file_size_bytes / 1024:,.1f} KB</span>',
        unsafe_allow_html=True,
    )
with clear_col:
    if st.button("Clear dataset", use_container_width=True):
        reset_dataset_state(st.session_state)
        st.session_state.pop(UPLOAD_SESSION_KEY, None)
        st.rerun()

# Metric row
type_counts = column_type_counts(schema)
m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    with st.container(border=True):
        st.metric("Rows", f"{len(df):,}")
        st.caption(f"{df.memory_usage(deep=True).sum() / 1024**2:,.1f} MB in memory")
with m2:
    with st.container(border=True):
        st.metric("Columns", f"{len(df.columns):,}")
        st.caption(format_type_counts(type_counts))
with m3:
    with st.container(border=True):
        has_profile = profile is not None and hasattr(profile, "quality_score")
        st.metric(
            "Quality Score",
            f"{profile.quality_score.overall_score:.0f}/100" if has_profile else "N/A",
        )
        st.caption(
            f"{len(profile.quality_issues)} issue(s) detected" if has_profile else "Not profiled"
        )
with m4:
    with st.container(border=True):
        st.metric(
            "Missing Cells", f"{profile.missing_cell_percentage:.1f}%" if has_profile else "N/A"
        )
        st.caption(
            f"{profile.duplicate_row_percentage:.1f}% duplicate rows" if has_profile else " "
        )
with m5:
    with st.container(border=True):
        st.metric("Key Findings", f"{len(findings):,}")
        st.caption(f"{len(display)} insight card(s)")

st.markdown("&nbsp;", unsafe_allow_html=True)
col_left, col_right = st.columns([3, 2], gap="large")

with col_left:
    st.subheader("Key Findings")
    if display:
        for idx, item in enumerate(display[:TOP_FINDINGS]):
            render_finding_card(item, key=f"top_{idx}")
        rest = display[TOP_FINDINGS:]
        if rest:
            with st.expander(f"Show {len(rest)} more"):
                for idx, item in enumerate(rest):
                    render_finding_card(item, key=f"rest_{idx}")
    else:
        st.info(
            "No findings yet. Open the **Exploration Studio** or **AI Analyst** to generate verified insights."
        )

    with st.expander("Dataset snapshot"):
        st.dataframe(df.head(5), use_container_width=True)

with col_right:
    st.subheader("Data Quality")
    if profile is not None and hasattr(profile, "quality_score"):
        if profile.quality_issues:
            ranked = sorted(
                profile.quality_issues,
                key=lambda i: ("critical", "high", "medium", "low").index(i.severity),
            )
            for iss in ranked[:3]:
                st.markdown(
                    f"{severity_chip(iss.severity)} **{esc(iss.title)}**<br>"
                    f"<span class='fcard-body'>{esc(iss.recommended_action)}</span>",
                    unsafe_allow_html=True,
                )
            st.page_link("pages/2_Data_Quality.py", label="Open Data Quality Lab →")
        else:
            st.success("No quality anomalies detected.")

    st.subheader("Recommended Next Steps")
    for step in recommend_next_steps(profile, findings, schema):
        with st.container(border=True):
            st.markdown(f"**{step.icon} {esc(step.title)}**")
            st.caption(step.detail)
            st.page_link(step.page, label="Go →")

    plan = st.session_state.get(KEY_ANALYSIS_PLAN)
    trace = st.session_state.get(KEY_EXECUTION_TRACE, [])
    if plan or trace:
        st.subheader("Investigation Progress")
        if plan:
            st.markdown(f"**Objective:** {plan.objective}")
            for step in plan.steps:
                st.write(f"- [{step.status.upper()}] {step.tool_name}: {step.purpose}")
        else:
            for item in trace:
                st.write(f"- {item}")

    st.subheader("Ask the AI Analyst")
    with st.form("overview_ask_ai", border=False):
        question = st.text_input(
            "Question", placeholder="e.g. What drives churn?", label_visibility="collapsed"
        )
        if st.form_submit_button("Ask →", use_container_width=True) and question.strip():
            st.session_state[KEY_ANALYST_PREFILL] = question.strip()
            st.switch_page("pages/5_AI_Analyst.py")

    render_code_button(
        "Dataset profile", profile_snippet(**load_kwargs(dataset)), key="code_overview"
    )
    st.download_button(
        "⬇️ Download summary (Markdown)",
        data=build_summary_markdown(dataset.filename, len(df), len(df.columns), profile, display),
        file_name=f"summary_{dataset.filename.rsplit('.', 1)[0]}.md",
        mime="text/markdown",
        use_container_width=True,
    )
