"""AI Data Analyst Agent - Main Workspace Entry Point."""

import streamlit as st
from src.config.settings import get_settings
from src.core.logging_config import setup_logging
from src.core.session_state import KEY_WORKSPACE_ID, init_session_state
from src.core.ui_components import (
    get_workspace_kpis,
    render_page_header,
    render_sidebar_status,
)

# Initialize application settings and logging
settings = get_settings()
setup_logging(settings.log_level)

# Page configuration
st.set_page_config(
    page_title=f"{settings.app_name} | Main Workspace",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize shared session state
init_session_state()

# Sidebar
render_sidebar_status()

# Main Header
render_page_header(
    title=f"📊 {settings.app_name}",
    subtitle="From raw data to validated insights and informed decisions.",
    badge="Phase 1 Foundation",
)

# Workspace Overview & KPI Indicators
kpis = get_workspace_kpis()
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="Dataset Status", value=kpis["dataset"])
with col2:
    st.metric(label="Rows Analyzed", value=kpis["rows"])
with col3:
    st.metric(label="Columns Detected", value=kpis["cols"])
with col4:
    st.metric(label="Insights Generated", value=kpis["insights"])

st.markdown("<br>", unsafe_allow_html=True)

# Workspace Modules Grid
st.subheader("Analytical Workspace Modules")
st.caption(
    "Explore dedicated modules designed for data quality profiling, deep exploratory analysis, "
    "statistical inference, and autonomous AI investigations."
)

m_col1, m_col2, m_col3 = st.columns(3)

with m_col1:
    st.markdown(
        """
        ### 🔍 Data Quality Lab
        **Scope:** Schema inference, missingness analysis, duplicate detection, and outlier audits.
        - **Status:** *Planned for Phase 2*
        - **Next Steps:** Automated quality scoring and repair recommendations.
        """
    )
    if st.button("Open Data Quality Lab", key="nav_dq"):
        st.switch_page("pages/2_Data_Quality.py")

with m_col2:
    st.markdown(
        """
        ### 📈 Exploration Studio
        **Scope:** Interactive distributions, correlation matrices, and automated chart selection.
        - **Status:** *Planned for Phase 2*
        - **Next Steps:** Plotly visualizations with written analytical context.
        """
    )
    if st.button("Open Exploration Studio", key="nav_eda"):
        st.switch_page("pages/3_Exploration_Studio.py")

with m_col3:
    st.markdown(
        """
        ### 🧪 Statistical Intelligence
        **Scope:** Hypothesis testing, effect size calculations, confidence intervals, and assumptions.
        - **Status:** *Planned for Phase 2*
        - **Next Steps:** SciPy/statsmodels rigorous verification.
        """
    )
    if st.button("Open Statistical Intelligence", key="nav_stats"):
        st.switch_page("pages/4_Statistical_Intelligence.py")

st.markdown("<br>", unsafe_allow_html=True)

m_col4, m_col5, m_col6 = st.columns(3)

with m_col4:
    st.markdown(
        """
        ### 🤖 AI Analyst Workspace
        **Scope:** Autonomous multi-step planning, evidence-linked findings, and conversational drill-down.
        - **Status:** *Planned for Phase 4*
        - **Next Steps:** 6-layer structured interpretation engine.
        """
    )
    if st.button("Open AI Analyst", key="nav_analyst"):
        st.switch_page("pages/5_AI_Analyst.py")

with m_col5:
    st.markdown(
        """
        ### 🔮 Predictive Lab
        **Scope:** Chronological time-series forecasting, machine learning models, and residual checks.
        - **Status:** *Planned for Phase 5*
        - **Next Steps:** Walk-forward validation and feature importance.
        """
    )
    if st.button("Open Predictive Lab", key="nav_pred"):
        st.switch_page("pages/6_Predictive_Lab.py")

with m_col6:
    st.markdown(
        """
        ### 📑 Intelligence Reports
        **Scope:** Exportable executive summaries, methodology logs, and decision-ready briefings.
        - **Status:** *Planned for Phase 6*
        - **Next Steps:** PDF/HTML report compilation.
        """
    )
    if st.button("Open Intelligence Reports", key="nav_reports"):
        st.switch_page("pages/7_Intelligence_Reports.py")

st.markdown("---")

# Getting Started Section
g_col1, g_col2 = st.columns([2, 1])

with g_col1:
    st.subheader("Analytical Workflow")
    st.markdown(
        """
        The platform enforces a rigorous **tool-first, evidence-backed** architecture:

        1. **Ingest & Profile:** Upload raw CSV, Excel, or Parquet datasets without modifying original records.
        2. **Inspect Quality:** Detect missing values, schema inconsistencies, and potential target columns.
        3. **Formulate Hypotheses:** Explore distributions, trends, and segment relationships.
        4. **Test & Model:** Run statistical tests and train cross-validated predictive models.
        5. **Interpret & Decide:** The agent translates validated evidence into actionable executive reports.
        """
    )

with g_col2:
    st.subheader("System Status")
    workspace_id = st.session_state.get(KEY_WORKSPACE_ID, "default")
    st.write(f"**Application Version:** `{settings.app_version}`")
    st.write(f"**Runtime Environment:** `{settings.app_env}`")
    st.write(f"**Active Workspace:** `{workspace_id}`")
    st.write(f"**Max Upload Size:** `{settings.max_upload_size_mb} MB`")
    st.write(
        f"**LLM Status:** `{'Configured' if settings.is_llm_configured else 'Local Mode (No Key)'}`"
    )
    st.write(f"**Log Level:** `{settings.log_level}`")
