"""Page 1: Executive Overview - High-level summary of datasets, metrics, and key findings."""

import streamlit as st
from src.config.settings import get_settings
from src.core.session_state import (
    KEY_ANALYSIS_PLAN,
    KEY_DATA_QUALITY_REPORT,
    KEY_EXECUTION_TRACE,
    KEY_GENERATED_FINDINGS,
    init_session_state,
)
from src.core.ui_components import (
    get_workspace_kpis,
    render_page_header,
    render_sidebar_status,
)

settings = get_settings()
st.set_page_config(
    page_title=f"{settings.app_name} | Executive Overview",
    page_icon="📈",
    layout="wide",
)

init_session_state()
render_sidebar_status()

render_page_header(
    title="📈 Executive Overview",
    subtitle="High-level synthesis of dataset health, analytical progress, and verified findings.",
    badge="Module Overview",
)

# Workspace Metrics
kpis = get_workspace_kpis()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Active Dataset", kpis["dataset"])
c2.metric("Observed Rows", kpis["rows"])
c3.metric("Observed Columns", kpis["cols"])
c4.metric("Key Findings", kpis["insights"])

st.markdown("---")

col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("Key Findings & Executive Insights")
    findings = st.session_state.get(KEY_GENERATED_FINDINGS, [])
    if findings:
        for f in findings:
            sev = getattr(f, "severity", getattr(f, "severity_or_importance", "medium"))
            st.markdown(f"**{f.title}** (`{str(sev).upper()}`)")
            st.write(f.summary)
    else:
        st.info(
            "**No Findings Generated Yet**\n\n"
            "Once a dataset is analyzed in the **Exploration Studio**, **AI Analyst**, or **Statistical Intelligence** "
            "workspaces, verified findings and executive takeaways will be synthesized here."
        )

    st.subheader("Data Quality Status")
    dq_report = st.session_state.get(KEY_DATA_QUALITY_REPORT)
    if dq_report and hasattr(dq_report, "quality_score"):
        qs = dq_report.quality_score
        dqc1, dqc2, dqc3 = st.columns(3)
        dqc1.metric("Overall Score", f"{qs.overall_score:.0f}/100")
        dqc2.metric("Missing Cells", f"{dq_report.missing_cell_percentage:.1f}%")
        dqc3.metric("Duplicate Rows", f"{dq_report.duplicate_row_percentage:.1f}%")

        if dq_report.quality_issues:
            st.caption(f"**Top Quality Issues ({len(dq_report.quality_issues)} total):**")
            for iss in dq_report.quality_issues[:3]:
                st.markdown(
                    f"- **[{iss.severity.upper()}]** {iss.title}: *{iss.recommended_action}*"
                )
        else:
            st.success("No critical or high-severity quality anomalies detected.")
    else:
        st.info(
            "Quality score and anomaly summary will be generated automatically upon data ingestion."
        )

with col_right:
    st.subheader("Investigation Progress")
    plan = st.session_state.get(KEY_ANALYSIS_PLAN)
    trace = st.session_state.get(KEY_EXECUTION_TRACE, [])

    if plan:
        st.markdown(f"**Objective:** {plan.objective}")
        for step in plan.steps:
            st.write(f"- [{step.status.upper()}] {step.tool_name}: {step.purpose}")
    elif trace:
        for item in trace:
            st.write(f"- {item}")
    else:
        st.caption("No active analysis plan running in the current session.")

    st.markdown("---")
    st.markdown("#### Quick Navigation")
    st.page_link("app.py", label="← Return to Main Workspace", icon="🏠")
    st.page_link("pages/2_Data_Quality.py", label="Go to Data Quality Lab", icon="🔍")
    st.page_link("pages/3_Exploration_Studio.py", label="Go to Exploration Studio", icon="📈")
    st.page_link("pages/5_AI_Analyst.py", label="Go to AI Analyst", icon="🤖")
