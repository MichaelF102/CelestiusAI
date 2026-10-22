"""Page 7: Intelligence Reports - Executive briefing, methodology audit, and exportable reports."""

import streamlit as st
from src.config.settings import get_settings
from src.core.session_state import (
    KEY_DATASET_METADATA,
    KEY_GENERATED_FINDINGS,
    init_session_state,
)
from src.core.ui_components import (
    render_empty_state_banner,
    render_page_header,
    render_sidebar_status,
)

settings = get_settings()
st.set_page_config(
    page_title=f"{settings.app_name} | Intelligence Reports",
    page_icon="📑",
    layout="wide",
)

init_session_state()
render_sidebar_status()

render_page_header(
    title="📑 Intelligence Reports",
    subtitle="Audit-trailed research logs, executive briefings, verified metrics, and exportable deliverables.",
    badge="Reporting & Export Engine",
)

findings = st.session_state.get(KEY_GENERATED_FINDINGS, [])
metadata = st.session_state.get(KEY_DATASET_METADATA)

if not findings and not metadata:
    render_empty_state_banner("Intelligence Reports", phase_label="Phase 6")
else:
    st.success("Report data compiled from current active session.")

st.markdown("### Planned Capabilities in Phase 6")
r_col1, r_col2 = st.columns(2)

with r_col1:
    st.markdown(
        """
        - **Executive Briefings:** High-level executive synthesis structured for decision-makers.
        - **Methodological Transparency:** Complete log of assumptions, tools, and parameters used.
        - **Evidence Auditing:** Direct links between every chart, metric, and finding.
        """
    )

with r_col2:
    st.markdown(
        """
        - **Export Formats:** Clean PDF generation and interactive standalone HTML dossiers.
        - **Audit Trails:** Immutable session logs recording all data transformations.
        - **Reproducibility Manifest:** Full analytical code export for local reproduction.
        """
    )

st.markdown("---")
st.page_link("app.py", label="← Return to Main Workspace", icon="🏠")
