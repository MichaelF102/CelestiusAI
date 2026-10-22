"""Page 2: Data Quality Lab - Missingness, duplicates, outliers, and schema profiling."""

import streamlit as st
from src.config.settings import get_settings
from src.core.session_state import (
    KEY_DATASET_METADATA,
    init_session_state,
    is_dataset_loaded,
)
from src.core.ui_components import (
    render_empty_state_banner,
    render_page_header,
    render_sidebar_status,
)

settings = get_settings()
st.set_page_config(
    page_title=f"{settings.app_name} | Data Quality Lab",
    page_icon="🔍",
    layout="wide",
)

init_session_state()
render_sidebar_status()

render_page_header(
    title="🔍 Data Quality Lab",
    subtitle="Automated schema inference, missingness analysis, duplicate detection, and outlier audits.",
    badge="Module 1 • Ingestion & Quality",
)

if not is_dataset_loaded(st.session_state):
    render_empty_state_banner("Data Quality Lab", phase_label="Phase 2")
else:
    metadata = st.session_state.get(KEY_DATASET_METADATA)
    st.success(f"Inspecting active dataset: **{metadata.get('filename', 'Unknown')}**")

st.markdown("### Planned Capabilities in Phase 2")
q_col1, q_col2 = st.columns(2)

with q_col1:
    st.markdown(
        """
        - **Format Ingestion:** Native reading of CSV, Excel (.xlsx/.xls), and Parquet files.
        - **Schema Inference:** Auto-detection of numeric, categorical, datetime, and identifier columns.
        - **Missingness Audit:** Distinguishing Missing Completely at Random (MCAR) vs. systemic gaps.
        - **Duplicate Detection:** Exact and near-duplicate entity identification.
        """
    )

with q_col2:
    st.markdown(
        """
        - **Outlier Detection:** IQR and z-score boundary diagnostics with distribution checks.
        - **Data Quality Score:** Holistic quality rating based on completeness, uniqueness, and validity.
        - **Correction Recommendations:** Transparent, user-approved cleaning suggestions.
        """
    )

st.markdown("---")
st.page_link("app.py", label="← Return to Main Workspace", icon="🏠")
