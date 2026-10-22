"""Page 3: Exploration Studio - Distributions, trends, relationships, and smart chart selection."""

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
    page_title=f"{settings.app_name} | Exploration Studio",
    page_icon="📈",
    layout="wide",
)

init_session_state()
render_sidebar_status()

render_page_header(
    title="📈 Exploration Studio",
    subtitle="Interactive distributions, bivariate relationships, and automated chart generation.",
    badge="Module 2 • Exploratory Analysis",
)

if not is_dataset_loaded(st.session_state):
    render_empty_state_banner("Exploration Studio", phase_label="Phase 2")
else:
    metadata = st.session_state.get(KEY_DATASET_METADATA)
    st.success(f"Exploring active dataset: **{metadata.get('filename', 'Unknown')}**")

st.markdown("### Planned Capabilities in Phase 2")
e_col1, e_col2 = st.columns(2)

with e_col1:
    st.markdown(
        """
        - **Distribution Diagnostics:** Mean, median, percentiles, skewness, and kurtosis.
        - **Interactive Plotly Visuals:** Density plots, box plots, histograms, and scatter plots.
        - **Automated Chart Selection:** Intelligently picks chart types based on variable cardinality.
        """
    )

with e_col2:
    st.markdown(
        """
        - **Correlation Matrix:** Pearson and Spearman pairwise correlation heatmaps.
        - **Group Comparisons:** Multi-segment breakdowns and pivot aggregations.
        - **Chart Narratives:** Written analytical context detailing what each visual confirms or refutes.
        """
    )

st.markdown("---")
st.page_link("app.py", label="← Return to Main Workspace", icon="🏠")
