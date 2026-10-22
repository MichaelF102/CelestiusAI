"""Page 4: Statistical Intelligence - Hypothesis testing, effect sizes, and assumption diagnostics."""

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
    page_title=f"{settings.app_name} | Statistical Intelligence",
    page_icon="🧪",
    layout="wide",
)

init_session_state()
render_sidebar_status()

render_page_header(
    title="🧪 Statistical Intelligence",
    subtitle="Hypothesis testing, effect sizes, confidence intervals, and rigorous assumption verification.",
    badge="Module 3 • Statistical Inference",
)

if not is_dataset_loaded(st.session_state):
    render_empty_state_banner("Statistical Intelligence", phase_label="Phase 2")
else:
    metadata = st.session_state.get(KEY_DATASET_METADATA)
    fname = getattr(metadata, "filename", None) or (
        metadata.get("filename", "Unknown") if isinstance(metadata, dict) else "Unknown"
    )
    st.success(f"Statistical workspace active for: **{fname}**")

st.markdown("### Planned Capabilities in Phase 2")
s_col1, s_col2 = st.columns(2)

with s_col1:
    st.markdown(
        """
        - **Parametric & Non-Parametric Tests:** Student's t-test, Mann–Whitney U, ANOVA, and Kruskal–Wallis.
        - **Proportion & Categorical Tests:** Chi-Square test of independence and Fisher's exact test.
        - **Assumption Diagnostics:** Normality (Shapiro-Wilk) and homoscedasticity (Levene's test).
        """
    )

with s_col2:
    st.markdown(
        """
        - **Effect Size Estimation:** Cohen's d, Cliff's delta, Cramér's V, and odds ratios.
        - **Confidence Intervals:** Bootstrap and analytic confidence intervals at configurable alpha levels.
        - **Multiple Testing Corrections:** False Discovery Rate (FDR / Benjamini–Hochberg) adjustments.
        """
    )

st.markdown("---")
st.page_link("app.py", label="← Return to Main Workspace", icon="🏠")
