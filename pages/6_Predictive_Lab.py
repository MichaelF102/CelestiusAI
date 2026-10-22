"""Page 6: Predictive Lab - Time-series forecasting, machine learning models, and residual diagnostics."""

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
    page_title=f"{settings.app_name} | Predictive Lab",
    page_icon="🔮",
    layout="wide",
)

init_session_state()
render_sidebar_status()

render_page_header(
    title="🔮 Predictive Lab",
    subtitle="Chronological forecasting, machine learning models, walk-forward validation, and residual audits.",
    badge="Modules 5 & 6 • Prediction & Forecasting",
)

if not is_dataset_loaded(st.session_state):
    render_empty_state_banner("Predictive Lab", phase_label="Phase 5")
else:
    metadata = st.session_state.get(KEY_DATASET_METADATA)
    fname = getattr(metadata, "filename", None) or (
        metadata.get("filename", "Unknown") if isinstance(metadata, dict) else "Unknown"
    )
    st.success(f"Modeling workspace active for: **{fname}**")

st.markdown("### Planned Capabilities in Phase 5")
p_col1, p_col2 = st.columns(2)

with p_col1:
    st.markdown(
        """
        - **Time-Series Decomposition:** Trend, seasonal, and cyclical component isolation.
        - **Chronological Validation:** Rolling-origin walk-forward evaluation (no temporal data leakage).
        - **Forecasting Models:** Baseline drift/naive models vs. ARIMA/SARIMA and Exponential Smoothing.
        """
    )

with p_col2:
    st.markdown(
        """
        - **Supervised ML Classification & Regression:** Automated feature preprocessing and cross-validation.
        - **Model Comparison:** Clear benchmark matrices against naive baselines.
        - **Residual Diagnostics & Interpretability:** Error distribution analysis and feature importance.
        """
    )

st.markdown("---")
st.page_link("app.py", label="← Return to Main Workspace", icon="🏠")
