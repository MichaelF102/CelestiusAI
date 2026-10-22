"""Page 5: AI Analyst - Conversational data analysis, multi-step planning, and evidence-backed interpretations."""

import streamlit as st
from src.config.settings import get_settings
from src.core.session_state import (
    init_session_state,
)
from src.core.ui_components import (
    render_empty_state_banner,
    render_page_header,
    render_sidebar_status,
)

settings = get_settings()
st.set_page_config(
    page_title=f"{settings.app_name} | AI Analyst",
    page_icon="🤖",
    layout="wide",
)

init_session_state()
render_sidebar_status()

render_page_header(
    title="🤖 AI Analyst Workspace",
    subtitle="Conversational investigations, dynamic analysis plans, and evidence-linked findings.",
    badge="Module 7 • Agent & Interpretation",
)

# LLM Status Banner
if not settings.is_llm_configured:
    st.warning(
        "**LLM Integration in Local Foundation Mode**\n\n"
        "No Gemini API key is configured. The application runs locally without external API dependencies. "
        "LLM-powered planning and natural-language narrative generation will be connected in **Phase 4**."
    )

render_empty_state_banner("AI Analyst Workspace", phase_label="Phase 4")

st.markdown("### The 6-Layer Interpretation Architecture")
st.markdown(
    """
    When active, every insight produced by the agent strictly adheres to our verifiable interpretation model:

    1. **Observation:** What precise anomaly or pattern was detected in the data?
    2. **Quantitative Evidence:** Measured figures, baseline comparisons, and percentage variances.
    3. **Statistical Confidence:** Sample size sufficiency, p-values, confidence intervals, and limitations.
    4. **Explanatory Breakdown:** Decomposition of primary contributors vs. secondary associations.
    5. **Practical Implication:** Real-world significance for business operations or strategic goals.
    6. **Next Recommended Action:** Concrete follow-up queries or empirical checks to reduce uncertainty.
    """
)

st.markdown("---")
st.page_link("app.py", label="← Return to Main Workspace", icon="🏠")
