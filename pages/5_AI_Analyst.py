"""Page 5: AI Analyst Workspace - Evidence-Grounded AI Interpretation Engine.

Connects deterministic profiling, EDA, and statistical inference to a configurable
Large Language Model (Gemini/Mock) with strict claim verification and evidence provenance.
"""

import pandas as pd
import streamlit as st
from src.agent.analyst_service import AnalystService
from src.agent.provider_factory import ProviderFactory
from src.config.settings import get_settings
from src.core.session_state import (
    KEY_CURRENT_DATASET,
    init_session_state,
    is_dataset_loaded,
)
from src.core.ui_components import (
    render_empty_state_banner,
    render_page_header,
    render_sidebar_status,
)
from src.ingestion.loader import ManagedDataset
from src.ingestion.schema import infer_dataset_schema

settings = get_settings()

st.set_page_config(
    page_title=f"{settings.app_name} | AI Analyst",
    page_icon="🤖",
    layout="wide",
)

init_session_state()
render_sidebar_status()

# Initialize AnalystService in session state if not present
if "analyst_service" not in st.session_state:
    st.session_state.analyst_service = AnalystService(settings=settings)
if "ai_chat_history" not in st.session_state:
    st.session_state.ai_chat_history = []  # list of tuples: (AnalystResponse, EvidencePack)

service: AnalystService = st.session_state.analyst_service

render_page_header(
    title="🤖 AI Analyst Workspace",
    subtitle="Evidence-grounded analytical investigations powered by deterministic calculations & AI synthesis.",
    badge="Phase 5 • AI Interpretation Engine",
)

# -----------------------------------------------------------------------------
# 1. Provider Status Panel
# -----------------------------------------------------------------------------
with st.expander(
    "⚙️ LLM Provider & Execution Engine Status", expanded=not settings.is_llm_configured
):
    col_p1, col_p2, col_p3 = st.columns(3)

    with col_p1:
        st.markdown("**Configured Provider**")
        provider_name = service.provider.provider_name.upper()
        if service.provider.provider_name == "gemini":
            st.info(f"✨ Google Gemini ({service.provider.model_name})")
        else:
            st.info(f"🧪 {provider_name} ({service.provider.model_name})")

    with col_p2:
        st.markdown("**Connection Status**")
        if service.provider.is_configured:
            st.success("🟢 Ready (Active Credentials)")
        else:
            st.warning("🟡 No-Provider Mode (Deterministic Only)")

    with col_p3:
        st.markdown("**Provider Controls**")
        # Allow toggling Mock mode for demonstration / evaluation
        use_mock = st.checkbox(
            "Enable Mock LLM Provider (Offline Testing)",
            value=(service.provider.provider_name == "mock"),
            help="Allows simulating AI interpretations offline without consuming API credits.",
        )
        if use_mock and service.provider.provider_name != "mock":
            service.set_provider(ProviderFactory.get_provider(provider_type="mock"))
            st.rerun()
        elif not use_mock and service.provider.provider_name == "mock":
            service.set_provider(ProviderFactory.get_provider(settings=settings))
            st.rerun()

    if not service.provider.is_configured and service.provider.provider_name != "mock":
        st.markdown(
            """
            > **ℹ️ No API Key Configured:** The AI Analyst will run in **Deterministic Offline Mode**.
            > All statistics, hypothesis tests, and evidence items will be computed exactly by deterministic Python services.
            > To unlock natural language LLM interpretations:
            > 1. Set `GEMINI_API_KEY="your-api-key"` in your `.env` file or environment.
            > 2. Or toggle **Enable Mock LLM Provider** above to test with simulated responses.
            """
        )

# -----------------------------------------------------------------------------
# Check Active Dataset
# -----------------------------------------------------------------------------
if not is_dataset_loaded(st.session_state):
    render_empty_state_banner("AI Analyst Workspace", phase_label="Phase 5")
    st.page_link("app.py", label="← Ingest Dataset", icon="📥")
    st.stop()

dataset_obj = st.session_state.get(KEY_CURRENT_DATASET)
if isinstance(dataset_obj, ManagedDataset):
    dataset = dataset_obj
elif isinstance(dataset_obj, pd.DataFrame):
    dataset = ManagedDataset(
        dataset_id="active_ds",
        filename="Active DataFrame",
        file_format="dataframe",
        file_size_bytes=int(dataset_obj.memory_usage(deep=True).sum()),
        fingerprint="fp_mem",
        raw_df=dataset_obj,
        working_df=dataset_obj,
        schema_info=infer_dataset_schema(dataset_obj),
        ingestion_report=None,  # type: ignore[arg-type]
    )
else:
    st.error("Invalid dataset representation in session state.")
    st.stop()

df = dataset.working_df
version = getattr(dataset, "version", 1)

# -----------------------------------------------------------------------------
# 2. Analytical Question Interface
# -----------------------------------------------------------------------------
st.markdown("### 💬 Ask an Analytical Question")

example_questions = [
    "What is the overall shape, distribution, and summary of this dataset?",
    "Which variables show the strongest correlation or statistical association?",
    "Is there a statistically significant difference between categories in our key metrics?",
    "Fit a regression model and check if assumptions hold.",
    "Could missing values or outliers be distorting the observed patterns?",
]

selected_example = st.selectbox(
    "💡 Pick an example analytical question:",
    ["-- Select an example or type your own below --"] + example_questions,
)

q_default = (
    selected_example if selected_example != "-- Select an example or type your own below --" else ""
)

col_q, col_btn = st.columns([5, 1])
with col_q:
    user_query = st.text_input(
        "Enter your question for the AI Analyst:",
        value=q_default,
        placeholder="e.g. Compare average revenue across customer segments and verify significance.",
        label_visibility="collapsed",
    )

with col_btn:
    submit_clicked = st.button("🚀 Analyze", type="primary", use_container_width=True)

col_ctrl1, col_ctrl2 = st.columns([4, 2])
with col_ctrl2:
    if st.button("🗑️ Reset Analytical Conversation", use_container_width=True):
        service.reset_conversation()
        st.session_state.ai_chat_history = []
        st.success("Conversation history cleared.")
        st.rerun()

# -----------------------------------------------------------------------------
# Execute Analysis on Submit
# -----------------------------------------------------------------------------
if submit_clicked and user_query.strip():
    with st.spinner(
        "🤖 Calculating deterministic evidence and generating evidence-grounded interpretation..."
    ):
        try:
            resp, pack = service.analyze_question(
                question=user_query.strip(),
                managed_dataset=dataset,
            )
            st.session_state.ai_chat_history.append((resp, pack))
        except Exception as err:
            st.error(f"❌ Analysis failed: {err}")

# -----------------------------------------------------------------------------
# 3. Answer Workspace & Session History
# -----------------------------------------------------------------------------
if st.session_state.ai_chat_history:
    st.markdown("---")
    st.markdown(f"### 📋 Analytical Answers ({len(st.session_state.ai_chat_history)} Exchanges)")

    # Display in reverse chronological order (newest first)
    for idx, (resp, pack) in enumerate(reversed(st.session_state.ai_chat_history)):
        turn_num = len(st.session_state.ai_chat_history) - idx
        with st.container(border=True):
            # Header
            col_h1, col_h2, col_h3 = st.columns([4, 2, 2])
            with col_h1:
                st.subheader(f"Q{turn_num}: {resp.question}")
            with col_h2:
                st.caption(f"🎯 Intent: `{resp.intent_category}` | v{resp.dataset_version}")
            with col_h3:
                st.caption(
                    f"⚡ {resp.trace_metadata.provider} ({resp.trace_metadata.execution_time_ms}ms)"
                )

            # A. Direct Answer
            st.markdown(f"**Direct Answer:** {resp.direct_answer}")

            # Contradiction warnings if any
            if resp.contradictions_detected:
                for c in resp.contradictions_detected:
                    if c.severity == "critical":
                        st.error(f"🚨 **Contradiction Detected:** {c.description}")
                    else:
                        st.warning(f"⚠️ **Reasoning Conflict:** {c.description}")

            # B. Key Findings
            if resp.key_findings:
                st.markdown("#### 🔍 Evidence-Backed Findings")
                for f_idx, finding in enumerate(resp.key_findings, 1):
                    col_f1, col_f2 = st.columns([5, 2])
                    with col_f1:
                        st.markdown(f"**{f_idx}. {finding.claim}**")
                        if finding.caveats:
                            st.caption(f"Caveats: {'; '.join(finding.caveats)}")
                    with col_f2:
                        if finding.validation_status == "verified":
                            st.success(f"✓ Verified ({', '.join(finding.evidence_ids)})")
                        elif finding.validation_status == "mismatched":
                            st.error(f"✗ Value Mismatch ({', '.join(finding.evidence_ids)})")
                        elif finding.validation_status == "stale_version":
                            st.warning(f"⚠ Stale Version ({', '.join(finding.evidence_ids)})")
                        elif finding.validation_status == "invalid_reference":
                            st.error("✗ Invalid Reference")
                        else:
                            st.info("ℹ Unverified Observation")

            # C. Deeper Interpretation & Mechanisms
            if resp.interpretation:
                with st.expander("📖 Deeper Interpretation & Mechanisms", expanded=True):
                    st.write(resp.interpretation)

            # D. Alternative Explanations & Uncertainty
            col_alt, col_unc = st.columns(2)
            with col_alt:
                with st.expander("⚖️ Plausible Alternative Explanations", expanded=False):
                    if resp.alternative_explanations:
                        for alt in resp.alternative_explanations:
                            st.markdown(f"- {alt}")
                    else:
                        st.caption("No competing alternative explanations flagged.")

            with col_unc:
                with st.expander("⚠️ Uncertainty & Data Limitations", expanded=False):
                    if resp.uncertainty_and_limitations:
                        for unc in resp.uncertainty_and_limitations:
                            st.markdown(f"- {unc}")
                    else:
                        st.caption("No specific limitations identified.")

            # E. Testable Hypotheses
            if resp.hypotheses:
                with st.expander("🔬 Testable Exploratory Hypotheses", expanded=False):
                    for h_idx, hyp in enumerate(resp.hypotheses, 1):
                        st.markdown(f"**Hypothesis {h_idx}: {hyp.hypothesis_statement}**")
                        st.markdown(f"- *Rationale:* {hyp.rationale}")
                        st.markdown(
                            f"- *Suggested Test:* `{hyp.suggested_method}` (Vars: {', '.join(hyp.required_variables)})"
                        )
                        st.markdown(f"- *Supporting Evidence:* {hyp.expected_support_evidence}")
                        st.markdown(f"- *Weakening Evidence:* {hyp.expected_weaken_evidence}")
                        st.caption(
                            "Status: Exploratory post-hoc conjecture (requires independent confirmatory testing)."
                        )
                        st.markdown("---")

            # F. Recommended Follow-up Analyses
            if resp.recommended_next_analyses:
                with st.expander("🚀 Recommended Follow-up Actions", expanded=False):
                    for rec in resp.recommended_next_analyses:
                        st.markdown(f"- {rec}")

            # G. Evidence Pack Inspector
            with st.expander(
                f"📦 Evidence Inspector ({len(pack.items)} Verified Items)", expanded=False
            ):
                st.caption(
                    f"Evidence Pack ID: `{pack.pack_id}` | Dataset: `{pack.dataset_name}` (v{pack.dataset_version}) "
                    f"| Dimensions: {pack.row_count:,} x {pack.column_count}"
                )
                evidence_data = []
                for it in pack.items:
                    evidence_data.append(
                        {
                            "Evidence ID": it.id,
                            "Type": it.evidence_type,
                            "Metric": it.metric_name,
                            "Computed Value": it.formatted_value,
                            "Variables": ", ".join(it.relevant_variables),
                            "Method": it.calculation_method,
                            "Sample Size N": it.sample_size or pack.row_count,
                        }
                    )
                st.dataframe(evidence_data, use_container_width=True)

else:
    st.info(
        "💡 Type a question above or select an example to initiate an evidence-grounded investigation."
    )
