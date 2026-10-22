"""Page 5: AI Analyst Workspace - Evidence-Grounded AI Interpretation Engine.

Connects deterministic profiling, EDA, and statistical inference to a configurable
Large Language Model (Gemini/Mock) with strict claim verification, evidence provenance,
and agentic multi-step investigation orchestration.
"""

import pandas as pd
import streamlit as st
from src.agent.analyst_service import AnalystService
from src.agent.orchestration import (
    InvestigationOrchestrator,
    InvestigationState,
    InvestigationStatus,
    ResourceBudget,
)
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

# Initialize AnalystService & Orchestrator in session state if not present
if "analyst_service" not in st.session_state:
    st.session_state.analyst_service = AnalystService(settings=settings)
if "ai_chat_history" not in st.session_state:
    st.session_state.ai_chat_history = []  # list of tuples: (AnalystResponse, EvidencePack)
if "investigation_history" not in st.session_state:
    st.session_state.investigation_history = []  # list of tuples: (InvestigationResult, EvidencePack)
if "pending_plan_state" not in st.session_state:
    st.session_state.pending_plan_state = None

service: AnalystService = st.session_state.analyst_service
orchestrator = InvestigationOrchestrator(
    llm_provider=service.provider,
    tool_registry=service.tool_registry,
)

render_page_header(
    title="🤖 AI Analyst Workspace",
    subtitle="Evidence-grounded analytical investigations powered by deterministic calculations & agentic orchestration.",
    badge="Phase 5 & 6 • Agentic Investigation Engine",
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
    render_empty_state_banner("AI Analyst Workspace", phase_label="Phase 5 & 6")
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
# 2. Analytical Mode & Question Interface
# -----------------------------------------------------------------------------
st.markdown("### 💬 Ask an Analytical Question")

col_mode1, col_mode2 = st.columns([3, 2])
with col_mode1:
    analysis_mode = st.radio(
        "Investigation Strategy:",
        ["⚡ Quick Answer", "🕵️ Deep Multi-Step Investigation"],
        horizontal=True,
        help="Quick Answer synthesizes evidence immediately. Deep Multi-Step Investigation constructs a DAG execution plan across multiple analytical tools.",
    )

with col_mode2:
    if analysis_mode == "🕵️ Deep Multi-Step Investigation":
        auto_approve = st.checkbox(
            "Auto-approve plan and execute immediately",
            value=False,
            help="When checked, bypasses manual plan inspection and executes immediately.",
        )
    else:
        auto_approve = False

example_questions = [
    "Why did revenue vary, which customer segments contributed most, and what should be investigated next?",
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
        placeholder="e.g. Why did revenue decline and which segments contributed most?",
        label_visibility="collapsed",
    )

with col_btn:
    btn_label = (
        "🚀 Plan Investigation"
        if analysis_mode == "🕵️ Deep Multi-Step Investigation" and not auto_approve
        else "🚀 Analyze"
    )
    submit_clicked = st.button(btn_label, type="primary", use_container_width=True)

col_ctrl1, col_ctrl2 = st.columns([4, 2])
with col_ctrl2:
    if st.button("🗑️ Reset Workspace History", use_container_width=True):
        service.reset_conversation()
        st.session_state.ai_chat_history = []
        st.session_state.investigation_history = []
        st.session_state.pending_plan_state = None
        st.success("Session history cleared.")
        st.rerun()

# -----------------------------------------------------------------------------
# 3. Execution Logic
# -----------------------------------------------------------------------------

# Mode A: Quick Answer (Prompt 5)
if submit_clicked and user_query.strip() and analysis_mode == "⚡ Quick Answer":
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

# Mode B: Deep Multi-Step Investigation (Prompt 6)
if submit_clicked and user_query.strip() and analysis_mode == "🕵️ Deep Multi-Step Investigation":
    with st.spinner("📐 Decomposing research question and validating analytical DAG plan..."):
        try:
            plan_state = orchestrator.plan_investigation(
                question=user_query.strip(),
                managed_dataset=dataset,
                budget=ResourceBudget(max_steps=8, max_plan_revisions=2),
            )
            if auto_approve:
                with st.spinner("⚙️ Executing multi-step analytical plan across approved tools..."):
                    result, final_pack = orchestrator.run_investigation(plan_state, dataset)
                    st.session_state.investigation_history.append((result, final_pack))
                    st.session_state.pending_plan_state = None
            else:
                st.session_state.pending_plan_state = plan_state
        except Exception as err:
            st.error(f"❌ Planning failed: {err}")

# -----------------------------------------------------------------------------
# 4. Plan Preview & User Approval Panel
# -----------------------------------------------------------------------------
if st.session_state.pending_plan_state is not None:
    p_state: InvestigationState = st.session_state.pending_plan_state
    plan = p_state.current_plan

    st.markdown("---")
    with st.container(border=True):
        st.subheader("📋 Investigation Plan Preview (Awaiting Approval)")
        st.markdown(f"**Research Objective:** {plan.normalized_objective}")
        st.caption(
            f"Plan ID: `{plan.plan_id}` | Dataset: `{plan.dataset_id}` (v{plan.dataset_version}) | Fingerprint: `{plan.signature_hash}`"
        )

        # Steps preview table
        step_rows = []
        for s in plan.steps:
            step_rows.append(
                {
                    "Step ID": s.step_id,
                    "Step Name": s.step_name,
                    "Tool": s.tool_name,
                    "Dependencies": ", ".join(s.dependencies) if s.dependencies else "(None)",
                    "Objective": s.objective,
                    "Reason": s.reason,
                }
            )
        st.dataframe(step_rows, use_container_width=True)

        col_app1, col_app2, col_app3 = st.columns([2, 2, 4])
        with col_app1:
            if st.button("▶️ Approve & Execute Plan", type="primary", use_container_width=True):
                with st.spinner("⚙️ Executing investigation steps..."):
                    try:
                        res, f_pack = orchestrator.run_investigation(p_state, dataset)
                        st.session_state.investigation_history.append((res, f_pack))
                        st.session_state.pending_plan_state = None
                        st.rerun()
                    except Exception as err:
                        st.error(f"Execution error: {err}")

        with col_app2:
            if st.button("❌ Cancel Plan", use_container_width=True):
                st.session_state.pending_plan_state = None
                st.rerun()

# -----------------------------------------------------------------------------
# 5. Multi-Step Investigation Results Workspace
# -----------------------------------------------------------------------------
if st.session_state.investigation_history:
    st.markdown("---")
    st.markdown(
        f"### 🕵️ Multi-Step Investigation Reports ({len(st.session_state.investigation_history)})"
    )

    for idx, (res, pack) in enumerate(reversed(st.session_state.investigation_history)):
        inv_num = len(st.session_state.investigation_history) - idx
        with st.container(border=True):
            col_rh1, col_rh2, col_rh3 = st.columns([5, 2, 2])
            with col_rh1:
                st.subheader(f"Investigation {inv_num}: {res.question}")
            with col_rh2:
                status_color = "🟢" if res.status == InvestigationStatus.COMPLETED else "🔴"
                st.caption(f"{status_color} Status: `{res.status.value}` (v{res.dataset_version})")
            with col_rh3:
                st.caption(
                    f"⚡ Duration: {res.total_duration_ms:.0f}ms | Trace ID: `{res.investigation_id}`"
                )

            # Executive Answer
            st.info(f"**Executive Answer:** {res.executive_answer}")

            # Key Findings
            if res.key_findings:
                st.markdown("#### 🔍 Evidence-Backed Findings")
                for f_idx, finding in enumerate(res.key_findings, 1):
                    col_kf1, col_kf2 = st.columns([5, 2])
                    with col_kf1:
                        st.markdown(f"**{f_idx}. {finding.claim}**")
                        if finding.caveats:
                            st.caption(f"Caveats: {'; '.join(finding.caveats)}")
                    with col_kf2:
                        if finding.validation_status == "verified":
                            st.success(f"✓ Verified ({', '.join(finding.evidence_ids)})")
                        else:
                            st.warning(f"⚠ {finding.validation_status}")

            # Execution Trace & Path
            with st.expander("🛤️ Analytical Investigation Path & Step Trace", expanded=True):
                trace_df = []
                for tr in res.execution_trace:
                    trace_df.append(
                        {
                            "Step": tr.get("step_id"),
                            "Action": tr.get("step_name"),
                            "Tool": tr.get("tool_name"),
                            "Status": tr.get("status"),
                            "Duration (ms)": tr.get("execution_time_ms"),
                            "Evidence IDs": ", ".join(tr.get("evidence_ids", [])),
                        }
                    )
                st.dataframe(trace_df, use_container_width=True)

            # Limitations & Next Actions
            col_l1, col_l2 = st.columns(2)
            with col_l1:
                with st.expander("⚠️ Limitations & Unresolved Questions", expanded=False):
                    for lim in res.limitations:
                        st.markdown(f"- {lim}")
                    if res.unresolved_questions:
                        st.markdown("**Unresolved:**")
                        for un in res.unresolved_questions:
                            st.markdown(f"- {un}")

            with col_l2:
                with st.expander("🚀 Recommended Next Actions", expanded=False):
                    for rec in res.recommended_next_actions:
                        st.markdown(f"- {rec}")

            # Evidence Inspector
            with st.expander(
                f"📦 Evidence Inspector ({len(pack.items)} Verified Items)", expanded=False
            ):
                ev_table = [
                    {
                        "Evidence ID": it.id,
                        "Type": it.evidence_type,
                        "Metric": it.metric_name,
                        "Computed Value": it.formatted_value,
                        "Variables": ", ".join(it.relevant_variables),
                        "Method": it.calculation_method,
                        "Sample Size N": it.sample_size or pack.row_count,
                    }
                    for it in pack.items
                ]
                st.dataframe(ev_table, use_container_width=True)

# -----------------------------------------------------------------------------
# 6. Quick Answer History Workspace (Prompt 5)
# -----------------------------------------------------------------------------
if st.session_state.ai_chat_history:
    st.markdown("---")
    st.markdown(f"### ⚡ Quick Answers ({len(st.session_state.ai_chat_history)} Exchanges)")

    for idx, (resp, pack) in enumerate(reversed(st.session_state.ai_chat_history)):
        turn_num = len(st.session_state.ai_chat_history) - idx
        with st.container(border=True):
            col_h1, col_h2, col_h3 = st.columns([4, 2, 2])
            with col_h1:
                st.subheader(f"Q{turn_num}: {resp.question}")
            with col_h2:
                st.caption(f"🎯 Intent: `{resp.intent_category}` | v{resp.dataset_version}")
            with col_h3:
                st.caption(
                    f"⚡ {resp.trace_metadata.provider} ({resp.trace_metadata.execution_time_ms}ms)"
                )

            st.markdown(f"**Direct Answer:** {resp.direct_answer}")

            if resp.contradictions_detected:
                for c in resp.contradictions_detected:
                    if c.severity == "critical":
                        st.error(f"🚨 **Contradiction Detected:** {c.description}")
                    else:
                        st.warning(f"⚠️ **Reasoning Conflict:** {c.description}")

            if resp.key_findings:
                st.markdown("#### 🔍 Evidence-Backed Findings")
                for f_idx, finding in enumerate(resp.key_findings, 1):
                    col_f1, col_f2 = st.columns([5, 2])
                    with col_f1:
                        st.markdown(f"**{f_idx}. {finding.claim}**")
                    with col_f2:
                        if finding.validation_status == "verified":
                            st.success(f"✓ Verified ({', '.join(finding.evidence_ids)})")
                        else:
                            st.warning(f"⚠ {finding.validation_status}")

            if resp.interpretation:
                with st.expander("📖 Deeper Interpretation & Mechanisms", expanded=False):
                    st.write(resp.interpretation)

            with st.expander(f"📦 Evidence Inspector ({len(pack.items)} Items)", expanded=False):
                evidence_data = [
                    {
                        "Evidence ID": it.id,
                        "Type": it.evidence_type,
                        "Metric": it.metric_name,
                        "Value": it.formatted_value,
                        "Variables": ", ".join(it.relevant_variables),
                    }
                    for it in pack.items
                ]
                st.dataframe(evidence_data, use_container_width=True)

if (
    not st.session_state.investigation_history
    and not st.session_state.ai_chat_history
    and st.session_state.pending_plan_state is None
):
    st.info("💡 Type an analytical question above to launch an evidence-grounded investigation.")
