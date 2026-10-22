"""Page 7: Intelligence Reports — Evidence-Backed Report Assembly and Multi-Format Export."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from src.config.settings import get_settings
from src.core.session_state import (
    KEY_ANALYSIS_RESULTS,
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
from src.predictive.model_registry import ModelRegistry
from src.reporting import (
    ExportService,
    ReportBuilder,
    ReportDepth,
    ReportDocument,
    ReportEvidenceCollector,
    ReportType,
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
    subtitle="Assemble verified analytical findings, quality audits, statistical inferences, and predictive evaluations into exportable executive reports.",
    badge="Module 6 • Intelligence Reporting Engine",
)

if not is_dataset_loaded(st.session_state):
    render_empty_state_banner("Intelligence Reports", phase_label="Phase 6")
    st.page_link(
        "pages/2_Data_Quality.py", label="Go to Data Quality Lab to Upload a Dataset", icon="📥"
    )
    st.stop()

dataset_obj = st.session_state.get(KEY_CURRENT_DATASET)
if isinstance(dataset_obj, ManagedDataset):
    managed_ds = dataset_obj
    dataset_name = dataset_obj.filename
    dataset_id = str(dataset_obj.dataset_id)
    dataset_version = getattr(dataset_obj, "version", 1)
elif isinstance(dataset_obj, pd.DataFrame):
    managed_ds = ManagedDataset(
        raw_df=dataset_obj,
        working_df=dataset_obj,
        dataset_id="default_dataset",
        filename="Active DataFrame",
    )
    dataset_name = "Active DataFrame"
    dataset_id = "default_dataset"
    dataset_version = 1
else:
    st.error("Invalid dataset representation in session state.")
    st.stop()

model_registry = st.session_state.get("model_registry")
if not isinstance(model_registry, ModelRegistry):
    model_registry = ModelRegistry()
    st.session_state["model_registry"] = model_registry

# Setup Report Evidence Collector
collector = ReportEvidenceCollector(managed_dataset=managed_ds, model_registry=model_registry)

# Get statistical and investigation results if available in session
stat_results = st.session_state.get(KEY_ANALYSIS_RESULTS, {}).get("statistical_tests", [])
inv_trace = st.session_state.get("latest_investigation_state")

evidence_bundle = collector.collect_all_evidence(
    investigation_trace=inv_trace,
    statistical_results=stat_results,
)

st.markdown("### 1. Analysis Inventory & Configuration")

inv_col1, inv_col2 = st.columns([1, 1])

with inv_col1:
    report_type = st.selectbox(
        "Select Report Type:",
        options=[
            ReportType.COMPREHENSIVE,
            ReportType.EXECUTIVE,
            ReportType.EDA,
            ReportType.STATISTICAL,
            ReportType.INVESTIGATION,
            ReportType.PREDICTIVE,
        ],
        format_func=lambda rt: rt.value,
        help="Select the scope and narrative focus of the report.",
    )

    report_depth = st.select_slider(
        "Report Depth:",
        options=[ReportDepth.SUMMARY, ReportDepth.STANDARD, ReportDepth.DETAILED],
        value=ReportDepth.STANDARD,
        format_func=lambda rd: rd.value,
    )

with inv_col2:
    st.markdown("#### Verified Analysis Inventory")
    st.markdown(
        f"- **Data Ingestion & Quality:** ✅ Active ({len(managed_ds.working_df):,} rows, v{dataset_version})"
    )

    corr_count = sum(
        1 for it in evidence_bundle["evidence_items"] if it.evidence_type == "correlation"
    )
    st.markdown(
        f"- **Exploratory Data Analysis:** {'✅' if corr_count > 0 else '⚠️'} ({corr_count} correlations indexed)"
    )

    st.markdown(
        f"- **Statistical Inference:** {'✅' if stat_results else '⚪'} ({len(stat_results)} tests recorded)"
    )

    inv_active = inv_trace is not None
    st.markdown(
        f"- **Autonomous Investigation:** {'✅' if inv_active else '⚪'} ({'1 active trace' if inv_active else 'Not executed'})"
    )

    pred_records = evidence_bundle["predictive_records"]
    st.markdown(
        f"- **Predictive Models:** {'✅' if pred_records else '⚪'} ({len(pred_records)} models evaluated)"
    )

st.markdown("---")

# Generate Action
gen_col1, gen_col2 = st.columns([1, 2])

with gen_col1:
    enable_ai = st.toggle(
        "Enrich with AI Narrative (if configured)",
        value=False,
        help="Synthesize executive narrative using configured LLM. Deterministic evidence remains the source of truth.",
    )

with gen_col2:
    generate_btn = st.button(
        "📑 Generate Intelligence Report", type="primary", use_container_width=True
    )

if generate_btn:
    with st.spinner("Compiling verified evidence into structured report..."):
        ai_narrative = None
        if enable_ai and getattr(settings, "gemini_api_key", None):
            try:
                from src.agent.analyst_service import AnalystService
                from src.agent.provider_factory import ProviderFactory

                provider = ProviderFactory.get_provider(settings=settings)
                analyst = AnalystService(provider=provider, settings=settings)
                # Quick prompt for executive briefing
                res, _ = analyst.analyze_question(
                    question="Provide a concise 2-paragraph executive overview summarizing the key metrics and verified data patterns.",
                    managed_dataset=managed_ds,
                )
                if res and getattr(res, "synthesis", None):
                    ai_narrative = res.synthesis
            except Exception as e:
                st.warning(f"Could not generate AI narrative ({e}). Using deterministic summary.")

        builder = ReportBuilder(evidence_bundle)
        doc: ReportDocument = builder.build_report(
            report_type=report_type,
            report_depth=report_depth,
            ai_narrative_text=ai_narrative,
        )
        st.session_state["active_report_doc"] = doc
        st.success(
            f"Report '{doc.metadata.title}' assembled with {doc.metadata.evidence_item_count} verified evidence items."
        )

active_doc: ReportDocument = st.session_state.get("active_report_doc")

if active_doc:
    st.markdown("---")
    st.subheader(f"Report Preview: {active_doc.metadata.title}")

    # Export Download Buttons
    exp_col1, exp_col2, exp_col3 = st.columns([1, 1, 2])

    html_str = ExportService.export_html(active_doc)

    with exp_col1:
        st.download_button(
            label="📥 Download HTML Report",
            data=html_str,
            file_name=f"{active_doc.metadata.report_id}.html",
            mime="text/html",
            use_container_width=True,
        )

    with exp_col2:
        pdf_bytes, pdf_err = ExportService.export_pdf(active_doc)
        if pdf_bytes:
            st.download_button(
                label="📥 Download PDF Report",
                data=pdf_bytes,
                file_name=f"{active_doc.metadata.report_id}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        else:
            st.warning(pdf_err or "PDF export unavailable. Use HTML download.")

    with exp_col3:
        st.caption(
            f"Report ID: `{active_doc.metadata.report_id}` | "
            f"Generated: {active_doc.metadata.generated_at.strftime('%Y-%m-%d %H:%M UTC')}"
        )

    # Document Visual Rendering
    with st.container(border=True):
        st.markdown(f"## {active_doc.metadata.title}")
        st.caption(
            f"**Dataset:** {active_doc.metadata.dataset_id} (v{active_doc.metadata.dataset_version}) &bull; "
            f"**Type:** {active_doc.metadata.report_type.value} &bull; "
            f"**Evidence Items:** {active_doc.metadata.evidence_item_count}"
        )

        st.markdown("### Executive Summary")
        st.info(active_doc.executive_summary)

        for sec in active_doc.sections:
            st.markdown(f"### {sec.title}")
            if not sec.was_performed:
                st.caption(f"⚠️ *{sec.omission_reason or 'Analysis omitted.'}*")
                continue

            st.markdown(sec.summary_markdown)

            for tbl in sec.tables:
                st.markdown(f"**{tbl.title}**")
                tbl_df = pd.DataFrame(tbl.rows, columns=tbl.headers)
                st.dataframe(tbl_df, use_container_width=True)

        if active_doc.unperformed_analyses:
            st.warning("**Analyses Not Executed:** " + ", ".join(active_doc.unperformed_analyses))

        st.markdown("### Methodology, Assumptions & Limitations")
        for lim in active_doc.limitations_and_assumptions:
            st.markdown(f"- {lim}")
