"""Page 19: Settings - AI provider, privacy controls, and audit logs."""

import pandas as pd
import streamlit as st
from src.agent.provider_factory import PROVIDER_CHOICES, ProviderFactory
from src.config.settings import get_settings
from src.core.session_state import KEY_CURRENT_DATASET, KEY_LLM_OVERRIDE, KEY_PRIVACY_REDACT
from src.core.ui_components import page_setup
from src.security import privacy

page_setup(
    "Settings",
    "⚙️",
    "Choose the AI provider, control what leaves your machine, and review activity.",
)
settings = get_settings()

tab_ai, tab_privacy, tab_audit, tab_about = st.tabs(
    ["🤖 AI provider", "🔒 Privacy", "🧾 Audit log", "ℹ️ About"]
)

with tab_ai:
    st.caption(
        "Overrides apply to this browser session only. Keys are never written to disk or logs."
    )
    override = st.session_state.get(KEY_LLM_OVERRIDE) or {}
    kind = st.selectbox(
        "Provider",
        PROVIDER_CHOICES,
        index=PROVIDER_CHOICES.index(override.get("provider", "gemini")),
    )
    defaults = {
        "gemini": settings.gemini_model,
        "anthropic": settings.anthropic_model,
        "ollama": settings.ollama_model,
        "mock": "mock-analyst-v1",
    }
    model = st.text_input("Model", value=override.get("model") or defaults[kind])
    key = host = ""
    if kind in ("gemini", "anthropic"):
        key = st.text_input(
            "API key", type="password", placeholder="Leave empty to use the environment variable"
        )
    if kind == "ollama":
        host = st.text_input("Ollama host", value=override.get("host") or settings.ollama_host)
    a, b, c = st.columns(3)
    if a.button("Apply for this session", type="primary"):
        st.session_state[KEY_LLM_OVERRIDE] = {
            "provider": kind,
            "model": model,
            "api_key": key or override.get("api_key", ""),
            "host": host,
        }
        st.success("Applied. The AI Analyst and SQL assistant will use it.")
    if b.button("Reset to defaults"):
        st.session_state[KEY_LLM_OVERRIDE] = None
        st.rerun()
    if c.button("Test connection"):
        provider = ProviderFactory.from_override(
            {
                "provider": kind,
                "model": model,
                "api_key": key or override.get("api_key", ""),
                "host": host,
            },
            settings,
        )
        ok, message = provider.check_availability()
        (st.success if ok else st.error)(message)
    active = st.session_state.get(KEY_LLM_OVERRIDE)
    st.info(
        f"Active: **{active['provider']}** · `{active['model']}`"
        if active
        else f"Active: environment defaults (Gemini `{settings.gemini_model}`)."
    )
    if kind == "anthropic":
        st.caption("Requires `pip install anthropic`.")

with tab_privacy:
    redact = st.toggle(
        "Redact personal data from text sent to an AI provider",
        value=st.session_state.get(KEY_PRIVACY_REDACT, True),
    )
    st.session_state[KEY_PRIVACY_REDACT] = redact
    st.caption(
        "Emails, phone numbers, SSNs, card numbers and IP addresses are replaced by placeholders before any prompt leaves the app."
    )
    sample = st.text_area(
        "Try it", "Contact ann@example.com or +1 415-555-0132 about card 4111 1111 1111 1111."
    )
    cleaned, counts = privacy.redact_text(sample)
    st.code(cleaned)
    if counts:
        st.caption("Found: " + ", ".join(f"{n} {k}" for k, n in counts.items()))
    dataset = st.session_state.get(KEY_CURRENT_DATASET)
    st.markdown("##### Scan the loaded dataset")
    if dataset is None:
        st.caption("Load a dataset to scan its columns.")
    else:
        found = privacy.detect_sensitive_columns(dataset.working_df)
        if not found:
            st.success("No columns look like personal data.")
        else:
            st.warning(f"{len(found)} column(s) may contain personal data.")
            st.dataframe(pd.DataFrame(found), hide_index=True, use_container_width=True)
            to_mask = st.multiselect(
                "Mask these columns in the working dataset (irreversible hashes)",
                [f["column"] for f in found],
            )
            if st.button("Mask selected columns", disabled=not to_mask):
                dataset.update_working_df(privacy.mask_frame(dataset.working_df, to_mask))
                st.success("Masked. Re-profile on the Data Quality page if needed.")

with tab_audit:
    service = st.session_state.get("analyst_service")
    rows = [vars(a) for a in service.tool_registry.audit_log] if service else []
    if not rows:
        st.info("No analytical tool calls yet. They appear here after you use the AI Analyst.")
    else:
        frame = pd.DataFrame(rows)
        frame["arguments"] = frame["arguments"].astype(str)
        st.dataframe(frame, use_container_width=True, hide_index=True)
        if service and service.last_redactions:
            st.caption(f"Last prompt redactions: {service.last_redactions}")

with tab_about:
    st.json(settings.safe_dict())
