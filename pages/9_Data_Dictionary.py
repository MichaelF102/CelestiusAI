"""Page 9: Data Dictionary & Glossary - document what each column means."""

import pandas as pd
import streamlit as st
from src.core.code_dq import derive_rules, dictionary_apply_snippet, dictionary_validation_snippet
from src.core.code_snippets import load_kwargs
from src.core.session_state import KEY_DATA_DICTIONARY, KEY_DATASET_SCHEMA
from src.core.ui_components import (
    page_setup,
    render_code_button,
    render_code_export,
    require_dataset,
)
from src.datascience import dictionary as dic

page_setup(
    "Data Dictionary",
    "📖",
    "Describe your columns once. The AI Analyst uses your wording, never your raw values.",
)
dataset = require_dataset("Data Dictionary")
df = dataset.working_df
schema = st.session_state.get(KEY_DATASET_SCHEMA) or dataset.schema_info

# Rebuild so renamed/added/dropped columns are reflected while user text is preserved.
dictionary = dic.build_dictionary(df, schema, st.session_state.get(KEY_DATA_DICTIONARY))
st.session_state[KEY_DATA_DICTIONARY] = dictionary

described = sum(1 for e in dic.columns_only(dictionary).values() if e.get("described_by_user"))
c1, c2, c3 = st.columns(3)
c1.metric("Columns", len(df.columns))
c2.metric("Described by you", f"{described}/{len(df.columns)}")
c3.metric("Glossary terms", len(dic.get_glossary(dictionary)))

tab_cols, tab_gloss, tab_ai, tab_export = st.tabs(
    ["📋 Columns", "🔤 Glossary", "🤖 What the AI sees", "⬇️ Export"]
)

with tab_cols:
    st.caption("Edit the description, role and unit. Grey columns are computed and read-only.")
    edited = st.data_editor(
        dic.to_frame(dictionary),
        use_container_width=True,
        hide_index=True,
        disabled=["column", "type", "missing_pct", "distinct", "example"],
        column_config={
            "rename_to": st.column_config.TextColumn(
                "Rename to", help="Optional new column name used by the generated code"
            ),
            "description": st.column_config.TextColumn("Description", width="large"),
            "role": st.column_config.SelectboxColumn("Role", options=list(dic.ROLES)),
            "missing_pct": st.column_config.NumberColumn("Missing %", format="%.1f"),
        },
        key="dict_editor",
    )
    if st.button("Save descriptions", type="primary"):
        st.session_state[KEY_DATA_DICTIONARY] = dic.apply_edits(dictionary, edited)
        st.toast("Dictionary saved", icon="✅")
        st.rerun()

with tab_gloss:
    st.caption("Define business terms (e.g. *churn*, *AOV*) so answers use your vocabulary.")
    glossary = pd.DataFrame(dic.get_glossary(dictionary) or [], columns=["term", "definition"])
    gloss_edit = st.data_editor(
        glossary, num_rows="dynamic", use_container_width=True, hide_index=True, key="gloss_editor"
    )
    if st.button("Save glossary"):
        st.session_state[KEY_DATA_DICTIONARY] = dic.set_glossary(
            dictionary, gloss_edit.fillna("").to_dict("records")
        )
        st.toast("Glossary saved", icon="✅")
        st.rerun()

with tab_ai:
    context = dic.context_for_llm(dictionary)
    if context:
        st.code(context, language="markdown")
        st.caption("Only descriptions you wrote and your glossary are shared. No data values.")
    else:
        st.info("Nothing to share yet. Write a description for at least one column.")

with tab_export:
    e1, e2 = st.columns(2)
    e1.download_button(
        "Markdown",
        dic.to_markdown(dictionary, dataset.filename),
        "data_dictionary.md",
        "text/markdown",
        use_container_width=True,
    )
    e2.download_button(
        "CSV", dic.to_csv(dictionary), "data_dictionary.csv", "text/csv", use_container_width=True
    )
    st.download_button(
        "Data card (README.md)",
        dic.data_card_markdown(
            dataset.filename,
            df,
            dictionary,
            getattr(
                getattr(st.session_state.get("data_quality_report"), "quality_score", None),
                "overall_score",
                None,
            ),
        ),
        "DATA_CARD.md",
        "text/markdown",
        use_container_width=True,
    )
    st.markdown("##### Code")
    LOAD = load_kwargs(dataset)
    rules = derive_rules(df)
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_code_button(
            "Data dictionary skeleton",
            lambda o: dic.dictionary_snippet(options=o, **LOAD),
            key="code_dict",
        )
    with k2:
        render_code_button(
            "Apply the dictionary (rename + cast)",
            lambda o: dictionary_apply_snippet(dictionary, options=o, **LOAD),
            key="code_dict_apply",
        )
    with k3:
        render_code_button(
            "Validation from the dictionary (asserts)",
            lambda o: dictionary_validation_snippet(dictionary, df, "asserts", options=o, **LOAD),
            key="code_dict_assert",
        )
    with k4:
        render_code_button(
            "Validation from the dictionary (pandera)",
            lambda o: dictionary_validation_snippet(dictionary, df, "pandera", options=o, **LOAD),
            key="code_dict_pandera",
        )

render_code_export()
