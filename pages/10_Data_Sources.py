"""Page 10: Data Sources - read-only SQL, URLs, Google Sheets, and extra tables."""

import streamlit as st
from src.agent.provider_factory import ProviderFactory
from src.config.settings import get_settings
from src.core.code_snippets import load_kwargs
from src.core.code_sources import (
    SQLALCHEMY_DIALECTS,
    added_table_snippet,
    chunked_read_snippet,
    duckdb_sql_snippet,
    gsheet_snippet,
    sqlalchemy_snippet,
    sqlite_snippet,
    url_snippet,
)
from src.core.exceptions import DataAnalystError
from src.core.library import ACTIVE, add_table, get_tables, remove_table, unique_name
from src.core.session_state import KEY_LLM_OVERRIDE
from src.core.ui_components import page_setup, render_code_button, render_code_export
from src.datascience import sources as src
from src.datascience.transform import slug_name
from src.ingestion.loader import load_dataset_from_stream
from src.ingestion.service import ingest_dataframe

page_setup(
    "Data Sources",
    "🔌",
    "Query with read-only SQL, pull from a URL or Google Sheet, and add side tables for joins.",
)
settings = get_settings()
tables = get_tables(st.session_state)


def _parse_bytes(data: bytes, name: str):
    return load_dataset_from_stream(data, name, max_size_mb=settings.max_upload_size_mb).working_df


tab_sql, tab_url, tab_upload, tab_db, tab_lib = st.tabs(
    [
        "🧮 SQL workspace",
        "🌐 URL / Google Sheet",
        "📎 Add a table",
        "🔗 Database connectors",
        "🗂️ Table library",
    ]
)
ACTIVE_LOAD = load_kwargs(st.session_state.get("current_dataset"))

# ------------------------------------------------------------------ SQL
with tab_sql:
    st.caption(
        "Queries run on an in-memory copy, are limited to a single SELECT, and can never change your data. "
        f"Refer to tables by name; the loaded dataset is called `{slug_name(ACTIVE)}`."
    )
    db_file = st.file_uploader("Optional: a SQLite database file", type=["db", "sqlite", "sqlite3"])
    try:
        workspace = src.SqlWorkspace(tables, db_file.getvalue() if db_file else None)
    except src.SourceError as err:
        st.error(str(err))
        st.stop()

    schema = workspace.describe()
    with st.expander(f"Tables ({len(schema)})"):
        for table, cols in schema.items():
            st.markdown(f"**{table}**: " + ", ".join(f"`{c}`" for c, _ in cols))

    first_table = next(iter(schema), "active")
    question = st.text_input(
        "Ask in plain English (optional)", placeholder="Total sales per region"
    )
    if st.button("✨ Draft SQL with AI", disabled=not question.strip()):
        provider = ProviderFactory.from_override(st.session_state.get(KEY_LLM_OVERRIDE), settings)
        if not provider.is_configured:
            st.warning(
                "No AI provider is configured (Settings page). Write the SQL yourself below."
            )
        else:
            with st.spinner("Drafting a query..."):
                try:
                    st.session_state["sql_text"] = src.nl_to_sql(provider, question, workspace)
                except src.SourceError as err:
                    st.error(str(err))
            st.info("Review the generated query before running it.")

    st.session_state.setdefault("sql_text", f'SELECT * FROM "{first_table}" LIMIT 100')
    query = st.text_area("SQL", height=140, key="sql_text")
    run_col, code_col = st.columns([1, 1])
    if run_col.button("▶️ Run query", type="primary"):
        try:
            st.session_state["sql_result"] = workspace.query(query)
        except src.SourceError as err:
            st.error(str(err))
    with code_col:
        engine = st.radio(
            "Code engine",
            ["sqlite3", "duckdb"],
            horizontal=True,
            key="sql_code_engine",
            label_visibility="collapsed",
        )
        _tbl = first_table
        render_code_button(
            "SQL on a DataFrame",
            (lambda o: src.sql_snippet(query, _tbl, options=o, **ACTIVE_LOAD))
            if engine == "sqlite3"
            else (lambda o: duckdb_sql_snippet(query, _tbl, options=o, **ACTIVE_LOAD)),
            key="code_sql",
            sql_builder=lambda: query,
        )

    result = st.session_state.get("sql_result")
    if result is not None:
        st.caption(f"{len(result):,} rows × {len(result.columns)} columns")
        st.dataframe(result.head(1000), use_container_width=True)
        s1, s2, s3 = st.columns([2, 1, 1])
        new_name = s1.text_input("Save result as", value="sql_result", key="sql_save_name")
        if s2.button("Add to library", disabled=result.empty):
            add_table(st.session_state, unique_name(tables, slug_name(new_name)), result)
            st.success("Added to the library.")
            st.rerun()
        if s3.button("Use as dataset", disabled=result.empty):
            try:
                ingest_dataframe(
                    st.session_state, result, slug_name(new_name), settings.max_upload_size_mb
                )
                st.switch_page("pages/1_Executive_Overview.py")
            except DataAnalystError as err:
                st.error(err.user_message)

# ------------------------------------------------------------------ URL
with tab_url:
    st.caption(
        "Public `http(s)` CSV/Excel/Parquet links or a Google Sheet shared as *Anyone with the link*. "
        "Private and internal addresses are blocked."
    )
    url = st.text_input("URL", placeholder="https://example.com/data.csv")
    if st.button("Fetch", disabled=not url.strip()):
        with st.spinner("Downloading..."):
            try:
                data, name = src.fetch_url(url.strip())
                st.session_state["url_df"] = (name, _parse_bytes(data, name))
            except (src.SourceError, DataAnalystError) as err:
                st.error(getattr(err, "user_message", None) or str(err))
    if url.strip():
        _u = url.strip()
        render_code_button(
            "Load from a Google Sheet"
            if "docs.google.com/spreadsheets" in _u
            else "Load from a URL",
            (lambda o: gsheet_snippet(_u, options=o))
            if "docs.google.com/spreadsheets" in _u
            else (lambda o: url_snippet(_u, options=o)),
            key="code_url",
        )
    fetched = st.session_state.get("url_df")
    if fetched:
        name, frame = fetched
        st.success(f"Fetched `{name}`: {len(frame):,} rows × {len(frame.columns)} columns")
        st.dataframe(frame.head(20), use_container_width=True)
        u1, u2 = st.columns(2)
        if u1.button("Add to library", key="url_add"):
            add_table(
                st.session_state, unique_name(tables, slug_name(name.rsplit(".", 1)[0])), frame
            )
            st.rerun()
        if u2.button("Use as dataset", key="url_use"):
            ingest_dataframe(
                st.session_state,
                frame,
                slug_name(name.rsplit(".", 1)[0]),
                settings.max_upload_size_mb,
            )
            st.switch_page("pages/1_Executive_Overview.py")

# ------------------------------------------------------------------ upload
with tab_upload:
    st.caption("Add another file as a side table (for joins in the Transformation Studio or SQL).")
    extra = st.file_uploader(
        "CSV / Excel / Parquet", type=["csv", "xlsx", "xls", "parquet"], key="extra_table"
    )
    if extra is not None:
        try:
            frame = _parse_bytes(extra.getvalue(), extra.name)
        except DataAnalystError as err:
            st.error(err.user_message)
        else:
            st.dataframe(frame.head(10), use_container_width=True)
            t_name = st.text_input("Table name", value=slug_name(extra.name.rsplit(".", 1)[0]))
            cc1, cc2 = st.columns(2)
            with cc1:
                render_code_button(
                    f"Load side table '{slug_name(t_name)}'",
                    lambda o: added_table_snippet(extra.name, slug_name(t_name), options=o),
                    key="code_added_table",
                )
            with cc2:
                chunk = st.number_input(
                    "Chunk size (rows)", 1000, 5_000_000, 100_000, 10_000, key="chunk_rows"
                )
                frac = st.slider("Sample fraction", 0.01, 1.0, 1.0, 0.01, key="chunk_frac")
                render_code_button(
                    "Chunked / sampled read",
                    lambda o: chunked_read_snippet(
                        extra.name, int(chunk), None if frac >= 1 else frac, options=o
                    ),
                    key="code_chunked",
                )
            if st.button("Add table", type="primary"):
                try:
                    add_table(st.session_state, unique_name(tables, slug_name(t_name)), frame)
                    st.rerun()
                except ValueError as err:
                    st.error(str(err))

# ------------------------------------------------------------------ connectors
with tab_db:
    st.caption(
        "Generates connection code for a database you run yourself. **Nothing is connected from this app**, "
        "and credentials are read from environment variables in the generated code, never typed here."
    )
    d1, d2, d3 = st.columns(3)
    dialect = d1.selectbox("Database", ["postgresql", "mysql", "sqlite", "bigquery"])
    host = d2.text_input("Host", value="localhost", disabled=dialect in ("sqlite", "bigquery"))
    port = d3.number_input(
        "Port",
        0,
        65535,
        SQLALCHEMY_DIALECTS.get(dialect, ("", 0, ""))[1],
        disabled=dialect in ("sqlite", "bigquery"),
    )
    database = st.text_input(
        "Database / file name", value="mydb.sqlite" if dialect == "sqlite" else "mydb"
    )
    db_query = st.text_area(
        "Query", value="SELECT * FROM my_table LIMIT 1000", height=100, key="db_query"
    )
    render_code_button(
        f"{dialect.title()} connector",
        lambda o: sqlalchemy_snippet(dialect, host, int(port), database, db_query, options=o),
        key="code_db",
    )
    st.markdown("##### SQLite file")
    if db_file is not None:
        render_code_button(
            "Query the uploaded SQLite file",
            lambda o: sqlite_snippet(db_file.name, db_query, options=o),
            key="code_sqlite",
        )
    else:
        st.caption("Upload a SQLite file on the SQL workspace tab to get code for it here.")

# ------------------------------------------------------------------ library
with tab_lib:
    if not tables:
        st.info("No tables yet.")
    for name, frame in tables.items():
        with st.container(border=True):
            l1, l2 = st.columns([5, 1])
            l1.markdown(f"**{name}** · {len(frame):,} rows × {len(frame.columns)} columns")
            if name != ACTIVE and l2.button("Remove", key=f"rm_{name}"):
                remove_table(st.session_state, name)
                st.rerun()
            st.caption(
                ", ".join(map(str, frame.columns[:12])) + (" …" if len(frame.columns) > 12 else "")
            )

render_code_export()
