"""Code generators for loading data from files, URLs, sheets, SQLite and databases."""

from __future__ import annotations

from typing import Any

from src.core.code_options import make_snippet, read_call, sheets_export_url
from src.core.code_options import q as _q

SQLALCHEMY_DIALECTS = {
    "postgresql": ("postgresql+psycopg2", 5432, "psycopg2-binary"),
    "mysql": ("mysql+pymysql", 3306, "pymysql"),
    "sqlite": ("sqlite", 0, ""),
}


def url_snippet(url: str, **load: Any) -> str:
    body = f"""url = {_q(url)}
df = {read_call("url", url)}
#? assert not df.empty, "the URL returned no rows"
print(df.shape)
df.head()
"""
    return make_snippet("Load from a URL", body, include_load=False, **load)


def gsheet_snippet(url: str, **load: Any) -> str:
    export = (
        sheets_export_url(url)
        if url.strip()
        else "https://docs.google.com/spreadsheets/d/<SHEET_ID>/export?format=csv"
    )
    body = f"""# The sheet must be shared as "Anyone with the link can view"
sheet_url = {_q(export)}
df = pd.read_csv(sheet_url)
print(df.shape)
df.head()
"""
    return make_snippet("Load a Google Sheet", body, include_load=False, **load)


def sqlite_snippet(db_filename: str, query: str, **load: Any) -> str:
    body = f"""import sqlite3

conn = sqlite3.connect({_q(db_filename)})
#~ List the tables first so you know what to query.
print(pd.read_sql_query("SELECT name FROM sqlite_master WHERE type IN ('table', 'view')", conn))

# %%
query = {_q(query)}
df = pd.read_sql_query(query, conn)
conn.close()
df.head()
"""
    return make_snippet("Query a SQLite database", body, include_load=False, **load)


def added_table_snippet(filename: str, table_name: str, **load: Any) -> str:
    body = f"""# A second table, ready to join with `df` (see the Transformation Studio code)
{table_name} = {read_call(_q(filename), filename)}
print({table_name}.shape)
{table_name}.head()
"""
    return make_snippet(f"Load side table '{table_name}'", body, include_load=False, **load)


def chunked_read_snippet(
    filename: str,
    chunksize: int = 100_000,
    sample_frac: float | None = None,
    nrows: int | None = None,
    **load: Any,
) -> str:
    """Read big files in pieces, optionally sampling, so they fit in memory."""
    lower = filename.lower()
    if lower.endswith(".parquet"):
        body = f"""import pyarrow.parquet as pq

pf = pq.ParquetFile({_q(filename)})
parts = []
for batch in pf.iter_batches(batch_size={chunksize}):
    chunk = batch.to_pandas()
    parts.append(chunk{f".sample(frac={sample_frac}, random_state=0)" if sample_frac else ""})
df = pd.concat(parts, ignore_index=True)
print(df.shape)
"""
    elif lower.endswith((".xlsx", ".xls")):
        body = f"""# Excel files cannot be streamed: read only the rows you need
df = pd.read_excel({_q(filename)}{f", nrows={nrows}" if nrows else ""})
print(df.shape)
"""
    else:
        limit = f", nrows={nrows}" if nrows else ""
        body = f"""#~ chunksize returns an iterator: only one chunk is in memory at a time.
parts = []
for chunk in pd.read_csv({_q(filename)}, chunksize={chunksize}{limit}):
    parts.append(chunk{f".sample(frac={sample_frac}, random_state=0)" if sample_frac else ""})
df = pd.concat(parts, ignore_index=True)
print(df.shape)
"""
    return make_snippet("Chunked / sampled read", body + "df.head()\n", include_load=False, **load)


def sqlalchemy_snippet(
    dialect: str, host: str, port: int, database: str, query: str, **load: Any
) -> str:
    """Database connector reading credentials from environment variables (never hard-coded)."""
    if dialect == "bigquery":
        body = f"""import os
import pandas_gbq

project_id = os.environ["GCP_PROJECT"]          # set this in your environment / Colab secrets
query = {_q(query)}
df = pandas_gbq.read_gbq(query, project_id=project_id)
df.head()
"""
        return make_snippet("BigQuery query", body, include_load=False, **load)
    if dialect not in SQLALCHEMY_DIALECTS:
        raise ValueError(f"Unsupported dialect '{dialect}'")
    driver, default_port, _ = SQLALCHEMY_DIALECTS[dialect]
    if dialect == "sqlite":
        url_expr = f'"sqlite:///{database}"'
        creds = ""
    else:
        creds = """user = os.environ["DB_USER"]            # never type passwords into a notebook
password = os.environ["DB_PASSWORD"]    # in Colab: use Secrets, or %env DB_PASSWORD=...
"""
        url_expr = f'f"{driver}://{{user}}:{{password}}@{host}:{port or default_port}/{database}"'
    body = f"""import os
from sqlalchemy import create_engine, text

{creds}engine = create_engine({url_expr})
query = {_q(query)}
with engine.connect() as conn:
    df = pd.read_sql_query(text(query), conn)
#? assert not df.empty, "the query returned no rows"
print(df.shape)
df.head()
"""
    return make_snippet(f"{dialect.title()} query via SQLAlchemy", body, include_load=False, **load)


def duckdb_sql_snippet(query: str, table: str = "data", **load: Any) -> str:
    body = f"""import duckdb

con = duckdb.connect()
con.register({_q(table)}, df)         # the DataFrame becomes a SQL table
query = {_q(query)}
result = con.execute(query).df()
result.head(20)
"""
    return make_snippet("SQL on a DataFrame (DuckDB)", body, **load)
