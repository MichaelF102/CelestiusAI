"""Read-only data sources: SQL over in-memory tables/SQLite files, URLs, and Google Sheets.

Safety model
- SQL runs on an in-memory SQLite copy; a SQLite *authorizer* denies everything except reads,
  and a text validator allows a single SELECT/WITH statement only.
- URL fetching allows http(s) only, refuses private/loopback/link-local addresses (also on
  every redirect), and enforces a size and time limit.
"""

from __future__ import annotations

import ipaddress
import re
import socket
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

import pandas as pd

from src.datascience.transform import slug_name

MAX_ROWS = 50_000
QUERY_TIMEOUT_S = 10
MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024


class SourceError(ValueError):
    """User-facing source/SQL problem."""


# --------------------------------------------------------------------------- SQL
_STRING_LITERAL = re.compile(r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"")
_FORBIDDEN_SQL = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|replace|vacuum|reindex|"
    r"truncate|load_extension|begin|commit|rollback|savepoint|grant|revoke)\b",
    re.I,
)


def validate_select_sql(sql: str) -> str:
    """Return a cleaned single SELECT/WITH statement or raise SourceError."""
    cleaned = sql.strip().rstrip(";").strip()
    if not cleaned:
        raise SourceError("Enter a SQL query.")
    stripped = _STRING_LITERAL.sub("''", cleaned)
    stripped = re.sub(r"--[^\n]*|/\*.*?\*/", " ", stripped, flags=re.S)
    if ";" in stripped:
        raise SourceError("Only one statement is allowed.")
    if not re.match(r"^\s*(select|with)\b", stripped, re.I):
        raise SourceError("Only SELECT queries are allowed.")
    hit = _FORBIDDEN_SQL.search(stripped)
    if hit:
        raise SourceError(f"'{hit.group(0).upper()}' is not allowed in a read-only query.")
    return cleaned


def _authorizer(action: int, *_: Any) -> int:
    allowed = {
        sqlite3.SQLITE_SELECT,
        sqlite3.SQLITE_READ,
        sqlite3.SQLITE_FUNCTION,
        sqlite3.SQLITE_RECURSIVE,
    }
    return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY


class SqlWorkspace:
    """In-memory SQLite holding copies of tables; queries are strictly read-only."""

    def __init__(
        self, tables: dict[str, pd.DataFrame] | None = None, sqlite_bytes: bytes | None = None
    ) -> None:
        self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        self.table_names: dict[str, str] = {}  # sql name -> display name
        if sqlite_bytes:
            try:
                self._conn.deserialize(sqlite_bytes)
                rows = self._conn.execute(
                    "SELECT name FROM sqlite_master WHERE type IN ('table','view')"
                ).fetchall()
                self.table_names.update({r[0]: r[0] for r in rows})
            except sqlite3.DatabaseError as err:
                raise SourceError(f"Not a valid SQLite database: {err}") from err
        for display, frame in (tables or {}).items():
            sql_name = slug_name(display)
            frame.to_sql(sql_name, self._conn, index=False, if_exists="replace")
            self.table_names[sql_name] = display
        self._conn.set_authorizer(_authorizer)

    def describe(self) -> dict[str, list[tuple[str, str]]]:
        """table -> [(column, declared type)], read through the authorizer-safe path."""
        result: dict[str, list[tuple[str, str]]] = {}
        for name in self.table_names:
            cur = self._conn.execute(f'SELECT * FROM "{name}" LIMIT 0')  # noqa: S608
            cols = [d[0] for d in cur.description]
            sample = pd.read_sql_query(f'SELECT * FROM "{name}" LIMIT 200', self._conn)  # noqa: S608
            result[name] = [(c, str(sample[c].dtype) if c in sample else "") for c in cols]
        return result

    def query(self, sql: str, limit: int = MAX_ROWS) -> pd.DataFrame:
        cleaned = validate_select_sql(sql)
        deadline = time.monotonic() + QUERY_TIMEOUT_S

        def stop() -> int:
            return 1 if time.monotonic() > deadline else 0

        self._conn.set_progress_handler(stop, 10_000)
        try:
            cur = self._conn.execute(cleaned)
            rows = cur.fetchmany(limit)
            cols = [d[0] for d in cur.description or []]
        except sqlite3.OperationalError as err:
            msg = (
                "The query took too long and was stopped."
                if "interrupted" in str(err)
                else f"SQL error: {err}"
            )
            raise SourceError(msg) from err
        except sqlite3.DatabaseError as err:
            raise SourceError(f"SQL error: {err}") from err
        finally:
            self._conn.set_progress_handler(None, 0)
        return pd.DataFrame(rows, columns=cols)

    def close(self) -> None:
        self._conn.close()


def schema_text(workspace: SqlWorkspace) -> str:
    """Compact schema description for prompts (names and types only, no data values)."""
    return "\n".join(
        f"{table}({', '.join(f'{c} {t}' for c, t in cols)})"
        for table, cols in workspace.describe().items()
    )


# --------------------------------------------------------------------------- natural language -> SQL
_SQL_FENCE = re.compile(r"```(?:sql)?\s*(.*?)```", re.S | re.I)


def build_sql_prompt(question: str, schema: str) -> str:
    return (
        "You write SQLite queries. Return ONLY one read-only SELECT statement in a ```sql block.\n"
        f"Schema:\n{schema}\n\nQuestion: {question}\n"
    )


def extract_sql(llm_text: str) -> str:
    match = _SQL_FENCE.search(llm_text)
    return validate_select_sql(match.group(1) if match else llm_text)


def nl_to_sql(provider: Any, question: str, workspace: SqlWorkspace) -> str:
    """Ask an LLM for SQL and validate it. The caller must still show it to the user first."""
    result = provider.generate_text(
        build_sql_prompt(question, schema_text(workspace)), temperature=0.0
    )
    if result.status != "success" or not result.raw_text.strip():
        raise SourceError(result.error_message or "The language model returned no query.")
    return extract_sql(result.raw_text)


# --------------------------------------------------------------------------- URLs & sheets
_SHEETS_RE = re.compile(r"https://docs\.google\.com/spreadsheets/d/([\w-]+)")


def normalize_sheets_url(url: str) -> str:
    """Turn a Google Sheets share link into its CSV export URL (sheet must be link-shared)."""
    match = _SHEETS_RE.match(url.strip())
    if not match:
        return url.strip()
    gid = re.search(r"[#&?]gid=(\d+)", url)
    suffix = f"&gid={gid.group(1)}" if gid else ""
    return f"https://docs.google.com/spreadsheets/d/{match.group(1)}/export?format=csv{suffix}"


def _is_public_address(host: str) -> bool:
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as err:
        raise SourceError(f"Could not resolve host '{host}'.") from err
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global:
            return False
    return True


def check_url(url: str, allow_private: bool = False) -> urllib.parse.ParseResult:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise SourceError("Only http(s) URLs are supported.")
    if not allow_private and not _is_public_address(parsed.hostname):
        raise SourceError("That address is private or internal and cannot be fetched.")
    return parsed


class _CheckedRedirect(urllib.request.HTTPRedirectHandler):
    def __init__(self, allow_private: bool) -> None:
        self.allow_private = allow_private

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        check_url(newurl, self.allow_private)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch_url(
    url: str,
    max_bytes: int = MAX_DOWNLOAD_BYTES,
    timeout: float = 15.0,
    allow_private: bool = False,
) -> tuple[bytes, str]:
    """Download a file; returns (bytes, filename)."""
    url = normalize_sheets_url(url)
    parsed = check_url(url, allow_private)
    opener = urllib.request.build_opener(_CheckedRedirect(allow_private))
    request = urllib.request.Request(url, headers={"User-Agent": "ai-data-analyst/1.0"})  # noqa: S310
    try:
        with opener.open(request, timeout=timeout) as resp:  # noqa: S310
            data = resp.read(max_bytes + 1)
            content_type = resp.headers.get("Content-Type", "")
    except urllib.error.HTTPError as err:
        raise SourceError(f"The server answered HTTP {err.code}.") from err
    except (urllib.error.URLError, TimeoutError, OSError) as err:
        raise SourceError(f"Could not download the file: {err}") from err
    if len(data) > max_bytes:
        raise SourceError(f"The file is larger than {max_bytes // (1024 * 1024)} MB.")
    name = parsed.path.rsplit("/", 1)[-1] or "download"
    if "export" in parsed.path and "csv" in url:
        name = "google_sheet.csv"
    if "." not in name:
        name += ".csv" if "csv" in content_type or "text" in content_type else ".csv"
    return data, name


def sql_snippet(query: str, table: str = "data", **load: object) -> str:
    """Pandas + sqlite3 code that runs `query` against the dataset (table name `table`)."""
    from src.core.code_snippets import make_snippet

    body = f"""import sqlite3

conn = sqlite3.connect(":memory:")
df.to_sql({table!r}, conn, index=False, if_exists="replace")   # the DataFrame becomes a SQL table
query = {query!r}
result = pd.read_sql_query(query, conn)
result.head(20)
"""
    return make_snippet("SQL on a DataFrame", body, **load)
