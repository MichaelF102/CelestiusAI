"""Tests for SQL read-only enforcement, NL-to-SQL, and safe URL fetching."""

import http.server
import sqlite3
import threading

import pandas as pd
import pytest
from src.agent.llm_provider import MockLLMProvider
from src.datascience import sources as s


@pytest.fixture
def ws():
    tables = {
        "Orders": pd.DataFrame({"id": [1, 2, 3], "cid": [1, 1, 2], "amount": [10.0, 20.0, 5.0]}),
        "Customers": pd.DataFrame({"cid": [1, 2], "name": ["Ann", "Bo"]}),
    }
    return s.SqlWorkspace(tables)


def test_select_and_join_and_aggregate(ws):
    out = ws.query(
        "SELECT c.name, SUM(o.amount) AS total FROM orders o JOIN customers c ON o.cid = c.cid "
        "GROUP BY c.name ORDER BY total DESC;"
    )
    assert out.to_dict("records") == [{"name": "Ann", "total": 30.0}, {"name": "Bo", "total": 5.0}]
    assert len(ws.query("WITH x AS (SELECT * FROM orders) SELECT * FROM x")) == 3
    assert ws.query("SELECT * FROM orders", limit=2).shape[0] == 2


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE orders",
        "DELETE FROM orders",
        "INSERT INTO orders VALUES (9, 9, 9)",
        "UPDATE orders SET amount = 0",
        "SELECT 1; DROP TABLE orders",
        "PRAGMA table_info(orders)",
        "ATTACH DATABASE 'x.db' AS x",
        "CREATE TABLE t (a int)",
        "",
        "   ;  ",
        "EXPLAIN SELECT 1",
    ],
)
def test_non_select_rejected(ws, sql):
    with pytest.raises(s.SourceError):
        ws.query(sql)
    assert len(ws.query("SELECT * FROM orders")) == 3  # data intact


def test_keywords_inside_strings_and_comments_are_fine(ws):
    assert len(ws.query("SELECT 'drop table; delete' AS x FROM orders -- insert here")) == 3


def test_authorizer_blocks_writes_even_if_validator_bypassed(ws):
    with pytest.raises(sqlite3.DatabaseError):
        ws._conn.execute("DELETE FROM orders")
    with pytest.raises(sqlite3.DatabaseError):
        ws._conn.execute("CREATE TABLE z (a int)")


def test_query_timeout(ws):
    runaway = "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM r) SELECT count(*) FROM r"
    old = s.QUERY_TIMEOUT_S
    s.QUERY_TIMEOUT_S = 0.3
    try:
        with pytest.raises(s.SourceError, match="too long"):
            ws.query(runaway)
    finally:
        s.QUERY_TIMEOUT_S = old


def test_sqlite_file_upload_and_schema(tmp_path):
    path = tmp_path / "x.db"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE pets (id INTEGER, name TEXT)")
    con.execute("INSERT INTO pets VALUES (1, 'Rex')")
    con.commit()
    con.close()
    ws = s.SqlWorkspace(sqlite_bytes=path.read_bytes())
    assert ws.query("SELECT name FROM pets")["name"].tolist() == ["Rex"]
    assert "pets(" in s.schema_text(ws)
    with pytest.raises(s.SourceError):
        s.SqlWorkspace(sqlite_bytes=b"not a database at all")


def test_nl_to_sql_validates_model_output(ws):
    good = MockLLMProvider(
        predefined_response="Here you go:\n```sql\nSELECT COUNT(*) AS n FROM orders;\n```"
    )
    sql = s.nl_to_sql(good, "how many orders?", ws)
    assert ws.query(sql)["n"].iloc[0] == 3
    evil = MockLLMProvider(predefined_response="```sql\nDROP TABLE orders\n```")
    with pytest.raises(s.SourceError):
        s.nl_to_sql(evil, "x", ws)
    assert "orders(" in s.build_sql_prompt("q", s.schema_text(ws))
    assert "price" not in s.build_sql_prompt("q", s.schema_text(ws))


def test_sheets_url_normalization():
    url = "https://docs.google.com/spreadsheets/d/AbC_123-x/edit#gid=456"
    assert (
        s.normalize_sheets_url(url)
        == "https://docs.google.com/spreadsheets/d/AbC_123-x/export?format=csv&gid=456"
    )
    assert s.normalize_sheets_url("https://example.com/a.csv") == "https://example.com/a.csv"


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/x.csv",
        "http://localhost/x.csv",
        "http://10.0.0.5/x.csv",
        "http://169.254.169.254/latest/meta-data",
        "http://192.168.1.1/x.csv",
        "ftp://example.com/x.csv",
        "file:///etc/passwd",
        "not a url",
    ],
)
def test_ssrf_and_scheme_guards(url):
    with pytest.raises(s.SourceError):
        s.fetch_url(url)


def test_fetch_from_local_server_with_limits():
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.path == "/redirect":
                self.send_response(302)
                self.send_header("Location", "http://127.0.0.1:1/secret")
                self.end_headers()
                return
            body = b"a,b\n1,2\n" if self.path.endswith(".csv") else b"x" * 5000
            self.send_response(200)
            self.send_header("Content-Type", "text/csv")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        data, name = s.fetch_url(f"{base}/t.csv", allow_private=True)
        assert data == b"a,b\n1,2\n" and name == "t.csv"
        with pytest.raises(s.SourceError, match="larger"):
            s.fetch_url(f"{base}/big.bin", max_bytes=100, allow_private=True)
        # redirect to a private address must be re-checked when private access is NOT allowed
        with pytest.raises(s.SourceError):
            s.fetch_url(f"{base}/redirect")
    finally:
        server.shutdown()
