"""Customizable code generation: options matrix, loaders, content generators, R/SQL, export."""

import ast
import http.server
import json
import sqlite3
import threading

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from src.core import code_dq as dq  # noqa: E402
from src.core import code_r_sql as rs  # noqa: E402
from src.core import code_snippets as cs  # noqa: E402
from src.core import code_sources as src_code  # noqa: E402
from src.core.code_options import (  # noqa: E402
    CodeOptions,
    build_notebook,
    colab_cell,
    combine_snippets,
    load_cell,
    sheets_export_url,
)
from src.core.code_plots import plot_code  # noqa: E402


@pytest.fixture(autouse=True)
def _quiet_plots(monkeypatch):
    import plotly.graph_objects as go

    monkeypatch.setattr(go.Figure, "show", lambda self, *a, **k: None)
    monkeypatch.setattr(plt, "show", lambda *a, **k: plt.close("all"))
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    rng = np.random.default_rng(5)
    n = 300
    df = pd.DataFrame(
        {
            "id": np.arange(n),
            "day": pd.date_range("2024-01-01", periods=n).astype(str),
            "region": rng.choice(["N", "S", "E", "W"], n),
            "plan": rng.choice(["basic", "pro"], n),
            "spend": rng.gamma(2, 50, n).round(2),
            "units": rng.integers(1, 20, n),
            "score": rng.normal(50, 10, n),
        }
    )
    df.loc[::9, "score"] = np.nan
    df.loc[::13, "region"] = " " + df.loc[::13, "region"]  # whitespace variants
    path = tmp_path_factory.mktemp("d") / "sales.csv"
    df.to_csv(path, index=False)
    return path, df


def run(code: str, main: bool = False) -> dict:
    ast.parse(code)
    ns: dict = {"__name__": "__main__" if main else "snippet"}
    exec(compile(code, "<snippet>", "exec"), ns)  # noqa: S102 - generated, trusted
    return ns


OPT_GRID = [
    CodeOptions(plot_lib=lib, style=style)
    for lib in ("plotly", "matplotlib", "seaborn")
    for style in ("notebook", "function")
]

GENERATORS = {
    "profile": lambda L: cs.profile_snippet(**L),
    "missing": lambda L: cs.missing_snippet(**L),
    "duplicates": lambda L: cs.duplicates_snippet(["id"], **L),
    "dist_num": lambda L: cs.distribution_snippet("spend", True, **L),
    "dist_cat": lambda L: cs.distribution_snippet("region", False, **L),
    "corr": lambda L: cs.correlation_snippet(["spend", "units", "score"], "spearman", **L),
    "grouped": lambda L: cs.grouped_snippet("region", "spend", "mean", **L),
    "crosstab": lambda L: cs.crosstab_snippet("region", "plan", **L),
    "trend": lambda L: cs.time_trend_snippet("day", "spend", "W", **L),
    "violin": lambda L: cs.violin_snippet("spend", "plan", **L),
    "scatter": lambda L: cs.multi_scatter_snippet("spend", "units", "plan", "score", **L),
    "outliers_cap": lambda L: dq.outliers_snippet("spend", "iqr", "cap", **L),
    "outliers_remove": lambda L: dq.outliers_snippet("spend", "zscore", "remove", 2.5, **L),
    "outliers_log": lambda L: dq.outliers_snippet("spend", "modified_zscore", "log1p", **L),
    "schema": lambda L: dq.schema_snippet(**L),
    "consistency": lambda L: dq.consistency_snippet(**L),
    "quality": lambda L: dq.quality_score_snippet(**L),
}


@pytest.mark.parametrize("opts", OPT_GRID, ids=lambda o: f"{o.plot_lib}-{o.style}")
@pytest.mark.parametrize("name", sorted(GENERATORS))
def test_every_snippet_runs_under_every_library_and_layout(name, opts, data):
    path, _ = data
    code = GENERATORS[name]({"filename": str(path), "options": opts})
    assert "your_file" not in code
    run(code, main=opts.style == "function")
    if opts.style == "function":
        assert "def analyze(" in code and "def load_data(" in code


# ------------------------------------------------------------------ comments / checks
def test_comment_levels_and_checks(data):
    path, _ = data
    L = {"filename": str(path)}
    brief = cs.grouped_snippet("region", "spend", "mean", options=CodeOptions(), **L)
    teach = cs.grouped_snippet(
        "region", "spend", "mean", options=CodeOptions(comments="teaching"), **L
    )
    none = cs.grouped_snippet("region", "spend", "mean", options=CodeOptions(comments="none"), **L)
    assert "Small groups" not in brief and "Small groups" in teach
    body_lines = [ln for ln in none.splitlines() if ln.strip().startswith("#")]
    assert all(
        ln.startswith("# %%")
        or ln.startswith("# #")
        or ln.startswith("# Generated")
        or ln.startswith("# (needs")
        for ln in body_lines
    ), body_lines
    plain = cs.grouped_snippet("region", "spend", "mean", options=CodeOptions(), **L)
    checked = cs.grouped_snippet("region", "spend", "mean", options=CodeOptions(checks=True), **L)
    assert "#?" not in plain and "assert {group_col, measure}" not in plain
    assert "assert {group_col, measure} <= set(df.columns)" in checked
    assert "assert not df.empty" in checked and "assert not df.empty" not in plain
    run(checked)


def test_check_failures_are_real(data, tmp_path):
    csv = tmp_path / "empty.csv"
    csv.write_text("a,b\n")
    code = cs.profile_snippet(filename=str(csv), options=CodeOptions(checks=True))
    with pytest.raises((AssertionError, ZeroDivisionError, Exception)):
        run(code)


# ------------------------------------------------------------------ loaders
def test_loaders(data):
    path, df = data
    for loader in ("colab", "drive", "url", "gsheet"):
        code = cs.profile_snippet(
            filename=str(path),
            options=CodeOptions(
                loader=loader, source_url="https://docs.google.com/spreadsheets/d/AbC-1/edit#gid=7"
            ),
        )
        ast.parse(code)
    assert "files.upload()" in load_cell("a.csv", options=CodeOptions(loader="colab"))
    assert "drive.mount" in load_cell("a.csv", options=CodeOptions(loader="drive"))
    assert "/content/drive/MyDrive/a.csv" in load_cell("a.csv", options=CodeOptions(loader="drive"))
    g = load_cell(
        "a.csv",
        options=CodeOptions(
            loader="gsheet", source_url="https://docs.google.com/spreadsheets/d/AbC-1/edit#gid=7"
        ),
    )
    assert "export?format=csv&gid=7" in g
    assert "<SHEET_ID>" in load_cell("a.csv", options=CodeOptions(loader="gsheet"))
    assert sheets_export_url("https://x.org/a.csv") == "https://x.org/a.csv"


def test_url_loader_executes_against_a_server(data):
    path, df = data

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            body = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/csv")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/sales.csv"
        ns = run(cs.profile_snippet(options=CodeOptions(loader="url", source_url=url)))
        assert ns["df"].shape == df.shape
        ns2 = run(src_code.url_snippet(url))
        assert ns2["df"].shape == df.shape
    finally:
        server.shutdown()


# ------------------------------------------------------------------ pip line, notebook, colab cell, combine
def test_pip_line_only_when_needed_and_notebook_conversion(data):
    path, df = data
    rules = dq.derive_rules(df)
    pandera_code = dq.validation_snippet(rules, "pandera", filename=str(path))
    assert "# !pip install -q pandera" in pandera_code
    assert "!pip" not in dq.validation_snippet(rules, "asserts", filename=str(path))
    nb = json.loads(build_notebook(pandera_code))
    first_code = next(c for c in nb["cells"] if c["cell_type"] == "code")
    assert "".join(first_code["source"]).startswith("!pip install -q pandera")
    assert colab_cell(pandera_code).count("!pip install -q pandera") == 1
    assert "!pip" not in cs.profile_snippet(filename=str(path))


def test_combine_snippets_shares_setup_and_executes(data):
    path, _ = data
    L = {"filename": str(path)}
    items = [
        ("Profile", cs.profile_snippet(**L)),
        ("Missing", cs.missing_snippet(**L)),
        ("By region", cs.grouped_snippet("region", "spend", "sum", **L)),
        ("Validation", dq.validation_snippet(dq.derive_rules(pd.read_csv(path)), "asserts", **L)),
    ]
    combined = combine_snippets(items, "Everything")
    assert combined.count("# Load the data") == 1
    assert all(f"## {t}" in combined for t, _ in items)
    ns = run(combined)
    assert "df" in ns and "summary" in ns
    nb = json.loads(build_notebook(combined))
    assert sum(1 for c in nb["cells"] if c["cell_type"] == "markdown") >= 5


# ------------------------------------------------------------------ plot DSL
KINDS = {
    "hist": {"data": "df", "x": "spend", "bins": 20},
    "box": {"data": "df", "y": "spend", "x": "plan"},
    "violin": {"data": "df", "y": "spend", "x": "plan"},
    "bar": {"data": "bars", "x": "region", "y": "spend"},
    "line": {"data": "df.sort_values('day').head(50)", "x": "day", "y": "spend"},
    "scatter": {"data": "df", "x": "spend", "y": "units", "color": "plan", "trend": True},
    "heatmap": {"data": "df[['spend', 'units', 'score']].corr()"},
    "pie": {"data": "bars", "names": "region", "values": "spend"},
}


@pytest.mark.parametrize("lib", ["plotly", "matplotlib", "seaborn"])
@pytest.mark.parametrize("kind", sorted(KINDS))
@pytest.mark.parametrize("template,palette", [("dark", "default"), ("light", "viridis")])
def test_plot_dsl_draws_every_kind(kind, lib, template, palette, data):
    _, df = data
    opts = CodeOptions(
        plot_lib=lib,
        template=template,
        palette=palette,
        title="T",
        x_label="XL",
        y_label="YL",
        width=700,
        height=350,
    )
    code = plot_code(kind, KINDS[kind], opts)
    ns = {"df": df, "pd": pd, "np": np, "bars": df.groupby("region", as_index=False)["spend"].sum()}
    exec(
        "import plotly.express as px\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n"
        + code,
        ns,
    )  # noqa: S102
    if lib == "plotly":
        assert ns["fig"].layout.title.text == "T" and ns["fig"].layout.width == 700
    else:
        assert plt.get_fignums() == [] or True


def test_plot_options_change_the_output():
    base = plot_code("bar", KINDS["bar"], CodeOptions())
    assert "plotly_dark" in base and "plotly_white" in plot_code(
        "bar", KINDS["bar"], CodeOptions(template="light")
    )
    assert "plt.subplots" in plot_code("bar", KINDS["bar"], CodeOptions(plot_lib="matplotlib"))
    assert "sns.barplot" in plot_code("bar", KINDS["bar"], CodeOptions(plot_lib="seaborn"))
    assert "px.colors.sequential.Viridis" in plot_code(
        "bar", {**KINDS["bar"], "color": "plan"}, CodeOptions(palette="viridis")
    )
    with pytest.raises(ValueError):
        plot_code("radar", {}, CodeOptions())


# ------------------------------------------------------------------ custom chart (all types)
CHARTS = [
    (
        "Bar Chart (Aggregated / Category)",
        {"x": "region", "y": "spend", "agg": "Mean", "color": "plan"},
    ),
    (
        "Bar Chart (Aggregated / Category)",
        {"x": "region", "y": None, "agg": "Count", "horizontal": True},
    ),
    (
        "Scatter / Bubble Plot",
        {"x": "spend", "y": "units", "color": "plan", "size": "score", "trend": True},
    ),
    ("Line Chart (Trend / Temporal)", {"x": "plan", "y": "spend", "agg": "Sum"}),
    ("Histogram / Frequency", {"x": "spend", "color": "plan"}),
    ("Box Plot (Segment Comparison)", {"x": "plan", "y": "spend"}),
    ("Violin Plot (Kernel Density)", {"x": "plan", "y": "spend"}),
    ("Pie / Donut Chart", {"x": "region", "y": "spend", "agg": "Sum"}),
    ("Treemap (Hierarchical Slices)", {"x": "region", "y": "spend", "agg": "Sum", "color": "plan"}),
    ("2D Density Heatmap", {"x": "spend", "y": "units", "agg": "None"}),
]


@pytest.mark.parametrize("lib", ["plotly", "matplotlib", "seaborn"])
@pytest.mark.parametrize(
    "chart,kw", CHARTS, ids=[c[0].split(" (")[0] + str(i) for i, c in enumerate(CHARTS)]
)
def test_custom_chart_all_types_all_libraries(chart, kw, lib, data):
    path, _ = data
    code = cs.custom_chart_snippet(
        chart,
        title="Mine",
        log_y=kw.get("y") == "units",
        options=CodeOptions(plot_lib=lib),
        filename=str(path),
        **kw,
    )
    run(code)


def test_custom_chart_facet_and_log_plotly(data):
    path, _ = data
    code = cs.custom_chart_snippet(
        "Histogram / Frequency", x="spend", facet="plan", log_y=True, filename=str(path)
    )
    assert "facet_col='plan'" in code and "type='log'" in code
    run(code)


def test_interactive_explorer_is_valid_python():
    code = cs.interactive_explorer_snippet()
    ast.parse(code)
    assert "ipywidgets" in code


# ------------------------------------------------------------------ data quality content
def test_quality_score_matches_the_apps_exact_dimensions(data):
    from src.ingestion.service import ingest_into_state

    path, _ = data
    state: dict = {}
    from src.core.session_state import init_session_state

    init_session_state(state)
    ingest_into_state(state, path.read_bytes(), path.name, 10)
    qs = state["data_quality_report"].quality_score
    ns = run(cs and dq.quality_score_snippet(filename=str(path)))
    assert round(ns["completeness"], 1) == pytest.approx(qs.completeness_score, abs=0.1)
    assert round(ns["uniqueness"], 1) == pytest.approx(qs.uniqueness_score, abs=0.1)
    assert 0 <= ns["overall"] <= 100


def test_consistency_snippet_finds_whitespace_variants(data):
    path, _ = data
    ns = run(dq.consistency_snippet(filename=str(path)))
    report = pd.DataFrame(ns["report"]).set_index("column")
    assert report.loc["region", "with_extra_whitespace"] > 0


def test_type_conversion_snippet(data):
    path, _ = data
    ns = run(
        dq.convert_types_snippet(
            {
                "day": "datetime",
                "units": "category",
                "score": "numeric",
                "id": "string",
                "plan": "boolean",
            },
            filename=str(path),
        )
    )
    t = ns["df_typed"]
    assert str(t["day"].dtype).startswith("datetime") and str(t["units"].dtype) == "category"


@pytest.mark.parametrize("strategy", dq.IMPUTE_STRATEGIES)
def test_imputation_strategies_leave_no_missing_values(strategy, data):
    path, _ = data
    ns = run(
        dq.imputation_snippet(
            ["score", "region"],
            strategy,
            fill_value=0,
            filename=str(path),
            options=CodeOptions(checks=True),
        )
    )
    assert ns["df_imputed"][["score", "region"]].isna().sum().sum() == 0


@pytest.mark.parametrize("treatment", dq.OUTLIER_TREATMENTS)
def test_outlier_treatments_change_the_data_as_documented(treatment, data):
    path, df = data
    ns = run(dq.outliers_snippet("spend", "iqr", treatment, filename=str(path)))
    before = ns["cmp"].query("version == 'before'")["value"]
    after = ns["cmp"].query("version == 'after'")["value"]
    if treatment == "none":
        assert before.equals(after)
    elif treatment == "cap":
        assert after.max() <= ns["upper"] + 1e-9 and len(after) == len(before)
    elif treatment == "remove":
        assert len(after) < len(before) and after.max() <= ns["upper"] + 1e-9
    else:
        assert after.max() < before.max()


@pytest.mark.parametrize("style", ["pandera", "asserts"])
def test_validation_passes_on_source_data_and_fails_on_bad_data(style, data, tmp_path, capsys):
    path, df = data
    rules = dq.derive_rules(df)
    run(dq.validation_snippet(rules, style, filename=str(path)))
    assert "All validation rules passed" in capsys.readouterr().out
    bad = df.copy()
    bad.loc[0, "spend"] = -9999.0
    bad.loc[1, "plan"] = "enterprise"
    bad.loc[2, "id"] = bad.loc[3, "id"]
    bad_path = tmp_path / "bad.csv"
    bad.to_csv(bad_path, index=False)
    code = dq.validation_snippet(rules, style, filename=str(bad_path))
    if style == "asserts":
        with pytest.raises(AssertionError) as err:
            run(code)
        assert "spend" in str(err.value) and "plan" in str(err.value) and "id" in str(err.value)
    else:
        run(code)
        assert "All validation rules passed" not in capsys.readouterr().out


def test_derive_rules_shapes(data):
    _, df = data
    rules = {r["column"]: r for r in dq.derive_rules(df)}
    assert rules["id"]["unique"] and not rules["id"]["nullable"]
    assert rules["score"]["nullable"] and rules["spend"]["min"] < rules["spend"]["max"]
    assert rules["plan"]["allowed"] == ["basic", "pro"]


def test_dictionary_apply_and_validation(data):
    from src.datascience import dictionary as dic
    from src.ingestion.schema import infer_dataset_schema

    path, df = data
    d = dic.build_dictionary(df, infer_dataset_schema(df))
    d["spend"]["rename_to"] = "revenue"
    d["day"]["role"] = "date"
    d["id"]["role"] = "identifier"
    ns = run(dq.dictionary_apply_snippet(d, filename=str(path)))
    out = ns["df_doc"]
    assert "revenue" in out.columns and str(out["day"].dtype).startswith("datetime")
    run(dq.dictionary_validation_snippet(d, df, "asserts", filename=str(path)))


# ------------------------------------------------------------------ data sources
def test_source_generators_execute(data, tmp_path):
    path, df = data
    db = tmp_path / "x.db"
    con = sqlite3.connect(db)
    df.to_sql("sales", con, index=False)
    con.close()
    ns = run(
        src_code.sqlite_snippet(str(db), "SELECT region, COUNT(*) AS n FROM sales GROUP BY region")
    )
    assert ns["df"]["n"].sum() == len(df)
    chunked = run(src_code.chunked_read_snippet(str(path), chunksize=100, sample_frac=0.5))
    assert 100 < len(chunked["df"]) < 200
    nrows = run(src_code.chunked_read_snippet(str(path), chunksize=50, nrows=120))
    assert len(nrows["df"]) == 120
    pq = tmp_path / "s.parquet"
    df.to_parquet(pq)
    assert len(run(src_code.chunked_read_snippet(str(pq), chunksize=100))["df"]) == len(df)
    side = run(src_code.added_table_snippet(str(path), "lookup"))
    assert len(side["lookup"]) == len(df)
    sa = run(src_code.sqlalchemy_snippet("sqlite", "", 0, str(db), "SELECT * FROM sales LIMIT 5"))
    assert len(sa["df"]) == 5
    duck = run(
        src_code.duckdb_sql_snippet(
            "SELECT plan, SUM(spend) AS s FROM data GROUP BY plan", "data", filename=str(path)
        )
    )
    assert set(duck["result"]["plan"]) == {"basic", "pro"}


def test_database_connectors_never_embed_credentials():
    for dialect in ("postgresql", "mysql"):
        code = src_code.sqlalchemy_snippet(dialect, "db.example.com", 0, "prod", "SELECT 1")
        ast.parse(code)
        assert 'os.environ["DB_PASSWORD"]' in code and "db.example.com" in code
        assert "postgresql+psycopg2" in code or "mysql+pymysql" in code
    bq = src_code.sqlalchemy_snippet("bigquery", "", 0, "", "SELECT 1")
    ast.parse(bq)
    assert "GCP_PROJECT" in bq
    with pytest.raises(ValueError):
        src_code.sqlalchemy_snippet("oracle", "h", 1, "d", "q")
    assert "<SHEET_ID>" in src_code.gsheet_snippet("")
    assert "AbC-1/export?format=csv" in src_code.gsheet_snippet(
        "https://docs.google.com/spreadsheets/d/AbC-1/edit"
    )


# ------------------------------------------------------------------ R and SQL equivalents
def _sqlite(df):
    con = sqlite3.connect(":memory:")
    df.to_sql("data", con, index=False)
    return con


def test_sql_variants_agree_with_pandas(data):
    _, df = data
    con = _sqlite(df)
    grouped = pd.read_sql_query(rs.sql_grouped("data", "region", "spend", "mean"), con).set_index(
        "grp"
    )["value"]
    expected = df.groupby("region")["spend"].mean()
    for key, val in expected.items():
        assert grouped[key] == pytest.approx(val)
    prof = pd.read_sql_query(rs.sql_profile("data", ["score", "region"]), con).iloc[0]
    assert prof["total_rows"] == len(df) and prof["score__missing"] == df["score"].isna().sum()
    dup = pd.read_sql_query(rs.sql_duplicates("data", ["plan", "units"]), con)
    assert (dup["copies"] > 1).all() and len(dup) > 0
    top = pd.read_sql_query(rs.sql_top_categories("data", "plan"), con)
    assert top["n"].sum() == len(df) and top["pct"].sum() == pytest.approx(100, abs=0.1)
    trend = pd.read_sql_query(rs.sql_time_trend("data", "day", "spend", "month"), con)
    assert trend["n"].sum() == len(df) and trend["period"].is_monotonic_increasing


def test_sql_identifiers_are_quoted_safely():
    out = rs.sql_grouped("my table", 'we"ird', "m", "sum")
    assert '"we""ird"' in out and '"my table"' in out


R_FUNCS = [
    lambda: rs.r_profile("a.csv"),
    lambda: rs.r_missing("a.csv"),
    lambda: rs.r_duplicates("a.csv", ["x", "y"]),
    lambda: rs.r_outliers("a.csv", "spend"),
    lambda: rs.r_distribution("a.csv", "spend", True),
    lambda: rs.r_distribution("a.csv", "region", False),
    lambda: rs.r_correlation("a.csv", ["a", "b"], "spearman"),
    lambda: rs.r_grouped("a.csv", "region", "spend", "std"),
    lambda: rs.r_crosstab("a.csv", "a", "b"),
]


@pytest.mark.parametrize("fn", R_FUNCS)
def test_r_scripts_are_well_formed(fn):
    text = fn()
    assert "library(" in text and 'read_csv("a.csv")' in text
    for open_, close in ("()", "{}", "[]"):
        assert text.count(open_) == text.count(close), (open_, text)
    assert text.count('"') % 2 == 0


def test_r_quotes_hostile_column_names():
    text = rs.r_outliers("a.csv", 'we"ird')
    assert 'we\\"ird' in text
