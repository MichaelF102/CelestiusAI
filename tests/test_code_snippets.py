"""Every generated snippet must be valid Python that actually runs on real data."""

import ast
import json
from pathlib import Path

import pandas as pd
import pytest
from src.core import code_snippets as cs
from src.ingestion.service import ingest_into_state
from src.ingestion.transformation_history import TransformationPipeline
from src.profiling.actions import actionable_issues, apply_issues

SAMPLE = Path(__file__).resolve().parent.parent / "sample_data" / "ecommerce_customer_churn.csv"
LOAD = {"filename": str(SAMPLE)}


@pytest.fixture(autouse=True)
def _no_plot_windows(monkeypatch):
    import plotly.graph_objects as go

    monkeypatch.setattr(go.Figure, "show", lambda self, *a, **k: None)


def run(code: str) -> dict:
    ast.parse(code)
    ns: dict = {"__name__": "snippet"}
    exec(compile(code, "<snippet>", "exec"), ns)  # noqa: S102 - generated, trusted test input
    return ns


SNIPPETS = {
    "profile": lambda: cs.profile_snippet(**LOAD),
    "missing": lambda: cs.missing_snippet(**LOAD),
    "duplicates": lambda: cs.duplicates_snippet(["customer_id"], **LOAD),
    "outliers_iqr": lambda: cs.outliers_snippet("monthly_spend", "iqr", **LOAD),
    "outliers_z": lambda: cs.outliers_snippet("monthly_spend", "zscore", **LOAD),
    "outliers_mz": lambda: cs.outliers_snippet("monthly_spend", "modified_zscore", **LOAD),
    "dist_num": lambda: cs.distribution_snippet("monthly_spend", True, **LOAD),
    "dist_cat": lambda: cs.distribution_snippet("country", False, **LOAD),
    "corr": lambda: cs.correlation_snippet(
        ["age", "monthly_spend", "total_orders"], "spearman", **LOAD
    ),
    "grouped": lambda: cs.grouped_snippet("country", "monthly_spend", "mean", **LOAD),
    "crosstab": lambda: cs.crosstab_snippet("country", "churn", **LOAD),
    "trend": lambda: cs.time_trend_snippet("signup_date", "monthly_spend", "MS", **LOAD),
    "t_two": lambda: cs.hypothesis_test_snippet("two_group", ["monthly_spend", "churn"], **LOAD),
    "t_multi": lambda: cs.hypothesis_test_snippet(
        "multi_group", ["monthly_spend", "country"], **LOAD
    ),
    "t_paired": lambda: cs.hypothesis_test_snippet(
        "paired", ["monthly_spend", "total_orders"], **LOAD
    ),
    "t_cat": lambda: cs.hypothesis_test_snippet("categorical", ["country", "churn"], **LOAD),
    "t_corr": lambda: cs.hypothesis_test_snippet("correlation", ["age", "monthly_spend"], **LOAD),
    "ols": lambda: cs.regression_snippet(
        "monthly_spend", ["age", "total_orders", "country"], **LOAD
    ),
    "model_reg": lambda: cs.model_training_snippet(
        "monthly_spend", ["age", "total_orders", "country"], "regression", **LOAD
    ),
    "model_clf": lambda: cs.model_training_snippet(
        "churn", ["age", "monthly_spend", "support_tickets"], "binary_classification", **LOAD
    ),
}


@pytest.mark.parametrize("name", sorted(SNIPPETS))
def test_snippet_runs(name):
    code = SNIPPETS[name]()
    assert "your_file" not in code
    run(code)


def test_cleaning_snippet_reproduces_pipeline():
    state: dict = {}
    from src.core.session_state import init_session_state

    init_session_state(state)
    ds = ingest_into_state(state, SAMPLE.read_bytes(), SAMPLE.name, max_size_mb=10)
    pipeline = TransformationPipeline()
    apply_issues(pipeline, ds, actionable_issues(state["data_quality_report"]))
    assert pipeline.specs
    ns = run(cs.cleaning_snippet(pipeline.specs, **LOAD))
    pd.testing.assert_frame_equal(
        ns["df_clean"].reset_index(drop=True),
        ds.working_df.reset_index(drop=True),
        check_dtype=False,
    )


def test_notebook_structure():
    nb = json.loads(cs.build_notebook(cs.profile_snippet()))
    assert nb["nbformat"] == 4
    kinds = [c["cell_type"] for c in nb["cells"]]
    assert kinds[0] == "markdown" and kinds.count("code") >= 3
    compile(
        "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"),
        "nb",
        "exec",
    )


def test_load_cell_variants():
    assert "pd.read_csv('d.csv', sep=';', encoding='latin-1')" in cs.load_cell(
        "d.csv", None, "latin-1", ";"
    )
    assert "pd.read_excel('a.xlsx', sheet_name='S1')" in cs.load_cell("a.xlsx", "S1")
    assert "pd.read_parquet('a.parquet')" in cs.load_cell("a.parquet")
    assert "pd.read_csv('d.csv')" in cs.load_cell("d.csv", None, "utf-8", ",")
