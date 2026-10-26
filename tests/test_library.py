"""Tests for the workspace table library helpers."""

import pandas as pd
import pytest
from src.core import library as lib
from src.core.session_state import init_session_state
from src.datascience.sources import sql_snippet


def test_library_add_remove_and_active():
    state = init_session_state({})
    assert lib.get_tables(state) == {}
    lib.add_table(state, "t1", pd.DataFrame({"a": [1]}))
    assert list(lib.get_tables(state)) == ["t1"]
    with pytest.raises(ValueError):
        lib.add_table(state, "active", pd.DataFrame())
    lib.remove_table(state, "t1")
    assert lib.get_tables(state) == {}
    assert lib.unique_name({"x": 1, "x_2": 1}, "x") == "x_3"
    assert lib.unique_name({}, "active") == "active_2"


def test_library_cap():
    state = init_session_state({})
    for i in range(lib.MAX_TABLES):
        lib.add_table(state, f"t{i}", pd.DataFrame({"a": [i]}))
    with pytest.raises(ValueError, match="full"):
        lib.add_table(state, "one_more", pd.DataFrame())
    lib.add_table(state, "t0", pd.DataFrame({"a": [99]}))  # replacing is fine


def test_sql_snippet_runs(tmp_path):
    csv = tmp_path / "d.csv"
    pd.DataFrame({"g": ["a", "b", "a"], "v": [1, 2, 3]}).to_csv(csv, index=False)
    code = sql_snippet("SELECT g, SUM(v) AS s FROM data GROUP BY g", filename=str(csv))
    ns: dict = {}
    exec(compile(code, "<sql>", "exec"), ns)  # noqa: S102
    assert ns["result"].set_index("g")["s"].to_dict() == {"a": 4, "b": 2}
