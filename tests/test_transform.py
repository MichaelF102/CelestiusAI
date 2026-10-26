"""Tests for the multi-table transformation engine."""

import pandas as pd
import pytest
from src.datascience import transform as t


@pytest.fixture
def tables():
    orders = pd.DataFrame(
        {
            "oid": [1, 2, 3, 4],
            "cid": [10, 10, 20, 30],
            "amount": [5.0, 7.5, 3.0, 9.0],
            "day": ["2024-01-01", "2024-01-02", "2024-01-02", "bad"],
        }
    )
    customers = pd.DataFrame({"cid": [10, 20, 40], "name": ["Ann", "Bo", "Cy"]})
    return {"orders": orders, "customers": customers}


def op(name, output, /, **params):
    return t.TableOp(op=name, output=output, params=params)


def test_merge_how_variants(tables):
    inner = t.apply_op(
        tables, op("merge", "j", left="orders", right="customers", on=["cid"], how="inner")
    )
    left = t.apply_op(
        tables, op("merge", "j", left="orders", right="customers", on=["cid"], how="left")
    )
    assert len(inner) == 3 and len(left) == 4 and left["name"].isna().sum() == 1
    diff_keys = t.apply_op(
        tables,
        op(
            "merge",
            "j",
            left="orders",
            right="customers",
            left_on="cid",
            right_on="cid",
            how="outer",
        ),
    )
    assert len(diff_keys) == 5


def test_groupby_pivot_melt_concat(tables):
    g = t.apply_op(tables, op("groupby", "g", table="orders", by=["cid"], aggs={"amount": "sum"}))
    assert g.set_index("cid")["amount"].to_dict() == {10: 12.5, 20: 3.0, 30: 9.0}
    pv = t.apply_op(
        tables,
        op(
            "pivot",
            "p",
            table="orders",
            index=["cid"],
            columns="day",
            values="amount",
            aggfunc="sum",
        ),
    )
    assert "cid" in pv.columns and "2024-01-02" in pv.columns
    wide = pd.DataFrame({"id": [1, 2], "a": [1, 2], "b": [3, 4]})
    melted = t.apply_op(
        {"w": wide}, op("melt", "m", table="w", id_vars=["id"], value_vars=["a", "b"])
    )
    assert len(melted) == 4 and set(melted["variable"]) == {"a", "b"}
    stacked = t.apply_op(tables, op("concat", "c", tables=["orders", "orders"]))
    assert len(stacked) == 8
    with pytest.raises(t.TransformError):
        t.apply_op(tables, op("concat", "c", tables=["orders"]))


def test_derive_filter_convert_rename(tables):
    d = t.apply_op(tables, op("derive", "d", table="orders", name="double", expr="amount * 2"))
    assert d["double"].tolist() == [10.0, 15.0, 6.0, 18.0]
    f = t.apply_op(tables, op("filter", "f", table="orders", expr="amount > 4"))
    assert len(f) == 3
    c = t.apply_op(tables, op("convert", "c", table="orders", column="day", to="datetime"))
    assert c["day"].isna().sum() == 1
    r = t.apply_op(tables, op("rename", "r", table="orders", mapping={"amount": "amt"}))
    assert "amt" in r.columns and "amount" in tables["orders"].columns


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os').system('x')",
        "amount.__class__",
        "@var + 1",
        "amount; 1",
        "(lambda: 1)()",
        "",
    ],
)
def test_dangerous_expressions_rejected(tables, expr):
    with pytest.raises(t.TransformError):
        t.apply_op(tables, op("derive", "d", table="orders", name="x", expr=expr))
    with pytest.raises(t.TransformError):
        t.apply_op(tables, op("filter", "f", table="orders", expr=expr))


def test_errors_are_user_facing(tables):
    with pytest.raises(t.TransformError, match="not found"):
        t.apply_op(tables, op("filter", "f", table="ghost", expr="a > 1"))
    with pytest.raises(t.TransformError):
        t.apply_op(tables, op("groupby", "g", table="orders", by=["nope"], aggs={"amount": "sum"}))
    with pytest.raises(t.TransformError, match="already exists"):
        t.validate_output_name(tables, "Orders")
    assert t.validate_output_name(tables, "New Table!") == "new_table"


def test_rollback_helper():
    ops = [op("filter", "a", table="x", expr="1"), op("filter", "b", table="a", expr="1")]
    assert t.removable_outputs(ops, 1) == ["b"] and t.removable_outputs(ops, 0) == ["a", "b"]


def test_generated_code_reproduces_results(tables, tmp_path):
    ops = [
        op("merge", "joined", left="active", right="customers", on=["cid"], how="left"),
        op("derive", "enriched", table="joined", name="amt2", expr="amount * 2"),
        op("convert", "typed", table="enriched", column="day", to="datetime"),
        op("filter", "big", table="typed", expr="amount > 4"),
        op("groupby", "agg", table="big", by=["cid"], aggs={"amt2": "sum"}),
        op(
            "pivot",
            "pv",
            table="big",
            index=["cid"],
            columns="name",
            values="amount",
            aggfunc="sum",
        ),
    ]
    live = {"active": tables["orders"], "customers": tables["customers"]}
    for o in ops:
        live[o.output] = t.apply_op(live, o)
    csv = tmp_path / "orders.csv"
    tables["orders"].to_csv(csv, index=False)
    code = t.ops_to_code(ops, external=["customers"], filename=str(csv)).replace(
        "# t_customers = ...",
        "t_customers = pd.DataFrame({'cid': [10, 20, 40], 'name': ['Ann', 'Bo', 'Cy']})  # ",
    )
    ns: dict = {}
    exec(compile(code, "<ops>", "exec"), ns)  # noqa: S102
    for name in ("joined", "big", "agg", "pv"):
        pd.testing.assert_frame_equal(
            ns[f"t_{name}"].reset_index(drop=True),
            live[name].reset_index(drop=True),
            check_dtype=False,
        )


# ------------------------------------------------------------------ new operations
@pytest.fixture
def sales():
    return pd.DataFrame(
        {
            "region": ["N", "N", "N", "S", "S", "S"],
            "day": [
                "2024-01-05",
                "2024-02-10",
                "2024-03-15",
                "2024-01-20",
                "2024-02-25",
                "2024-04-30",
            ],
            "amount": [10.0, 20.0, 40.0, 5.0, 15.0, 25.0],
            "note": ["  Alpha ", "beta-1", "GAMMA-22", " delta", "eps-3", "zeta"],
            "cat": ["a", "b", "a", "c", "b", "a"],
        }
    )


@pytest.mark.parametrize(
    "func,expected",
    [
        ("rolling_mean", [10, 15, 30, 5, 10, 20]),
        ("rolling_sum", [10, 30, 60, 5, 20, 40]),
        ("rank", [1, 2, 3, 1, 2, 3]),
        ("lag", [None, 10, 20, None, 5, 15]),
        ("lead", [20, 40, None, 15, 25, None]),
        ("cumsum", [10, 30, 70, 5, 20, 45]),
    ],
)
def test_window_functions_per_group(sales, func, expected):
    out = t.apply_op(
        {"s": sales},
        op(
            "window",
            "w",
            table="s",
            column="amount",
            func=func,
            window=1 if func in ("lag", "lead") else 2,
            by=["region"],
            order_by=["day"],
            name="x",
        ),
    )
    pd.testing.assert_series_equal(
        out["x"].reset_index(drop=True),
        pd.Series(expected, dtype=float, name="x"),
        check_names=False,
    )


def test_window_pct_change_and_errors(sales):
    out = t.apply_op(
        {"s": sales},
        op(
            "window",
            "w",
            table="s",
            column="amount",
            func="pct_change",
            window=1,
            by=["region"],
            name="x",
        ),
    )
    assert out["x"].iloc[1] == pytest.approx(1.0)
    with pytest.raises(t.TransformError):
        t.apply_op(
            {"s": sales},
            op("window", "w", table="s", column="amount", func="median", window=2, name="x"),
        )


@pytest.mark.parametrize(
    "part,first",
    [("year", 2024), ("quarter", 1), ("month", 1), ("day", 5), ("dayofweek", 4), ("week", 1)],
)
def test_date_buckets(sales, part, first):
    out = t.apply_op(
        {"s": sales}, op("datebucket", "d", table="s", column="day", part=part, name="p")
    )
    assert int(out["p"].iloc[0]) == first
    with pytest.raises(t.TransformError):
        t.apply_op(
            {"s": sales.assign(day="nope")},
            op("datebucket", "d", table="s", column="day", part="year", name="p"),
        )


@pytest.mark.parametrize(
    "kind,extra,check",
    [
        ("strip", {}, lambda s: s.iloc[0] == "Alpha"),
        ("lower", {}, lambda s: s.iloc[2] == "gamma-22"),
        ("upper", {}, lambda s: s.iloc[1] == "BETA-1"),
        ("title", {}, lambda s: s.iloc[2] == "Gamma-22"),
        ("replace", {"find": "-", "replace": "_"}, lambda s: s.iloc[1] == "beta_1"),
        (
            "replace",
            {"find": r"\d+", "replace": "#", "regex": True},
            lambda s: s.iloc[2] == "GAMMA-#",
        ),
        ("extract", {"find": r"(\d+)"}, lambda s: s.iloc[1] == "1" and pd.isna(s.iloc[0])),
        ("length", {}, lambda s: s.iloc[5] == 4),
    ],
)
def test_string_cleaning(sales, kind, extra, check):
    out = t.apply_op(
        {"s": sales},
        op("strclean", "c", table="s", column="note", kind=kind, name="clean", **extra),
    )
    assert check(out["clean"])
    assert out["note"].equals(sales["note"].astype("string")) or out["note"].equals(sales["note"])


def test_onehot_and_bin(sales):
    oh = t.apply_op({"s": sales}, op("onehot", "o", table="s", columns=["cat"], drop_first=True))
    assert {"cat_b", "cat_c"} <= set(oh.columns) and "cat" not in oh.columns
    q = t.apply_op(
        {"s": sales},
        op("bin", "b", table="s", column="amount", bins=3, kind="quantile", name="amount_bin"),
    )
    w = t.apply_op(
        {"s": sales},
        op("bin", "b", table="s", column="amount", bins=3, kind="width", name="amount_bin"),
    )
    assert q["amount_bin"].nunique() == 3 and w["amount_bin"].nunique() <= 3


def test_merge_validation_indicator_and_key_diagnostics():
    left = pd.DataFrame({"k": [1, 1, 2, 3], "v": list("abcd")})
    right = pd.DataFrame({"k": [1, 2, 2, 4], "w": list("wxyz")})
    tables = {"l": left, "r": right}
    with pytest.raises(t.TransformError, match="merge failed"):
        t.apply_op(
            tables,
            op("merge", "j", left="l", right="r", on=["k"], how="left", validate="many_to_one"),
        )
    ok = t.apply_op(
        tables,
        op(
            "merge",
            "j",
            left="l",
            right="r",
            on=["k"],
            how="outer",
            validate="many_to_many",
            indicator=True,
        ),
    )
    assert set(ok["_merge"].astype(str)) == {"both", "left_only", "right_only"}
    diag = t.key_diagnostics(left, right, ["k"])
    assert diag == {
        "left_duplicate_keys": 1,
        "right_duplicate_keys": 1,
        "left_unmatched_keys": 1,
        "right_unmatched_keys": 1,
    }


def test_pivot_margins_and_fill(sales):
    out = t.apply_op(
        {"s": sales},
        op(
            "pivot",
            "p",
            table="s",
            index=["region"],
            columns="cat",
            values="amount",
            aggfunc="sum",
            fill_value=0,
            margins=True,
        ),
    )
    assert "All" in out["region"].tolist() and not out.isna().any().any()


# ------------------------------------------------------------------ code styles reproduce results
STYLE_OPS = [
    op(
        "merge",
        "joined",
        left="active",
        right="regions",
        on=["region"],
        how="left",
        validate="many_to_one",
    ),
    op("datebucket", "dated", table="joined", column="day", part="month", name="month"),
    op(
        "window",
        "win",
        table="dated",
        column="amount",
        func="rolling_mean",
        window=2,
        by=["region"],
        order_by=["day"],
        name="amt_ma",
    ),
    op("strclean", "clean", table="win", column="note", kind="upper", name="note_up"),
    op("bin", "binned", table="clean", column="amount", bins=2, kind="width", name="size"),
    op("filter", "final", table="binned", expr="amount > 5"),
]


@pytest.mark.parametrize("style", t.CODE_STYLES)
def test_every_code_style_reproduces_the_live_result(style, sales, tmp_path, monkeypatch):
    regions = pd.DataFrame({"region": ["N", "S"], "zone": ["north", "south"]})
    (tmp_path / "sales.csv").write_text(sales.to_csv(index=False))
    (tmp_path / "regions.csv").write_text(regions.to_csv(index=False))
    monkeypatch.chdir(tmp_path)
    live = {"active": sales, "regions": regions}
    for o in STYLE_OPS:
        live[o.output] = t.apply_op(live, o)
    code = t.ops_to_code(
        STYLE_OPS, external=["regions"], style=style, filename="sales.csv", options=None
    )
    if style == "steps":
        code = code.replace("# t_regions = ...", "t_regions = pd.read_csv('regions.csv')  #")
    ns: dict = {"__name__": "snippet"}
    exec(compile(code, "<ops>", "exec"), ns)  # noqa: S102
    final = {
        "steps": lambda: ns["t_final"],
        "chain": lambda: ns["result"],
        "function": lambda: ns["result"],
    }[style]()
    pd.testing.assert_frame_equal(
        final.reset_index(drop=True), live["final"].reset_index(drop=True), check_dtype=False
    )
    if style == "function":
        ns["test_transform"]()
        assert "def transform(tables)" in code


def test_chain_style_falls_back_for_branching_logs(sales):
    ops = [
        op("filter", "a", table="active", expr="amount > 5"),
        op("filter", "b", table="active", expr="amount > 10"),
    ]
    assert t.is_linear_chain(ops[:1]) and not t.is_linear_chain(ops)
    code = t.ops_to_code(ops, style="chain", filename="x.csv")
    assert "branches" in code and ".pipe(" not in code
    assert "(no steps yet)" in t.ops_to_code([], style="function")


def test_merge_code_includes_checks_only_when_enabled(sales):
    from src.core.code_options import CodeOptions

    plain = t.ops_to_code(STYLE_OPS[:1], external=["regions"], filename="s.csv")
    checked = t.ops_to_code(
        STYLE_OPS[:1], external=["regions"], filename="s.csv", options=CodeOptions(checks=True)
    )
    assert "must keep every left row" not in plain and "must keep every left row" in checked
