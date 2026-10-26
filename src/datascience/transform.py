"""Multi-table transformation operations (joins, pivots, derived columns) with replayable code."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field

from src.core.code_snippets import make_snippet

TableOpName = Literal[
    "merge",
    "concat",
    "pivot",
    "melt",
    "groupby",
    "derive",
    "convert",
    "rename",
    "filter",
    "window",
    "datebucket",
    "strclean",
    "onehot",
    "bin",
]

OP_LABELS = {
    "merge": "Join / merge two tables",
    "concat": "Stack tables (union)",
    "pivot": "Pivot (long to wide)",
    "melt": "Unpivot (wide to long)",
    "groupby": "Group & aggregate",
    "derive": "Derived column (expression)",
    "convert": "Convert column type",
    "rename": "Rename columns",
    "filter": "Filter rows (expression)",
    "window": "Window function (rolling, rank, lag, cumulative)",
    "datebucket": "Date bucket (year, quarter, month, weekday...)",
    "strclean": "Clean text (trim, case, replace, extract)",
    "onehot": "One-hot encode categories",
    "bin": "Bin a numeric column",
}
WINDOW_FUNCS = ("rolling_mean", "rolling_sum", "rank", "lag", "lead", "cumsum", "pct_change")
DATE_BUCKETS = ("year", "quarter", "month", "week", "day", "dayofweek", "hour")
STR_OPS = ("strip", "lower", "upper", "title", "replace", "extract", "length")
JOIN_VALIDATIONS = (None, "one_to_one", "one_to_many", "many_to_one", "many_to_many")
AGG_FUNCS = ("sum", "mean", "median", "min", "max", "count", "nunique", "std")
JOIN_HOWS = ("inner", "left", "right", "outer")
CONVERT_TARGETS = ("numeric", "datetime", "category", "string", "integer", "float")

_FORBIDDEN_EXPR = re.compile(r"__|@|\bimport\b|\blambda\b|\bexec\b|\beval\b|\bopen\b|[;`]", re.I)


class TransformError(ValueError):
    """Invalid or failing table operation (message is user-facing)."""


class TableOp(BaseModel):
    op: TableOpName
    output: str
    params: dict[str, Any] = Field(default_factory=dict)

    @property
    def inputs(self) -> list[str]:
        p = self.params
        if self.op == "merge":
            return [p.get("left", ""), p.get("right", "")]
        if self.op == "concat":
            return list(p.get("tables", []))
        return [p.get("table", "")]

    @property
    def label(self) -> str:
        return f"{self.op}: {', '.join(self.inputs)} -> {self.output}"


def slug_name(name: str) -> str:
    out = re.sub(r"\W+", "_", name.strip()).strip("_").lower()
    if not out:
        raise TransformError("Table names need letters or digits.")
    return out if not out[0].isdigit() else f"t_{out}"


def validate_expression(expr: str) -> str:
    expr = expr.strip()
    if not expr:
        raise TransformError("Enter an expression.")
    if len(expr) > 500 or _FORBIDDEN_EXPR.search(expr):
        raise TransformError(
            "That expression is not allowed (no attribute tricks, imports, or '@')."
        )
    return expr


def _table(tables: dict[str, pd.DataFrame], name: str) -> pd.DataFrame:
    if name not in tables:
        raise TransformError(f"Table '{name}' not found.")
    return tables[name]


def apply_op(tables: dict[str, pd.DataFrame], op: TableOp) -> pd.DataFrame:
    """Run one operation against `tables` and return the result (inputs are not modified)."""
    p = op.params
    try:
        if op.op == "merge":
            left, right = _table(tables, p["left"]), _table(tables, p["right"])
            on = p.get("on")
            kwargs: dict[str, Any] = (
                {"on": on} if on else {"left_on": p["left_on"], "right_on": p["right_on"]}
            )
            return left.merge(
                right,
                how=p.get("how", "inner"),
                suffixes=("", f"_{slug_name(p['right'])}"),
                validate=p.get("validate"),
                indicator=bool(p.get("indicator", False)),
                **kwargs,
            )
        if op.op == "concat":
            frames = [_table(tables, t) for t in p["tables"]]
            if len(frames) < 2:
                raise TransformError("Pick at least two tables to stack.")
            return pd.concat(frames, ignore_index=True, sort=False)
        df = _table(tables, p["table"]).copy()
        if op.op == "pivot":
            out = df.pivot_table(
                index=p["index"],
                columns=p["columns"],
                values=p["values"],
                aggfunc=p.get("aggfunc", "sum"),
                fill_value=p.get("fill_value"),
                margins=bool(p.get("margins", False)),
            )
            out.columns = [
                str(c) if not isinstance(c, tuple) else "_".join(map(str, c)) for c in out.columns
            ]
            return out.reset_index()
        if op.op == "melt":
            return df.melt(
                id_vars=p.get("id_vars") or None,
                value_vars=p.get("value_vars") or None,
                var_name=p.get("var_name", "variable"),
                value_name=p.get("value_name", "value"),
            )
        if op.op == "groupby":
            aggs = dict(p["aggs"])
            if not aggs:
                raise TransformError("Choose at least one aggregation.")
            out = df.groupby(p["by"], dropna=False).agg(aggs)
            out.columns = (
                [f"{c}_{f}" for c, f in out.columns]
                if isinstance(out.columns, pd.MultiIndex)
                else list(out.columns)
            )
            return out.reset_index()
        if op.op == "derive":
            expr = validate_expression(p["expr"])
            df[p["name"]] = df.eval(expr)
            return df
        if op.op == "convert":
            col, target = p["column"], p["to"]
            if target == "numeric":
                df[col] = pd.to_numeric(df[col], errors="coerce")
            elif target == "datetime":
                df[col] = pd.to_datetime(df[col], errors="coerce")
            elif target == "category":
                df[col] = df[col].astype("category")
            elif target == "string":
                df[col] = df[col].astype("string")
            elif target == "integer":
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
            elif target == "float":
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
            else:
                raise TransformError(f"Unknown target type '{target}'.")
            return df
        if op.op == "window":
            col, func, w = p["column"], p["func"], int(p.get("window", 3))
            by = p.get("by") or []
            ordered = df.sort_values(p["order_by"], kind="stable") if p.get("order_by") else df
            g = ordered.groupby(by)[col] if by else ordered[col]
            if func == "rolling_mean":
                res = g.transform(lambda s: s.rolling(w, min_periods=1).mean())
            elif func == "rolling_sum":
                res = g.transform(lambda s: s.rolling(w, min_periods=1).sum())
            elif func == "rank":
                res = g.rank(method="average")
            elif func == "lag":
                res = g.shift(w)
            elif func == "lead":
                res = g.shift(-w)
            elif func == "cumsum":
                res = g.cumsum()
            elif func == "pct_change":
                res = g.pct_change()
            else:
                raise TransformError(f"Unknown window function '{func}'.")
            df[p["name"]] = res
            return df
        if op.op == "datebucket":
            col, part = p["column"], p["part"]
            dt = pd.to_datetime(df[col], errors="coerce")
            if dt.notna().sum() == 0:
                raise TransformError(f"'{col}' could not be parsed as dates.")
            if part == "week":
                df[p["name"]] = dt.dt.isocalendar().week.astype("Int64")
            elif part in DATE_BUCKETS:
                df[p["name"]] = getattr(dt.dt, part)
            else:
                raise TransformError(f"Unknown date part '{part}'.")
            return df
        if op.op == "strclean":
            col, kind = p["column"], p["kind"]
            text = df[col].astype("string")
            target = p.get("name") or col
            if kind == "strip":
                df[target] = text.str.strip()
            elif kind in ("lower", "upper", "title"):
                df[target] = getattr(text.str, kind)()
            elif kind == "replace":
                df[target] = text.str.replace(
                    p["find"], p.get("replace", ""), regex=bool(p.get("regex", False))
                )
            elif kind == "extract":
                df[target] = text.str.extract(p["find"], expand=False)
            elif kind == "length":
                df[target] = text.str.len()
            else:
                raise TransformError(f"Unknown text operation '{kind}'.")
            return df
        if op.op == "onehot":
            return pd.get_dummies(
                df, columns=p["columns"], drop_first=bool(p.get("drop_first", True)), dtype=int
            )
        if op.op == "bin":
            col, n = p["column"], int(p.get("bins", 4))
            num = pd.to_numeric(df[col], errors="coerce")
            name = p.get("name") or f"{col}_bin"
            df[name] = (
                pd.qcut(num, q=n, duplicates="drop")
                if p.get("kind", "quantile") == "quantile"
                else pd.cut(num, bins=n)
            )
            df[name] = df[name].astype(str)
            return df
        if op.op == "rename":
            return df.rename(columns=p["mapping"])
        if op.op == "filter":
            return df.query(validate_expression(p["expr"]), engine="python")
    except TransformError:
        raise
    except KeyError as err:
        raise TransformError(f"Missing column or setting: {err}") from err
    except Exception as err:
        raise TransformError(f"{op.op} failed: {err}") from err
    raise TransformError(f"Unsupported operation '{op.op}'.")


def validate_output_name(tables: dict[str, pd.DataFrame], name: str) -> str:
    slug = slug_name(name)
    if slug in tables:
        raise TransformError(f"A table named '{slug}' already exists - choose a new name.")
    return slug


def removable_outputs(ops: list[TableOp], keep: int) -> list[str]:
    """Output tables produced by steps after the first `keep` (to delete on rollback)."""
    return [o.output for o in ops[keep:]]


# --------------------------------------------------------------------------- code
def key_diagnostics(left: pd.DataFrame, right: pd.DataFrame, keys: list[str]) -> dict[str, Any]:
    """Facts to check before a join: duplicate keys and unmatched rows."""
    lk, rk = left[keys].drop_duplicates(), right[keys].drop_duplicates()
    matched = lk.merge(rk, on=keys, how="inner")
    return {
        "left_duplicate_keys": int(left.duplicated(keys).sum()),
        "right_duplicate_keys": int(right.duplicated(keys).sum()),
        "left_unmatched_keys": int(len(lk) - len(matched)),
        "right_unmatched_keys": int(len(rk) - len(matched)),
    }


def _primary_input(op: TableOp) -> str:
    return op.inputs[0]


def _op_lines(op: TableOp, tv: Callable[[str], str], out: str) -> list[str]:
    """Statements that compute table `out` from the tables named by `tv`."""
    p = op.params
    lines: list[str] = []
    if op.op == "merge":
        on = (
            f"on={p['on']!r}"
            if p.get("on")
            else f"left_on={p['left_on']!r}, right_on={p['right_on']!r}"
        )
        extra = ""
        if p.get("validate"):
            extra += f", validate={p['validate']!r}"
        if p.get("indicator"):
            extra += ", indicator=True"
        keys = p.get("on") or [p.get("left_on")]
        lines += [
            "#~ Before joining, count duplicated keys: duplicates multiply rows.",
            f"print('duplicate keys  left:', {tv(p['left'])}.duplicated({keys!r}).sum(), ' right:', {tv(p['right'])}.duplicated({p.get('on') or [p.get('right_on')]!r}).sum())",
            f"{out} = {tv(p['left'])}.merge({tv(p['right'])}, how={p.get('how', 'inner')!r}, {on}, suffixes=('', '_{slug_name(p['right'])}'){extra})",
        ]
        if p.get("how") == "left" and p.get("validate") in ("one_to_one", "many_to_one"):
            lines.append(
                f"#? assert len({out}) == len({tv(p['left'])}), 'a left join with unique right keys must keep every left row once'"
            )
        elif p.get("how") == "left":
            lines.append(
                f"#? assert len({out}) >= len({tv(p['left'])}), 'a left join cannot drop left rows'"
            )
        else:
            lines.append(f"#? assert len({out}) > 0, 'the join produced no rows'")
    elif op.op == "concat":
        lines.append(
            f"{out} = pd.concat([{', '.join(tv(t) for t in p['tables'])}], ignore_index=True, sort=False)"
        )
        lines.append(
            f"#? assert len({out}) == sum(len(t) for t in [{', '.join(tv(t) for t in p['tables'])}])"
        )
    elif op.op == "pivot":
        extra = ""
        if p.get("fill_value") is not None:
            extra += f", fill_value={p['fill_value']!r}"
        if p.get("margins"):
            extra += ", margins=True"
        lines += [
            f"{out} = {tv(p['table'])}.pivot_table(index={p['index']!r}, columns={p['columns']!r}, values={p['values']!r}, aggfunc={p.get('aggfunc', 'sum')!r}{extra})",
            f"{out}.columns = [c if not isinstance(c, tuple) else '_'.join(map(str, c)) for c in {out}.columns]",
            f"{out} = {out}.reset_index()",
        ]
    elif op.op == "melt":
        lines.append(
            f"{out} = {tv(p['table'])}.melt(id_vars={(p.get('id_vars') or None)!r}, value_vars={(p.get('value_vars') or None)!r}, var_name={p.get('var_name', 'variable')!r}, value_name={p.get('value_name', 'value')!r})"
        )
    elif op.op == "groupby":
        lines += [
            f"{out} = {tv(p['table'])}.groupby({p['by']!r}, dropna=False).agg({p['aggs']!r})",
            f"{out}.columns = [f'{{c}}_{{f}}' for c, f in {out}.columns] if isinstance({out}.columns, pd.MultiIndex) else list({out}.columns)",
            f"{out} = {out}.reset_index()",
        ]
    elif op.op == "derive":
        lines += [
            f"{out} = {tv(p['table'])}.copy()",
            f"{out}[{p['name']!r}] = {out}.eval({p['expr']!r})",
        ]
    elif op.op == "convert":
        conv = {
            "numeric": "pd.to_numeric({c}, errors='coerce')",
            "datetime": "pd.to_datetime({c}, errors='coerce')",
            "category": "{c}.astype('category')",
            "string": "{c}.astype('string')",
            "integer": "pd.to_numeric({c}, errors='coerce').astype('Int64')",
            "float": "pd.to_numeric({c}, errors='coerce').astype(float)",
        }[p["to"]]
        lines += [
            f"{out} = {tv(p['table'])}.copy()",
            f"{out}[{p['column']!r}] = " + conv.format(c=f"{out}[{p['column']!r}]"),
        ]
    elif op.op == "rename":
        lines.append(f"{out} = {tv(p['table'])}.rename(columns={p['mapping']!r})")
    elif op.op == "filter":
        lines.append(f"{out} = {tv(p['table'])}.query({p['expr']!r}, engine='python')")
    elif op.op == "window":
        col, func, w, by = p["column"], p["func"], int(p.get("window", 3)), p.get("by") or []
        lines.append(f"{out} = {tv(p['table'])}.copy()")
        src_frame = out
        if p.get("order_by"):
            lines.append(
                f"_ordered = {out}.sort_values({p['order_by']!r}, kind='stable')   # compute in this order; rows keep their position"
            )
            src_frame = "_ordered"
        grp = f"{src_frame}.groupby({by!r})[{col!r}]" if by else f"{src_frame}[{col!r}]"
        expr = {
            "rolling_mean": f"{grp}.transform(lambda s: s.rolling({w}, min_periods=1).mean())",
            "rolling_sum": f"{grp}.transform(lambda s: s.rolling({w}, min_periods=1).sum())",
            "rank": f"{grp}.rank(method='average')",
            "lag": f"{grp}.shift({w})",
            "lead": f"{grp}.shift(-{w})",
            "cumsum": f"{grp}.cumsum()",
            "pct_change": f"{grp}.pct_change()",
        }[func]
        lines.append(
            f"#~ Window functions look at neighbouring rows ({func}); sort first if order matters."
        )
        lines.append(f"{out}[{p['name']!r}] = {expr}")
    elif op.op == "datebucket":
        part = p["part"]
        lines += [
            f"{out} = {tv(p['table'])}.copy()",
            f"_dt = pd.to_datetime({out}[{p['column']!r}], errors='coerce')",
            f"{out}[{p['name']!r}] = "
            + ("_dt.dt.isocalendar().week.astype('Int64')" if part == "week" else f"_dt.dt.{part}"),
        ]
    elif op.op == "strclean":
        col, kind = p["column"], p["kind"]
        target = p.get("name") or col
        text = f"{out}[{col!r}].astype('string')"
        expr = {
            "strip": f"{text}.str.strip()",
            "lower": f"{text}.str.lower()",
            "upper": f"{text}.str.upper()",
            "title": f"{text}.str.title()",
            "replace": f"{text}.str.replace({p.get('find')!r}, {p.get('replace', '')!r}, regex={bool(p.get('regex', False))})",
            "extract": f"{text}.str.extract({p.get('find')!r}, expand=False)",
            "length": f"{text}.str.len()",
        }[kind]
        lines += [f"{out} = {tv(p['table'])}.copy()", f"{out}[{target!r}] = {expr}"]
    elif op.op == "onehot":
        lines.append(
            f"{out} = pd.get_dummies({tv(p['table'])}, columns={p['columns']!r}, drop_first={bool(p.get('drop_first', True))}, dtype=int)"
        )
    elif op.op == "bin":
        col, n = p["column"], int(p.get("bins", 4))
        name = p.get("name") or f"{col}_bin"
        fn = "pd.qcut" if p.get("kind", "quantile") == "quantile" else "pd.cut"
        extra = ", duplicates='drop'" if fn == "pd.qcut" else ""
        lines += [
            f"{out} = {tv(p['table'])}.copy()",
            f"{out}[{name!r}] = {fn}(pd.to_numeric({out}[{col!r}], errors='coerce'), {'q' if fn == 'pd.qcut' else 'bins'}={n}{extra}).astype(str)",
        ]
    return lines


def is_linear_chain(ops: list[TableOp]) -> bool:
    """True when every step consumes the previous step's output (so it can be a .pipe chain)."""
    return bool(ops) and all(
        _primary_input(ops[i]) == ops[i - 1].output for i in range(1, len(ops))
    )


CODE_STYLES = ("steps", "chain", "function")


def ops_to_code(
    ops: list[TableOp],
    external: list[str] | None = None,
    style: str = "steps",
    **load: Any,
) -> str:
    """Pandas code for the operation log.

    style: ``steps`` (one variable per table), ``chain`` (``.pipe`` pipeline; needs a linear log)
    or ``function`` (a reusable ``transform(tables)`` plus a test).
    """
    if style == "chain" and not is_linear_chain(ops):
        style = "steps"
        note = "# (this log branches, so it is shown step by step instead of as one chain)\n"
    else:
        note = ""
    ext = external or []
    if not ops:
        return make_snippet("Table transformations", "# (no steps yet)", **load)

    if style == "function":
        lines = [
            "def transform(tables):",
            '    """Apply every step. `tables` maps table name -> DataFrame; returns the final table."""',
            "    t = dict(tables)",
        ]
        for n, op in enumerate(ops, start=1):
            lines.append(f"    # Step {n}: {OP_LABELS[op.op]}")
            lines += [
                "    " + ln
                for ln in _op_lines(op, lambda name: f"t[{name!r}]", f"t[{op.output!r}]")
            ]
        lines += [f"    return t[{ops[-1].output!r}]", "", ""]
        ext_dict = "".join(f", {n!r}: pd.read_csv('{n}.csv')" for n in ext)
        lines += [
            "def test_transform():",
            f"    result = transform({{'active': df{ext_dict}}})",
            "    assert len(result) > 0",
            "    assert result.columns.is_unique",
            "",
            "",
            f"result = transform({{'active': df{ext_dict}}})",
            "result.head()",
        ]
        return make_snippet("Reusable transform function", "\n".join(lines), **load)

    if style == "chain":
        tables_init = (
            "tables = {'active': df.copy()"
            + "".join(f", {n!r}: pd.read_csv('{n}.csv')" for n in ext)
            + "}"
        )
        lines = [note + tables_init, ""]
        prev = "active"
        for n, op in enumerate(ops, start=1):

            def tv(name: str, prev_name: str = prev) -> str:
                return "d" if name == prev_name else f"tables[{name!r}]"

            lines.append(f"def step_{n}(d):")
            lines.append(f"    # {OP_LABELS[op.op]}")
            lines += ["    " + ln for ln in _op_lines(op, tv, "out")]
            lines += ["    return out", ""]
            prev = op.output
        chain = "tables['active']" + "".join(f".pipe(step_{n})" for n in range(1, len(ops) + 1))
        lines += [f"result = ({chain})", "print(result.shape)", "result.head()"]
        return make_snippet("Transformation pipeline", "\n".join(lines), **load)

    lines = ["t_active = df.copy()   # the active dataset"]
    for name in ext:
        lines.append(
            f"# t_{slug_name(name)} = ...   # load table '{name}' from its original source"
        )
    lines.append("")
    for n, op in enumerate(ops, start=1):
        lines.append(f"# Step {n}: {OP_LABELS[op.op]}")
        lines += _op_lines(op, lambda name: f"t_{slug_name(name)}", f"t_{slug_name(op.output)}")
        lines.append(f"print({op.output!r}, t_{slug_name(op.output)}.shape)")
        lines.append("")
    lines.append(f"t_{slug_name(ops[-1].output)}.head()")
    return make_snippet("Table transformations", note + "\n".join(lines), **load)
