"""Page 11: Transformation Studio - joins, pivots, derived columns, with preview and rollback."""

import streamlit as st
from src.config.settings import get_settings
from src.core.code_snippets import load_kwargs
from src.core.exceptions import DataAnalystError
from src.core.library import ACTIVE, add_table, get_tables, remove_table
from src.core.session_state import KEY_TRANSFORM_OPS
from src.core.ui_components import (
    esc,
    page_setup,
    render_code_button,
    render_code_export,
    require_dataset,
)
from src.datascience import transform as tf
from src.ingestion.service import ingest_dataframe

page_setup(
    "Transformation Studio",
    "🧱",
    "Combine and reshape tables. Every step is previewed, logged, and reversible.",
)
settings = get_settings()
dataset = require_dataset("Transformation Studio")
tables = get_tables(st.session_state)
ops: list[tf.TableOp] = st.session_state.get(KEY_TRANSFORM_OPS) or []
names = list(tables)

st.caption(
    "Tables: "
    + " · ".join(f"`{n}` ({len(t):,}×{len(t.columns)})" for n, t in tables.items())
    + ".  Add side tables on the **Data Sources** page."
)

op_name = st.selectbox(
    "Operation", list(tf.OP_LABELS), format_func=tf.OP_LABELS.get, key="studio_op"
)


def cols_of(table: str) -> list[str]:
    return [str(c) for c in tables[table].columns]


params: dict = {}
ready = True
if op_name == "merge":
    c1, c2, c3 = st.columns(3)
    left = c1.selectbox("Left table", names, key="m_left")
    right = c2.selectbox("Right table", [n for n in names if n != left] or names, key="m_right")
    how = c3.selectbox("Join type", tf.JOIN_HOWS, key="m_how")
    shared = [c for c in cols_of(left) if c in cols_of(right)]
    same = st.checkbox("Join on same-named columns", value=bool(shared), key="m_same")
    if same:
        on = st.multiselect("Key column(s)", shared, default=shared[:1], key="m_on")
        v1, v2 = st.columns(2)
        validate = v1.selectbox(
            "Expect key relationship",
            tf.JOIN_VALIDATIONS,
            format_func=lambda v: v or "(don't check)",
            key="m_val",
            help="Pandas raises an error if the keys do not match this relationship.",
        )
        indicator = v2.checkbox(
            "Add a '_merge' column showing where each row came from", key="m_ind"
        )
        params = {
            "left": left,
            "right": right,
            "how": how,
            "on": on,
            "validate": validate,
            "indicator": indicator,
        }
        ready = bool(on)
        if on:
            diag = tf.key_diagnostics(tables[left], tables[right], on)
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Duplicate keys (left)", f"{diag['left_duplicate_keys']:,}")
            d2.metric("Duplicate keys (right)", f"{diag['right_duplicate_keys']:,}")
            d3.metric("Unmatched keys (left)", f"{diag['left_unmatched_keys']:,}")
            d4.metric("Unmatched keys (right)", f"{diag['right_unmatched_keys']:,}")
            if diag["right_duplicate_keys"]:
                st.warning(
                    "The right table has duplicate keys: a join will multiply the matching left rows."
                )
    else:
        k1, k2 = st.columns(2)
        lo = k1.selectbox("Left key", cols_of(left), key="m_lk")
        ro = k2.selectbox("Right key", cols_of(right), key="m_rk")
        params = {"left": left, "right": right, "how": how, "left_on": lo, "right_on": ro}
elif op_name == "concat":
    sel = st.multiselect("Tables to stack (same columns)", names, default=names[:2], key="c_tables")
    params, ready = {"tables": sel}, len(sel) >= 2
else:
    table = st.selectbox("Table", names, key="t_table")
    cols = cols_of(table)
    params = {"table": table}
    if op_name == "pivot":
        a, b, c = st.columns(3)
        idx = a.multiselect("Rows (index)", cols, key="p_idx")
        col = b.selectbox("Columns", cols, key="p_col")
        val = c.selectbox("Values", cols, key="p_val")
        agg = st.selectbox("Aggregate", tf.AGG_FUNCS, key="p_agg")
        pf1, pf2 = st.columns(2)
        fill = pf1.text_input("Fill empty cells with (optional)", key="p_fill")
        margins = pf2.checkbox("Add row/column totals", key="p_margins")
        params |= {"index": idx, "columns": col, "values": val, "aggfunc": agg, "margins": margins}
        if fill.strip():
            try:
                params["fill_value"] = float(fill)
            except ValueError:
                params["fill_value"] = fill
        ready = bool(idx)
    elif op_name == "melt":
        ids = st.multiselect("Keep as identifiers", cols, key="me_ids")
        vals = st.multiselect(
            "Columns to unpivot", [c for c in cols if c not in ids], key="me_vals"
        )
        params |= {"id_vars": ids, "value_vars": vals}
        ready = bool(vals)
    elif op_name == "groupby":
        by = st.multiselect("Group by", cols, key="g_by")
        agg_cols = st.multiselect(
            "Aggregate columns", [c for c in cols if c not in by], key="g_cols"
        )
        func = st.selectbox("Function", tf.AGG_FUNCS, key="g_func")
        params |= {"by": by, "aggs": dict.fromkeys(agg_cols, func)}
        ready = bool(by and agg_cols)
    elif op_name == "derive":
        n1, n2 = st.columns([1, 3])
        new_col = n1.text_input("New column name", key="d_name")
        expr = n2.text_input("Expression", placeholder="price * quantity", key="d_expr")
        st.caption(
            "Use column names and + - * / ** comparisons. Wrap odd names in backticks: `unit price`."
        )
        params |= {"name": new_col, "expr": expr}
        ready = bool(new_col.strip() and expr.strip())
    elif op_name == "convert":
        a, b = st.columns(2)
        col = a.selectbox("Column", cols, key="cv_col")
        target = b.selectbox("Convert to", tf.CONVERT_TARGETS, key="cv_to")
        params |= {"column": col, "to": target}
    elif op_name == "rename":
        pick = st.multiselect("Columns to rename", cols, key="rn_cols")
        mapping = {c: st.text_input(f"New name for {c}", value=c, key=f"rn_{c}") for c in pick}
        params |= {"mapping": {k: v for k, v in mapping.items() if v and v != k}}
        ready = bool(params["mapping"])
    elif op_name == "window":
        a, b, c = st.columns(3)
        col = a.selectbox("Column", cols, key="w_col")
        func = b.selectbox("Function", tf.WINDOW_FUNCS, key="w_func")
        win = c.number_input("Window / offset", 1, 365, 3, key="w_win")
        by = st.multiselect("Per group (optional)", cols, key="w_by")
        order = st.multiselect("Sort by first (optional)", cols, key="w_order")
        name = st.text_input("New column name", value=f"{col}_{func}", key="w_name")
        params |= {
            "column": col,
            "func": func,
            "window": int(win),
            "by": by,
            "order_by": order,
            "name": name,
        }
        ready = bool(name.strip())
    elif op_name == "datebucket":
        a, b = st.columns(2)
        col = a.selectbox("Date column", cols, key="db_col")
        part = b.selectbox("Bucket", tf.DATE_BUCKETS, key="db_part")
        name = st.text_input("New column name", value=f"{col}_{part}", key="db_name")
        params |= {"column": col, "part": part, "name": name}
        ready = bool(name.strip())
    elif op_name == "strclean":
        a, b = st.columns(2)
        col = a.selectbox("Text column", cols, key="sc_col")
        kind = b.selectbox("Operation", tf.STR_OPS, key="sc_kind")
        extra = {}
        if kind in ("replace", "extract"):
            x1, x2, x3 = st.columns(3)
            extra["find"] = x1.text_input("Find (text or regex)", key="sc_find")
            if kind == "replace":
                extra["replace"] = x2.text_input("Replace with", key="sc_rep")
                extra["regex"] = x3.checkbox("Regex", key="sc_regex")
            ready = bool(extra["find"])
        target = st.text_input("Write result to column (blank = overwrite)", key="sc_target")
        params |= {"column": col, "kind": kind, "name": target.strip() or None, **extra}
    elif op_name == "onehot":
        cat = st.multiselect("Categorical columns", cols, key="oh_cols")
        params |= {
            "columns": cat,
            "drop_first": st.checkbox("Drop first level", value=True, key="oh_drop"),
        }
        ready = bool(cat)
    elif op_name == "bin":
        a, b, c = st.columns(3)
        col = a.selectbox("Numeric column", cols, key="bn_col")
        kind = b.radio("Bins", ["quantile", "width"], horizontal=True, key="bn_kind")
        n = c.slider("Number of bins", 2, 20, 4, key="bn_n")
        params |= {"column": col, "kind": kind, "bins": n, "name": f"{col}_bin"}
    elif op_name == "filter":
        expr = st.text_input(
            "Keep rows where", placeholder="amount > 100 and region == 'N'", key="f_expr"
        )
        params |= {"expr": expr}
        ready = bool(expr.strip())

default_out = f"{op_name}_{len(ops) + 1}"
out_name = st.text_input(
    "Name of the resulting table", value=default_out, key=f"out_{op_name}_{len(ops)}"
)

preview_col, apply_col = st.columns([1, 1])
op = None
if ready:
    try:
        out_slug = tf.validate_output_name(tables, out_name)
        op = tf.TableOp(op=op_name, output=out_slug, params=params)
    except tf.TransformError as err:
        st.error(str(err))

if op is not None:
    try:
        preview = tf.apply_op(tables, op)
        st.markdown(f"**Preview** · {len(preview):,} rows × {len(preview.columns)} columns")
        st.dataframe(preview.head(10), use_container_width=True)
    except tf.TransformError as err:
        st.error(str(err))
        preview = None
    if preview is not None and apply_col.button("✅ Apply and save as a new table", type="primary"):
        add_table(st.session_state, op.output, preview)
        st.session_state[KEY_TRANSFORM_OPS] = [*ops, op]
        st.rerun()

# ------------------------------------------------------------------ timeline
st.markdown("---")
st.subheader("Timeline")
if not ops:
    st.caption(
        "No steps yet. Results are stored as new tables; your original data is never modified."
    )
for n, step in enumerate(ops, start=1):
    with st.container(border=True):
        t1, t2 = st.columns([5, 1])
        result = tables.get(step.output)
        shape = f"{len(result):,}×{len(result.columns)}" if result is not None else "removed"
        t1.markdown(
            f"**Step {n}** · {esc(tf.OP_LABELS[step.op])}  \n`{esc(step.label)}` → {shape}",
            unsafe_allow_html=True,
        )
        if n < len(ops) and t2.button("⏪ Roll back here", key=f"rb_{n}"):
            for gone in tf.removable_outputs(ops, n):
                remove_table(st.session_state, gone)
            st.session_state[KEY_TRANSFORM_OPS] = ops[:n]
            st.rerun()
if ops:
    b1, b2, b3 = st.columns(3)
    if b1.button("↩️ Undo last step"):
        remove_table(st.session_state, ops[-1].output)
        st.session_state[KEY_TRANSFORM_OPS] = ops[:-1]
        st.rerun()
    final = tables.get(ops[-1].output)
    if final is not None and b2.button(f"Use `{ops[-1].output}` as the active dataset"):
        try:
            ingest_dataframe(st.session_state, final, ops[-1].output, settings.max_upload_size_mb)
            st.switch_page("pages/1_Executive_Overview.py")
        except DataAnalystError as err:
            st.error(err.user_message)
    with b3:
        style = st.radio(
            "Code style",
            tf.CODE_STYLES,
            horizontal=True,
            key="studio_code_style",
            label_visibility="collapsed",
            help="steps: one variable per table. chain: a .pipe() pipeline (linear logs only). function: a reusable transform(tables) with a test.",
        )
        _external = [n for n in tables if n not in {o.output for o in ops} and n != ACTIVE]
        render_code_button(
            "Table transformations",
            lambda o: tf.ops_to_code(
                ops, external=_external, style=style, options=o, **load_kwargs(dataset)
            ),
            key="code_studio",
        )

render_code_export()
