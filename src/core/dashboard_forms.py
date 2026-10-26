"""Streamlit forms for creating and editing dashboard tiles (shared by 'Add' and 'Edit')."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from src.datascience import dashboard as db

NONE = "(none)"
_FREQS = {"": "(raw)", "D": "Day", "W": "Week", "MS": "Month", "QS": "Quarter", "YS": "Year"}
_OPS = ["<", "<=", ">", ">=", "=="]
_COLORS = ["red", "orange", "yellow", "green", "blue"]


def _idx(options: list[Any], value: Any) -> int:
    return options.index(value) if value in options else 0


def col_pick(
    label: str,
    options: list[str],
    current: Any,
    key: str,
    params: list[str],
    optional: bool = False,
) -> Any:
    """Select a column (optionally '(none)'), or '↳ parameter' for a choice parameter."""
    opts = ([NONE] if optional else []) + list(options) + [f"↳ {p}" for p in params]
    if isinstance(current, dict) and "$param" in current:
        cur: Any = f"↳ {current['$param']}"
    else:
        cur = current if current else (NONE if optional else None)
    value = st.selectbox(label, opts, index=_idx(opts, cur), key=key)
    if value == NONE:
        return None
    return {"$param": value[2:]} if value.startswith("↳ ") else value


def num_pick(label: str, current: Any, key: str, params: list[str], **kw: Any) -> Any:
    """A number input that may instead be driven by a number parameter."""
    if params:
        opts = ["(fixed)", *params]
        cur = current["$param"] if isinstance(current, dict) else "(fixed)"
        source = st.selectbox(f"{label}: source", opts, index=_idx(opts, cur), key=f"{key}_src")
        if source != "(fixed)":
            return {"$param": source}
    fixed = (
        current
        if isinstance(current, (int, float)) and not isinstance(current, bool)
        else kw.pop("value", 0)
    )
    kw.pop("value", None)
    return st.number_input(label, value=fixed, key=key, **kw)


def tile_form(
    kind: str,
    tables: dict[str, pd.DataFrame],
    spec: db.DashboardSpec,
    key: str,
    tile: db.Tile | None = None,
    source: str | None = None,
) -> dict[str, Any]:
    """Render inputs for a tile of `kind`; returns the pieces needed to build/update a Tile."""
    p0 = tile.params if tile else {}
    o0 = tile.options if tile else {}
    k = lambda s: f"{key}_{s}"  # noqa: E731
    names = list(tables)
    src = (
        st.selectbox(
            "Data table",
            names,
            index=_idx(names, source or (tile.source if tile else "active")),
            key=k("src"),
        )
        if len(names) > 1
        else names[0]
    )
    base, _ = db.apply_calculated(tables[src], spec.calculated)
    all_cols = [str(c) for c in base.columns]
    num_cols = [c for c in all_cols if pd.api.types.is_numeric_dtype(base[c])]
    cat_cols = [c for c in all_cols if c not in num_cols]
    param_cols = [p.name for p in spec.parameters if p.kind == "choice"]
    param_nums = [p.name for p in spec.parameters if p.kind == "number"]
    p: dict[str, Any] = {}
    o: dict[str, Any] = {}
    default_title = tile.title if tile else ""

    def agg_pick(
        label: str = "Aggregate",
        field: str = "agg",
        default: str = "sum",
        allow: tuple[str, ...] = db.AGGS,
    ) -> str:
        return st.selectbox(
            label, list(allow), index=_idx(list(allow), p0.get(field, default)), key=k(field)
        )

    if kind in ("kpi", "gauge"):
        c1, c2 = st.columns(2)
        with c1:
            p["agg"] = agg_pick("Measure", default="count")
        with c2:
            col = col_pick(
                "Column",
                all_cols if p["agg"] in ("count", "nunique") else num_cols,
                p0.get("column"),
                k("col"),
                param_cols,
                optional=p["agg"] == "count",
            )
            p["column"] = col
        default_title = default_title or (
            f"{p['agg'].title()} of {p['column']}" if isinstance(p["column"], str) else "Rows"
        )
        f1, f2, f3, f4 = st.columns(4)
        o["format"] = f1.selectbox(
            "Format",
            list(db.VALUE_FORMATS),
            index=_idx(list(db.VALUE_FORMATS), o0.get("format", "auto")),
            key=k("fmt"),
        )
        o["prefix"] = f2.text_input("Prefix", value=o0.get("prefix", ""), key=k("pre"))
        o["suffix"] = f3.text_input("Suffix", value=o0.get("suffix", ""), key=k("suf"))
        dec = f4.number_input(
            "Decimals",
            0,
            6,
            int(o0["decimals"]) if o0.get("decimals") is not None else 2,
            key=k("dec"),
        )
        o["decimals"] = int(dec)
        if kind == "gauge":
            g1, g2 = st.columns(2)
            p["min"] = g1.number_input(
                "Gauge minimum", value=float(p0.get("min", 0)), key=k("gmin")
            )
            p["max"] = num_pick("Gauge maximum", p0.get("max", 100.0), k("gmax"), param_nums)
        date_cols = all_cols
        mode = st.radio(
            "Compare with",
            ["nothing", "previous period", "target"],
            index=["nothing", "previous_period", "target"].index(
                (o0.get("compare") or {}).get("kind", "nothing")
            )
            if (o0.get("compare") or {}).get("kind") in ("previous_period", "target")
            else 0,
            horizontal=True,
            key=k("cmp"),
        )
        if mode == "previous period":
            d1, d2 = st.columns(2)
            o["compare"] = {
                "kind": "previous_period",
                "date_col": d1.selectbox(
                    "Date column",
                    date_cols,
                    index=_idx(date_cols, (o0.get("compare") or {}).get("date_col")),
                    key=k("cmp_d"),
                ),
                "days": int(
                    d2.number_input(
                        "Period length (days)",
                        1,
                        3650,
                        int((o0.get("compare") or {}).get("days", 30)),
                        key=k("cmp_n"),
                    )
                ),
            }
        elif mode == "target":
            o["compare"] = {
                "kind": "target",
                "target": num_pick(
                    "Target", (o0.get("compare") or {}).get("target", 0.0), k("tgt"), param_nums
                ),
            }
        s1, s2 = st.columns(2)
        if s1.checkbox("Show a sparkline", value=bool(o0.get("sparkline")), key=k("spk")):
            o["sparkline"] = {
                "date_col": s2.selectbox(
                    "Sparkline date",
                    date_cols,
                    index=_idx(date_cols, (o0.get("sparkline") or {}).get("date_col")),
                    key=k("spk_d"),
                ),
                "freq": "W",
            }
        goal = st.number_input(
            "Goal (adds a progress bar; 0 = none)",
            value=float(o0.get("goal", 0) or 0),
            key=k("goal"),
        )
        if goal:
            o["goal"] = goal
        rule = (o0.get("thresholds") or [{}])[0]
        if st.checkbox("Colour when a threshold is crossed", value=bool(rule), key=k("thr")):
            t1, t2, t3 = st.columns(3)
            o["thresholds"] = [
                {
                    "op": t1.selectbox(
                        "When value is",
                        _OPS,
                        index=_idx(_OPS, rule.get("op", "<")),
                        key=k("thr_op"),
                    ),
                    "value": t2.number_input(
                        "Threshold", value=float(rule.get("value", 0)), key=k("thr_v")
                    ),
                    "color": t3.selectbox(
                        "Colour",
                        _COLORS,
                        index=_idx(_COLORS, rule.get("color", "red")),
                        key=k("thr_c"),
                    ),
                }
            ]
    elif kind in ("bar", "area"):
        c1, c2, c3 = st.columns(3)
        with c1:
            p["x"] = col_pick("Category (x)", all_cols, p0.get("x"), k("x"), param_cols)
        with c2:
            p["agg"] = agg_pick(default="count")
        with c3:
            p["y"] = (
                col_pick("Value (y)", num_cols, p0.get("y"), k("y"), param_cols)
                if p["agg"] != "count" and num_cols
                else None
            )
        p["color"] = col_pick(
            "Split by",
            [c for c in cat_cols if c != p["x"]],
            p0.get("color"),
            k("color"),
            [],
            optional=True,
        )
        stack_opts = list(db.STACKING)
        o["stacked"] = st.selectbox(
            "Multiple series",
            stack_opts,
            index=_idx(stack_opts, o0.get("stacked", "group")),
            key=k("stack"),
        )
        if kind == "bar":
            b1, b2, b3 = st.columns(3)
            o["orientation"] = (
                "h"
                if b1.checkbox("Horizontal", value=o0.get("orientation") == "h", key=k("hor"))
                else "v"
            )
            o["data_labels"] = b2.checkbox(
                "Value labels", value=bool(o0.get("data_labels")), key=k("lab")
            )
            drill = b3.multiselect(
                "Drill path (optional)",
                all_cols,
                default=[c for c in p0.get("drill", []) if c in all_cols],
                key=k("drill"),
            )
            if drill:
                p["drill"] = drill
                p["x"] = drill[0]
        default_title = (
            default_title or f"{p['agg'].title()} {p.get('y') or ''} by {p['x']}".replace("  ", " ")
        )
    elif kind == "line":
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            p["x"] = col_pick("Time / x", all_cols, p0.get("x"), k("x"), param_cols)
        y_default = p0.get("y") or next((c for c in num_cols if c != p["x"]), None)
        with c2:
            p["y"] = col_pick("Value (y)", num_cols, y_default, k("y"), param_cols)
        with c3:
            p["agg"] = agg_pick(allow=db.AGGS[:5], default="mean")
        freqs = list(_FREQS)
        freq = c4.selectbox(
            "Group time by",
            freqs,
            index=_idx(freqs, p0.get("freq", "")),
            format_func=_FREQS.get,
            key=k("freq"),
        )
        p["freq"] = freq or None
        p["color"] = col_pick("Split by", cat_cols, p0.get("color"), k("color"), [], optional=True)
        o["data_labels"] = st.checkbox(
            "Point labels", value=bool(o0.get("data_labels")), key=k("lab")
        )
        default_title = default_title or f"{p['y']} over {p['x']}"
    elif kind == "combo":
        c1, c2, c3 = st.columns(3)
        p["x"] = c1.selectbox("Time / x", all_cols, index=_idx(all_cols, p0.get("x")), key=k("x"))
        p["y"] = c2.selectbox(
            "Bars (left axis)", num_cols, index=_idx(num_cols, p0.get("y")), key=k("y")
        )
        p["y2"] = c3.selectbox(
            "Line (right axis)",
            num_cols,
            index=_idx(
                num_cols, p0.get("y2", num_cols[min(1, len(num_cols) - 1)] if num_cols else None)
            ),
            key=k("y2"),
        )
        d1, d2, d3 = st.columns(3)
        p["agg"] = d1.selectbox(
            "Bars aggregate",
            list(db.AGGS[:5]),
            index=_idx(list(db.AGGS[:5]), p0.get("agg", "sum")),
            key=k("agg"),
        )
        p["agg2"] = d2.selectbox(
            "Line aggregate",
            list(db.AGGS[:5]),
            index=_idx(list(db.AGGS[:5]), p0.get("agg2", "mean")),
            key=k("agg2"),
        )
        freqs = list(_FREQS)
        p["freq"] = (
            d3.selectbox(
                "Group time by",
                freqs,
                index=_idx(freqs, p0.get("freq", "MS")),
                format_func=_FREQS.get,
                key=k("freq"),
            )
            or None
        )
        default_title = default_title or f"{p['y']} and {p['y2']}"
    elif kind == "scatter":
        c1, c2 = st.columns(2)
        p["x"] = c1.selectbox("x", num_cols, index=_idx(num_cols, p0.get("x")), key=k("x"))
        p["y"] = c2.selectbox(
            "y",
            num_cols,
            index=_idx(
                num_cols, p0.get("y", num_cols[min(1, len(num_cols) - 1)] if num_cols else None)
            ),
            key=k("y"),
        )
        p["color"] = col_pick("Colour by", cat_cols, p0.get("color"), k("color"), [], optional=True)
        p["size"] = col_pick("Size by", num_cols, p0.get("size"), k("size"), [], optional=True)
        o["trendline"] = st.checkbox(
            "Trendline (OLS)", value=bool(o0.get("trendline")), key=k("trend")
        )
        default_title = default_title or f"{p['y']} vs {p['x']}"
    elif kind == "hist":
        c1, c2 = st.columns(2)
        p["column"] = col_pick("Column", num_cols, p0.get("column"), k("col"), param_cols)
        p["bins"] = int(c2.slider("Bins", 5, 80, int(p0.get("bins", 30)), key=k("bins")))
        p["color"] = col_pick("Colour by", cat_cols, p0.get("color"), k("color"), [], optional=True)
        default_title = default_title or f"Distribution of {p['column']}"
    elif kind == "pie":
        c1, c2, c3 = st.columns(3)
        p["names"] = c1.selectbox(
            "Slices", all_cols, index=_idx(all_cols, p0.get("names")), key=k("names")
        )
        vals = [NONE, *num_cols]
        v = c2.selectbox("Size by", vals, index=_idx(vals, p0.get("values") or NONE), key=k("vals"))
        p["values"] = None if v == NONE else v
        p["agg"] = c3.selectbox(
            "Aggregate",
            list(db.AGGS[:5]),
            index=_idx(list(db.AGGS[:5]), p0.get("agg", "sum")),
            key=k("agg"),
        )
        o["top_n"] = num_pick(
            "Show top N slices",
            o0.get("top_n", 8),
            k("topn"),
            param_nums,
            min_value=2,
            max_value=50,
            step=1,
            value=8,
        )
        o["donut"] = st.checkbox("Donut", value=bool(o0.get("donut")), key=k("donut"))
        default_title = default_title or f"Share by {p['names']}"
    elif kind == "box":
        c1, c2, c3 = st.columns(3)
        p["y"] = col_pick("Value", num_cols, p0.get("y"), k("y"), param_cols)
        p["x"] = col_pick("Group by", cat_cols, p0.get("x"), k("x"), [], optional=True)
        p["color"] = col_pick("Colour by", cat_cols, p0.get("color"), k("color"), [], optional=True)
        default_title = default_title or f"{p['y']} spread"
    elif kind == "heatmap":
        c1, c2, c3, c4 = st.columns(4)
        p["x"] = c1.selectbox(
            "Columns (x)", all_cols, index=_idx(all_cols, p0.get("x")), key=k("x")
        )
        p["y"] = c2.selectbox("Rows (y)", [c for c in all_cols if c != p["x"]], index=0, key=k("y"))
        v = c3.selectbox(
            "Value",
            [NONE, *num_cols],
            index=_idx([NONE, *num_cols], p0.get("value") or NONE),
            key=k("val"),
        )
        p["value"] = None if v == NONE else v
        p["agg"] = c4.selectbox(
            "Aggregate",
            list(db.AGGS[:5]),
            index=_idx(list(db.AGGS[:5]), p0.get("agg", "mean")),
            key=k("agg"),
            disabled=p["value"] is None,
        )
        default_title = default_title or f"{p['y']} x {p['x']}"
    elif kind in ("funnel", "waterfall"):
        c1, c2, c3 = st.columns(3)
        field = "names" if kind == "funnel" else "x"
        p[field] = c1.selectbox(
            "Stage / category", all_cols, index=_idx(all_cols, p0.get(field)), key=k("cat")
        )
        vfield = "values" if kind == "funnel" else "y"
        v = c2.selectbox(
            "Value",
            [NONE, *num_cols],
            index=_idx([NONE, *num_cols], p0.get(vfield) or NONE),
            key=k("val"),
        )
        p[vfield] = None if v == NONE else v
        p["agg"] = c3.selectbox(
            "Aggregate",
            list(db.AGGS[:5]),
            index=_idx(list(db.AGGS[:5]), p0.get("agg", "sum")),
            key=k("agg"),
        )
        default_title = default_title or f"{kind.title()} of {p.get(vfield) or 'records'}"
    elif kind == "treemap":
        path = st.multiselect(
            "Hierarchy (outer to inner)",
            all_cols,
            default=[c for c in p0.get("path", []) if c in all_cols] or all_cols[:1],
            key=k("path"),
        )
        p["path"] = path or all_cols[:1]
        v = st.selectbox(
            "Size by",
            [NONE, *num_cols],
            index=_idx([NONE, *num_cols], p0.get("values") or NONE),
            key=k("val"),
        )
        p["values"] = None if v == NONE else v
        p["agg"] = st.selectbox(
            "Aggregate",
            list(db.AGGS[:5]),
            index=_idx(list(db.AGGS[:5]), p0.get("agg", "sum")),
            key=k("agg"),
        )
        default_title = default_title or "Treemap of " + " > ".join(p["path"])
    elif kind == "map":
        mode = st.radio(
            "Map by",
            ["country", "latlon"],
            index=_idx(["country", "latlon"], p0.get("mode", "country")),
            horizontal=True,
            key=k("mode"),
            format_func={"country": "Country names", "latlon": "Latitude / longitude"}.get,
        )
        p["mode"] = mode
        if mode == "country":
            c1, c2 = st.columns(2)
            p["names"] = c1.selectbox(
                "Country column", all_cols, index=_idx(all_cols, p0.get("names")), key=k("names")
            )
            v = c2.selectbox(
                "Colour by",
                [NONE, *num_cols],
                index=_idx([NONE, *num_cols], p0.get("values") or NONE),
                key=k("vals"),
            )
            p["values"] = None if v == NONE else v
            p["agg"] = "sum"
        else:
            c1, c2 = st.columns(2)
            p["lat"] = c1.selectbox(
                "Latitude", num_cols, index=_idx(num_cols, p0.get("lat")), key=k("lat")
            )
            p["lon"] = c2.selectbox(
                "Longitude", num_cols, index=_idx(num_cols, p0.get("lon")), key=k("lon")
            )
            p["size"] = col_pick("Size by", num_cols, p0.get("size"), k("size"), [], optional=True)
            p["color"] = col_pick(
                "Colour by", all_cols, p0.get("color"), k("color"), [], optional=True
            )
        default_title = default_title or "Map"
    elif kind == "table":
        cols = st.multiselect(
            "Columns",
            all_cols,
            default=[c for c in p0.get("columns", []) if c in all_cols] or all_cols[:6],
            key=k("cols"),
        )
        p["columns"] = cols or all_cols[:6]
        p["rows"] = int(st.slider("Rows", 5, 200, int(p0.get("rows", 20)), key=k("rows")))
        formats = {}
        for c in [c for c in p["columns"] if c in num_cols][:6]:
            current = (o0.get("formats") or {}).get(c, "auto")
            formats[c] = st.selectbox(
                f"Format: {c}",
                list(db.VALUE_FORMATS),
                index=_idx(list(db.VALUE_FORMATS), current),
                key=k(f"fmt_{c}"),
            )
        o["formats"] = {c: f for c, f in formats.items() if f != "auto"}
        rule = (o0.get("highlight") or [{}])[0]
        if num_cols and st.checkbox("Highlight cells", value=bool(rule), key=k("hl")):
            h1, h2, h3, h4 = st.columns(4)
            hcols = [c for c in p["columns"] if c in num_cols] or num_cols
            o["highlight"] = [
                {
                    "column": h1.selectbox(
                        "Column", hcols, index=_idx(hcols, rule.get("column")), key=k("hl_c")
                    ),
                    "op": h2.selectbox(
                        "When", _OPS, index=_idx(_OPS, rule.get("op", ">")), key=k("hl_o")
                    ),
                    "value": h3.number_input(
                        "Value", value=float(rule.get("value", 0)), key=k("hl_v")
                    ),
                    "color": h4.selectbox(
                        "Colour",
                        _COLORS,
                        index=_idx(_COLORS, rule.get("color", "red")),
                        key=k("hl_k"),
                    ),
                }
            ]
        default_title = default_title or "Data table"
    elif kind == "text":
        p["markdown"] = st.text_area(
            "Text (Markdown: # headings, **bold**, - lists, [links](https://...))",
            value=p0.get("markdown", ""),
            height=140,
            key=k("md"),
        )
        default_title = default_title or "Notes"

    # ---- common display options (charts)
    if kind in db.CHART_KINDS:
        with st.expander("Display options"):
            d1, d2 = st.columns(2)
            o["x_title"] = d1.text_input("X axis title", value=o0.get("x_title", ""), key=k("xt"))
            o["y_title"] = d2.text_input("Y axis title", value=o0.get("y_title", ""), key=k("yt"))
            e1, e2, e3 = st.columns(3)
            if kind in ("bar", "area", "line", "combo", "scatter", "box", "hist"):
                o["log_y"] = e1.checkbox("Log scale (y)", value=bool(o0.get("log_y")), key=k("log"))
            if kind in ("bar", "waterfall", "funnel"):
                o["sort"] = e2.selectbox(
                    "Sort",
                    list(db.SORTS),
                    index=_idx(list(db.SORTS), o0.get("sort", "none")),
                    key=k("sort"),
                )
            if kind in ("bar", "waterfall"):
                tn = e3.number_input(
                    "Top N (0 = all)", 0, 100, int(o0.get("top_n", 0) or 0), key=k("topn2")
                )
                if tn:
                    o["top_n"] = int(tn)
                    o["other_group"] = st.checkbox(
                        "Group the rest as 'Other' (sums/counts only)",
                        value=bool(o0.get("other_group")),
                        key=k("other"),
                    )
            pal = ["(theme)", *db.PALETTES]
            chosen = st.selectbox(
                "Colours", pal, index=_idx(pal, o0.get("colors", "(theme)")), key=k("pal")
            )
            if chosen != "(theme)":
                o["colors"] = chosen
            ref = (o0.get("ref_lines") or [{}])[0]
            if kind in ("bar", "area", "line", "combo", "scatter") and st.checkbox(
                "Reference line", value=bool(ref), key=k("ref")
            ):
                r1, r2, r3 = st.columns(3)
                kinds = ["mean", "median", "value"]
                rk = r1.selectbox(
                    "Line at", kinds, index=_idx(kinds, ref.get("kind", "mean")), key=k("ref_k")
                )
                line: dict[str, Any] = {
                    "kind": rk,
                    "label": r3.text_input("Label", value=ref.get("label", ""), key=k("ref_l")),
                }
                if rk == "value":
                    line["value"] = num_pick("Value", ref.get("value", 0.0), k("ref_v"), param_nums)
                o["ref_lines"] = [line]

    # ---- layout
    with st.expander("Layout & filtering"):
        l1, l2, l3 = st.columns(3)
        title = l1.text_input("Tile title", value=default_title, key=k("title"))
        section = l2.text_input(
            "Section", value=tile.section if tile else "", key=k("section"), placeholder="(none)"
        )
        width = int(
            l3.slider("Width (grid columns)", 1, 3, tile.width if tile else 1, key=k("width"))
        )
        height = int(
            st.slider("Height (px)", 150, 800, tile.height if tile else 340, 10, key=k("height"))
        )
        use_global = st.checkbox(
            "Apply the dashboard filters",
            value=tile.apply_global_filters if tile else True,
            key=k("gf"),
        )
        local: list[db.Filter] = []
        lf = tile.local_filters[0] if tile and tile.local_filters else None
        if st.checkbox("Add a filter only for this tile", value=lf is not None, key=k("lf")):
            lc = st.selectbox(
                "Filter column",
                cat_cols or all_cols,
                index=_idx(cat_cols or all_cols, lf.column if lf else None),
                key=k("lf_c"),
            )
            opts = sorted(base[lc].dropna().astype(str).unique())[:200]
            vals = st.multiselect(
                "Keep values",
                opts,
                default=[v for v in (lf.values if lf and lf.values else []) if v in opts],
                key=k("lf_v"),
            )
            if vals:
                local = [db.Filter(column=lc, kind="categorical", values=vals)]
    return {
        "title": title or default_title or kind,
        "params": p,
        "options": {a: b for a, b in o.items() if b not in (None, "", [], {})},
        "width": width,
        "height": height,
        "section": section.strip(),
        "source": src,
        "apply_global_filters": use_global,
        "local_filters": local,
    }
