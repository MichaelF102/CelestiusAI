"""Code generation for dashboards: a Colab/Jupyter notebook and a runnable Streamlit app."""

from __future__ import annotations

from typing import Any

from src.core.code_options import make_snippet, read_call
from src.core.code_options import q as _q
from src.datascience import dashboard as db

_FORMAT_HELPER = '''def fmt(v, kind="auto", prefix="", suffix=""):
    """Format numbers like the dashboard does."""
    if v is None or v != v:
        return "-"
    if kind == "percent":
        s = f"{v * 100:.1f}%"
    elif kind == "currency":
        s = f"${v:,.0f}"
    elif kind == "integer":
        s = f"{v:,.0f}"
    elif kind == "decimal":
        s = f"{v:,.2f}"
    elif abs(v) >= 1_000_000:
        s = f"{v / 1_000_000:.2f}M"
    elif abs(v) >= 10_000:
        s = f"{v / 1_000:.1f}K"
    else:
        s = f"{v:,.0f}" if float(v).is_integer() else f"{v:,.2f}"
    return f"{prefix}{s}{suffix}"
'''


# --------------------------------------------------------------------------- filters
def filter_lines(f: db.Filter, var: str = "data") -> list[str]:
    """Static pandas lines applying one filter to `var`."""
    c = _q(f.column)
    if f.kind == "categorical" and f.values:
        return [f"{var} = {var}[{var}[{c}].astype(str).isin({[str(v) for v in f.values]!r})]"]
    if f.kind == "numeric":
        lines = []
        if f.low is not None:
            lines.append(f"{var} = {var}[pd.to_numeric({var}[{c}], errors='coerce') >= {f.low!r}]")
        if f.high is not None:
            lines.append(f"{var} = {var}[pd.to_numeric({var}[{c}], errors='coerce') <= {f.high!r}]")
        return lines
    if f.kind == "date":
        if f.preset and f.preset != "all":
            days = {"last_7": 7, "last_30": 30, "last_90": 90}.get(f.preset)
            start = (
                f"_end - pd.Timedelta(days={days})"
                if days
                else "pd.Timestamp(year=_end.year, month=1, day=1)"
            )
            return [
                f"_dates = pd.to_datetime({var}[{c}], errors='coerce')",
                "_end = _dates.max()",
                f"{var} = {var}[_dates >= {start}]",
            ]
        lines = []
        if f.start:
            lines.append(
                f"{var} = {var}[pd.to_datetime({var}[{c}], errors='coerce') >= {f.start!r}]"
            )
        if f.end:
            lines.append(f"{var} = {var}[pd.to_datetime({var}[{c}], errors='coerce') <= {f.end!r}]")
        return lines
    if f.kind == "text" and f.contains:
        return [
            f"{var} = {var}[{var}[{c}].astype(str).str.contains({f.contains!r}, case=False, regex=False, na=False)]"
        ]
    return []


def _applicable(spec: db.DashboardSpec, tile: db.Tile) -> list[db.Filter]:
    if not tile.apply_global_filters:
        return list(tile.local_filters)
    return [f for f in spec.filters if db.filter_applies(f, tile.id)] + list(tile.local_filters)


# --------------------------------------------------------------------------- per-tile code
def _title_args(tile: db.Tile, spec: db.DashboardSpec) -> str:
    o = tile.options
    parts = [f"template={'plotly_dark' if spec.theme.template == 'dark' else 'plotly_white'!r}"]
    pal = db._palette(o.get("colors") or spec.theme.palette)
    if pal:
        parts.append(f"color_discrete_sequence={pal!r}")
    return ", ".join(parts)


def _frame_lines(tile: db.Tile) -> list[str]:
    """Statements that build `frame` (mirrors db.tile_frame) from `data`."""
    p, o, kind = tile.params, tile.options, tile.kind
    agg = p.get("agg", "count")

    def aggregate(keys: list[str], value: str | None) -> list[str]:
        if not value or agg == "count":
            return [
                f"frame = data.groupby({keys!r}, dropna=False).size().reset_index(name='count')",
                "value_col = 'count'",
            ]
        coerce = (
            ""
            if agg in ("count", "nunique")
            else f"data = data.assign(**{{{value!r}: pd.to_numeric(data[{value!r}], errors='coerce')}})\n"
        )
        return [
            *([coerce.rstrip("\n")] if coerce else []),
            f"frame = data.groupby({keys!r}, dropna=False)[{value!r}].agg({agg!r}).reset_index()",
            f"value_col = {value!r}",
        ]

    def finish(key: str) -> list[str]:
        lines = []
        if o.get("sort") in ("asc", "desc"):
            lines.append(f"frame = frame.sort_values(value_col, ascending={o['sort'] == 'asc'})")
        if o.get("top_n"):
            lines += [
                f"_order = frame.groupby({key!r})[value_col].sum().sort_values(ascending=False)",
                f"_keep = _order.head({int(o['top_n'])}).index",
                f"_rest = frame[~frame[{key!r}].isin(_keep)]",
                f"frame = frame[frame[{key!r}].isin(_keep)]",
            ]
            if o.get("other_group") and agg in ("sum", "count"):
                lines.append(
                    f"frame = pd.concat([frame, pd.DataFrame({{{key!r}: ['Other'], value_col: [_rest[value_col].sum()]}})], ignore_index=True) if len(_rest) else frame"
                )
        return lines

    if kind in ("bar", "area"):
        keys = [p["x"]] + ([p["color"]] if p.get("color") else [])
        return (
            aggregate(keys, p.get("y"))
            + ([f"frame = frame.sort_values({p['x']!r})"] if kind == "area" else [])
            + finish(p["x"])
        )
    if kind == "line":
        grouper = f"pd.Grouper(key={p['x']!r}, freq={p['freq']!r})" if p.get("freq") else _q(p["x"])
        keys = [grouper] + ([_q(p["color"])] if p.get("color") else [])
        if not p.get("freq"):
            keys = [_q(p["x"])] + ([_q(p["color"])] if p.get("color") else [])
        return [
            "data = data.copy()",
            f"data[{p['x']!r}] = data[{p['x']!r}] if pd.api.types.is_numeric_dtype(data[{p['x']!r}]) else pd.to_datetime(data[{p['x']!r}], errors='coerce')",
            f"data = data.dropna(subset=[{p['x']!r}])",
            f"data[{p['y']!r}] = pd.to_numeric(data[{p['y']!r}], errors='coerce')",
            f"frame = data.groupby([{', '.join(keys)}])[{p['y']!r}].agg({agg!r}).reset_index().sort_values({p['x']!r})",
        ]
    if kind == "combo":
        key = f"pd.Grouper(key={p['x']!r}, freq={p['freq']!r})" if p.get("freq") else _q(p["x"])
        return [
            "data = data.copy()",
            f"data[{p['x']!r}] = pd.to_datetime(data[{p['x']!r}], errors='coerce') if {bool(p.get('freq'))} else data[{p['x']!r}]",
            f"data[{p['y']!r}] = pd.to_numeric(data[{p['y']!r}], errors='coerce'); data[{p['y2']!r}] = pd.to_numeric(data[{p['y2']!r}], errors='coerce')",
            f"frame = data.groupby({key}).agg(**{{{p['y']!r}: ({p['y']!r}, {agg!r}), {p['y2']!r}: ({p['y2']!r}, {p.get('agg2', 'mean')!r})}}).reset_index().sort_values({p['x']!r})",
        ]
    if kind == "scatter":
        cols = [c for c in (p["x"], p["y"], p.get("color"), p.get("size")) if c]
        return [
            f"frame = data[{cols!r}].dropna()",
            f"frame = frame.sample({db.MAX_POINTS}, random_state=0) if len(frame) > {db.MAX_POINTS} else frame",
        ]
    if kind == "hist":
        return [f"frame = data[{[c for c in (p['column'], p.get('color')) if c]!r}]"]
    if kind == "pie":
        a = p.get("agg", "sum")
        top = int(o.get("top_n", p.get("top_n", 8)))
        base = (
            [
                f"frame = data.groupby({p['names']!r}, dropna=False).size().reset_index(name='count')",
                "value_col = 'count'",
            ]
            if not p.get("values") or a == "count"
            else [
                f"frame = data.assign(**{{{p['values']!r}: pd.to_numeric(data[{p['values']!r}], errors='coerce')}}).groupby({p['names']!r}, dropna=False)[{p['values']!r}].agg({a!r}).reset_index()",
                f"value_col = {p['values']!r}",
            ]
        )
        lines = base + [
            "frame = frame.sort_values(value_col, ascending=False)",
            f"_rest = frame.iloc[{top}:][value_col].sum()",
            f"frame = frame.head({top})",
        ]
        if o.get("other_group", True):
            lines.append(
                f"frame = pd.concat([frame, pd.DataFrame({{{p['names']!r}: ['Other'], value_col: [_rest]}})], ignore_index=True) if _rest else frame"
            )
        return lines
    if kind == "box":
        return [f"frame = data[{[c for c in (p.get('x'), p['y'], p.get('color')) if c]!r}]"]
    if kind == "heatmap":
        if p.get("value") and agg != "count":
            return [
                f"frame = data.assign(**{{{p['value']!r}: pd.to_numeric(data[{p['value']!r}], errors='coerce')}}).pivot_table(index={p['y']!r}, columns={p['x']!r}, values={p['value']!r}, aggfunc={agg!r})"
            ]
        return [f"frame = pd.crosstab(data[{p['y']!r}], data[{p['x']!r}])"]
    if kind == "funnel":
        return aggregate([p["names"]], p.get("values")) + [
            "frame = frame.sort_values(value_col, ascending=False)"
        ]
    if kind == "waterfall":
        return aggregate([p["x"]], p.get("y")) + finish(p["x"])
    if kind == "treemap":
        path = p.get("path") or [p["names"]]
        if p.get("values"):
            return [
                f"frame = data.assign(**{{{p['values']!r}: pd.to_numeric(data[{p['values']!r}], errors='coerce')}}).groupby({path!r}, dropna=False)[{p['values']!r}].agg({(agg if agg != 'count' else 'sum')!r}).reset_index()"
            ]
        return [f"frame = data.groupby({path!r}, dropna=False).size().reset_index(name='count')"]
    if kind == "map":
        if p.get("mode", "country") == "latlon":
            cols = [c for c in (p["lat"], p["lon"], p.get("size"), p.get("color")) if c]
            return [f"frame = data[{cols!r}].dropna(subset=[{p['lat']!r}, {p['lon']!r}])"]
        return aggregate([p["names"]], p.get("values"))
    if kind == "table":
        cols = p.get("columns") or []
        return [
            f"frame = data[{cols!r}].head({int(p.get('rows', 20))})"
            if cols
            else f"frame = data.head({int(p.get('rows', 20))})"
        ]
    return []


def _figure_lines(tile: db.Tile, spec: db.DashboardSpec) -> list[str]:
    """Statements that define `fig` from `frame`."""
    p, o, kind = tile.params, tile.options, tile.kind
    common = _title_args(tile, spec)
    color = _q(p.get("color")) if p.get("color") else "None"
    lines: list[str] = []
    if kind == "bar":
        horizontal = o.get("orientation") == "h"
        value = "value_col"
        mode = {"group": "group", "stack": "stack", "percent": "relative"}.get(
            o.get("stacked", "group"), "group"
        )
        x, y = (value, _q(p["x"])) if horizontal else (_q(p["x"]), value)
        lines.append(
            f"fig = px.bar(frame, x={x}, y={y}, color={color}, barmode={mode!r}, orientation={'h' if horizontal else 'v'!r}, text_auto={bool(o.get('data_labels'))}, {common})"
        )
        if o.get("stacked") == "percent":
            lines.append("fig.update_layout(barnorm='percent')")
    elif kind == "area":
        lines.append(f"fig = px.area(frame, x={p['x']!r}, y=value_col, color={color}, {common})")
    elif kind == "line":
        lines.append(
            f"fig = px.line(frame, x={p['x']!r}, y={p['y']!r}, color={color}, markers={bool(o.get('data_labels'))}, {common})"
        )
    elif kind == "combo":
        lines += [
            "fig = go.Figure()",
            f"fig.add_bar(x=frame[{p['x']!r}], y=frame[{p['y']!r}], name={p['y']!r})",
            f"fig.add_scatter(x=frame[{p['x']!r}], y=frame[{p['y2']!r}], name={p['y2']!r}, yaxis='y2', mode='lines+markers')",
            f"fig.update_layout(template={'plotly_dark' if spec.theme.template == 'dark' else 'plotly_white'!r}, yaxis2={{'overlaying': 'y', 'side': 'right', 'title': {p['y2']!r}}})",
        ]
    elif kind == "scatter":
        trend = ", trendline='ols'" if o.get("trendline") else ""
        size = f", size={p['size']!r}" if p.get("size") else ""
        lines.append(
            f"fig = px.scatter(frame, x={p['x']!r}, y={p['y']!r}, color={color}{size}{trend}, {common})"
        )
    elif kind == "hist":
        lines.append(
            f"fig = px.histogram(frame, x={p['column']!r}, nbins={int(p.get('bins', 30))}, color={color}, {common})"
        )
    elif kind == "pie":
        lines.append(
            f"fig = px.pie(frame, names={p['names']!r}, values=value_col, hole={0.4 if o.get('donut') else 0}, {common})"
        )
    elif kind == "box":
        lines.append(
            f"fig = px.box(frame, y={p['y']!r}, x={p.get('x')!r}, color={color}, {common})"
        )
    elif kind == "heatmap":
        lines.append(
            f"fig = px.imshow(frame, text_auto='.0f', aspect='auto', template={'plotly_dark' if spec.theme.template == 'dark' else 'plotly_white'!r})"
        )
    elif kind == "funnel":
        lines.append(f"fig = px.funnel(frame, x=value_col, y={p['names']!r}, {common})")
    elif kind == "waterfall":
        lines.append(
            f"fig = go.Figure(go.Waterfall(x=[*frame[{p['x']!r}].astype(str), 'Total'], y=[*frame[value_col], 0], measure=['relative'] * len(frame) + ['total']))"
        )
    elif kind == "treemap":
        path = p.get("path") or [p["names"]]
        lines.append(
            f"fig = px.treemap(frame, path={path!r}, values={(p.get('values') or 'count')!r}, template={'plotly_dark' if spec.theme.template == 'dark' else 'plotly_white'!r})"
        )
    elif kind == "map":
        if p.get("mode", "country") == "latlon":
            lines.append(
                f"fig = px.scatter_geo(frame, lat={p['lat']!r}, lon={p['lon']!r}, size={p.get('size')!r}, color={p.get('color')!r})"
            )
        else:
            lines.append(
                f"fig = px.choropleth(frame, locations={p['names']!r}, locationmode='country names', color=value_col)"
            )
    if not lines:
        return []
    if o.get("log_y"):
        lines.append("fig.update_yaxes(type='log')")
    for ref in o.get("ref_lines", []) or []:
        if ref.get("kind", "value") == "value" and ref.get("value") is not None:
            lines.append(
                f"fig.add_hline(y={float(ref['value'])!r}, line_dash='dash', annotation_text={(ref.get('label') or 'target')!r})"
            )
    title_bits = [f"title={tile.title!r}", f"height={tile.height}"]
    if o.get("x_title"):
        title_bits.append(f"xaxis_title={o['x_title']!r}")
    if o.get("y_title"):
        title_bits.append(f"yaxis_title={o['y_title']!r}")
    lines.append(f"fig.update_layout({', '.join(title_bits)})")
    return lines


def _kpi_lines(tile: db.Tile) -> list[str]:
    """Statements defining `kpi_text` and optional `kpi_delta` for a KPI tile."""
    p, o = tile.params, tile.options
    agg, col = p.get("agg", "count"), p.get("column")

    def value_expr(frame_var: str) -> str:
        if agg == "count" and not col:
            return f"float(len({frame_var}))"
        if agg in ("count", "nunique"):
            return f"float({frame_var}[{col!r}].{agg}())"
        return f"float(pd.to_numeric({frame_var}[{col!r}], errors='coerce').{agg}())"

    lines = [f"kpi_value = {value_expr('data')} if len(data) else None", "kpi_delta = None"]
    compare = o.get("compare") or {}
    if compare.get("kind") == "previous_period":
        days = int(compare.get("days", 30))
        lines += [
            f"_d = pd.to_datetime(data[{compare['date_col']!r}], errors='coerce')",
            "_end = _d.max()",
            f"_cur = data[(_d > _end - pd.Timedelta(days={days})) & (_d <= _end)]",
            f"_prev = data[(_d > _end - pd.Timedelta(days={2 * days})) & (_d <= _end - pd.Timedelta(days={days}))]",
            f"kpi_value = {value_expr('_cur')} if len(_cur) else None",
            f"_pv = {value_expr('_prev')} if len(_prev) else None",
            "kpi_delta = f'{(kpi_value - _pv) / abs(_pv):+.1%}' if kpi_value is not None and _pv else None",
        ]
    elif compare.get("kind") == "target" and compare.get("target"):
        lines.append(
            f"kpi_delta = f'{{(kpi_value - {float(compare['target'])!r}) / abs({float(compare['target'])!r}):+.1%}} vs target' if kpi_value is not None else None"
        )
    lines.append(
        f"kpi_text = fmt(kpi_value, {o.get('format', 'auto')!r}, {o.get('prefix', '')!r}, {o.get('suffix', '')!r})"
    )
    return lines


# --------------------------------------------------------------------------- notebook
def dashboard_snippet(spec: db.DashboardSpec, **load: Any) -> str:
    lines = ["import plotly.graph_objects as go", "import plotly.express as px", "", _FORMAT_HELPER]
    for f in spec.calculated:
        lines.append(f"df[{f.name!r}] = df.eval({f.expr!r})   # calculated field")
    if spec.calculated:
        lines.append("")
    lines += [f"# {spec.title}" + (f" - {spec.description}" if spec.description else "")]
    cur_section = None
    for tile in (resolved for resolved in (db.resolve_tile(t, spec) for t in spec.tiles)):
        if tile.section != cur_section:
            cur_section = tile.section
            if cur_section:
                lines += ["", f"print('=== {cur_section} ===')"]
        lines.append(f"\n# {tile.title}")
        if tile.source != "active":
            lines.append(
                f"data = tables[{tile.source!r}]   # load this table first (see the Data Sources page)"
            )
        else:
            lines.append("data = df")
        for f in _applicable(spec, tile):
            lines += filter_lines(f)
        if tile.kind == "text":
            lines.append(f"print({tile.params.get('markdown', '')!r})")
            continue
        if tile.kind == "kpi":
            lines += _kpi_lines(tile) + [f"print({tile.title!r}, kpi_text, kpi_delta or '')"]
            continue
        if tile.kind == "gauge":
            lines += _kpi_lines(tile) + [f"print({tile.title!r}, kpi_text)"]
            continue
        lines += _frame_lines(tile)
        if tile.kind == "table":
            lines.append("display(frame)")
        else:
            lines += _figure_lines(tile, spec) + ["fig.show()"]
    return make_snippet(spec.title, "\n".join(lines), **load)


# --------------------------------------------------------------------------- streamlit app
def streamlit_app_code(
    spec: db.DashboardSpec, filename: str = "your_file.csv", sheet: str | None = None
) -> str:
    """A complete, runnable Streamlit app that rebuilds the dashboard with live filters."""
    out = [
        f'"""{spec.title} - generated by CelestiusAI."""',
        "import pandas as pd",
        "import plotly.express as px",
        "import plotly.graph_objects as go",
        "import streamlit as st",
        "",
        f"st.set_page_config(page_title={spec.title!r}, layout='wide')",
        f"st.title({spec.title!r})",
    ]
    if spec.description:
        out.append(f"st.caption({spec.description!r})")
    out += [
        "",
        _FORMAT_HELPER,
        "",
        "@st.cache_data",
        "def load_data():",
        f"    return {read_call(_q(filename), filename, sheet)}",
        "",
        "df = load_data()",
    ]
    for f in spec.calculated:
        out.append(f"df[{f.name!r}] = df.eval({f.expr!r})")
    out += ["", "# ---- sidebar filters"]
    fn_names: dict[int, str] = {}
    for i, f in enumerate(spec.filters, start=1):
        c, var = _q(f.column), f"flt_{i}"
        fn = f"filter_{i}"
        fn_names[i - 1] = fn
        if f.kind == "categorical":
            opts = f"sorted(df[{c}].dropna().astype(str).unique())"
            default = [str(v) for v in (f.values or [])]
            widget = (
                f"st.sidebar.selectbox({f.column!r}, {opts}, index=0)"
                if f.single
                else f"st.sidebar.multiselect({f.column!r}, {opts}, default={default!r})"
            )
            out += [
                f"{var} = {widget}",
                f"def {fn}(d):",
                f"    return d[d[{c}].astype(str).isin([{var}] if isinstance({var}, str) else {var})] if {var} else d",
            ]
        elif f.kind == "numeric":
            out += [
                f"_n{i} = pd.to_numeric(df[{c}], errors='coerce').dropna()",
                f"{var} = st.sidebar.slider({f.column!r}, float(_n{i}.min()), float(_n{i}.max()), ({(f.low if f.low is not None else 0.0)!r} if {f.low is not None} else float(_n{i}.min()), {(f.high if f.high is not None else 0.0)!r} if {f.high is not None} else float(_n{i}.max())))",
                f"def {fn}(d):",
                f"    n = pd.to_numeric(d[{c}], errors='coerce')",
                f"    return d[(n >= {var}[0]) & (n <= {var}[1])]",
            ]
        elif f.kind == "date":
            presets = list(db.DATE_PRESETS)
            default_idx = presets.index(f.preset) if f.preset in presets else 0
            out += [
                f"{var} = st.sidebar.selectbox({f.column + ' range'!r}, {presets!r}, index={default_idx}, format_func={db.DATE_PRESETS!r}.get)",
                f"def {fn}(d):",
                f"    dates = pd.to_datetime(d[{c}], errors='coerce')",
                f"    days = {{'last_7': 7, 'last_30': 30, 'last_90': 90}}.get({var})",
                "    end = dates.max()",
                f"    if {var} == 'this_year':",
                "        return d[dates >= pd.Timestamp(year=end.year, month=1, day=1)]",
                "    return d[dates >= end - pd.Timedelta(days=days)] if days else d",
            ]
        else:
            out += [
                f"{var} = st.sidebar.text_input({f.column + ' contains'!r}, value={(f.contains or '')!r})",
                f"def {fn}(d):",
                f"    return d[d[{c}].astype(str).str.contains({var}, case=False, regex=False, na=False)] if {var} else d",
            ]
    out += ["", "# ---- tiles"]
    cur_section = None
    for tile in (db.resolve_tile(t, spec) for t in spec.tiles):
        if tile.section != cur_section:
            cur_section = tile.section
            if cur_section:
                out += ["", f"st.header({cur_section!r})"]
        out += ["", f"# {tile.title}", "data = df"]
        if tile.apply_global_filters:
            for i, f in enumerate(spec.filters):
                if db.filter_applies(f, tile.id):
                    out.append(f"data = {fn_names[i]}(data)")
        for f in tile.local_filters:
            out += filter_lines(f)
        if tile.kind == "text":
            out.append(f"st.markdown({tile.params.get('markdown', '')!r})")
        elif tile.kind in ("kpi",):
            out += _kpi_lines(tile) + [f"st.metric({tile.title!r}, kpi_text, kpi_delta)"]
        elif tile.kind == "gauge":
            out += _kpi_lines(tile) + [f"st.metric({tile.title!r}, kpi_text)"]
        elif tile.kind == "table":
            out += _frame_lines(tile) + [
                "st.dataframe(frame, use_container_width=True, hide_index=True)"
            ]
        else:
            body = _frame_lines(tile) + _figure_lines(tile, spec)
            out += [
                "try:",
                *[f"    {ln}" for ln in body],
                "    st.plotly_chart(fig, use_container_width=True)",
                "except Exception as err:",
                f"    st.warning({tile.title!r} + ': ' + str(err))",
            ]
    return "\n".join(out) + "\n"
