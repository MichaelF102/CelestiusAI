"""Page 12: Dashboard Builder - tiles, sections, themes, cross-filtering, exports, and code."""

import pandas as pd
import streamlit as st
from src.agent.provider_factory import ProviderFactory
from src.config.settings import get_settings
from src.core.code_snippets import load_kwargs
from src.core.dashboard_forms import tile_form
from src.core.library import get_tables
from src.core.session_state import (
    KEY_DASHBOARD_SPEC,
    KEY_DATASET_SCHEMA,
    KEY_GENERATED_FINDINGS,
    KEY_LLM_OVERRIDE,
)
from src.core.ui_components import (
    esc,
    page_setup,
    render_code_button,
    render_code_export,
    require_dataset,
)
from src.datascience import dashboard as db
from src.datascience import dashboard_ai as ai
from src.datascience import dashboard_code as dcode
from src.datascience import dashboard_export as dx
from src.datascience.dashboard_code import streamlit_app_code

page_setup(
    "Dashboard Builder",
    "🧩",
    "Compose KPIs and charts, filter and drill into them, and export or rebuild them anywhere.",
)
settings = get_settings()
dataset = require_dataset("Dashboard Builder")
tables = dict(get_tables(st.session_state))
df = dataset.working_df
schema = st.session_state.get(KEY_DATASET_SCHEMA) or dataset.schema_info

# ------------------------------------------------------------------ state
if st.session_state.get(KEY_DASHBOARD_SPEC) is None:
    st.session_state[KEY_DASHBOARD_SPEC] = db.DashboardSpec(
        title=dataset.filename.rsplit(".", 1)[0]
    )
raw = st.session_state[KEY_DASHBOARD_SPEC]
spec: db.DashboardSpec = (
    raw if isinstance(raw, db.DashboardSpec) else db.DashboardSpec.model_validate(raw)
)
st.session_state[KEY_DASHBOARD_SPEC] = spec
history: db.History = st.session_state.setdefault("dash_history", db.History())
cross: dict[str, db.Filter] = st.session_state.setdefault("dash_cross", {})
drills: dict[str, db.DrillState] = st.session_state.setdefault("dash_drill", {})


def commit(new_spec: db.DashboardSpec) -> None:
    """Record the current spec for undo, then replace it and rerun."""
    history.push(spec)
    st.session_state[KEY_DASHBOARD_SPEC] = new_spec
    st.rerun()


all_cols = [str(c) for c in db.apply_calculated(df, spec.calculated)[0].columns]
num_cols = [
    c
    for c in all_cols
    if pd.api.types.is_numeric_dtype(db.apply_calculated(df, spec.calculated)[0][c])
]
tile_titles = {t.id: t.title for t in spec.tiles}

# ------------------------------------------------------------------ sidebar: filters, parameters, calculated fields, theme
with st.sidebar:
    with st.expander("🎛️ Filters", expanded=True):
        add_col = st.selectbox("Add a filter on", ["-"] + all_cols, key="dash_new_filter")
        if add_col != "-" and st.button("Add filter", key="dash_add_filter"):
            series = db.apply_calculated(df, spec.calculated)[0][add_col]
            if add_col in num_cols:
                kind = "numeric"
            elif (
                "date" in add_col.lower()
                or str(series.dtype).startswith("datetime")
                or ai._looks_like_date(series)
            ):
                kind = "date"
            else:
                kind = "categorical"
            new = db.clone_spec(spec)
            new.filters.append(db.Filter(column=add_col, kind=kind))
            commit(new)
        base_df = db.apply_calculated(df, spec.calculated)[0]
        for i, f in enumerate(list(spec.filters)):
            st.markdown(f"**{f.column}**")
            if f.kind == "categorical":
                opts = sorted(base_df[f.column].dropna().astype(str).unique())[:300]
                f.single = st.checkbox("Single choice", value=f.single, key=f"flt_single_{i}")
                if f.single:
                    current = (f.values or [opts[0]])[0] if opts else None
                    pick = st.selectbox(
                        "Value",
                        opts,
                        index=opts.index(current) if current in opts else 0,
                        key=f"flt_{i}",
                        label_visibility="collapsed",
                    )
                    f.values = [pick]
                else:
                    f.values = st.multiselect(
                        "Values",
                        opts,
                        default=[v for v in (f.values or []) if v in opts],
                        key=f"flt_{i}",
                        label_visibility="collapsed",
                    )
            elif f.kind == "numeric":
                series = pd.to_numeric(base_df[f.column], errors="coerce").dropna()
                lo, hi = float(series.min()), float(series.max())
                if lo < hi:
                    f.low, f.high = st.slider(
                        "Range",
                        lo,
                        hi,
                        (f.low if f.low is not None else lo, f.high if f.high is not None else hi),
                        key=f"flt_{i}",
                        label_visibility="collapsed",
                    )
            elif f.kind == "date":
                presets = list(db.DATE_PRESETS)
                f.preset = st.selectbox(
                    "Period",
                    presets,
                    index=presets.index(f.preset) if f.preset in presets else 0,
                    format_func=db.DATE_PRESETS.get,
                    key=f"flt_p_{i}",
                )
                if f.preset == "all":
                    d = pd.to_datetime(base_df[f.column], errors="coerce").dropna()
                    if len(d):
                        span = st.date_input(
                            "Custom range", (d.min().date(), d.max().date()), key=f"flt_{i}"
                        )
                        if isinstance(span, tuple) and len(span) == 2:
                            f.start, f.end = str(span[0]), str(span[1])
            else:
                f.contains = (
                    st.text_input(
                        "Contains",
                        value=f.contains or "",
                        key=f"flt_{i}",
                        label_visibility="collapsed",
                    )
                    or None
                )
            if len(spec.tiles) > 1:
                scoped = st.multiselect(
                    "Apply to tiles (empty = all)",
                    list(tile_titles),
                    default=[t for t in (f.tiles or []) if t in tile_titles],
                    format_func=tile_titles.get,
                    key=f"flt_tiles_{i}",
                )
                f.tiles = scoped or None
            if st.button("Remove filter", key=f"flt_rm_{i}"):
                new = db.clone_spec(spec)
                new.filters.pop(i)
                commit(new)
        txt_col = st.selectbox(
            "Add a text search on",
            ["-"] + [c for c in all_cols if c not in num_cols],
            key="dash_new_text",
        )
        if txt_col != "-" and st.button("Add search box", key="dash_add_text"):
            new = db.clone_spec(spec)
            new.filters.append(db.Filter(column=txt_col, kind="text"))
            commit(new)

    with st.expander("🧮 Parameters"):
        st.caption("Values your tiles can reference (a goal, a top-N, or a switchable measure).")
        for i, prm in enumerate(spec.parameters):
            if prm.kind == "number":
                prm.value = st.number_input(
                    prm.label or prm.name, value=float(prm.value or 0), key=f"prm_{i}"
                )
            else:
                opts = prm.options or []
                prm.value = st.selectbox(
                    prm.label or prm.name,
                    opts,
                    index=opts.index(prm.value) if prm.value in opts else 0,
                    key=f"prm_{i}",
                )
            if st.button("Remove", key=f"prm_rm_{i}"):
                new = db.clone_spec(spec)
                new.parameters.pop(i)
                commit(new)
        n1, n2 = st.columns(2)
        pname = n1.text_input("Name", key="prm_new_name", placeholder="target")
        pkind = n2.selectbox("Type", ["number", "choice"], key="prm_new_kind")
        if pkind == "choice":
            chosen = st.multiselect("Choices (columns)", all_cols, key="prm_new_opts")
        if st.button("Add parameter", disabled=not pname.strip()):
            new = db.clone_spec(spec)
            slug = pname.strip().replace(" ", "_")
            if any(p.name == slug for p in new.parameters):
                st.error("A parameter with that name exists.")
            else:
                new.parameters.append(
                    db.Parameter(
                        name=slug,
                        kind=pkind,
                        value=0.0 if pkind == "number" else (chosen[0] if chosen else None),
                        options=chosen if pkind == "choice" else None,
                    )
                )
                commit(new)

    with st.expander("➕ Calculated fields"):
        st.caption("New columns computed from existing ones, usable in every tile and filter.")
        _, calc_errors = db.apply_calculated(df, spec.calculated)
        for i, cf in enumerate(spec.calculated):
            st.markdown(
                f"`{cf.name}` = `{cf.expr}`"
                + (f"  ⚠️ {calc_errors[cf.name]}" if cf.name in calc_errors else "")
            )
            if st.button("Remove", key=f"calc_rm_{i}"):
                new = db.clone_spec(spec)
                new.calculated.pop(i)
                commit(new)
        c_name = st.text_input("Field name", key="calc_new_name", placeholder="profit")
        c_expr = st.text_input("Expression", key="calc_new_expr", placeholder="revenue - cost")
        if st.button("Add field", disabled=not (c_name.strip() and c_expr.strip())):
            trial = db.CalcField(name=c_name.strip(), expr=c_expr.strip())
            _, errs = db.apply_calculated(df, [trial])
            if errs:
                st.error(errs[trial.name])
            else:
                new = db.clone_spec(spec)
                new.calculated.append(trial)
                commit(new)

    with st.expander("🎨 Theme"):
        th = spec.theme
        th.template = st.radio(
            "Mode",
            ["dark", "light"],
            index=["dark", "light"].index(th.template),
            horizontal=True,
            key="th_mode",
        )
        th.palette = st.selectbox(
            "Colours",
            list(db.PALETTES),
            index=list(db.PALETTES).index(th.palette) if th.palette in db.PALETTES else 0,
            key="th_pal",
            help="'colorblind' uses the Okabe-Ito palette.",
        )
        th.font_size = st.slider("Chart font size", 9, 20, th.font_size, key="th_font")
        th.accent = st.color_picker("Accent colour", th.accent, key="th_accent")
        th.tile_border = st.checkbox("Tile borders", value=th.tile_border, key="th_border")
        show_captions = st.checkbox("Show captions under charts", value=True, key="th_captions")

# ------------------------------------------------------------------ header controls
h1, h2, h3, h4, h5 = st.columns([4, 1, 1, 1, 1])
spec.title = h1.text_input("Title", value=spec.title, key="dash_title")
spec.columns = int(h2.slider("Grid columns", 1, 3, spec.columns, key="dash_cols"))
if h3.button("↩️ Undo", disabled=not history.undo_stack, use_container_width=True):
    prev = history.undo(spec)
    if prev is not None:
        st.session_state[KEY_DASHBOARD_SPEC] = prev
        st.rerun()
if h4.button("↪️ Redo", disabled=not history.redo_stack, use_container_width=True):
    nxt = history.redo(spec)
    if nxt is not None:
        st.session_state[KEY_DASHBOARD_SPEC] = nxt
        st.rerun()
if h5.button(
    "🔄 Refresh",
    use_container_width=True,
    help="Recompute from the current data and re-check the layout",
):
    st.rerun()
spec.description = st.text_input("Description (optional)", value=spec.description, key="dash_desc")

problems = db.validate_spec(spec, tables)
if problems:
    with st.expander(f"⚠️ {len(problems)} layout problem(s)", expanded=True):
        for msg in problems:
            st.warning(msg)
st.caption(
    f"Data: **{dataset.filename}** · {len(df):,} rows · version {dataset.version} · {len(tables)} table(s) available"
)

# ------------------------------------------------------------------ start: templates, suggestions, AI
with st.expander("✨ Start from a template or describe a tile", expanded=not spec.tiles):
    t1, t2, t3 = st.columns([2, 1, 1])
    template = t1.selectbox(
        "Template", list(ai.TEMPLATES), format_func=ai.TEMPLATES.get, key="dash_template"
    )
    if t2.button("Use template", use_container_width=True):
        try:
            if template == "quality":
                from src.core.library import add_table

                add_table(st.session_state, "quality_summary", ai.quality_table(df))
            new = ai.template_spec(template, df, schema, title=spec.title)
            new.theme = spec.theme
            commit(new)
        except db.DashboardError as err:
            st.error(str(err))
    if t3.button(
        "Suggest from my data",
        use_container_width=True,
        help="Uses column types and the findings already generated",
    ):
        try:
            new = ai.suggest_spec(
                df, schema, st.session_state.get(KEY_GENERATED_FINDINGS), title=spec.title
            )
            new.theme = spec.theme
            commit(new)
        except db.DashboardError as err:
            st.error(str(err))
    ask = st.text_input(
        "Describe a tile in plain English",
        placeholder="average revenue by region as a bar chart",
        key="dash_ai_text",
    )
    if st.button("Draft tile with AI", disabled=not ask.strip()):
        provider = ProviderFactory.from_override(st.session_state.get(KEY_LLM_OVERRIDE), settings)
        if not provider.is_configured:
            st.warning(
                "No AI provider is configured (Settings page). Use a template or the Add tile form instead."
            )
        else:
            try:
                st.session_state["dash_pending"] = ai.nl_to_tile(provider, ask, df).model_dump()
            except db.DashboardError as err:
                st.error(str(err))
    pending = st.session_state.get("dash_pending")
    if pending:
        tile_p = db.Tile.model_validate(pending)
        st.info(
            f"Proposed: **{tile_p.title}** ({db.TILE_LABELS[tile_p.kind]}) with {tile_p.params}"
        )
        a1, a2 = st.columns(2)
        if a1.button("Add this tile", type="primary"):
            new = db.clone_spec(spec)
            new.tiles.append(tile_p)
            st.session_state.pop("dash_pending")
            commit(new)
        if a2.button("Discard"):
            st.session_state.pop("dash_pending")
            st.rerun()

with st.expander("➕ Add a tile", expanded=False):
    kind = st.selectbox(
        "Tile type", list(db.TILE_LABELS), format_func=db.TILE_LABELS.get, key="tile_kind"
    )
    form = tile_form(kind, tables, spec, key=f"add_{kind}")
    if st.button("Add tile", type="primary", key="dash_add_tile"):
        new = db.clone_spec(spec)
        new.tiles.append(db.Tile(kind=kind, **form))
        commit(new)

# ------------------------------------------------------------------ cross-filter chips
if cross:
    chip_cols = st.columns(min(4, len(cross)) + 1)
    for (tid, flt), col in zip(list(cross.items()), chip_cols, strict=False):
        label = f"{flt.column} = {', '.join(map(str, flt.values or []))}"
        if col.button(f"✕ {label}", key=f"clear_{tid}", help="Remove this cross-filter"):
            cross.pop(tid)
            st.rerun()
    if chip_cols[-1].button("Clear all", key="clear_cross"):
        cross.clear()
        st.rerun()

# ------------------------------------------------------------------ render tiles
SELECTABLE = ("bar", "pie", "treemap", "funnel")


def table_style(frame: pd.DataFrame, options: dict):
    shown = frame.copy()
    styler = shown.style
    fmts = {
        c: (lambda v, f=f: db.format_value(v, f))
        for c, f in (options.get("formats") or {}).items()
        if c in shown.columns
    }
    if fmts:
        styler = styler.format(fmts, na_rep="-")
    marks = dx.highlight_style(frame, options)
    if marks:
        col_marks = {}
        for (pos, col), color in marks.items():
            col_marks.setdefault(col, {})[pos] = color

        def paint(column: pd.Series) -> list[str]:
            cells = col_marks.get(column.name, {})
            return [
                f"background-color: {cells[i]}55; font-weight: 600" if i in cells else ""
                for i in range(len(column))
            ]

        styler = styler.apply(paint, axis=0)
    return styler


def render_tile(tile: db.Tile) -> None:
    resolved = db.resolve_tile(tile, spec)
    state = drills.setdefault(tile.id, db.DrillState())
    draw_tile, drill_filters = db.drill_apply(resolved, state)
    others = [f for tid, f in cross.items() if tid != tile.id]
    with st.container(border=spec.theme.tile_border):
        head, menu = st.columns([6, 1])
        head.markdown(f"**{esc(tile.title)}**")
        with menu.popover("⋯"):
            idx = next(i for i, t in enumerate(spec.tiles) if t.id == tile.id)
            m1, m2, m3 = st.columns(3)
            if m1.button("⬆", key=f"up_{tile.id}", disabled=idx == 0, help="Move earlier"):
                commit(db.move_tile(spec, tile.id, -1))
            if m2.button(
                "⬇", key=f"dn_{tile.id}", disabled=idx == len(spec.tiles) - 1, help="Move later"
            ):
                commit(db.move_tile(spec, tile.id, 1))
            if m3.button("⧉", key=f"dup_{tile.id}", help="Duplicate"):
                commit(db.duplicate_tile(spec, tile.id))
            if st.checkbox("✏️ Edit this tile", key=f"edit_on_{tile.id}"):
                form = tile_form(tile.kind, tables, spec, key=f"edit_{tile.id}", tile=tile)
                if st.button("Save changes", key=f"save_{tile.id}", type="primary"):
                    new = db.clone_spec(spec)
                    target = next(t for t in new.tiles if t.id == tile.id)
                    for key_, val in form.items():
                        setattr(target, key_, val)
                    commit(new)
            if tile.kind not in ("text", "kpi", "gauge") and st.checkbox(
                "📄 Show the data behind this tile", key=f"data_on_{tile.id}"
            ):
                try:
                    frame_view = db.tile_frame(
                        db.tile_data(tables, spec, draw_tile, [*drill_filters, *others]), draw_tile
                    )
                    st.dataframe(
                        frame_view.reset_index() if tile.kind == "heatmap" else frame_view,
                        use_container_width=True,
                    )
                    st.download_button(
                        "Download CSV",
                        frame_view.to_csv(),
                        f"{tile.title}.csv",
                        "text/csv",
                        key=f"dl_{tile.id}",
                    )
                except db.DashboardError as err:
                    st.caption(str(err))
            if st.button("🗑️ Remove tile", key=f"rm_{tile.id}"):
                new = db.clone_spec(spec)
                new.tiles = [t for t in new.tiles if t.id != tile.id]
                cross.pop(tile.id, None)
                commit(new)
        try:
            data = db.tile_data(tables, spec, draw_tile, [*drill_filters, *others])
            if tile.kind == "text":
                st.markdown(str(resolved.params.get("markdown", "")))
            elif tile.kind == "kpi":
                card = db.kpi_card(data, resolved)
                dot = {
                    "red": "🔴 ",
                    "orange": "🟠 ",
                    "yellow": "🟡 ",
                    "green": "🟢 ",
                    "blue": "🔵 ",
                }.get(card["color"] or "", "")
                st.metric(
                    dot + resolved.title,
                    card["text"],
                    delta=card["delta_text"],
                    label_visibility="collapsed",
                )
                if card["progress"] is not None:
                    st.progress(card["progress"], text=card["note"])
                elif card["note"]:
                    st.caption(card["note"])
                if card["spark"] is not None:
                    st.area_chart(card["spark"].rename("trend"), height=60)
            elif tile.kind == "table":
                st.dataframe(
                    table_style(db.tile_frame(data, resolved), resolved.options),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                if state.level or (tile.params.get("drill") and len(tile.params["drill"]) > 1):
                    path = tile.params["drill"]
                    crumb = " › ".join(
                        [
                            *(
                                f"{c}: {state.picked[c]}"
                                for c in path[: state.level]
                                if c in state.picked
                            ),
                            path[min(state.level, len(path) - 1)],
                        ]
                    )
                    b1, b2 = st.columns([1, 5])
                    if b1.button("⬆ Up", key=f"up_drill_{tile.id}", disabled=state.level == 0):
                        drills[tile.id] = db.drill_up(state)
                        st.rerun()
                    b2.caption(f"Drill-down: {crumb}")
                fig = db.build_figure(data, draw_tile, theme=spec.theme)
                if tile.kind in SELECTABLE:
                    event = st.plotly_chart(
                        fig,
                        use_container_width=True,
                        key=f"fig_{tile.id}_{state.level}",
                        on_select="rerun",
                        selection_mode="points",
                    )
                    picked = db.selection_to_filter(
                        draw_tile,
                        {"selection": {"points": list(event.selection.points)}}
                        if event and event.selection
                        else None,
                    )
                    if picked and picked.values:
                        if tile.params.get("drill") and state.level < len(tile.params["drill"]) - 1:
                            drills[tile.id] = db.drill_down(tile, state, picked.values[0])
                            st.rerun()
                        if cross.get(tile.id) != picked:
                            cross[tile.id] = picked
                            st.rerun()
                else:
                    st.plotly_chart(fig, use_container_width=True, key=f"fig_{tile.id}")
                caption = db.caption_for(data, draw_tile)
                if show_captions and caption:
                    st.caption(caption)
                if tile.kind in ("scatter", "map") and len(data) > db.MAX_POINTS:
                    st.caption(f"Showing a sample of {db.MAX_POINTS:,} of {len(data):,} points.")
        except db.DashboardError as err:
            st.warning(str(err))


if not spec.tiles:
    st.info("Add your first tile, or start from a template above.")
for section in spec.sections:
    if section:
        st.subheader(section)
    for row in db.layout_rows([t for t in spec.tiles if t.section == section], spec.columns):
        cols = st.columns([max(1, min(t.width, spec.columns)) for t in row])
        for col, tile in zip(cols, row, strict=True):
            with col:
                render_tile(tile)

# ------------------------------------------------------------------ export & code
st.markdown("---")
st.subheader("Export & share")
cross_list = list(cross.values())
e1, e2, e3, e4 = st.columns(4)
e1.download_button(
    "⬇️ HTML page",
    dx.export_html(tables, spec, cross_list),
    f"{spec.title or 'dashboard'}.html",
    "text/html",
    use_container_width=True,
    disabled=not spec.tiles,
)
e2.download_button(
    "⬇️ Excel (data per tile)",
    dx.export_xlsx(tables, spec, cross_list),
    f"{spec.title or 'dashboard'}.xlsx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    use_container_width=True,
    disabled=not spec.tiles,
)
if e3.button("Prepare PDF", use_container_width=True, disabled=not spec.tiles):
    with st.spinner("Building the PDF..."):
        pdf_bytes, charts_in = dx.export_pdf(tables, spec, cross_list)
    st.session_state["dash_pdf"] = (pdf_bytes, charts_in)
if st.session_state.get("dash_pdf"):
    pdf_bytes, charts_in = st.session_state["dash_pdf"]
    e3.download_button(
        "⬇️ Download PDF",
        pdf_bytes,
        f"{spec.title or 'dashboard'}.pdf",
        "application/pdf",
        use_container_width=True,
    )
    if not charts_in:
        e3.caption("Charts are shown as data tables. Install `kaleido` to include chart images.")
e4.download_button(
    "💾 Layout (JSON)",
    db.spec_to_json(spec),
    "dashboard.json",
    "application/json",
    use_container_width=True,
)

c1, c2 = st.columns(2)
with c1:
    render_code_button(
        spec.title or "Dashboard",
        lambda o: dcode.dashboard_snippet(spec, options=o, **load_kwargs(dataset)),
        key="code_dash",
    )
with c2:
    app_code = streamlit_app_code(spec, dataset.filename)
    render_code_button("Streamlit app", app_code, key="code_dash_app", label="</> Streamlit app")
    st.download_button(
        "⬇️ dashboard_app.py",
        app_code,
        "dashboard_app.py",
        "text/x-python",
        key="dl_dash_app",
        use_container_width=True,
    )

upload = st.file_uploader("Load a saved layout", type=["json"], key="dash_upload")
if upload is not None and st.button("Load this layout"):
    try:
        commit(db.spec_from_json(upload.getvalue()))
    except db.DashboardError as err:
        st.error(str(err))

render_code_export()
