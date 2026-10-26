"""Dashboard exports: standalone HTML, Excel workbook, PDF summary, and PNG snapshots."""

from __future__ import annotations

import html
import io
import re
from collections.abc import Mapping
from typing import Any

import pandas as pd

from src.datascience import dashboard as db


# --------------------------------------------------------------------------- helpers
def _tables(df_or_tables: Any) -> dict[str, pd.DataFrame]:
    if isinstance(df_or_tables, pd.DataFrame):
        return {"active": df_or_tables}
    return dict(df_or_tables)


_COLOR_NAMES = {
    "red": "#ef4444",
    "green": "#22c55e",
    "orange": "#f97316",
    "yellow": "#eab308",
    "blue": "#38bdf8",
}


def _css_color(name: str) -> str:
    return _COLOR_NAMES.get(
        name, name if re.fullmatch(r"#[0-9a-fA-F]{3,8}", name or "") else "#ef4444"
    )


def md_to_html(text: str) -> str:
    """Tiny, safe Markdown subset: headings, bold, italics, lists, http(s) links."""
    escaped = html.escape(text)
    out: list[str] = []
    in_list = False
    for raw in escaped.splitlines():
        line = raw.strip()
        if line.startswith("- ") or line.startswith("* "):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{_inline(line[2:])}</li>")
            continue
        if in_list:
            out.append("</ul>")
            in_list = False
        if not line:
            continue
        match = re.match(r"^(#{1,3})\s+(.*)$", line)
        if match:
            level = len(match.group(1)) + 2
            out.append(f"<h{level}>{_inline(match.group(2))}</h{level}>")
        else:
            out.append(f"<p>{_inline(line)}</p>")
    if in_list:
        out.append("</ul>")
    return "\n".join(out)


def _inline(s: str) -> str:
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<!\*)\*(?!\*)(.+?)\*(?!\*)", r"<em>\1</em>", s)
    return re.sub(
        r"\[([^\]]+)\]\((https?://[^\s)]+)\)", r'<a href="\2" rel="noopener noreferrer">\1</a>', s
    )


def format_frame(frame: pd.DataFrame, options: Mapping[str, Any]) -> pd.DataFrame:
    """String-formatted copy of a table using ``options['formats']`` {column: format}."""
    out = frame.copy()
    for col, fmt in (options.get("formats") or {}).items():
        if col in out.columns and pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].map(lambda v, f=fmt: db.format_value(v, f))
    return out


def highlight_style(frame: pd.DataFrame, options: Mapping[str, Any]) -> dict[tuple[int, str], str]:
    """{(row position, column): css colour} for ``options['highlight']`` rules."""
    cells: dict[tuple[int, str], str] = {}
    for rule in options.get("highlight") or []:
        col, op = rule.get("column"), db._OPS.get(rule.get("op", ">"))
        if col not in frame.columns or op is None:
            continue
        nums = pd.to_numeric(frame[col], errors="coerce")
        for pos, hit in enumerate(op(nums.to_numpy(dtype=float), float(rule["value"]))):
            if hit and not pd.isna(nums.iloc[pos]):
                cells[(pos, col)] = _css_color(rule.get("color", "red"))
    return cells


def table_html(frame: pd.DataFrame, options: Mapping[str, Any]) -> str:
    shown = format_frame(frame, options)
    marks = highlight_style(frame, options)
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in shown.columns)
    rows = []
    for pos in range(len(shown)):
        cells = []
        for col in shown.columns:
            style = (
                f' style="background:{marks[(pos, col)]}33;font-weight:600"'
                if (pos, col) in marks
                else ""
            )
            cells.append(f"<td{style}>{html.escape(str(shown.iloc[pos][col]))}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return (
        f'<table class="tbl"><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>'
    )


def _sparkline_svg(series: pd.Series, color: str = "#38bdf8", w: int = 120, h: int = 28) -> str:
    vals = series.dropna().to_numpy(dtype=float)
    if len(vals) < 2:
        return ""
    lo, hi = vals.min(), vals.max()
    span = (hi - lo) or 1.0
    pts = " ".join(
        f"{i * w / (len(vals) - 1):.1f},{h - (v - lo) / span * (h - 4) - 2:.1f}"
        for i, v in enumerate(vals)
    )
    return f'<svg width="{w}" height="{h}" role="img" aria-label="trend"><polyline fill="none" stroke="{color}" stroke-width="2" points="{pts}"/></svg>'


def kpi_html(card: Mapping[str, Any], title: str, accent: str) -> str:
    delta = ""
    if card.get("delta_text"):
        arrow = {"up": "▲", "down": "▼"}.get(card.get("direction") or "", "■")
        tone = {"up": "#22c55e", "down": "#ef4444"}.get(card.get("direction") or "", "#94a3b8")
        delta = (
            f'<div class="d" style="color:{tone}">{arrow} {html.escape(card["delta_text"])}</div>'
        )
    progress = ""
    if card.get("progress") is not None:
        progress = f'<div class="bar"><div style="width:{card["progress"] * 100:.0f}%;background:{accent}"></div></div>'
    note = f'<div class="n">{html.escape(card["note"])}</div>' if card.get("note") else ""
    spark = _sparkline_svg(card["spark"], accent) if card.get("spark") is not None else ""
    color = f' style="color:{_css_color(card["color"])}"' if card.get("color") else ""
    return (
        f'<div class="kpi" role="group" aria-label="{html.escape(title)}"><div class="v"{color}>{html.escape(card["text"])}</div>'
        f'<div class="l">{html.escape(title)}</div>{delta}{progress}{note}{spark}</div>'
    )


# --------------------------------------------------------------------------- HTML
def export_html(
    df_or_tables: Any, spec: db.DashboardSpec, cross_filters: list[db.Filter] | None = None
) -> str:
    """Self-contained HTML page (plotly.js from CDN) reflecting the filtered data."""
    tables = _tables(df_or_tables)
    theme = spec.theme
    dark = theme.template == "dark"
    bg, fg, card_bg, muted = (
        ("#0b0f17", "#f1f5f9", "#131b2e", "#94a3b8")
        if dark
        else ("#f8fafc", "#0f172a", "#ffffff", "#64748b")
    )
    border = f"1px solid {'#263248' if dark else '#e2e8f0'}" if theme.tile_border else "none"
    css = (
        f"body{{font-family:system-ui,sans-serif;margin:24px;background:{bg};color:{fg};font-size:{theme.font_size + 2}px}}"
        f"h1,h2{{margin:.3em 0}}h2{{border-bottom:2px solid {theme.accent};padding-bottom:4px;margin-top:1.4em}}"
        ".grid{display:grid;grid-template-columns:repeat(__COLS__,1fr);gap:16px;margin-top:10px}"
        f".card{{background:{card_bg};border-radius:10px;padding:12px 16px;border:{border}}}"
        f".kpi{{background:{card_bg};border-radius:10px;padding:14px 22px;border:{border};min-width:150px}}"
        f".kpi .v{{font-size:2rem;font-weight:700}}.kpi .l{{color:{muted};font-size:.85rem}}"
        ".kpi .d{font-size:.85rem;margin-top:2px}.kpi .n{font-size:.75rem;color:#94a3b8}"
        f".bar{{height:6px;background:#33415555;border-radius:3px;margin:6px 0}}.bar div{{height:6px;border-radius:3px}}"
        f"figcaption{{color:{muted};font-size:.8rem;margin-top:4px}}figure{{margin:0}}"
        f".tbl{{border-collapse:collapse;font-size:.85rem}}.tbl td,.tbl th{{padding:4px 10px;border-bottom:1px solid {'#263248' if dark else '#e2e8f0'};text-align:left}}"
        ".kpis{display:flex;gap:16px;flex-wrap:wrap}"
    ).replace("__COLS__", str(max(1, spec.columns)))

    first_chart = True
    body: list[str] = []
    for section in spec.sections:
        if section:
            body.append(f"<h2>{html.escape(section)}</h2>")
        kpis, items = [], []
        for tile in (t for t in spec.tiles if t.section == section):
            resolved = db.resolve_tile(tile, spec)
            span = f' style="grid-column: span {max(1, min(tile.width, spec.columns))}"'
            try:
                data = db.tile_data(tables, spec, resolved, cross_filters)
                if resolved.kind == "kpi":
                    kpis.append(kpi_html(db.kpi_card(data, resolved), resolved.title, theme.accent))
                    continue
                caption = db.caption_for(data, resolved)
                if resolved.kind == "text":
                    inner = md_to_html(str(resolved.params.get("markdown", "")))
                    items.append(f'<section class="card"{span}>{inner}</section>')
                    continue
                if resolved.kind == "table":
                    inner = table_html(db.tile_frame(data, resolved), resolved.options)
                else:
                    fig = db.build_figure(data, resolved, theme=theme)
                    inner = fig.to_html(
                        full_html=False, include_plotlyjs="cdn" if first_chart else False
                    )
                    first_chart = False
                cap = f"<figcaption>{html.escape(caption)}</figcaption>" if caption else ""
                items.append(
                    f'<section class="card"{span}><figure role="img" aria-label="{html.escape(resolved.title + ". " + caption)}">'
                    f"<h3>{html.escape(resolved.title)}</h3>{inner}{cap}</figure></section>"
                )
            except db.DashboardError as err:
                items.append(
                    f'<section class="card"{span}><h3>{html.escape(tile.title)}</h3><p><em>{html.escape(str(err))}</em></p></section>'
                )
        if kpis:
            body.append(f'<div class="kpis">{"".join(kpis)}</div>')
        if items:
            body.append(f'<div class="grid">{"".join(items)}</div>')

    base = tables.get("active", next(iter(tables.values()), pd.DataFrame()))
    shown = len(db.apply_filters(base, spec.filters))
    flt = ", ".join(html.escape(f.column) for f in spec.filters if f.column in base.columns)
    desc = f"<p>{html.escape(spec.description)}</p>" if spec.description else ""
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        f"<title>{html.escape(spec.title)}</title><style>{css}</style></head><body>"
        f"<h1>{html.escape(spec.title)}</h1>{desc}"
        f"<p style='color:{muted}'>{shown:,} of {len(base):,} rows{' (filtered on: ' + flt + ')' if flt else ''}</p>"
        + "".join(body)
        + "</body></html>"
    )


# --------------------------------------------------------------------------- Excel
def _sheet_name(name: str, used: set[str]) -> str:
    base = re.sub(r"[\[\]:*?/\\]", "_", name).strip()[:28] or "Sheet"
    candidate, n = base, 2
    while candidate.lower() in used:
        candidate = f"{base[:25]}_{n}"
        n += 1
    used.add(candidate.lower())
    return candidate


def export_xlsx(
    df_or_tables: Any, spec: db.DashboardSpec, cross_filters: list[db.Filter] | None = None
) -> bytes:
    """Workbook with one sheet per data-backed tile (the exact rows behind each chart)."""
    tables = _tables(df_or_tables)
    used: set[str] = {"summary"}
    summary_rows = []
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for tile in spec.tiles:
            resolved = db.resolve_tile(tile, spec)
            if resolved.kind == "text":
                continue
            try:
                data = db.tile_data(tables, spec, resolved, cross_filters)
                frame = db.tile_frame(data, resolved)
                if (
                    isinstance(frame.index, pd.MultiIndex)
                    or frame.index.name
                    or resolved.kind == "heatmap"
                ):
                    frame = frame.reset_index()
                frame.columns = [str(c) for c in frame.columns]
                name = _sheet_name(resolved.title, used)
                frame.to_excel(writer, sheet_name=name, index=False)
                summary_rows.append(
                    {
                        "tile": resolved.title,
                        "type": resolved.kind,
                        "rows": len(frame),
                        "sheet": name,
                        "caption": db.caption_for(data, resolved),
                    }
                )
            except db.DashboardError as err:
                summary_rows.append(
                    {
                        "tile": resolved.title,
                        "type": resolved.kind,
                        "rows": 0,
                        "sheet": "",
                        "caption": f"skipped: {err}",
                    }
                )
        filters = [
            {
                "filter": f.column,
                "kind": f.kind,
                "values": str(
                    f.values or f.preset or f.contains or (f.low, f.high) or (f.start, f.end)
                ),
            }
            for f in spec.filters
        ]
        pd.DataFrame(summary_rows or [{"tile": "(no data tiles)"}]).to_excel(
            writer, sheet_name="Summary", index=False
        )
        if filters:
            pd.DataFrame(filters).to_excel(
                writer, sheet_name="Summary", index=False, startrow=len(summary_rows) + 3
            )
        writer.book.move_sheet("Summary", offset=-len(writer.book.sheetnames) + 1)
    return buffer.getvalue()


# --------------------------------------------------------------------------- PNG / PDF
def figure_png(fig: Any, width: int = 900, height: int = 450) -> bytes | None:
    """PNG bytes via kaleido, or None when static export is unavailable."""
    try:
        return fig.to_image(format="png", width=width, height=height, scale=2)
    except Exception:
        return None


def export_pdf(
    df_or_tables: Any,
    spec: db.DashboardSpec,
    cross_filters: list[db.Filter] | None = None,
    with_images: bool = True,
) -> tuple[bytes, bool]:
    """PDF summary: KPIs, captions, data tables, and (if kaleido works) chart images.

    Returns (pdf bytes, charts_included).
    """
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    tables = _tables(df_or_tables)
    styles = getSampleStyleSheet()
    story: list[Any] = [Paragraph(html.escape(spec.title), styles["Title"])]
    if spec.description:
        story.append(Paragraph(html.escape(spec.description), styles["Normal"]))
    base = tables.get("active", next(iter(tables.values()), pd.DataFrame()))
    story += [
        Paragraph(
            f"{len(db.apply_filters(base, spec.filters)):,} of {len(base):,} rows", styles["Italic"]
        ),
        Spacer(1, 0.4 * cm),
    ]
    charts = False
    for section in spec.sections:
        if section:
            story.append(Paragraph(html.escape(section), styles["Heading2"]))
        for tile in (t for t in spec.tiles if t.section == section):
            resolved = db.resolve_tile(tile, spec)
            try:
                data = db.tile_data(tables, spec, resolved, cross_filters)
                if resolved.kind == "text":
                    story.append(
                        Paragraph(
                            html.escape(str(resolved.params.get("markdown", ""))).replace(
                                "\n", "<br/>"
                            ),
                            styles["Normal"],
                        )
                    )
                    continue
                story.append(Paragraph(f"<b>{html.escape(resolved.title)}</b>", styles["Heading3"]))
                caption = db.caption_for(data, resolved)
                if caption:
                    story.append(Paragraph(html.escape(caption), styles["Italic"]))
                if resolved.kind in ("kpi",):
                    continue
                fig = (
                    db.build_figure(data, resolved, template="plotly_white")
                    if with_images
                    else None
                )
                png = figure_png(fig) if fig is not None else None
                if png:
                    charts = True
                    story.append(Image(io.BytesIO(png), width=16 * cm, height=8 * cm))
                else:
                    frame = db.tile_frame(data, resolved)
                    if resolved.kind == "heatmap":
                        frame = frame.reset_index()
                    head = frame.head(12)
                    rows = [[str(c) for c in head.columns]] + [
                        [f"{v:,.4g}" if isinstance(v, float) else str(v) for v in r]
                        for r in head.itertuples(index=False)
                    ]
                    table = Table(rows, repeatRows=1)
                    table.setStyle(
                        TableStyle(
                            [
                                ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                                ("FONTSIZE", (0, 0), (-1, -1), 7),
                            ]
                        )
                    )
                    story.append(table)
                story.append(Spacer(1, 0.3 * cm))
            except db.DashboardError as err:
                story.append(
                    Paragraph(
                        f"{html.escape(tile.title)}: {html.escape(str(err))}", styles["Italic"]
                    )
                )
    buffer = io.BytesIO()
    SimpleDocTemplate(buffer, pagesize=A4, title=spec.title).build(story)
    return buffer.getvalue(), charts
