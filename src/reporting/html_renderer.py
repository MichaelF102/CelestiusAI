"""HTML Report Renderer.

Renders ReportDocument into a self-contained, beautifully styled HTML document
with responsive executive design and print stylesheets.
"""

from __future__ import annotations

import html

from src.reporting.report_models import ReportDocument, ReportSection, ReportTable


def render_report_to_html(doc: ReportDocument) -> str:
    """Render a ReportDocument to a complete standalone HTML string."""
    meta = doc.metadata
    gen_time_str = meta.generated_at.strftime("%Y-%m-%d %H:%M UTC")

    # Render sections
    sections_html = []
    for sec in doc.sections:
        sections_html.append(_render_section_html(sec))

    # Render limitations
    limitations_html = "".join(
        f"<li>{html.escape(lim)}</li>" for lim in doc.limitations_and_assumptions
    )

    # Render unperformed
    unperformed_html = ""
    if doc.unperformed_analyses:
        unperformed_items = "".join(f"<li>{html.escape(u)}</li>" for u in doc.unperformed_analyses)
        unperformed_html = f"""
        <div class="card warning-card">
            <h3>Analyses Not Executed</h3>
            <p>The following analytical modules were omitted because their corresponding workflows have not been run on this dataset version:</p>
            <ul>{unperformed_items}</ul>
        </div>
        """

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(meta.title)}</title>
    <style>
        :root {{
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --bg-card: #1e293b;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --accent: #38bdf8;
            --accent-gradient: linear-gradient(135deg, #38bdf8, #818cf8);
            --border: #334155;
            --success: #34d399;
            --warning: #fbbf24;
        }}
        @media print {{
            :root {{
                --bg-primary: #ffffff;
                --bg-secondary: #f8fafc;
                --bg-card: #ffffff;
                --text-primary: #0f172a;
                --text-secondary: #475569;
                --accent: #0284c7;
                --border: #cbd5e1;
            }}
            body {{ font-size: 11pt; }}
            .no-print {{ display: none; }}
            .page-break {{ page-break-after: always; }}
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            margin: 0;
            padding: 2rem;
            line-height: 1.6;
        }}
        .container {{
            max-width: 960px;
            margin: 0 auto;
        }}
        header {{
            border-bottom: 2px solid var(--border);
            padding-bottom: 1.5rem;
            margin-bottom: 2rem;
        }}
        .badge {{
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.85rem;
            font-weight: 600;
            background: rgba(56, 189, 248, 0.15);
            color: var(--accent);
            border: 1px solid var(--accent);
            margin-bottom: 0.5rem;
        }}
        h1 {{
            font-size: 2.2rem;
            margin: 0.5rem 0;
            background: var(--accent-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .meta-bar {{
            display: flex;
            flex-wrap: wrap;
            gap: 1.5rem;
            color: var(--text-secondary);
            font-size: 0.9rem;
            margin-top: 0.5rem;
        }}
        .card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }}
        .warning-card {{
            border-left: 4px solid var(--warning);
            background: rgba(251, 191, 36, 0.05);
        }}
        h2 {{
            font-size: 1.4rem;
            color: var(--accent);
            margin-top: 0;
            border-bottom: 1px solid var(--border);
            padding-bottom: 0.5rem;
        }}
        h3 {{
            font-size: 1.15rem;
            margin-top: 1rem;
            color: var(--text-primary);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
            font-size: 0.9rem;
        }}
        th, td {{
            padding: 0.6rem 0.8rem;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}
        th {{
            background: rgba(255, 255, 255, 0.03);
            color: var(--accent);
            font-weight: 600;
        }}
        tr:hover td {{
            background: rgba(255, 255, 255, 0.02);
        }}
        footer {{
            margin-top: 3rem;
            border-top: 1px solid var(--border);
            padding-top: 1.5rem;
            font-size: 0.85rem;
            color: var(--text-secondary);
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="badge">{html.escape(meta.report_type.value)}</div>
            <h1>{html.escape(meta.title)}</h1>
            <div class="meta-bar">
                <div><strong>Report ID:</strong> {html.escape(meta.report_id)}</div>
                <div><strong>Dataset:</strong> {html.escape(meta.dataset_id)} (v{meta.dataset_version})</div>
                <div><strong>Generated:</strong> {gen_time_str}</div>
                <div><strong>Depth:</strong> {html.escape(meta.report_depth.value)}</div>
                <div><strong>Evidence Items:</strong> {meta.evidence_item_count}</div>
            </div>
        </header>

        <section class="card">
            <h2>Executive Summary</h2>
            <p>{html.escape(doc.executive_summary).replace("\n", "<br>")}</p>
        </section>

        {"".join(sections_html)}

        {unperformed_html}

        <section class="card">
            <h2>Methodology, Assumptions & Limitations</h2>
            <ul>{limitations_html}</ul>
        </section>

        <footer>
            Generated deterministically by <strong>AI Data Analyst Agent</strong> &bull; All quantitative findings backed by verified evidence items.
        </footer>
    </div>
</body>
</html>
"""
    return html_content


def _render_section_html(sec: ReportSection) -> str:
    """Render an individual section to HTML."""
    if not sec.was_performed:
        return f"""
        <section class="card" style="opacity: 0.7;">
            <h2>{html.escape(sec.title)}</h2>
            <p><em>{html.escape(sec.omission_reason or "Analysis not performed.")}</em></p>
        </section>
        """

    tables_html = "".join(_render_table_html(tbl) for tbl in sec.tables)

    return f"""
    <section class="card">
        <h2>{html.escape(sec.title)}</h2>
        <p>{html.escape(sec.summary_markdown).replace("\n", "<br>")}</p>
        {tables_html}
    </section>
    """


def _render_table_html(tbl: ReportTable) -> str:
    """Render a ReportTable to HTML table element."""
    header_cols = "".join(f"<th>{html.escape(h)}</th>" for h in tbl.headers)
    rows_html = []
    for row in tbl.rows:
        tds = "".join(f"<td>{html.escape(str(cell))}</td>" for cell in row)
        rows_html.append(f"<tr>{tds}</tr>")

    caption_html = f"<div style='font-size: 0.8rem; color: var(--text-secondary); margin-bottom: 0.4rem;'>{html.escape(tbl.title)}</div>"
    return f"""
    <div style="margin: 1.2rem 0;">
        {caption_html}
        <table>
            <thead><tr>{header_cols}</tr></thead>
            <tbody>{"".join(rows_html)}</tbody>
        </table>
    </div>
    """
