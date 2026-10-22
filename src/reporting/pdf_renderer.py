"""PDF Report Renderer.

Generates professional PDF intelligence reports using ReportLab.
Handles page layout, typography, tabular formatting, and numbered canvases.
"""

from __future__ import annotations

import io
import logging

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from src.reporting.report_models import ReportDocument, ReportTable

logger = logging.getLogger(__name__)


def render_report_to_pdf(doc: ReportDocument) -> bytes:
    """Render a ReportDocument to binary PDF bytes."""
    buffer = io.BytesIO()
    pdf = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=6,
    )
    meta_style = ParagraphStyle(
        "DocMeta",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#475569"),
        spaceAfter=12,
    )
    h2_style = ParagraphStyle(
        "SectionH2",
        parent=styles["Heading2"],
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#0284c7"),
        spaceBefore=14,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=6,
    )
    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#0f172a"),
    )
    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontSize=8.5,
        leading=11,
        fontName="Helvetica-Bold",
        textColor=colors.HexColor("#ffffff"),
    )

    story = []

    # Title & Header
    meta = doc.metadata
    story.append(Paragraph(meta.title, title_style))
    meta_text = (
        f"<b>Report Type:</b> {meta.report_type.value} &nbsp;|&nbsp; "
        f"<b>Dataset:</b> {meta.dataset_id} (v{meta.dataset_version}) &nbsp;|&nbsp; "
        f"<b>Generated:</b> {meta.generated_at.strftime('%Y-%m-%d %H:%M UTC')}"
    )
    story.append(Paragraph(meta_text, meta_style))
    story.append(
        HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0284c7"), spaceAfter=14)
    )

    # Executive Summary
    story.append(Paragraph("Executive Summary", h2_style))
    story.append(Paragraph(doc.executive_summary.replace("\n", "<br/>"), body_style))
    story.append(Spacer(1, 8))

    # Sections
    for sec in doc.sections:
        story.append(Paragraph(sec.title, h2_style))
        if not sec.was_performed:
            story.append(
                Paragraph(f"<i>{sec.omission_reason or 'Analysis not performed.'}</i>", body_style)
            )
            continue

        story.append(Paragraph(sec.summary_markdown.replace("\n", "<br/>"), body_style))

        # Tables
        for tbl in sec.tables:
            story.append(Spacer(1, 4))
            table_flowable = _build_pdf_table(tbl, table_header_style, table_cell_style)
            if table_flowable:
                story.append(table_flowable)
                story.append(Spacer(1, 6))

    # Unperformed analyses
    if doc.unperformed_analyses:
        story.append(Paragraph("Analyses Omitted / Not Executed", h2_style))
        for u in doc.unperformed_analyses:
            story.append(Paragraph(f"&bull; {u}", body_style))

    # Methodology & Limitations
    story.append(Paragraph("Methodology, Assumptions & Limitations", h2_style))
    for lim in doc.limitations_and_assumptions:
        story.append(Paragraph(f"&bull; {lim}", body_style))

    story.append(Spacer(1, 14))
    story.append(
        HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cbd5e1"), spaceAfter=8)
    )
    footer_text = "Generated deterministically by AI Data Analyst Agent. All claims verifiable against dataset evidence items."
    story.append(Paragraph(f"<i>{footer_text}</i>", meta_style))

    pdf.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def _build_pdf_table(
    tbl: ReportTable, header_style: ParagraphStyle, cell_style: ParagraphStyle
) -> Table | None:
    """Convert ReportTable to a ReportLab Platypus Table."""
    if not tbl.headers or not tbl.rows:
        return None

    data = []
    # Header row
    header_row = [Paragraph(str(h), header_style) for h in tbl.headers]
    data.append(header_row)

    # Data rows (max 15 rows to fit PDF gracefully)
    for row in tbl.rows[:15]:
        data_row = [Paragraph(str(c), cell_style) for c in row]
        data.append(data_row)

    t = Table(data, hAlign="LEFT")
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0284c7")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [colors.HexColor("#f8fafc"), colors.HexColor("#ffffff")],
                ),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ]
        )
    )
    return t
