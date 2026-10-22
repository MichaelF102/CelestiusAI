"""Export Service.

Coordinates rendering ReportDocument into HTML and PDF export formats,
providing graceful degradation if PDF generation fails.
"""

from __future__ import annotations

import logging

from src.reporting.html_renderer import render_report_to_html
from src.reporting.pdf_renderer import render_report_to_pdf
from src.reporting.report_models import ReportDocument

logger = logging.getLogger(__name__)


class ExportService:
    """Manages document export pipelines for HTML and PDF outputs."""

    @staticmethod
    def export_html(doc: ReportDocument) -> str:
        """Render document to HTML string."""
        return render_report_to_html(doc)

    @staticmethod
    def export_pdf(doc: ReportDocument) -> tuple[bytes | None, str | None]:
        """Render document to binary PDF.

        Returns (pdf_bytes, error_message).
        If rendering fails, returns (None, error_message) so HTML can be used.
        """
        try:
            pdf_bytes = render_report_to_pdf(doc)
            return pdf_bytes, None
        except Exception as exc:
            logger.error(f"PDF rendering failed: {exc}", exc_info=True)
            return (
                None,
                f"PDF export encountered an error: {str(exc)}. Please download the HTML version instead.",
            )
