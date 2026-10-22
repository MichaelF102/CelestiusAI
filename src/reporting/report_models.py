"""Report data models and schemas.

Defines report types, depths, sections, evidence bindings, and document containers.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ReportType(StrEnum):
    """Supported intelligence report categories."""

    EXECUTIVE = "Executive Analytics Report"
    EDA = "Exploratory Data Analysis Report"
    STATISTICAL = "Statistical Intelligence Report"
    INVESTIGATION = "AI Investigation Report"
    PREDICTIVE = "Predictive Model Evaluation Report"
    COMPREHENSIVE = "Comprehensive Dataset Intelligence Report"


class ReportDepth(StrEnum):
    """Depth level for generated report content."""

    SUMMARY = "Executive Summary"
    STANDARD = "Standard Briefing"
    DETAILED = "Comprehensive Technical Deep Dive"


class ReportTable(BaseModel):
    """Structured tabular data for inclusion in reports."""

    title: str
    headers: list[str]
    rows: list[list[Any]]
    caption: str | None = None


class ReportSection(BaseModel):
    """An individual section within an intelligence report."""

    section_id: str
    title: str
    category: str
    summary_markdown: str
    detailed_markdown: str | None = None
    tables: list[ReportTable] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    was_performed: bool = True
    omission_reason: str | None = None


class ReportMetadata(BaseModel):
    """Metadata describing report generation, provenance, and parameters."""

    report_id: str
    title: str
    report_type: ReportType
    report_depth: ReportDepth
    dataset_id: str
    dataset_version: int
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    generator: str = "AI Data Analyst Agent"
    ai_narrative_applied: bool = False
    evidence_item_count: int = 0


class ReportDocument(BaseModel):
    """Complete assembled report ready for export and presentation."""

    metadata: ReportMetadata
    executive_summary: str
    sections: list[ReportSection] = Field(default_factory=list)
    limitations_and_assumptions: list[str] = Field(default_factory=list)
    unperformed_analyses: list[str] = Field(default_factory=list)

    def get_performed_sections(self) -> list[ReportSection]:
        """Return only sections that contain verified completed analysis."""
        return [s for s in self.sections if s.was_performed]
