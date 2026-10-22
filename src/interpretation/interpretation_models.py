"""Validated Pydantic data schemas for the Evidence-Grounded AI Interpretation Engine.

Distinguishes:
1. Observed evidence (deterministic calculations with stable IDs)
2. Statistical inference (formal test results and estimates)
3. Interpretation (model-generated analytical reasoning)
4. Hypotheses (testable future conjectures)
5. Recommendations (prioritized next analytical steps)
"""

import uuid
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

EvidenceType = Literal[
    "dataset_overview",
    "descriptive_statistic",
    "group_comparison",
    "correlation",
    "hypothesis_test",
    "regression",
    "data_quality",
    "distribution",
    "temporal",
    "metadata",
]

ClaimValidationStatus = Literal[
    "verified",
    "unverified",
    "mismatched",
    "stale_version",
    "invalid_reference",
]


class EvidenceItem(BaseModel):
    """Atomic unit of deterministic analytical evidence with verifiable provenance."""

    id: str = Field(description="Unique stable identifier, e.g. ev_desc_revenue_mean")
    evidence_type: EvidenceType = Field(description="Category of deterministic evidence")
    dataset_version: int = Field(description="Version of working dataset at computation time")
    source_service: str = Field(description="Service that calculated this result")
    calculation_method: str = Field(description="Statistical or computational method used")
    relevant_variables: list[str] = Field(default_factory=list, description="Columns involved")
    metric_name: str = Field(description="Name of the computed metric")
    actual_value: Any = Field(description="Exact computed numerical or structured value")
    formatted_value: str = Field(description="Human-readable representation")
    units: str | None = Field(default=None, description="Physical or currency units if known")
    sample_size: int | None = Field(default=None, description="Sample size N used")
    caveats: list[str] = Field(default_factory=list, description="Assumptions or warnings")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="UTC creation timestamp",
    )


class EvidencePack(BaseModel):
    """Scoped, token-efficient collection of verified analytical evidence for a query."""

    pack_id: str = Field(default_factory=lambda: f"pack_{uuid.uuid4().hex[:8]}")
    dataset_id: str
    dataset_name: str
    dataset_version: int
    row_count: int
    column_count: int
    column_types: dict[str, str] = Field(default_factory=dict)
    quality_warnings: list[str] = Field(default_factory=list)
    items: list[EvidenceItem] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

    def get_item(self, item_id: str) -> EvidenceItem | None:
        """Find an evidence item by its stable ID."""
        for item in self.items:
            if item.id == item_id:
                return item
        return None

    def item_ids(self) -> set[str]:
        """Return the set of all evidence IDs in this pack."""
        return {item.id for item in self.items}

    def summary_for_prompt(self, max_items: int = 40) -> str:
        """Render a compact, structured text representation for LLM ingestion."""
        lines = [
            f"=== EVIDENCE PACK [{self.pack_id}] ===",
            f"Dataset: '{self.dataset_name}' (Version: {self.dataset_version})",
            f"Dimensions: {self.row_count:,} rows x {self.column_count} columns",
        ]
        if self.quality_warnings:
            lines.append("Data Quality Warnings:")
            for w in self.quality_warnings[:5]:
                lines.append(f"  * {w}")

        lines.append("\nComputed Evidence Items (Deterministic & Verified):")
        for idx, item in enumerate(self.items[:max_items], 1):
            vars_str = (
                f" [vars: {', '.join(item.relevant_variables)}]" if item.relevant_variables else ""
            )
            n_str = f" (N={item.sample_size:,})" if item.sample_size else ""
            lines.append(
                f"{idx}. [{item.id}] ({item.evidence_type}): {item.metric_name} = "
                f"{item.formatted_value}{vars_str}{n_str} (Method: {item.calculation_method})"
            )
            if item.caveats:
                lines.append(f"   Caveats: {'; '.join(item.caveats)}")

        if len(self.items) > max_items:
            lines.append(f"... and {len(self.items) - max_items} additional verified items.")

        lines.append("=== END EVIDENCE PACK ===")
        return "\n".join(lines)


class AnalystFindingClaim(BaseModel):
    """Specific, evidence-backed finding returned by the analyst."""

    claim_id: str = Field(default_factory=lambda: f"claim_{uuid.uuid4().hex[:6]}")
    claim: str = Field(description="Statement of fact or observed finding")
    evidence_ids: list[str] = Field(
        default_factory=list,
        description="IDs of supporting evidence items from the pack",
    )
    reported_values: dict[str, Any] = Field(
        default_factory=dict,
        description="Specific numbers cited in the claim, e.g. {'mean_revenue': 45200.5}",
    )
    relevant_variables: list[str] = Field(default_factory=list)
    sample_size: int | None = None
    method: str | None = None
    caveats: list[str] = Field(default_factory=list)
    validation_status: ClaimValidationStatus = Field(
        default="unverified",
        description="Veracity status evaluated by claim validator",
    )
    validation_notes: list[str] = Field(default_factory=list)


class HypothesisProposal(BaseModel):
    """Testable future hypothesis generated from data patterns, clearly labeled as unverified."""

    hypothesis_statement: str = Field(description="Testable proposition")
    rationale: str = Field(description="Data patterns motivating this conjecture")
    required_variables: list[str] = Field(default_factory=list)
    suggested_method: str = Field(description="Proposed statistical test or modeling analysis")
    expected_support_evidence: str = Field(description="What result would support this hypothesis")
    expected_weaken_evidence: str = Field(description="What result would weaken this hypothesis")
    is_exploratory: bool = Field(
        default=True,
        description="True = generated post-hoc from data; False = pre-specified",
    )
    known_limitations: list[str] = Field(default_factory=list)


class ContradictionFlag(BaseModel):
    """Flagged conflict between narrative claims and underlying evidence or sound logic."""

    flag_id: str = Field(default_factory=lambda: f"flag_{uuid.uuid4().hex[:6]}")
    severity: Literal["critical", "warning", "info"] = "warning"
    description: str
    conflicting_claim: str
    conflicting_evidence_id: str | None = None
    resolved: bool = False


class TraceMetadata(BaseModel):
    """Audit metadata recording provider, model, execution time, and tool traces."""

    provider: str
    model: str
    execution_time_ms: float
    evidence_ids_used: list[str] = Field(default_factory=list)
    tool_executions: list[dict[str, Any]] = Field(default_factory=list)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class AnalystResponse(BaseModel):
    """Complete, validated structured response adhering to the Deep Interpretation Framework."""

    question: str
    intent_category: str = Field(
        default="exploratory",
        description="Classified analytical category (e.g. group_comparison, descriptive)",
    )
    direct_answer: str = Field(description="Concise direct answer to the user query")
    key_findings: list[AnalystFindingClaim] = Field(
        default_factory=list,
        description="Structured, evidence-linked findings",
    )
    interpretation: str = Field(
        default="",
        description="Deeper explanation of mechanisms, practical relevance, and context",
    )
    alternative_explanations: list[str] = Field(
        default_factory=list,
        description="Competing explanations (confounding, selection, missingness)",
    )
    uncertainty_and_limitations: list[str] = Field(
        default_factory=list,
        description="Statistical caveats, data limitations, and scope restrictions",
    )
    recommended_next_analyses: list[str] = Field(
        default_factory=list,
        description="Prioritized concrete follow-up analyses",
    )
    hypotheses: list[HypothesisProposal] = Field(
        default_factory=list,
        description="Testable exploratory hypotheses",
    )
    contradictions_detected: list[ContradictionFlag] = Field(
        default_factory=list,
        description="Identified inconsistencies between claims and evidence",
    )
    trace_metadata: TraceMetadata
    dataset_version: int
