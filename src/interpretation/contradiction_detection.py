"""Contradiction and Reasoning Conflict Detection Layer.

Deterministically scans analytical answers and findings for:
1. Spurious causal assertions on observational/correlational evidence.
2. Inverted directional claims (e.g. claiming increase when metric is negative).
3. False significance claims (claiming p < 0.05 when p >= 0.05).
4. Stale evidence version references.
5. Inconsistencies between stated conclusions and underlying evidence.
"""

import logging
import re

from src.interpretation.interpretation_models import (
    AnalystResponse,
    ContradictionFlag,
    EvidencePack,
)

logger = logging.getLogger(__name__)

CAUSAL_PHRASES = [
    r"\bcauses?\b",
    r"\bcaused by\b",
    r"\bleads?\s+to\b",
    r"\bproves causality\b",
    r"\bcausal relationship\b",
    r"\bdirectly produces?\b",
]


class ContradictionDetector:
    """Deterministic rule-based detector for reasoning and quantitative contradictions."""

    def detect_contradictions(
        self,
        response: AnalystResponse,
        evidence_pack: EvidencePack,
        active_dataset_version: int,
    ) -> list[ContradictionFlag]:
        """Inspect response and return any flagged contradictions or logical fallacies."""
        flags: list[ContradictionFlag] = []

        all_text = " ".join(
            [
                response.direct_answer,
                response.interpretation,
                " ".join(f.claim for f in response.key_findings),
            ]
        ).lower()

        # 1. Check for unjustified causal language on observational evidence
        has_correlation_only = any(
            item.evidence_type in ("correlation", "regression", "descriptive_statistic")
            for item in evidence_pack.items
        )
        if has_correlation_only:
            for phrase in CAUSAL_PHRASES:
                if re.search(phrase, all_text):
                    flags.append(
                        ContradictionFlag(
                            severity="warning",
                            description=(
                                f"Detected potential correlation-to-causation fallacy (phrase: '{phrase.strip(r'\b')}'). "
                                "Observational data and regression models establish conditional associations, not proven causality."
                            ),
                            conflicting_claim=f"Text matching pattern: {phrase}",
                        )
                    )
                    break

        # 2. Check significance vs p-value
        for finding in response.key_findings:
            claim_lower = finding.claim.lower()
            claims_significant = (
                "significant" in claim_lower and "not significant" not in claim_lower
            )
            claims_not_significant = (
                "not significant" in claim_lower or "no significant" in claim_lower
            )

            for eid in finding.evidence_ids:
                item = evidence_pack.get_item(eid)
                if not item:
                    continue

                if "p_value" in item.metric_name or "pvalue" in item.metric_name:
                    p_val = float(item.actual_value)
                    if claims_significant and p_val >= 0.05:
                        flags.append(
                            ContradictionFlag(
                                severity="critical",
                                description=(
                                    f"Claim asserts statistical significance, but cited evidence '{eid}' "
                                    f"has p-value = {p_val:.4f} (>= 0.05)."
                                ),
                                conflicting_claim=finding.claim,
                                conflicting_evidence_id=eid,
                            )
                        )
                    elif claims_not_significant and p_val < 0.001:
                        flags.append(
                            ContradictionFlag(
                                severity="warning",
                                description=(
                                    f"Claim asserts lack of significance, but cited evidence '{eid}' "
                                    f"has highly significant p-value = {p_val:.4e}."
                                ),
                                conflicting_claim=finding.claim,
                                conflicting_evidence_id=eid,
                            )
                        )

                # 3. Check directional conflict
                if "negative" in claim_lower or "decrease" in claim_lower or "lower" in claim_lower:
                    if isinstance(item.actual_value, (int, float)) and item.actual_value > 0.05:
                        if (
                            "coef" in item.metric_name
                            or "diff" in item.metric_name
                            or "r" in item.metric_name
                        ):
                            flags.append(
                                ContradictionFlag(
                                    severity="critical",
                                    description=(
                                        f"Claim asserts decrease/negative trend, but metric '{item.metric_name}' "
                                        f"has positive value: {item.actual_value}."
                                    ),
                                    conflicting_claim=finding.claim,
                                    conflicting_evidence_id=eid,
                                )
                            )

                elif (
                    "positive" in claim_lower
                    or "increase" in claim_lower
                    or "higher" in claim_lower
                ):
                    if isinstance(item.actual_value, (int, float)) and item.actual_value < -0.05:
                        if (
                            "coef" in item.metric_name
                            or "diff" in item.metric_name
                            or "r" in item.metric_name
                        ):
                            flags.append(
                                ContradictionFlag(
                                    severity="critical",
                                    description=(
                                        f"Claim asserts increase/positive trend, but metric '{item.metric_name}' "
                                        f"has negative value: {item.actual_value}."
                                    ),
                                    conflicting_claim=finding.claim,
                                    conflicting_evidence_id=eid,
                                )
                            )

        # 4. Check for stale dataset versions
        for finding in response.key_findings:
            if finding.validation_status == "stale_version":
                flags.append(
                    ContradictionFlag(
                        severity="critical",
                        description=(
                            f"Finding '{finding.claim}' references evidence from an older dataset version. "
                            f"Active version is {active_dataset_version}."
                        ),
                        conflicting_claim=finding.claim,
                    )
                )

        return flags
