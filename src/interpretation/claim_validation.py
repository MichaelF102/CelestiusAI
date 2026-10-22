"""Numerical Claim Validation Layer.

Deterministic verification ensuring that all quantitative statements made in
analyst findings correspond precisely to verified calculation results in the
active Evidence Pack. Never relies on LLMs to self-validate claims.
"""

import logging
import math
from typing import Any

from src.interpretation.interpretation_models import (
    AnalystFindingClaim,
    ClaimValidationStatus,
    EvidencePack,
)

logger = logging.getLogger(__name__)


class ClaimValidator:
    """Deterministic validator verifying claims against active Evidence Pack."""

    def __init__(self, relative_tolerance: float = 0.02, absolute_tolerance: float = 0.05) -> None:
        self.rel_tol = relative_tolerance
        self.abs_tol = absolute_tolerance

    def validate_claim(
        self,
        claim: AnalystFindingClaim,
        evidence_pack: EvidencePack,
        active_dataset_version: int,
    ) -> AnalystFindingClaim:
        """Validate a single finding claim against the provided Evidence Pack."""
        notes: list[str] = []
        status: ClaimValidationStatus = "unverified"

        # 1. Check for evidence IDs
        if not claim.evidence_ids:
            notes.append("No evidence ID cited for this claim. Marked unverified.")
            claim.validation_status = "unverified"
            claim.validation_notes = notes
            return claim

        # 2. Check existence of cited evidence IDs
        valid_items = []
        for eid in claim.evidence_ids:
            item = evidence_pack.get_item(eid)
            if item is None:
                notes.append(f"Referenced evidence ID '{eid}' does not exist in active pack.")
                claim.validation_status = "invalid_reference"
                claim.validation_notes = notes
                return claim
            valid_items.append(item)

        # 3. Check for stale dataset versions
        for item in valid_items:
            if item.dataset_version != active_dataset_version:
                notes.append(
                    f"Evidence '{item.id}' belongs to dataset version {item.dataset_version}, "
                    f"but active version is {active_dataset_version}."
                )
                claim.validation_status = "stale_version"
                claim.validation_notes = notes
                return claim

        # 4. Compare quantitative reported values if provided
        if claim.reported_values:
            all_matched = True
            for key, rep_val in claim.reported_values.items():
                matched_val = False
                for item in valid_items:
                    actual = item.actual_value
                    if self._values_match(rep_val, actual):
                        matched_val = True
                        break

                if not matched_val:
                    all_matched = False
                    notes.append(
                        f"Reported value '{key}={rep_val}' could not be verified against cited evidence values."
                    )

            if all_matched:
                status = "verified"
                notes.append("All reported numerical values match active deterministic evidence.")
            else:
                status = "mismatched"
        else:
            # Evidence IDs exist and are fresh, but no structured numeric fields were extracted
            status = "verified"
            notes.append("Supporting evidence references confirmed present and fresh.")

        claim.validation_status = status
        claim.validation_notes = notes
        return claim

    def validate_all(
        self,
        claims: list[AnalystFindingClaim],
        evidence_pack: EvidencePack,
        active_dataset_version: int,
    ) -> list[AnalystFindingClaim]:
        """Validate a collection of findings."""
        return [self.validate_claim(c, evidence_pack, active_dataset_version) for c in claims]

    def _values_match(self, reported: Any, actual: Any) -> bool:
        """Deterministic numerical comparison with tolerance for rounding."""
        if reported is None or actual is None:
            return False

        # Numeric to numeric
        try:
            r_num = float(reported)
            if isinstance(actual, (int, float)):
                a_num = float(actual)
                if math.isnan(r_num) or math.isnan(a_num):
                    return False
                return math.isclose(r_num, a_num, rel_tol=self.rel_tol, abs_tol=self.abs_tol)
            elif isinstance(actual, dict):
                # Search if numeric value is in dict values
                for v in actual.values():
                    if isinstance(v, (int, float)) and math.isclose(
                        r_num, float(v), rel_tol=self.rel_tol, abs_tol=self.abs_tol
                    ):
                        return True
        except (ValueError, TypeError):
            pass

        # String to string
        if isinstance(reported, str) and isinstance(actual, str):
            return reported.strip().lower() == actual.strip().lower()

        return False
