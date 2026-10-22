"""Exploratory Hypothesis Generation Service.

Derives testable exploratory hypotheses from deterministic evidence items,
distribution anomalies, and correlation patterns. Explicitly marks them as
post-hoc exploratory propositions requiring independent confirmatory validation.
"""

from src.interpretation.interpretation_models import (
    EvidencePack,
    HypothesisProposal,
)


class HypothesisGenerator:
    """Generates testable candidate hypotheses grounded in observed evidence."""

    @staticmethod
    def generate_hypotheses_from_evidence(
        evidence_pack: EvidencePack,
        max_hypotheses: int = 3,
    ) -> list[HypothesisProposal]:
        """Produce structured hypothesis proposals from active evidence items."""
        proposals: list[HypothesisProposal] = []

        # 1. Inspect correlation evidence
        corr_items = [
            item
            for item in evidence_pack.items
            if item.evidence_type == "correlation" or "r" in item.metric_name
        ]
        for c in corr_items:
            vars_involved = c.relevant_variables
            if len(vars_involved) >= 2:
                v1, v2 = vars_involved[0], vars_involved[1]
                proposals.append(
                    HypothesisProposal(
                        hypothesis_statement=(
                            f"The observed correlation between '{v1}' and '{v2}' is driven by an underlying "
                            f"latent or confounding factor rather than a direct relationship."
                        ),
                        rationale=f"Observed correlation value: {c.formatted_value} [{c.id}].",
                        required_variables=[v1, v2],
                        suggested_method="Multiple regression or partial correlation controlling for other covariates.",
                        expected_support_evidence=f"Partial correlation between '{v1}' and '{v2}' approaches zero after conditioning on covariates.",
                        expected_weaken_evidence="Association remains statistically significant (p < 0.05) and stable in magnitude.",
                        is_exploratory=True,
                        known_limitations=["Observational data cannot establish causal direction."],
                    )
                )

        # 2. Inspect group difference evidence
        comp_items = [
            item
            for item in evidence_pack.items
            if item.evidence_type in ("group_comparison", "hypothesis_test")
        ]
        for g in comp_items:
            vars_involved = g.relevant_variables
            if vars_involved:
                metric = vars_involved[0]
                proposals.append(
                    HypothesisProposal(
                        hypothesis_statement=(
                            f"Subgroup disparities observed in '{metric}' reflect structural segment differences "
                            f"that persist across time periods."
                        ),
                        rationale=f"Observed statistical disparity: {g.formatted_value} [{g.id}].",
                        required_variables=vars_involved,
                        suggested_method="Longitudinal panel regression or two-way ANOVA across time and segment.",
                        expected_support_evidence="Segment main effect remains significant across all observed subgroups.",
                        expected_weaken_evidence="Interaction term dominates or segment difference vanishes when adjusting for baseline covariates.",
                        is_exploratory=True,
                        known_limitations=[
                            "Requires representative subgroup sampling without attrition bias."
                        ],
                    )
                )

        # 3. Fallback general hypothesis if specific evidence is sparse
        if not proposals and len(evidence_pack.items) >= 2:
            num_items = [
                i for i in evidence_pack.items if i.evidence_type == "descriptive_statistic"
            ]
            if len(num_items) >= 2:
                v1 = (
                    num_items[0].relevant_variables[0]
                    if num_items[0].relevant_variables
                    else "Var1"
                )
                v2 = (
                    num_items[1].relevant_variables[0]
                    if num_items[1].relevant_variables
                    else "Var2"
                )
                proposals.append(
                    HypothesisProposal(
                        hypothesis_statement=f"High values of '{v1}' are positively associated with '{v2}'.",
                        rationale="Exploratory bivariate hypothesis based on initial distribution inspection.",
                        required_variables=[v1, v2],
                        suggested_method="Spearman rank correlation or OLS regression.",
                        expected_support_evidence="Statistically significant monotonic association (p < 0.05).",
                        expected_weaken_evidence="No statistically significant association (p >= 0.05).",
                        is_exploratory=True,
                        known_limitations=[
                            "Data-dredging risk; requires confirmatory holdout validation."
                        ],
                    )
                )

        return proposals[:max_hypotheses]
