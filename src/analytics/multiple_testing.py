"""Multiple-testing correction procedures (Benjamini-Hochberg FDR, Bonferroni, Holm)."""

from typing import Literal

import numpy as np

from src.analytics.statistical_models import StatisticalTestResult

CorrectionMethod = Literal["fdr_bh", "bonferroni", "holm", "none"]


def adjust_p_values(
    p_values: list[float],
    method: CorrectionMethod = "fdr_bh",
) -> list[float]:
    """Calculate multiplicity-adjusted p-values with robust handling of bounds and edge cases.

    Methods:
        - 'fdr_bh': Benjamini-Hochberg False Discovery Rate control
        - 'bonferroni': Family-Wise Error Rate (FWER) single-step control
        - 'holm': Holm-Bonferroni step-down FWER control
        - 'none': Unadjusted (exploratory)
    """
    m = len(p_values)
    if m <= 1 or method == "none":
        return [float(np.clip(p, 0.0, 1.0)) for p in p_values]

    p_arr = np.array([np.clip(p, 0.0, 1.0) for p in p_values], dtype=float)

    if method == "bonferroni":
        adjusted = np.minimum(1.0, p_arr * m)
        return [round(float(x), 6) for x in adjusted]

    if method == "holm":
        # Step-down Bonferroni
        sort_indices = np.argsort(p_arr)
        p_sorted = p_arr[sort_indices]
        adj_sorted = np.zeros(m, dtype=float)
        running_max = 0.0

        for i in range(m):
            step_multiplier = m - i
            val = p_sorted[i] * step_multiplier
            running_max = max(running_max, val)
            adj_sorted[i] = min(1.0, running_max)

        # Invert sort back to original order
        adj_original = np.zeros(m, dtype=float)
        adj_original[sort_indices] = adj_sorted
        return [round(float(x), 6) for x in adj_original]

    if method == "fdr_bh":
        # Benjamini-Hochberg step-up procedure
        sort_indices = np.argsort(p_arr)
        p_sorted = p_arr[sort_indices]
        ranks = np.arange(1, m + 1, dtype=float)
        adj_sorted = np.zeros(m, dtype=float)

        raw_adj = (p_sorted * m) / ranks
        # Step-up enforces monotonicity: adj[i] = min(adj[i], adj[i+1])
        running_min = 1.0
        for i in range(m - 1, -1, -1):
            running_min = min(running_min, raw_adj[i])
            adj_sorted[i] = min(1.0, running_min)

        adj_original = np.zeros(m, dtype=float)
        adj_original[sort_indices] = adj_sorted
        return [round(float(x), 6) for x in adj_original]

    return [float(p) for p in p_values]


def apply_multiple_testing_correction(
    results: list[StatisticalTestResult],
    method: CorrectionMethod = "fdr_bh",
    family_name: str = "Comparison Family",
) -> list[StatisticalTestResult]:
    """Apply multiple-testing adjustment across a family of StatisticalTestResult objects.

    Preserves individual result immutability and test mappings.
    """
    if not results:
        return []

    raw_p_values = [r.p_value for r in results]
    adjusted_p_values = adjust_p_values(raw_p_values, method=method)

    updated_results: list[StatisticalTestResult] = []
    method_labels = {
        "fdr_bh": "Benjamini-Hochberg (FDR)",
        "bonferroni": "Bonferroni (FWER)",
        "holm": "Holm-Bonferroni (FWER)",
        "none": "Unadjusted (Exploratory)",
    }
    label = method_labels.get(method, method)

    for res, adj_p in zip(results, adjusted_p_values, strict=True):
        # Create updated model with adjusted p-value
        updated_dict = res.model_dump()
        updated_dict["adjusted_p_value"] = adj_p
        updated_dict["adjustment_method"] = (
            f"{label} across {len(results)} tests in '{family_name}'"
        )

        # Update p_value_display to indicate both
        disp = res.p_value_display
        if method != "none":
            adj_disp = "< 0.001" if adj_p < 0.001 else f"= {adj_p:.4f}"
            updated_dict["p_value_display"] = f"Raw p {disp} | Adj. p {adj_disp} ({method})"

        updated_results.append(StatisticalTestResult(**updated_dict))

    return updated_results
