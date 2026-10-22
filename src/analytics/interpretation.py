"""Deterministic calculation-backed 6-layer structured interpretation engine."""

import uuid

from src.analytics.analysis_models import (
    CategoricalSummary,
    CorrelationPair,
    EDAFinding,
    GroupAggregationResult,
    NumericSummary,
)


def generate_skewness_finding(summary: NumericSummary, shape_diag: dict[str, object]) -> EDAFinding:
    """Generate 6-layer finding for a continuous feature's distribution shape and central tendency."""
    col = summary.column_name
    skew = summary.skewness or 0.0
    mean_val = summary.mean
    median_val = summary.median
    divergence = shape_diag.get("mean_median_divergence_pct") or 0.0
    preferred_metric = shape_diag.get("preferred_central_metric", "median")

    is_right_skewed = skew > 0.5
    direction_word = "right" if is_right_skewed else "left"

    obs = (
        f"Column '{col}' exhibits marked {direction_word}-skewness (skewness = {skew:.2f}) "
        f"with the mean ({mean_val:,.2f}) diverging from the median ({median_val:,.2f}) by {abs(float(divergence)):.1f}%."
    )
    evidence = (
        f"Sample size: N = {summary.count:,}. Mean = {mean_val:,.2f}, Median = {median_val:,.2f}, "
        f"Std Dev = {summary.std:,.2f}, IQR = [{summary.q25:,.2f}, {summary.q75:,.2f}]. "
        f"Identified {summary.outlier_count_iqr:,} values outside Tukey's IQR boundaries."
    )
    stat_context = (
        f"Because the distribution is {direction_word}-skewed, extreme values in the tail disproportionately pull the arithmetic mean. "
        f"Under these distributional conditions, the {preferred_metric} represents central tendency more faithfully than the mean."
    )
    implication = (
        f"Setting operational benchmarks or forecasting models based on the mean '{col}' would overstate "
        "typical performance. Resource allocations should be guided by median values."
    )
    limitations = (
        f"Evaluation assumes observations are independent and un-truncated. "
        f"{summary.missing_percentage:.1f}% missing values were excluded from this calculation."
    )
    next_step = (
        f"Inspect the {summary.outlier_count_iqr} values in the upper tail to determine if they represent data entry errors "
        "or a distinct high-value subgroup that warrants separate segmentation."
    )

    return EDAFinding(
        finding_id=f"fnd_dist_{uuid.uuid4().hex[:6]}",
        title=f"Distribution of '{col}' is {direction_word}-skewed (mean diverges from median by {abs(float(divergence)):.1f}%)",
        category="distribution",
        summary=obs,
        observation=obs,
        quantitative_evidence=evidence,
        statistical_context=stat_context,
        practical_implication=implication,
        limitations=limitations,
        next_investigation=next_step,
        severity="medium" if abs(float(divergence)) < 25.0 else "high",
        relevant_columns=[col],
    )


def generate_correlation_finding(pair: CorrelationPair) -> EDAFinding:
    """Generate 6-layer finding for pairwise correlation strictly enforcing non-causal boundaries."""
    c1 = pair.var1
    c2 = pair.var2
    r = pair.coefficient
    method_name = pair.method.capitalize()

    obs = (
        f"A {pair.strength.replace('_', ' ')} {pair.direction} association exists between '{c1}' and '{c2}' "
        f"({method_name} r = {r:+.2f}, N = {pair.sample_size:,})."
    )
    p_text = f"p < {pair.p_value:.4f}" if pair.p_value is not None else "p-value not computed"
    evidence = (
        f"{method_name} correlation coefficient: r = {r:+.4f}. "
        f"Effective complete pairwise observations: N = {pair.sample_size:,}. "
        f"Significance indicator: {p_text}."
    )
    stat_context = (
        f"The coefficient measures linear/monotonic co-movement across {pair.sample_size:,} records. "
        "This association does NOT prove or imply causation. The relationship may be driven by confounding variables "
        "or non-linear dynamics not captured by bivariate correlation."
    )
    implication = (
        f"Increases in '{c1}' are consistently associated with {'higher' if r > 0 else 'lower'} values of '{c2}'. "
        "This pairing represents a promising candidate for predictive modeling and scenario planning."
    )
    limitations = (
        "Bivariate correlation is vulnerable to omitted variable bias (lurking confounders) and outlier leverage. "
        "It does not demonstrate that manipulating one variable will alter the other."
    )
    next_step = (
        "Fit a multivariate regression model or perform stratified subgroup analysis controlling for "
        "potential confounders such as customer segment, region, or time period."
    )

    return EDAFinding(
        finding_id=f"fnd_corr_{uuid.uuid4().hex[:6]}",
        title=f"{pair.strength.capitalize()} {pair.direction} association between '{c1}' and '{c2}' (r = {r:+.2f})",
        category="correlation",
        summary=obs,
        observation=obs,
        quantitative_evidence=evidence,
        statistical_context=stat_context,
        practical_implication=implication,
        limitations=limitations,
        next_investigation=next_step,
        severity="high" if abs(r) >= 0.7 else "medium",
        relevant_columns=[c1, c2],
    )


def generate_grouped_disparity_finding(result: GroupAggregationResult) -> EDAFinding:
    """Generate 6-layer finding for segment disparity across groups."""
    grp_col = result.group_column
    meas_col = result.measure_column
    agg = result.aggregation
    top_g = result.top_group or "Top"
    bot_g = result.bottom_group or "Bottom"
    ratio = result.max_to_min_ratio or 1.0

    top_val = result.groups[0]["value"] if result.groups else 0.0
    bot_val = result.groups[-1]["value"] if result.groups else 0.0

    obs = (
        f"Prominent disparity in {agg} '{meas_col}' across '{grp_col}' segments: "
        f"'{top_g}' averages {top_val:,.2f} versus {bot_val:,.2f} for '{bot_g}' (a {ratio:.1f}x spread)."
    )
    evidence = (
        f"Total groups evaluated: {result.total_groups_count}. "
        f"Top group: '{top_g}' ({agg} = {top_val:,.2f}, N = {result.groups[0]['sample_size']:,}, share = {result.groups[0]['share_pct']:.1f}%). "
        f"Bottom group: '{bot_g}' ({agg} = {bot_val:,.2f}, N = {result.groups[-1]['sample_size']:,}, share = {result.groups[-1]['share_pct']:.1f}%)."
    )
    stat_context = (
        f"Descriptive group comparison shows substantial between-group variance. "
        f"{result.notes}"
        "Formal hypothesis testing (e.g., ANOVA or Kruskal–Wallis in Phase 4) is required to determine whether "
        "this disparity exceeds expected random sampling fluctuation."
    )
    implication = (
        f"Performance is concentrated in '{top_g}'. A single uniform policy across all '{grp_col}' groups "
        "is unlikely to be optimal. Strategy should differentiate high-performing segments from laggards."
    )
    limitations = "Group sizes may be imbalanced, and aggregate averages may hide wide intra-group variation or outliers."
    next_step = (
        f"Conduct a two-sample statistical test comparing '{top_g}' against other segments, "
        "and inspect boxplots to evaluate whether the difference is driven by a few extreme records."
    )

    return EDAFinding(
        finding_id=f"fnd_grp_{uuid.uuid4().hex[:6]}",
        title=f"'{top_g}' leads '{grp_col}' in {agg} '{meas_col}' by {ratio:.1f}x over '{bot_g}'",
        category="grouped_disparity",
        summary=obs,
        observation=obs,
        quantitative_evidence=evidence,
        statistical_context=stat_context,
        practical_implication=implication,
        limitations=limitations,
        next_investigation=next_step,
        severity="medium" if ratio < 3.0 else "high",
        relevant_columns=[grp_col, meas_col],
    )


def generate_category_concentration_finding(summary: CategoricalSummary) -> EDAFinding:
    """Generate 6-layer finding for high category concentration in discrete columns."""
    col = summary.column_name
    top_1 = (
        summary.top_categories[0]
        if summary.top_categories
        else {"label": "None", "percentage": 0.0, "count": 0}
    )
    top_1_pct = summary.top_1_concentration_pct
    top_3_pct = summary.top_3_concentration_pct

    obs = (
        f"High concentration in categorical variable '{col}': "
        f"'{top_1['label']}' accounts for {top_1_pct:.1f}% of all records alone (top 3 represent {top_3_pct:.1f}%)."
    )
    evidence = (
        f"Total distinct categories: {summary.distinct_count}. Non-null records: {summary.total_count:,}. "
        f"Top category '{top_1['label']}': {top_1['count']:,} rows ({top_1_pct:.1f}%). "
        f"Identified {summary.rare_categories_count} rare categories with < 1% representation."
    )
    stat_context = (
        "Extreme category imbalance compresses effective degrees of freedom and creates severe class imbalance "
        "if this variable is used as a stratifier or prediction target."
    )
    implication = (
        f"Operational focus is predominantly tied to '{top_1['label']}'. Business models must account for "
        "vulnerability to changes within this single dominant category."
    )
    limitations = "Concentration figures are descriptive of the current dataset sample and depend on data collection filters."
    next_step = (
        "Investigate whether rare categories can be grouped into an 'Other' bucket, "
        "or whether the dominant category should be analyzed in isolation."
    )

    return EDAFinding(
        finding_id=f"fnd_conc_{uuid.uuid4().hex[:6]}",
        title=f"Category '{top_1['label']}' dominates '{col}' with {top_1_pct:.1f}% of records",
        category="concentration",
        summary=obs,
        observation=obs,
        quantitative_evidence=evidence,
        statistical_context=stat_context,
        practical_implication=implication,
        limitations=limitations,
        next_investigation=next_step,
        severity="medium" if top_1_pct < 70.0 else "high",
        relevant_columns=[col],
    )
