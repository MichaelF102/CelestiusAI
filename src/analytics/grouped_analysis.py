"""Grouped aggregations, segment disparity analysis, and group imbalance diagnostics."""

from typing import Any

import pandas as pd

from src.analytics.analysis_models import AggregationFunc, GroupAggregationResult


def analyze_grouped_metric(
    df: pd.DataFrame,
    group_column: str,
    measure_column: str,
    aggregation: AggregationFunc = "mean",
    max_groups: int = 15,
) -> GroupAggregationResult:
    """Aggregate a quantitative measure across discrete groups and evaluate group disparities."""
    if group_column not in df.columns or measure_column not in df.columns or df.empty:
        return GroupAggregationResult(
            group_column=group_column,
            measure_column=measure_column,
            aggregation=aggregation,
            groups=[],
            total_groups_count=0,
            top_group=None,
            bottom_group=None,
            max_to_min_ratio=None,
            is_imbalanced=False,
            notes="Missing columns or empty dataset.",
        )

    clean = df[[group_column, measure_column]].dropna()
    clean[measure_column] = pd.to_numeric(clean[measure_column], errors="coerce")
    clean = clean.dropna()

    if len(clean) == 0:
        return GroupAggregationResult(
            group_column=group_column,
            measure_column=measure_column,
            aggregation=aggregation,
            groups=[],
            total_groups_count=0,
            top_group=None,
            bottom_group=None,
            max_to_min_ratio=None,
            is_imbalanced=False,
            notes="No valid numeric records after cleaning.",
        )

    # Compute group statistics
    agg_funcs = {
        "mean": lambda s: s.mean(),
        "median": lambda s: s.median(),
        "sum": lambda s: s.sum(),
        "count": lambda s: len(s),
        "std": lambda s: s.std(ddof=1) if len(s) > 1 else 0.0,
        "min": lambda s: s.min(),
        "max": lambda s: s.max(),
    }
    target_func = agg_funcs.get(aggregation, agg_funcs["mean"])

    grouped = clean.groupby(group_column)[measure_column].agg(["count", target_func]).reset_index()
    grouped.columns = [group_column, "sample_size", "aggregated_value"]

    # Filter out empty groups and sort descending by aggregated value
    grouped = grouped[grouped["sample_size"] > 0].sort_values(
        by="aggregated_value", ascending=False
    )
    total_groups = len(grouped)

    if total_groups == 0:
        return GroupAggregationResult(
            group_column=group_column,
            measure_column=measure_column,
            aggregation=aggregation,
            groups=[],
            total_groups_count=0,
            top_group=None,
            bottom_group=None,
            max_to_min_ratio=None,
            is_imbalanced=False,
            notes="No non-empty groups found.",
        )

    # Check for group size imbalance
    min_size = int(grouped["sample_size"].min())
    max_size = int(grouped["sample_size"].max())
    is_imbalanced = (max_size / min_size) > 5.0 if min_size > 0 else True

    # Build group breakdown list
    total_val_sum = grouped["aggregated_value"].abs().sum()
    groups_data: list[dict[str, Any]] = []

    for _, row in grouped.head(max_groups).iterrows():
        val = float(row["aggregated_value"])
        cnt = int(row["sample_size"])
        share_pct = round((abs(val) / total_val_sum * 100.0), 2) if total_val_sum > 0 else 0.0
        groups_data.append(
            {
                "group": str(row[group_column])[:40],
                "value": round(val, 4),
                "sample_size": cnt,
                "share_pct": share_pct,
            }
        )

    top_grp = groups_data[0]["group"] if groups_data else None
    bottom_grp = groups_data[-1]["group"] if groups_data else None

    # Ratio between top and bottom group
    max_to_min_ratio: float | None = None
    if groups_data and abs(groups_data[-1]["value"]) > 1e-6:
        max_to_min_ratio = round(groups_data[0]["value"] / groups_data[-1]["value"], 2)

    notes = ""
    if is_imbalanced:
        notes += f"Sample sizes across groups are unequal (ranging from {min_size:,} to {max_size:,} rows). "

    return GroupAggregationResult(
        group_column=group_column,
        measure_column=measure_column,
        aggregation=aggregation,
        groups=groups_data,
        total_groups_count=total_groups,
        top_group=top_grp,
        bottom_group=bottom_grp,
        max_to_min_ratio=max_to_min_ratio,
        is_imbalanced=is_imbalanced,
        notes=notes.strip(),
    )


def analyze_grouped_disparity(
    df: pd.DataFrame,
    group_col: str,
    measure_col: str,
    agg: AggregationFunc = "mean",
    max_groups: int = 15,
) -> GroupAggregationResult:
    """Analyze group disparity with flexible parameter naming."""
    return analyze_grouped_metric(
        df=df,
        group_column=group_col,
        measure_column=measure_col,
        aggregation=agg,
        max_groups=max_groups,
    )


def compute_two_way_contingency(
    df: pd.DataFrame,
    row_var: str,
    col_var: str,
) -> pd.DataFrame:
    """Compute a 2-way cross-tabulation table between two categorical variables."""
    if row_var not in df.columns or col_var not in df.columns or df.empty:
        return pd.DataFrame()
    return pd.crosstab(df[row_var], df[col_var])
