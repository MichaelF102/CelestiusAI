"""Temporal trend analysis, calendar cyclicality, and time-series aggregation."""

import pandas as pd

from src.analytics.analysis_models import TemporalSummary


def analyze_temporal_series(
    df: pd.DataFrame,
    date_column: str,
    value_column: str | None = None,
    frequency: str = "ME",  # month end
) -> TemporalSummary:
    """Analyze temporal coverage, resampled periodic aggregations, and cyclical distributions."""
    if date_column not in df.columns:
        return TemporalSummary(date_column=date_column)

    clean = df[[date_column]].copy()
    if value_column and value_column in df.columns:
        clean["val"] = pd.to_numeric(df[value_column], errors="coerce")
    else:
        clean["val"] = 1.0

    clean["dt"] = pd.to_datetime(clean[date_column], errors="coerce")
    clean = clean.dropna(subset=["dt"])
    total_records = len(clean)

    if total_records == 0:
        return TemporalSummary(date_column=date_column, value_column=value_column)

    clean = clean.sort_values(by="dt")
    min_dt = clean["dt"].min()
    max_dt = clean["dt"].max()
    span_days = (max_dt - min_dt).days

    if total_records < 3:
        return TemporalSummary(
            date_column=date_column,
            value_column=value_column,
            total_records=total_records,
            earliest_date=min_dt.strftime("%Y-%m-%d"),
            latest_date=max_dt.strftime("%Y-%m-%d"),
            span_days=span_days,
            trend_direction="stable",
            timeline_data=[],
            day_of_week_distribution=[],
        )

    # Resample or group by date intervals
    clean.set_index("dt", inplace=True)
    try:
        # Month-end aggregation
        resampled = clean["val"].resample(frequency).sum().reset_index()
        timeline_data = [
            {"date": r["dt"].strftime("%Y-%m-%d"), "value": round(float(r["val"]), 2)}
            for _, r in resampled.iterrows()
        ]
    except Exception:
        # Fallback to date grouping
        grouped = clean.groupby(clean.index.date)["val"].sum().reset_index()
        timeline_data = [
            {"date": str(r["dt"]), "value": round(float(r["val"]), 2)}
            for _, r in grouped.iterrows()
        ]

    # Cyclical analysis: observations by Day of Week
    dow_counts = clean.index.day_name().value_counts()
    dow_distribution = [{"day": str(day), "count": int(cnt)} for day, cnt in dow_counts.items()]

    # Overall trend direction (first vs last period)
    trend_direction = "stable"
    if len(timeline_data) >= 2:
        v_start = timeline_data[0]["value"]
        v_end = timeline_data[-1]["value"]
        if v_start > 0:
            growth_pct = ((v_end - v_start) / v_start) * 100.0
            if growth_pct >= 15.0:
                trend_direction = "upward"
            elif growth_pct <= -15.0:
                trend_direction = "downward"

    return TemporalSummary(
        date_column=date_column,
        value_column=value_column,
        total_records=total_records,
        earliest_date=min_dt.strftime("%Y-%m-%d"),
        latest_date=max_dt.strftime("%Y-%m-%d"),
        span_days=span_days,
        trend_direction=trend_direction,
        timeline_data=timeline_data,
        day_of_week_distribution=dow_distribution,
    )
