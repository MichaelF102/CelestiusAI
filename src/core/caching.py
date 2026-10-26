"""Streamlit cache wrappers for deterministic, expensive analytics.

Streamlit reruns a page script on every widget interaction. These wrappers memoize the
heavy pure computations so changing an unrelated control does not recompute them.
DataFrame arguments are hashed by content, so a cleaned/filtered frame never serves stale
results. Cached values must be picklable (pydantic models, dataclasses, DataFrames).
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from src.analytics.eda_service import EDAService
from src.analytics.grouped_analysis import analyze_grouped_disparity
from src.analytics.relationships import compute_cramers_v, compute_pairwise_correlations
from src.analytics.temporal import analyze_temporal_series

_CACHE = {"show_spinner": False, "max_entries": 64}


@st.cache_data(**_CACHE)
def cached_pairwise_correlations(
    df: pd.DataFrame, numeric_cols: tuple[str, ...], method: str, min_sample: int = 10
) -> Any:
    return compute_pairwise_correlations(
        df, list(numeric_cols), method=method, min_sample=min_sample
    )


@st.cache_data(**_CACHE)
def cached_cramers_v(df: pd.DataFrame, col_a: str, col_b: str) -> Any:
    return compute_cramers_v(df, col_a, col_b)


@st.cache_data(**_CACHE)
def cached_grouped_disparity(df: pd.DataFrame, group_col: str, measure_col: str, agg: str) -> Any:
    return analyze_grouped_disparity(df, group_col, measure_col, agg=agg)


@st.cache_data(**_CACHE)
def cached_temporal_series(df: pd.DataFrame, column: str) -> Any:
    return analyze_temporal_series(df, column)


@st.cache_data(**_CACHE)
def cached_correlation_matrix(df: pd.DataFrame, columns: tuple[str, ...], method: str) -> Any:
    return df[list(columns)].corr(method=method)


@st.cache_data(**_CACHE)
def cached_eda_summary(_dataset: Any, dataset_id: str, version: int, fingerprint: str) -> Any:
    """EDA summary keyed on dataset identity + version (bumped by every transformation)."""
    return EDAService.generate_eda_summary(_dataset)
