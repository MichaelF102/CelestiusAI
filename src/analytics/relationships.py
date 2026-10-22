"""Pairwise correlation analysis, ranked relationship discovery, and categorical association."""

import numpy as np
import pandas as pd
from scipy import stats

from src.analytics.analysis_models import CorrelationMethod, CorrelationPair
from src.core.logging_config import get_logger

logger = get_logger("analytics.relationships")


def _classify_correlation_strength(r: float) -> str:
    abs_r = abs(r)
    if abs_r >= 0.8:
        return "very_strong"
    if abs_r >= 0.6:
        return "strong"
    if abs_r >= 0.4:
        return "moderate"
    if abs_r >= 0.2:
        return "weak"
    return "negligible"


def calculate_pairwise_correlation(
    series1: pd.Series,
    series2: pd.Series,
    method: CorrelationMethod = "pearson",
    min_sample: int = 5,
) -> CorrelationPair | None:
    """Calculate pairwise Pearson or Spearman correlation with p-values and effective sample size."""
    col1 = str(series1.name)
    col2 = str(series2.name)

    df_pair = pd.DataFrame(
        {
            "v1": pd.to_numeric(series1, errors="coerce"),
            "v2": pd.to_numeric(series2, errors="coerce"),
        }
    ).dropna()
    df_pair = df_pair[np.isfinite(df_pair["v1"]) & np.isfinite(df_pair["v2"])]

    n = len(df_pair)
    if n < min_sample:
        return None

    # Check for zero variance
    if df_pair["v1"].std() == 0 or df_pair["v2"].std() == 0:
        return None

    try:
        if method == "spearman":
            res = stats.spearmanr(df_pair["v1"], df_pair["v2"])
            r_val = float(res.statistic)  # type: ignore[attr-defined]
            p_val = float(res.pvalue)  # type: ignore[attr-defined]
        else:
            res = stats.pearsonr(df_pair["v1"], df_pair["v2"])
            r_val = float(res.statistic)
            p_val = float(res.pvalue)

        if np.isnan(r_val):
            return None

        strength = _classify_correlation_strength(r_val)
        direction = "positive" if r_val > 0.05 else ("negative" if r_val < -0.05 else "none")

        return CorrelationPair(
            var1=col1,
            var2=col2,
            method=method,
            coefficient=round(r_val, 4),
            sample_size=n,
            p_value=round(p_val, 6),
            strength=strength,  # type: ignore[arg-type]
            direction=direction,  # type: ignore[arg-type]
        )
    except Exception as e:
        logger.debug(f"Failed correlation between '{col1}' and '{col2}': {e}")
        return None


def calculate_correlation_matrix(
    df: pd.DataFrame,
    numeric_columns: list[str],
    method: CorrelationMethod = "pearson",
    min_sample: int = 5,
) -> tuple[pd.DataFrame, list[CorrelationPair]]:
    """Compute full correlation matrix and ranked list of strongest pairwise relationships."""
    valid_cols = [c for c in numeric_columns if c in df.columns and df[c].dropna().nunique() > 1]
    if len(valid_cols) < 2:
        return pd.DataFrame(), []

    corr_matrix = pd.DataFrame(index=valid_cols, columns=valid_cols, dtype=float)
    pairs: list[CorrelationPair] = []

    for i in range(len(valid_cols)):
        for j in range(i, len(valid_cols)):
            c1 = valid_cols[i]
            c2 = valid_cols[j]
            if c1 == c2:
                corr_matrix.loc[c1, c2] = 1.0
                continue

            pair = calculate_pairwise_correlation(
                df[c1], df[c2], method=method, min_sample=min_sample
            )
            if pair is not None:
                corr_matrix.loc[c1, c2] = pair.coefficient
                corr_matrix.loc[c2, c1] = pair.coefficient
                pairs.append(pair)
            else:
                corr_matrix.loc[c1, c2] = np.nan
                corr_matrix.loc[c2, c1] = np.nan

    # Sort pairs by absolute correlation strength descending
    pairs.sort(key=lambda p: abs(p.coefficient), reverse=True)
    return corr_matrix, pairs


def compute_pairwise_correlations(
    df: pd.DataFrame,
    numeric_columns: list[str],
    method: CorrelationMethod = "pearson",
    min_sample: int = 5,
) -> list[CorrelationPair]:
    """Compute and return ranked list of pairwise correlations."""
    _, pairs = calculate_correlation_matrix(
        df=df,
        numeric_columns=numeric_columns,
        method=method,
        min_sample=min_sample,
    )
    return pairs


def calculate_cramers_v(
    series1: pd.Series,
    series2: pd.Series,
) -> tuple[float, pd.DataFrame]:
    """Calculate Cramér's V association metric and contingency matrix for two categorical features."""
    clean = pd.DataFrame({"s1": series1.astype(str), "s2": series2.astype(str)}).dropna()
    if len(clean) < 5 or clean["s1"].nunique() < 2 or clean["s2"].nunique() < 2:
        return 0.0, pd.DataFrame()

    contingency = pd.crosstab(clean["s1"], clean["s2"])
    try:
        chi2, _, _, _ = stats.chi2_contingency(contingency)
        n = len(clean)
        r, k = contingency.shape
        min_dim = min(r - 1, k - 1)
        if min_dim == 0 or n == 0:
            return 0.0, contingency
        v = np.sqrt(chi2 / (n * min_dim))
        return round(float(v), 4), contingency
    except Exception:
        return 0.0, contingency


def compute_cramers_v(
    df: pd.DataFrame,
    cat_col1: str,
    cat_col2: str,
) -> tuple[float, pd.DataFrame]:
    """Compute Cramér's V for two categorical columns of a DataFrame."""
    if cat_col1 not in df.columns or cat_col2 not in df.columns:
        return 0.0, pd.DataFrame()
    return calculate_cramers_v(df[cat_col1], df[cat_col2])


compute_pairwise_correlation = calculate_pairwise_correlation
compute_correlation_matrix = calculate_correlation_matrix
