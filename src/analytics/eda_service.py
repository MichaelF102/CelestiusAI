"""Automated Exploratory Data Analysis (EDA) orchestration service."""

import logging
from typing import TYPE_CHECKING

import pandas as pd

from src.analytics.analysis_models import (
    CategoricalSummary,
    EDAFinding,
    EDASummary,
    NumericSummary,
)
from src.analytics.chart_recommender import recommend_charts
from src.analytics.descriptive import compute_categorical_summary, compute_numeric_summary
from src.analytics.distributions import diagnose_distribution_shape
from src.analytics.grouped_analysis import analyze_grouped_disparity
from src.analytics.interpretation import (
    generate_category_concentration_finding,
    generate_correlation_finding,
    generate_grouped_disparity_finding,
    generate_skewness_finding,
)
from src.analytics.relationships import compute_pairwise_correlations
from src.ingestion.schema import infer_dataset_schema

if TYPE_CHECKING:
    from src.ingestion.loader import ManagedDataset

logger = logging.getLogger(__name__)


class EDAService:
    """Orchestrates comprehensive, deterministic EDA and 6-layer findings on managed datasets."""

    @staticmethod
    def generate_eda_summary(
        managed_dataset: "ManagedDataset",
        max_correlations: int = 5,
        max_findings: int = 12,
    ) -> EDASummary:
        """Run complete automated exploratory analysis across the active working dataframe.

        Derives grounded numerical findings without LLM dependencies.
        """
        df: pd.DataFrame = getattr(managed_dataset, "working_df", pd.DataFrame())
        dataset_id = getattr(
            managed_dataset, "dataset_id", getattr(managed_dataset, "id", "dataset")
        )

        if df.empty:
            return EDASummary(
                dataset_id=dataset_id,
                row_count=0,
                column_count=0,
                findings=[],
                recommendations=[],
                top_correlations=[],
                skewed_columns=[],
                concentrated_categories=[],
            )

        schema = getattr(managed_dataset, "schema_info", None)
        if not schema:
            schema = infer_dataset_schema(df)

        findings: list[EDAFinding] = []
        skewed_cols: list[str] = []
        concentrated_cats: list[str] = []

        numeric_cols = [
            col
            for col, s in schema.items()
            if s.inferred_type in ("numeric", "integer", "float")
            and not s.is_constant
            and not s.is_identifier
            and col in df.columns
        ]
        categorical_cols = [
            col
            for col, s in schema.items()
            if s.inferred_type in ("categorical", "string", "category", "boolean")
            and not s.is_constant
            and not s.is_identifier
            and col in df.columns
        ]

        # 1. Numeric Summaries & Skewness Findings
        numeric_summaries: dict[str, NumericSummary] = {}
        for col in numeric_cols:
            num_sum = compute_numeric_summary(df[col], col)
            numeric_summaries[col] = num_sum
            shape_diag = diagnose_distribution_shape(df[col])
            if shape_diag.get("is_skewed"):
                skewed_cols.append(col)
                if len(findings) < max_findings:
                    findings.append(generate_skewness_finding(num_sum, shape_diag))

        # 2. Categorical Summaries & Concentration Findings
        categorical_summaries: dict[str, CategoricalSummary] = {}
        for col in categorical_cols:
            cat_sum = compute_categorical_summary(df[col], col)
            categorical_summaries[col] = cat_sum
            if cat_sum.top_1_concentration_pct >= 50.0 and cat_sum.distinct_count > 1:
                concentrated_cats.append(col)
                if len(findings) < max_findings:
                    findings.append(generate_category_concentration_finding(cat_sum))

        # 3. Pairwise Correlations & Findings
        top_correlations = []
        if len(numeric_cols) >= 2:
            all_corrs = compute_pairwise_correlations(
                df, numeric_cols, method="pearson", min_sample=10
            )
            top_correlations = all_corrs[:max_correlations]
            for pair in top_correlations:
                if abs(pair.coefficient) >= 0.5 and len(findings) < max_findings:
                    findings.append(generate_correlation_finding(pair))

        # 4. Grouped Disparity Findings
        if categorical_cols and numeric_cols:
            best_cat = categorical_cols[0]
            best_num = numeric_cols[0]
            for c in categorical_cols:
                distinct = df[c].nunique(dropna=True)
                if 2 <= distinct <= 15:
                    best_cat = c
                    break

            try:
                grp_res = analyze_grouped_disparity(df, group_col=best_cat, measure_col=best_num)
                if (
                    grp_res
                    and grp_res.max_to_min_ratio
                    and grp_res.max_to_min_ratio >= 1.5
                    and len(findings) < max_findings
                ):
                    findings.append(generate_grouped_disparity_finding(grp_res))
            except Exception as e:
                logger.warning("Grouped disparity derivation failed: %s", e)

        # 5. Chart Recommendations
        recommendations = recommend_charts(df, schema, max_recommendations=8)

        # Sort findings by severity
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        findings.sort(key=lambda f: severity_order.get(f.severity, 4))

        return EDASummary(
            dataset_id=dataset_id,
            row_count=len(df),
            column_count=len(df.columns),
            findings=findings,
            recommendations=recommendations,
            top_correlations=top_correlations,
            skewed_columns=skewed_cols,
            concentrated_categories=concentrated_cats,
        )
