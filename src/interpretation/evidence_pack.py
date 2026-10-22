"""Evidence Pack construction and serialization for AI analytical reasoning.

The Evidence Pack represents the immutable, verified source of truth for the LLM.
All quantitative claims in generated interpretations must cite stable evidence IDs
present in this pack.
"""

import logging
from typing import Any

import pandas as pd

from src.ingestion.loader import ManagedDataset
from src.interpretation.interpretation_models import EvidenceItem, EvidencePack

logger = logging.getLogger(__name__)


class EvidencePackBuilder:
    """Assembles scoped, token-efficient evidence packs from managed dataset state."""

    @staticmethod
    def build_pack(
        managed_dataset: ManagedDataset,
        relevant_columns: list[str] | None = None,
        tool_results: list[dict[str, Any]] | None = None,
        max_items: int = 50,
    ) -> EvidencePack:
        """Construct an EvidencePack for the given dataset version and query scope."""
        df: pd.DataFrame = managed_dataset.working_df
        dataset_version = getattr(managed_dataset, "version", 1)
        dataset_id = str(getattr(managed_dataset, "dataset_id", "dataset"))
        dataset_name = getattr(managed_dataset, "filename", dataset_id)

        schema = getattr(managed_dataset, "schema_info", {})
        col_types = {col: s.inferred_type for col, s in schema.items()}

        warnings: list[str] = []
        if hasattr(managed_dataset, "profile") and managed_dataset.profile:
            for issue in managed_dataset.profile.quality_issues[:5]:
                warnings.append(f"{issue.issue_type}: {issue.description}")

        items: list[EvidenceItem] = []

        # 1. Dimension Evidence
        items.append(
            EvidenceItem(
                id="ev_dim_rows",
                evidence_type="dataset_overview",
                dataset_version=dataset_version,
                source_service="ingestion.loader",
                calculation_method="len(dataframe)",
                relevant_variables=[],
                metric_name="row_count",
                actual_value=int(len(df)),
                formatted_value=f"{len(df):,} rows",
                sample_size=int(len(df)),
            )
        )
        items.append(
            EvidenceItem(
                id="ev_dim_cols",
                evidence_type="dataset_overview",
                dataset_version=dataset_version,
                source_service="ingestion.loader",
                calculation_method="len(columns)",
                relevant_variables=[],
                metric_name="column_count",
                actual_value=int(len(df.columns)),
                formatted_value=f"{len(df.columns)} columns",
            )
        )

        if hasattr(managed_dataset, "profile") and managed_dataset.profile:
            q_score = managed_dataset.profile.overall_quality_score
            items.append(
                EvidenceItem(
                    id="ev_quality_score",
                    evidence_type="data_quality",
                    dataset_version=dataset_version,
                    source_service="profiling.quality_score",
                    calculation_method="composite_weighted_penalty",
                    relevant_variables=[],
                    metric_name="overall_quality_score",
                    actual_value=float(q_score),
                    formatted_value=f"{q_score:.1f} / 100",
                    caveats=[
                        "Composite metric based on missingness, duplicates, and type validity."
                    ],
                )
            )

        # 2. Column Descriptive Evidence for relevant or selected columns
        cols_to_profile = relevant_columns or list(df.columns[:8])
        for col in cols_to_profile:
            if col not in df.columns:
                continue

            series = df[col]
            if pd.api.types.is_numeric_dtype(series):
                clean = series.dropna().astype(float)
                n_obs = len(clean)
                if n_obs > 0:
                    mean_val = float(clean.mean())
                    std_val = float(clean.std(ddof=1)) if n_obs > 1 else 0.0
                    median_val = float(clean.median())
                    min_val = float(clean.min())
                    max_val = float(clean.max())

                    items.append(
                        EvidenceItem(
                            id=f"ev_desc_{col}_mean",
                            evidence_type="descriptive_statistic",
                            dataset_version=dataset_version,
                            source_service="analytics.descriptive",
                            calculation_method="sample_arithmetic_mean",
                            relevant_variables=[col],
                            metric_name=f"{col}_mean",
                            actual_value=round(mean_val, 4),
                            formatted_value=f"{mean_val:,.2f}",
                            sample_size=n_obs,
                        )
                    )
                    items.append(
                        EvidenceItem(
                            id=f"ev_desc_{col}_median",
                            evidence_type="descriptive_statistic",
                            dataset_version=dataset_version,
                            source_service="analytics.descriptive",
                            calculation_method="sample_median",
                            relevant_variables=[col],
                            metric_name=f"{col}_median",
                            actual_value=round(median_val, 4),
                            formatted_value=f"{median_val:,.2f}",
                            sample_size=n_obs,
                        )
                    )
                    items.append(
                        EvidenceItem(
                            id=f"ev_desc_{col}_std",
                            evidence_type="descriptive_statistic",
                            dataset_version=dataset_version,
                            source_service="analytics.descriptive",
                            calculation_method="unbiased_standard_deviation",
                            relevant_variables=[col],
                            metric_name=f"{col}_std",
                            actual_value=round(std_val, 4),
                            formatted_value=f"{std_val:,.2f}",
                            sample_size=n_obs,
                        )
                    )
                    items.append(
                        EvidenceItem(
                            id=f"ev_desc_{col}_range",
                            evidence_type="descriptive_statistic",
                            dataset_version=dataset_version,
                            source_service="analytics.descriptive",
                            calculation_method="min_max_span",
                            relevant_variables=[col],
                            metric_name=f"{col}_range",
                            actual_value={"min": min_val, "max": max_val},
                            formatted_value=f"[{min_val:,.2f} to {max_val:,.2f}]",
                            sample_size=n_obs,
                        )
                    )
            else:
                # Categorical column summary
                clean = series.dropna().astype(str)
                n_obs = len(clean)
                if n_obs > 0:
                    counts = clean.value_counts()
                    top_mode = str(counts.index[0])
                    top_count = int(counts.iloc[0])
                    top_pct = round(top_count / n_obs * 100, 2)
                    items.append(
                        EvidenceItem(
                            id=f"ev_cat_{col}_mode",
                            evidence_type="descriptive_statistic",
                            dataset_version=dataset_version,
                            source_service="analytics.descriptive",
                            calculation_method="mode_frequency",
                            relevant_variables=[col],
                            metric_name=f"{col}_mode",
                            actual_value={"mode": top_mode, "count": top_count, "pct": top_pct},
                            formatted_value=f"'{top_mode}' ({top_pct}% of N={n_obs:,})",
                            sample_size=n_obs,
                        )
                    )

        # 3. Incorporate tool results dynamically if available
        if tool_results:
            for tr in tool_results:
                EvidencePackBuilder._absorb_tool_result(items, tr, dataset_version)

        # Limit items to max_items to prevent context overload
        packed_items = items[:max_items]

        return EvidencePack(
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            row_count=len(df),
            column_count=len(df.columns),
            column_types=col_types,
            quality_warnings=warnings,
            items=packed_items,
        )

    @staticmethod
    def _absorb_tool_result(
        items: list[EvidenceItem],
        tool_data: dict[str, Any],
        dataset_version: int,
    ) -> None:
        """Extract verifiable evidence items from executed analytical tool results."""
        # Check if it's a StatisticalTestResult
        if "test_name" in tool_data and "p_value" in tool_data:
            test_name = tool_data.get("test_name", "statistical_test")
            var_names = tool_data.get("variable_names", [])
            tag = "_".join(var_names) if var_names else "test"
            p_val = tool_data.get("p_value", 1.0)
            stat_val = tool_data.get("statistic_value", 0.0)
            alpha = tool_data.get("alpha", 0.05)
            n_obs = tool_data.get("sample_size_total")

            items.append(
                EvidenceItem(
                    id=f"ev_stat_{tag}_stat",
                    evidence_type="hypothesis_test",
                    dataset_version=dataset_version,
                    source_service="analytics.statistical_service",
                    calculation_method=test_name,
                    relevant_variables=var_names,
                    metric_name=f"{test_name}_statistic",
                    actual_value=stat_val,
                    formatted_value=f"{stat_val:.4f}",
                    sample_size=n_obs,
                )
            )
            items.append(
                EvidenceItem(
                    id=f"ev_stat_{tag}_pvalue",
                    evidence_type="hypothesis_test",
                    dataset_version=dataset_version,
                    source_service="analytics.statistical_service",
                    calculation_method=test_name,
                    relevant_variables=var_names,
                    metric_name=f"{test_name}_p_value",
                    actual_value=p_val,
                    formatted_value=f"{p_val:.4e}" if p_val < 0.001 else f"{p_val:.4f}",
                    sample_size=n_obs,
                    caveats=[f"Significance evaluated at alpha={alpha}"],
                )
            )

            # Effect size if available
            eff = tool_data.get("effect_size")
            if isinstance(eff, dict) and "estimate" in eff:
                eff_name = eff.get("name", "effect_size")
                eff_est = eff.get("estimate")
                eff_mag = eff.get("magnitude", "observed")
                items.append(
                    EvidenceItem(
                        id=f"ev_stat_{tag}_effect",
                        evidence_type="group_comparison",
                        dataset_version=dataset_version,
                        source_service="analytics.effect_sizes",
                        calculation_method=eff_name,
                        relevant_variables=var_names,
                        metric_name=eff_name,
                        actual_value=eff_est,
                        formatted_value=f"{eff_est:.4f} ({eff_mag})",
                        sample_size=n_obs,
                    )
                )

        # Check if it's RegressionModelResult
        elif "r_squared" in tool_data and "coefficients" in tool_data:
            outcome = tool_data.get("outcome_variable", tool_data.get("outcome_name", "outcome"))
            r2 = tool_data.get("r_squared", 0.0)
            adj_r2 = tool_data.get("adj_r_squared", 0.0)
            n_obs = tool_data.get("n_observations", tool_data.get("sample_size", 0))

            items.append(
                EvidenceItem(
                    id=f"ev_reg_{outcome}_r2",
                    evidence_type="regression",
                    dataset_version=dataset_version,
                    source_service="analytics.regression_inference",
                    calculation_method="OLS_coefficient_of_determination",
                    relevant_variables=[outcome],
                    metric_name="r_squared",
                    actual_value=r2,
                    formatted_value=f"{r2:.4f} (Adjusted R²: {adj_r2:.4f})",
                    sample_size=n_obs,
                )
            )

            for coef in tool_data.get("coefficients", []):
                feature = coef.get("variable", coef.get("feature_name", "feature"))
                val = coef.get("estimate", coef.get("coefficient", 0.0))
                p = coef.get("p_value", 1.0)
                items.append(
                    EvidenceItem(
                        id=f"ev_reg_{outcome}_coef_{feature}",
                        evidence_type="regression",
                        dataset_version=dataset_version,
                        source_service="analytics.regression_inference",
                        calculation_method="OLS_t_test",
                        relevant_variables=[outcome, feature],
                        metric_name=f"coef_{feature}",
                        actual_value=val,
                        formatted_value=f"beta={val:.4f} (p={p:.4f})",
                        sample_size=n_obs,
                    )
                )
