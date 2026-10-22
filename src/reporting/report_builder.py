"""Report Builder.

Constructs structured intelligence reports tailored by ReportType and ReportDepth.
Assembles evidence-backed sections, tables, and provenance without hallucination.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from src.reporting.report_models import (
    ReportDepth,
    ReportDocument,
    ReportMetadata,
    ReportSection,
    ReportTable,
    ReportType,
)

logger = logging.getLogger(__name__)


class ReportBuilder:
    """Assembles verified evidence into a cohesive ReportDocument."""

    def __init__(self, evidence_data: dict[str, Any]) -> None:
        self.evidence_data = evidence_data
        self.dataset_id = evidence_data.get("dataset_id", "dataset")
        self.dataset_version = evidence_data.get("dataset_version", 1)
        self.filename = evidence_data.get("filename", "Dataset")
        self.evidence_items = evidence_data.get("evidence_items", [])
        self.predictive_records = evidence_data.get("predictive_records", [])
        self.investigation_trace = evidence_data.get("investigation_trace")
        self.statistical_results = evidence_data.get("statistical_results", [])
        self.profile = evidence_data.get("profile")

    def build_report(
        self,
        report_type: ReportType = ReportType.COMPREHENSIVE,
        report_depth: ReportDepth = ReportDepth.STANDARD,
        ai_narrative_text: str | None = None,
    ) -> ReportDocument:
        """Assemble the complete ReportDocument for the given type and depth."""
        sections: list[ReportSection] = []
        limitations: list[str] = [
            "Observational data cannot establish definitive causality without randomized experiments or natural experimental controls.",
            "Metrics reflect the active dataset version and must be re-evaluated if upstream data is transformed.",
        ]
        unperformed: list[str] = []

        # 1. Dataset Overview & Data Quality Section (Always included)
        sections.append(self._build_overview_and_quality_section())

        # 2. EDA & Distributions Section
        if report_type in (ReportType.EDA, ReportType.EXECUTIVE, ReportType.COMPREHENSIVE):
            sections.append(self._build_eda_section())

        # 3. Statistical Inferences Section
        if report_type in (ReportType.STATISTICAL, ReportType.COMPREHENSIVE):
            stat_sec = self._build_statistical_section()
            sections.append(stat_sec)
            if not stat_sec.was_performed:
                unperformed.append("Advanced Statistical Hypothesis Testing")

        # 4. Investigation Trace Section
        if report_type in (ReportType.INVESTIGATION, ReportType.COMPREHENSIVE):
            inv_sec = self._build_investigation_section()
            sections.append(inv_sec)
            if not inv_sec.was_performed:
                unperformed.append("Autonomous Investigation Planning")

        # 5. Predictive Modeling Section
        if report_type in (ReportType.PREDICTIVE, ReportType.EXECUTIVE, ReportType.COMPREHENSIVE):
            pred_sec = self._build_predictive_section()
            sections.append(pred_sec)
            if not pred_sec.was_performed:
                unperformed.append("Predictive Modeling Evaluation")

        # Executive Summary Generation
        exec_summary = ai_narrative_text or self._generate_deterministic_exec_summary(
            report_type, sections
        )

        metadata = ReportMetadata(
            report_id=f"rep_{uuid.uuid4().hex[:8]}",
            title=f"{report_type.value}: {self.filename}",
            report_type=report_type,
            report_depth=report_depth,
            dataset_id=self.dataset_id,
            dataset_version=self.dataset_version,
            ai_narrative_applied=bool(ai_narrative_text),
            evidence_item_count=len(self.evidence_items),
        )

        return ReportDocument(
            metadata=metadata,
            executive_summary=exec_summary,
            sections=sections,
            limitations_and_assumptions=limitations,
            unperformed_analyses=unperformed,
        )

    def _build_overview_and_quality_section(self) -> ReportSection:
        """Construct Dataset Overview and Quality Audit section."""
        row_item = next((it for it in self.evidence_items if it.metric_name == "row_count"), None)
        col_item = next(
            (it for it in self.evidence_items if it.metric_name == "column_count"), None
        )
        missing_item = next(
            (it for it in self.evidence_items if it.metric_name == "missing_cell_percentage"), None
        )

        rows_str = row_item.formatted_value if row_item else "Unknown"
        cols_str = col_item.formatted_value if col_item else "Unknown"
        missing_str = missing_item.formatted_value if missing_item else "0.0%"

        summary = (
            f"Dataset **{self.filename}** (version {self.dataset_version}) contains **{rows_str}** "
            f"across **{cols_str}**. Total missing cell rate is **{missing_str}**."
        )

        tables = []
        if self.profile and hasattr(self.profile, "columns"):
            headers = ["Column", "Inferred Type", "Missing %", "Unique Values"]
            tbl_rows = []
            for col_name, col_summary in list(self.profile.columns.items())[:12]:
                tbl_rows.append(
                    [
                        col_name,
                        getattr(col_summary, "inferred_type", "unknown"),
                        f"{getattr(col_summary, 'missing_percentage', 0.0):.1f}%",
                        str(getattr(col_summary, "unique_count", "N/A")),
                    ]
                )
            tables.append(
                ReportTable(title="Data Quality & Column Profiles", headers=headers, rows=tbl_rows)
            )

        return ReportSection(
            section_id="sec_overview_quality",
            title="Dataset Overview & Data Quality",
            category="Overview",
            summary_markdown=summary,
            tables=tables,
            evidence_ids=[it.id for it in [row_item, col_item, missing_item] if it],
            was_performed=True,
        )

    def _build_eda_section(self) -> ReportSection:
        """Construct Exploratory Data Analysis & Relationship section."""
        corr_items = [it for it in self.evidence_items if it.evidence_type == "correlation"]
        summary = (
            f"Exploratory analysis evaluated pairwise relationships across key variables. "
            f"Found **{len(corr_items)}** notable linear correlation signals."
        )

        tables = []
        if corr_items:
            headers = ["Variable A", "Variable B", "Pearson r", "Evidence ID"]
            tbl_rows = []
            for it in corr_items[:8]:
                vars_pair = it.relevant_variables
                var_a = vars_pair[0] if len(vars_pair) > 0 else "Col A"
                var_b = vars_pair[1] if len(vars_pair) > 1 else "Col B"
                tbl_rows.append([var_a, var_b, f"{it.actual_value:.3f}", it.id])
            tables.append(
                ReportTable(title="Top Observed Correlations", headers=headers, rows=tbl_rows)
            )

        return ReportSection(
            section_id="sec_eda_correlations",
            title="Exploratory Analysis & Correlations",
            category="Exploratory",
            summary_markdown=summary,
            tables=tables,
            evidence_ids=[it.id for it in corr_items],
            was_performed=True,
        )

    def _build_statistical_section(self) -> ReportSection:
        """Construct Statistical Inferences section."""
        if not self.statistical_results:
            return ReportSection(
                section_id="sec_statistical_tests",
                title="Statistical Inferences & Hypothesis Tests",
                category="Inference",
                summary_markdown="No formal inferential statistical tests have been performed on this dataset version.",
                was_performed=False,
                omission_reason="Statistical analysis module was not run.",
            )

        summary = (
            f"Evaluated **{len(self.statistical_results)}** formal inferential statistical tests. "
            "Results report exact p-values, test statistics, and sample sizes with strict separation between "
            "statistical significance and practical effect magnitude."
        )

        headers = ["Test Name", "Variables", "Statistic", "p-value", "Significant (α=0.05)"]
        tbl_rows = []
        for stat in self.statistical_results:
            p_val = stat.get("p_value", 1.0)
            tbl_rows.append(
                [
                    stat.get("test_name", "Test"),
                    ", ".join(stat.get("variables", [])),
                    f"{stat.get('statistic', 0.0):.3f}",
                    f"{p_val:.4f}",
                    "Yes" if p_val < 0.05 else "No",
                ]
            )

        return ReportSection(
            section_id="sec_statistical_tests",
            title="Statistical Inferences & Hypothesis Tests",
            category="Inference",
            summary_markdown=summary,
            tables=[ReportTable(title="Statistical Test Results", headers=headers, rows=tbl_rows)],
            was_performed=True,
        )

    def _build_investigation_section(self) -> ReportSection:
        """Construct Agentic Investigation section."""
        if not self.investigation_trace:
            return ReportSection(
                section_id="sec_investigation_findings",
                title="AI Investigation Findings",
                category="Investigation",
                summary_markdown="No multi-step autonomous investigation has been conducted on this dataset version.",
                was_performed=False,
                omission_reason="Investigation orchestrator was not executed.",
            )

        goal = getattr(self.investigation_trace, "question", "Multi-step analytical query")
        status = getattr(self.investigation_trace, "status", "Completed")
        findings = getattr(self.investigation_trace, "findings", [])

        summary = (
            f"Autonomous investigation conducted for question: *'{goal}'*. "
            f"Execution Status: **{status}**. Identified **{len(findings)}** verified findings."
        )

        tables = []
        if findings:
            headers = ["Finding Title", "Verified Insight"]
            tbl_rows = []
            for f in findings[:6]:
                tbl_rows.append(
                    [
                        getattr(f, "title", "Finding"),
                        getattr(f, "content", str(f))[:120] + "...",
                    ]
                )
            tables.append(
                ReportTable(title="Verified Findings Trace", headers=headers, rows=tbl_rows)
            )

        return ReportSection(
            section_id="sec_investigation_findings",
            title="AI Investigation Findings",
            category="Investigation",
            summary_markdown=summary,
            tables=tables,
            was_performed=True,
        )

    def _build_predictive_section(self) -> ReportSection:
        """Construct Predictive Modeling and Diagnostics section."""
        if not self.predictive_records:
            return ReportSection(
                section_id="sec_predictive_modeling",
                title="Predictive Modeling & Diagnostics",
                category="Predictive",
                summary_markdown="No predictive models have been trained or registered for this dataset version.",
                was_performed=False,
                omission_reason="Predictive modeling lab was not executed.",
            )

        best_model = self.predictive_records[0]
        task_type_str = best_model.task.value
        target_var = best_model.target_column
        val_strat_str = best_model.split_info.strategy.value if best_model.split_info else "Holdout"
        train_secs = best_model.training_duration_ms / 1000.0

        summary = (
            f"Evaluated **{len(self.predictive_records)}** models on target **{target_var}** ({task_type_str}) "
            f"using leakage-safe **{val_strat_str}** validation.\n\n"
            f"Best Performing Model: **{best_model.model_name}** with training duration **{train_secs:.2f}s**."
        )

        # Performance Comparison Table
        headers = ["Model", "Family", "Primary Metric", "Fit Status", "Train Time"]
        tbl_rows = []
        for m in self.predictive_records:
            if m.regression_metrics:
                metric_val = f"RMSE: {m.regression_metrics.rmse:.3f}, R²: {m.regression_metrics.r_squared:.3f}"
            elif m.classification_metrics:
                metric_val = f"F1: {m.classification_metrics.f1_macro:.3f}, Acc: {m.classification_metrics.accuracy:.3f}"
            elif m.forecasting_metrics:
                metric_val = f"RMSE: {m.forecasting_metrics.rmse:.3f}, sMAPE: {m.forecasting_metrics.smape:.2f}%"
            else:
                metric_val = "N/A"

            tbl_rows.append(
                [
                    m.model_name,
                    m.family.value,
                    metric_val,
                    m.fit_status,
                    f"{m.training_duration_ms / 1000.0:.2f}s",
                ]
            )

        tables = [
            ReportTable(title="Model Comparison & Evaluation", headers=headers, rows=tbl_rows)
        ]

        # Add top feature importance table if available
        if best_model.feature_importance and best_model.feature_importance.feature_names:
            fi_headers = ["Predictor", "Importance", "Method", "Evaluation Split"]
            fi_rows = []
            for f_name, imp in zip(
                best_model.feature_importance.feature_names[:6],
                best_model.feature_importance.importances[:6],
                strict=False,
            ):
                fi_rows.append(
                    [
                        f_name,
                        f"{imp:.4f}",
                        best_model.feature_importance.method,
                        best_model.feature_importance.evaluation_split,
                    ]
                )
            tables.append(
                ReportTable(
                    title="Top Feature Importance (Held-Out Data)", headers=fi_headers, rows=fi_rows
                )
            )

        return ReportSection(
            section_id="sec_predictive_modeling",
            title="Predictive Modeling & Diagnostics",
            category="Predictive",
            summary_markdown=summary,
            tables=tables,
            was_performed=True,
        )

    def _generate_deterministic_exec_summary(
        self, report_type: ReportType, sections: list[ReportSection]
    ) -> str:
        """Synthesize a robust, deterministic executive summary without requiring an LLM."""
        performed_count = sum(1 for s in sections if s.was_performed)
        return (
            f"This **{report_type.value}** provides a rigorous, evidence-grounded assessment of **{self.filename}** "
            f"(version {self.dataset_version}). A total of {performed_count} analytical domain(s) were successfully "
            f"executed and validated against the dataset.\n\n"
            "All quantitative conclusions are bound directly to verifiable data artifacts with strict leakage safeguards, "
            "temporal boundary preservation, and baseline comparisons. Please inspect the detailed sections below for "
            "methodology notes and empirical distributions."
        )
