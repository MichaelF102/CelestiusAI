"""Report Evidence Collector.

Gathers deterministic evidence across ingestion, profiling, statistical inference,
orchestrated investigations, and predictive modeling runs to ensure reports
only cite verified findings with strict dataset-version provenance.
"""

from __future__ import annotations

import logging
from typing import Any

from src.ingestion.loader import ManagedDataset
from src.interpretation.evidence_pack import EvidencePackBuilder
from src.interpretation.interpretation_models import EvidenceItem, EvidencePack
from src.predictive.model_registry import ModelRegistry
from src.predictive.predictive_models import TrainedModelRecord

logger = logging.getLogger(__name__)


class ReportEvidenceCollector:
    """Collects and organizes verified evidence items for intelligence reporting."""

    def __init__(
        self, managed_dataset: ManagedDataset, model_registry: ModelRegistry | None = None
    ) -> None:
        self.managed_dataset = managed_dataset
        self.model_registry = model_registry
        self.dataset_version = getattr(managed_dataset, "version", 1)
        self.dataset_id = str(getattr(managed_dataset, "dataset_id", "dataset"))

    def collect_all_evidence(
        self,
        investigation_trace: Any | None = None,
        statistical_results: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Aggregate evidence into organized categories."""
        # 1. Base EvidencePack (dimensions, missingness, numeric summaries, correlations)
        base_pack: EvidencePack = EvidencePackBuilder.build_pack(self.managed_dataset)
        evidence_by_id: dict[str, EvidenceItem] = {item.id: item for item in base_pack.items}

        # 2. Add Predictive Model Evidence
        predictive_records: list[TrainedModelRecord] = []
        if self.model_registry:
            predictive_records = self.model_registry.list_models(
                dataset_id=self.dataset_id,
                active_version=self.dataset_version,
            )

        for rec in predictive_records:
            ev_id = f"ev_pred_{rec.model_id}"
            primary_metric_val = None
            if rec.regression_metrics:
                primary_metric_val = rec.regression_metrics.rmse
            elif rec.classification_metrics:
                primary_metric_val = rec.classification_metrics.f1_macro
            elif rec.forecasting_metrics:
                primary_metric_val = rec.forecasting_metrics.rmse

            split_strat = rec.split_info.strategy.value if rec.split_info else "holdout"
            test_n = rec.split_info.test_size if rec.split_info else None

            ev_item = EvidenceItem(
                id=ev_id,
                evidence_type="metadata",
                dataset_version=rec.dataset_version,
                source_service="predictive.training_service",
                calculation_method=f"{rec.model_name} on {split_strat}",
                relevant_variables=[rec.target_column] + rec.predictor_columns,
                metric_name=f"{rec.model_name}_score",
                actual_value=primary_metric_val if primary_metric_val is not None else 0.0,
                formatted_value=f"Primary Score: {primary_metric_val}",
                sample_size=test_n,
                caveats=rec.warnings,
            )
            evidence_by_id[ev_id] = ev_item

        # 3. Add Statistical Results if provided
        if statistical_results:
            for idx, stat in enumerate(statistical_results):
                stat_id = f"ev_stat_{idx}_{stat.get('test_name', 'test')}"
                ev_item = EvidenceItem(
                    id=stat_id,
                    evidence_type="hypothesis_test",
                    dataset_version=self.dataset_version,
                    source_service="analytics.statistical",
                    calculation_method=stat.get("method", "statistical_test"),
                    relevant_variables=stat.get("variables", []),
                    metric_name=stat.get("test_name", "test"),
                    actual_value=stat.get("p_value", 1.0),
                    formatted_value=f"p={stat.get('p_value', 1.0):.4f}, stat={stat.get('statistic', 0.0):.3f}",
                    sample_size=stat.get("sample_size"),
                    caveats=stat.get("assumptions_violated", []),
                )
                evidence_by_id[stat_id] = ev_item

        return {
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "filename": getattr(self.managed_dataset, "filename", self.dataset_id),
            "evidence_items": list(evidence_by_id.values()),
            "evidence_by_id": evidence_by_id,
            "profile": getattr(self.managed_dataset, "profile", None),
            "predictive_records": predictive_records,
            "investigation_trace": investigation_trace,
            "statistical_results": statistical_results or [],
        }
