"""Predictive Model Registry.

Tracks trained models, version provenance, configuration metadata, and performance.
Flags models as stale if the active dataset version changes.
"""

from __future__ import annotations

import logging

from src.predictive.predictive_models import TrainedModelRecord

logger = logging.getLogger(__name__)


class ModelRegistry:
    """In-memory and session-compatible model registry with version awareness."""

    def __init__(self) -> None:
        self._models: dict[str, TrainedModelRecord] = {}

    def register_model(self, record: TrainedModelRecord) -> str:
        """Register a newly trained model record.

        Returns the model_id.
        """
        self._models[record.model_id] = record
        logger.info(
            f"Registered model {record.model_id} ({record.model_name}) "
            f"for dataset {record.dataset_id} v{record.dataset_version}"
        )
        return record.model_id

    def register_models(self, records: list[TrainedModelRecord]) -> list[str]:
        """Register multiple model records."""
        return [self.register_model(rec) for rec in records]

    def get_model(self, model_id: str) -> TrainedModelRecord | None:
        """Retrieve a model record by ID."""
        return self._models.get(model_id)

    def list_models(
        self,
        dataset_id: str | None = None,
        active_version: int | None = None,
    ) -> list[TrainedModelRecord]:
        """List registered models, optionally filtering by dataset and updating staleness."""
        models = list(self._models.values())
        if dataset_id:
            models = [m for m in models if m.dataset_id == dataset_id]

        if active_version is not None:
            for m in models:
                if m.dataset_version != active_version:
                    m.is_stale = True

        # Sort by creation time descending
        models.sort(key=lambda m: m.created_at, reverse=True)
        return models

    def get_best_model(
        self,
        dataset_id: str | None = None,
        active_version: int | None = None,
    ) -> TrainedModelRecord | None:
        """Return the best non-baseline model for the dataset."""
        candidates = self.list_models(dataset_id=dataset_id, active_version=active_version)
        non_baselines = [
            c for c in candidates if "Baseline" not in c.model_name and c.fit_status == "SUCCESS"
        ]
        if not non_baselines:
            return candidates[0] if candidates else None
        return non_baselines[0]

    def clear(self) -> None:
        """Clear all registered models."""
        self._models.clear()
