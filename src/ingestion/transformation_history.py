"""Transformation pipeline manager supporting execution, audit history, undo, and reset."""

from src.ingestion.loader import ManagedDataset
from src.ingestion.transformations import apply_transformation
from src.profiling.models import TransformationRecord, TransformationSpec


class TransformationPipeline:
    """Manages an ordered sequence of approved cleaning operations on a ManagedDataset."""

    def __init__(self) -> None:
        self.specs: list[TransformationSpec] = []
        self.records: list[TransformationRecord] = []

    def apply(
        self,
        spec: TransformationSpec,
        managed_dataset: ManagedDataset,
    ) -> TransformationRecord:
        """Apply a new transformation to the active working dataset and record audit history."""
        new_df, record = apply_transformation(managed_dataset.working_df, spec)
        managed_dataset.update_working_df(new_df)
        self.specs.append(spec)
        self.records.append(record)
        return record

    def undo(self, managed_dataset: ManagedDataset) -> TransformationRecord | None:
        """Revert the most recent transformation by replaying remaining specs from original raw_df."""
        if not self.specs:
            return None
        removed = self.records[-1]
        self.rollback_to(managed_dataset, len(self.specs) - 1)
        return removed

    def rollback_to(self, managed_dataset: ManagedDataset, keep: int) -> list[TransformationRecord]:
        """Keep the first `keep` transformations, revert the rest, and return the removed records.

        The working dataset is rebuilt deterministically from the immutable raw copy.
        """
        keep = max(0, min(keep, len(self.specs)))
        removed = self.records[keep:]
        if not removed:
            return []

        self.specs = self.specs[:keep]
        current_df = managed_dataset.raw_df
        new_records: list[TransformationRecord] = []
        for spec in self.specs:
            current_df, rec = apply_transformation(current_df, spec)
            new_records.append(rec)

        managed_dataset.update_working_df(current_df)
        self.records = new_records
        return removed

    def reset(self, managed_dataset: ManagedDataset) -> None:
        """Clear all transformation history and restore working dataset to the immutable raw copy."""
        self.specs.clear()
        self.records.clear()
        managed_dataset.reset_working_df()
