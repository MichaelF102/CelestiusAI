"""Export and replay of cleaning transformation history as a portable JSON recipe."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from src.ingestion.loader import ManagedDataset
from src.ingestion.transformation_history import TransformationPipeline
from src.profiling.models import TransformationSpec

RECIPE_VERSION = 1


def export_recipe(pipeline: TransformationPipeline, source_filename: str = "") -> str:
    """Serialize the applied transformation specs to a JSON recipe."""
    payload = {
        "recipe_version": RECIPE_VERSION,
        "source": source_filename,
        "created_at": datetime.now(UTC).isoformat(),
        "steps": [spec.model_dump(mode="json") for spec in pipeline.specs],
    }
    return json.dumps(payload, indent=2)


def replay_recipe(
    recipe_json: str | bytes, pipeline: TransformationPipeline, dataset: ManagedDataset
) -> tuple[int, list[str]]:
    """Apply every step of a recipe to `dataset`; failing steps are skipped and reported.

    Returns (applied_count, error_messages). Raises ValueError for a malformed recipe.
    """
    try:
        payload = json.loads(recipe_json)
        steps = payload["steps"]
        specs = [TransformationSpec.model_validate(step) for step in steps]
    except (ValueError, KeyError, TypeError) as err:
        raise ValueError(f"Not a valid cleaning recipe: {err}") from err

    applied = 0
    errors: list[str] = []
    for idx, spec in enumerate(specs, start=1):
        try:
            pipeline.apply(spec, dataset)
            applied += 1
        except Exception as err:
            errors.append(
                f"Step {idx} ({spec.action_type} on {spec.column_name or 'dataset'}): {err}"
            )
    return applied, errors
