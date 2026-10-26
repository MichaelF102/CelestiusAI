"""Save and restore whole analysis workspaces (data, cleaning recipe, dictionary, dashboard, tables).

Layout::

    <root>/<slug>/
        data/<fingerprint>.parquet      immutable raw data (stored once per distinct file)
        versions/v001/ meta.json recipe.json dictionary.json dashboard.json library/*.parquet
        events.jsonl                    run history (saved / loaded)
"""

from __future__ import annotations

import io
import json
import re
import shutil
from collections.abc import MutableMapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.core.session_state import (
    KEY_BASELINE_QUALITY_SCORE,
    KEY_CURRENT_DATASET,
    KEY_DASHBOARD_SPEC,
    KEY_DATA_DICTIONARY,
    KEY_DATA_QUALITY_REPORT,
    KEY_DATASET_LIBRARY,
    KEY_DATASET_METADATA,
    KEY_DATASET_SCHEMA,
    KEY_EXPERIMENT_LOG,
    get_transformation_pipeline,
)
from src.ingestion.loader import ManagedDataset
from src.ingestion.recipe import export_recipe, replay_recipe
from src.ingestion.service import ingest_into_state
from src.profiling.profiler import profile_dataset

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "artifacts" / "projects"
_SLUG_RE = re.compile(r"[^a-z0-9_-]+")


class ProjectError(ValueError):
    """Raised for invalid project names or missing project data."""


def slugify(name: str) -> str:
    slug = _SLUG_RE.sub("-", name.strip().lower()).strip("-_")
    if not slug:
        raise ProjectError("Project name must contain letters or digits.")
    return slug[:60]


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _to_parquet_bytes(df: pd.DataFrame) -> bytes:
    def dump(frame: pd.DataFrame) -> bytes:
        buf = io.BytesIO()
        frame.to_parquet(buf, engine="pyarrow")
        return buf.getvalue()

    try:
        return dump(df)
    except Exception:
        safe = df.copy()
        for col in safe.select_dtypes("object").columns:
            safe[col] = safe[col].map(lambda v: None if v is None or v != v else str(v))
        return dump(safe)


@dataclass
class ProjectInfo:
    name: str
    slug: str
    updated_at: str
    versions: int


class ProjectStore:
    def __init__(self, root: Path | str | None = None) -> None:
        self.root = Path(root) if root else DEFAULT_ROOT

    # ------------------------------------------------------------------ paths
    def _dir(self, slug: str) -> Path:
        slug = slugify(slug)
        path = (self.root / slug).resolve()
        if self.root.resolve() not in path.parents:
            raise ProjectError("Invalid project path.")
        return path

    def _log(self, slug: str, event: str, **extra: Any) -> None:
        path = self._dir(slug)
        path.mkdir(parents=True, exist_ok=True)
        with (path / "events.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"ts": _now(), "event": event, **extra}) + "\n")

    # ------------------------------------------------------------------ queries
    def list_projects(self) -> list[ProjectInfo]:
        if not self.root.exists():
            return []
        infos = []
        for path in sorted(self.root.iterdir()):
            if not (path / "versions").is_dir():
                continue
            metas = self.versions(path.name)
            if not metas:
                continue
            infos.append(
                ProjectInfo(
                    name=metas[-1].get("project_name", path.name),
                    slug=path.name,
                    updated_at=metas[-1]["saved_at"],
                    versions=len(metas),
                )
            )
        return sorted(infos, key=lambda i: i.updated_at, reverse=True)

    def versions(self, slug: str) -> list[dict[str, Any]]:
        vdir = self._dir(slug) / "versions"
        if not vdir.is_dir():
            return []
        metas = []
        for sub in sorted(vdir.iterdir()):
            meta_file = sub / "meta.json"
            if meta_file.exists():
                metas.append(json.loads(meta_file.read_text(encoding="utf-8")))
        return metas

    def events(self, slug: str) -> list[dict[str, Any]]:
        path = self._dir(slug) / "events.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

    # ------------------------------------------------------------------ save
    def save(self, state: MutableMapping[str, Any], name: str, note: str = "") -> int:
        dataset: ManagedDataset | None = state.get(KEY_CURRENT_DATASET)
        if not isinstance(dataset, ManagedDataset):
            raise ProjectError("Load a dataset before saving a project.")
        slug = slugify(name)
        base = self._dir(slug)
        (base / "data").mkdir(parents=True, exist_ok=True)

        raw_name = f"{dataset.fingerprint}.parquet"
        raw_path = base / "data" / raw_name
        if not raw_path.exists():
            raw_path.write_bytes(_to_parquet_bytes(dataset.raw_df))

        number = len(self.versions(slug)) + 1
        vdir = base / "versions" / f"v{number:03d}"
        (vdir / "library").mkdir(parents=True, exist_ok=True)

        pipeline = get_transformation_pipeline(state)
        (vdir / "recipe.json").write_text(
            export_recipe(pipeline, dataset.filename), encoding="utf-8"
        )
        (vdir / "dictionary.json").write_text(
            json.dumps(state.get(KEY_DATA_DICTIONARY) or {}, indent=2), encoding="utf-8"
        )
        (vdir / "dashboard.json").write_text(
            json.dumps(state.get(KEY_DASHBOARD_SPEC) or {}, indent=2), encoding="utf-8"
        )
        runs = state.get(KEY_EXPERIMENT_LOG)
        (vdir / "runs.json").write_text(
            runs.to_json() if runs is not None else "[]", encoding="utf-8"
        )
        library: dict[str, pd.DataFrame] = state.get(KEY_DATASET_LIBRARY) or {}
        table_names = []
        for table_name, frame in library.items():
            table_slug = slugify(table_name)
            (vdir / "library" / f"{table_slug}.parquet").write_bytes(_to_parquet_bytes(frame))
            table_names.append({"name": table_name, "file": f"{table_slug}.parquet"})

        profile = state.get(KEY_DATA_QUALITY_REPORT)
        report = dataset.ingestion_report
        meta = {
            "project_name": name.strip(),
            "version": number,
            "saved_at": _now(),
            "note": note,
            "filename": dataset.filename,
            "fingerprint": dataset.fingerprint,
            "raw_file": raw_name,
            "rows": len(dataset.working_df),
            "columns": len(dataset.working_df.columns),
            "transformations": len(pipeline.specs),
            "quality_score": getattr(
                getattr(profile, "quality_score", None), "overall_score", None
            ),
            "tables": table_names,
            "ingestion": {
                "sheet": getattr(report, "sheet_name_used", None),
                "encoding": getattr(report, "encoding_used", None),
                "delimiter": getattr(report, "delimiter_detected", None),
            },
        }
        (vdir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        self._log(slug, "saved", version=number, note=note)
        return number

    # ------------------------------------------------------------------ load
    def load(
        self,
        state: MutableMapping[str, Any],
        slug: str,
        version: int | None = None,
        max_size_mb: int = 100,
    ) -> dict[str, Any]:
        metas = self.versions(slug)
        if not metas:
            raise ProjectError(f"Project '{slug}' has no saved versions.")
        meta = next((m for m in metas if m["version"] == version), None) if version else metas[-1]
        if meta is None:
            raise ProjectError(f"Version {version} not found.")

        base = self._dir(slug)
        vdir = base / "versions" / f"v{meta['version']:03d}"
        raw_path = base / "data" / meta["raw_file"]
        if not raw_path.exists():
            raise ProjectError("The raw data file for this version is missing.")

        dataset = ingest_into_state(
            state, raw_path.read_bytes(), f"{Path(meta['filename']).stem}.parquet", max_size_mb
        )
        dataset.filename = meta["filename"]
        dataset.fingerprint = meta["fingerprint"]
        metadata = state.get(KEY_DATASET_METADATA)
        if metadata is not None:
            state[KEY_DATASET_METADATA] = metadata.model_copy(update={"filename": meta["filename"]})

        errors: list[str] = []
        recipe = (vdir / "recipe.json").read_text(encoding="utf-8")
        if json.loads(recipe).get("steps"):
            _, errors = replay_recipe(recipe, get_transformation_pipeline(state), dataset)
            profile = profile_dataset(dataset.working_df, dataset_id=dataset.dataset_id)
            state[KEY_DATA_QUALITY_REPORT] = profile
            state[KEY_DATASET_SCHEMA] = dataset.schema_info
        # Raw-file score is the baseline for the before/after comparison.
        raw_profile = profile_dataset(dataset.raw_df, dataset_id=dataset.dataset_id)
        state[KEY_BASELINE_QUALITY_SCORE] = raw_profile.quality_score.overall_score

        state[KEY_DATA_DICTIONARY] = json.loads((vdir / "dictionary.json").read_text("utf-8"))
        dashboard = json.loads((vdir / "dashboard.json").read_text("utf-8"))
        state[KEY_DASHBOARD_SPEC] = dashboard or None
        runs_file = vdir / "runs.json"
        if runs_file.exists():
            from src.datascience.tracking import ExperimentLog

            state[KEY_EXPERIMENT_LOG] = ExperimentLog.from_json(runs_file.read_text("utf-8"))
        library = {}
        for table in meta.get("tables", []):
            library[table["name"]] = pd.read_parquet(vdir / "library" / table["file"])
        state[KEY_DATASET_LIBRARY] = library

        self._log(slug, "loaded", version=meta["version"])
        return {"meta": meta, "replay_errors": errors}

    def delete(self, slug: str) -> None:
        path = self._dir(slug)
        if path.exists():
            shutil.rmtree(path)
