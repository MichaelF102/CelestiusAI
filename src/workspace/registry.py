"""On-disk model registry: versions, stages, model cards. Models are joblib files (trusted source only)."""

from __future__ import annotations

import io
import json
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib

from src.datascience.modeling import FittedModel, export_model_bytes, model_from_bundle

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "artifacts" / "models"
STAGES = ("draft", "staging", "production", "archived")
_SLUG = re.compile(r"[^a-z0-9_-]+")


class RegistryError(ValueError):
    """Invalid registry operation (message is user-facing)."""


def slugify(name: str) -> str:
    slug = _SLUG.sub("-", name.strip().lower()).strip("-_")
    if not slug:
        raise RegistryError("Model name must contain letters or digits.")
    return slug[:60]


def model_card_markdown(
    model: FittedModel,
    name: str,
    version: int,
    stage: str,
    notes: str = "",
    extras: dict[str, Any] | None = None,
) -> str:
    """A model card: what it is, how it was trained, how it performs, and where it can fail."""
    extras = extras or {}
    lines = [
        f"# Model card: {name} (v{version})",
        "",
        f"- **Stage:** {stage}",
        f"- **Algorithm:** {model.estimator_name}",
        f"- **Task:** {model.task.replace('_', ' ')}",
        f"- **Target:** `{model.target}`",
        f"- **Trained on:** {model.rows_used:,} rows ({model.trained_at})",
    ]
    if extras.get("data_fingerprint"):
        lines.append(
            f"- **Data fingerprint:** `{extras['data_fingerprint']}` (dataset version {extras.get('data_version', '?')})"
        )
    lines += [
        "",
        "## Intended use",
        notes.strip() or "_Describe who uses this model and for which decisions._",
        "",
        "## Performance (held-out data)",
        "",
    ]
    lines += [f"- {k}: {v:.4f}" for k, v in model.metrics.items()] or ["- _not recorded_"]
    lines += [
        "",
        "## Features",
        "",
        "| Feature | Type | Training range / values |",
        "|---|---|---|",
    ]
    for feat in model.features:
        info = model.feature_info.get(feat, {})
        if info.get("kind") == "numeric":
            rng = f"{info['min']:g} to {info['max']:g} (median {info['default']:g})"
        else:
            rng = ", ".join(map(str, info.get("choices", [])[:8])) + (
                "..." if len(info.get("choices", [])) > 8 else ""
            )
        lines.append(f"| {feat} | {info.get('kind', '?')} | {rng} |")
    if model.params:
        lines += ["", "## Hyperparameters", ""] + [
            f"- `{k}` = `{v}`" for k, v in model.params.items()
        ]
    fairness = extras.get("fairness")
    if fairness:
        lines += ["", "## Fairness summary", ""] + [
            f"- {w}"
            for w in (
                fairness.get("warnings") or ["No disparities flagged at the checked thresholds."]
            )
        ]
        if fairness.get("disparate_impact") is not None:
            lines.append(f"- Disparate impact ratio: {fairness['disparate_impact']:.2f}")
    limits = [
        "Predictions outside the training ranges above are extrapolations.",
        "Performance was measured on one held-out split; monitor it after deployment.",
    ]
    if model.rows_used < 1000:
        limits.insert(0, f"Trained on only {model.rows_used:,} rows: estimates are noisy.")
    lines += ["", "## Limitations", ""] + [f"- {x}" for x in limits]
    lines += [
        "",
        "## Loading",
        "```python",
        "import joblib",
        'bundle = joblib.load("model.joblib")   # only load files you trust (pickle)',
        'pipeline = bundle["pipeline"]',
        "```",
        "",
    ]
    return "\n".join(lines)


class ModelRegistry:
    def __init__(self, root: Path | str | None = None) -> None:
        self.root = Path(root) if root else DEFAULT_ROOT

    def _dir(self, slug: str) -> Path:
        path = (self.root / slugify(slug)).resolve()
        if self.root.resolve() not in path.parents:
            raise RegistryError("Invalid model path.")
        return path

    def versions(self, slug: str) -> list[dict[str, Any]]:
        base = self._dir(slug)
        if not base.is_dir():
            return []
        metas = []
        for sub in sorted(base.iterdir()):
            meta = sub / "meta.json"
            if meta.exists():
                metas.append(json.loads(meta.read_text("utf-8")))
        return sorted(metas, key=lambda m: m["version"])

    def list_models(self) -> list[dict[str, Any]]:
        if not self.root.exists():
            return []
        out = []
        for path in sorted(self.root.iterdir()):
            metas = self.versions(path.name)
            if not metas:
                continue
            latest = metas[-1]
            prod = next((m for m in reversed(metas) if m["stage"] == "production"), None)
            out.append(
                {
                    "slug": path.name,
                    "name": latest["name"],
                    "versions": len(metas),
                    "latest_version": latest["version"],
                    "latest_stage": latest["stage"],
                    "production_version": prod["version"] if prod else None,
                    "task": latest["task"],
                    "updated": latest["registered_at"],
                }
            )
        return sorted(out, key=lambda m: m["updated"], reverse=True)

    def register(
        self,
        model: FittedModel,
        name: str,
        notes: str = "",
        stage: str = "draft",
        extras: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if stage not in STAGES:
            raise RegistryError(f"Unknown stage '{stage}'.")
        slug = slugify(name)
        number = len(self.versions(slug)) + 1
        vdir = self._dir(slug) / f"v{number:03d}"
        vdir.mkdir(parents=True, exist_ok=True)
        (vdir / "model.joblib").write_bytes(export_model_bytes(model))
        meta = {
            "name": name.strip(),
            "slug": slug,
            "version": number,
            "stage": stage,
            "registered_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "notes": notes,
            "task": model.task,
            "target": model.target,
            "estimator": model.estimator_name,
            "metrics": model.metrics,
            "features": model.features,
            "extras": {
                k: v for k, v in (extras or {}).items() if k in ("data_fingerprint", "data_version")
            },
        }
        (vdir / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        (vdir / "MODEL_CARD.md").write_text(
            model_card_markdown(model, name.strip(), number, stage, notes, extras), encoding="utf-8"
        )
        if stage == "production":
            self.set_stage(slug, number, "production")
        return meta

    def load(self, slug: str, version: int | None = None) -> FittedModel:
        metas = self.versions(slug)
        if not metas:
            raise RegistryError(f"No model named '{slug}'.")
        meta = next((m for m in metas if m["version"] == version), None) if version else metas[-1]
        if meta is None:
            raise RegistryError(f"Version {version} not found.")
        path = self._dir(slug) / f"v{meta['version']:03d}" / "model.joblib"
        return model_from_bundle(joblib.load(io.BytesIO(path.read_bytes())))

    def card(self, slug: str, version: int | None = None) -> str:
        metas = self.versions(slug)
        if not metas:
            raise RegistryError(f"No model named '{slug}'.")
        number = version or metas[-1]["version"]
        return (self._dir(slug) / f"v{number:03d}" / "MODEL_CARD.md").read_text("utf-8")

    def set_stage(self, slug: str, version: int, stage: str) -> None:
        if stage not in STAGES:
            raise RegistryError(f"Unknown stage '{stage}'.")
        metas = self.versions(slug)
        if not any(m["version"] == version for m in metas):
            raise RegistryError(f"Version {version} not found.")
        for m in metas:
            vdir = self._dir(slug) / f"v{m['version']:03d}"
            new_stage = m["stage"]
            if m["version"] == version:
                new_stage = stage
            elif stage == "production" and m["stage"] == "production":
                new_stage = "archived"  # only one production version at a time
            if new_stage != m["stage"]:
                m["stage"] = new_stage
                (vdir / "meta.json").write_text(
                    json.dumps(m, indent=2, default=str), encoding="utf-8"
                )
                card_path = vdir / "MODEL_CARD.md"
                if card_path.exists():
                    card_path.write_text(
                        re.sub(
                            r"- \*\*Stage:\*\* \w+",
                            f"- **Stage:** {new_stage}",
                            card_path.read_text("utf-8"),
                            count=1,
                        ),
                        encoding="utf-8",
                    )

    def delete(self, slug: str) -> None:
        path = self._dir(slug)
        if path.exists():
            shutil.rmtree(path)
