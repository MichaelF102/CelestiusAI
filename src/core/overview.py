"""Streamlit-independent view logic for the Executive Overview page."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}

# Finding category -> page where the user can dig deeper.
CATEGORY_PAGE = {
    "anomaly": "pages/2_Data_Quality.py",
}
DEFAULT_FINDING_PAGE = "pages/3_Exploration_Studio.py"

# Categories whose findings are collapsed into a single card once there are several.
_GROUP_LABELS = {
    "correlation": ("notable relationship", "notable relationships"),
    "distribution": ("skewed distribution", "skewed distributions"),
}
_GROUP_THRESHOLD = 2


@dataclass
class DisplayFinding:
    """A finding (or a group of similar findings) ready to render as a card."""

    title: str
    summary: str
    severity: str
    category: str
    page: str
    details: list[str] = field(default_factory=list)


def finding_severity(finding: Any) -> str:
    raw = getattr(finding, "severity", None) or getattr(finding, "severity_or_importance", None)
    raw = str(raw or "medium").lower()
    return raw if raw in SEVERITY_ORDER else "medium"


def _worst(severities: Iterable[str]) -> str:
    return min(severities, key=lambda s: SEVERITY_ORDER.get(s, 2))


def group_findings(findings: Iterable[Any]) -> list[DisplayFinding]:
    """Sort findings by severity and collapse repetitive categories into one card each."""
    buckets: dict[str, list[Any]] = {}
    for f in findings:
        buckets.setdefault(str(getattr(f, "category", "general")), []).append(f)

    items: list[DisplayFinding] = []
    for category, group in buckets.items():
        page = CATEGORY_PAGE.get(category, DEFAULT_FINDING_PAGE)
        if category in _GROUP_LABELS and len(group) > _GROUP_THRESHOLD:
            singular, plural = _GROUP_LABELS[category]
            ranked = sorted(group, key=lambda f: SEVERITY_ORDER[finding_severity(f)])
            items.append(
                DisplayFinding(
                    title=f"{len(group)} {plural}",
                    summary=ranked[0].title,
                    severity=_worst(finding_severity(f) for f in group),
                    category=category,
                    page=page,
                    details=[f.title for f in ranked],
                )
            )
            continue
        for f in group:
            items.append(
                DisplayFinding(
                    title=f.title,
                    summary=f.summary,
                    severity=finding_severity(f),
                    category=category,
                    page=page,
                )
            )
    items.sort(key=lambda i: SEVERITY_ORDER[i.severity])
    return items


def column_type_counts(schema: Mapping[str, Any] | None) -> dict[str, int]:
    """Count columns per inferred semantic type, e.g. {'numeric': 14, 'categorical': 4}."""
    counts = Counter(getattr(s, "inferred_type", "unknown") for s in (schema or {}).values())
    return dict(counts.most_common())


def format_type_counts(counts: Mapping[str, int]) -> str:
    return " · ".join(f"{n} {t}" for t, n in counts.items()) or "—"


@dataclass(frozen=True)
class NextStep:
    icon: str
    title: str
    detail: str
    page: str


def recommend_next_steps(
    profile: Any | None,
    findings: list[Any],
    schema: Mapping[str, Any] | None,
    limit: int = 4,
) -> list[NextStep]:
    """Context-aware suggestions for what to do next, most urgent first."""
    steps: list[NextStep] = []
    dq_page = "pages/2_Data_Quality.py"

    if profile is not None:
        issues = list(getattr(profile, "quality_issues", []) or [])
        urgent = [i for i in issues if i.severity in ("critical", "high")]
        if urgent:
            steps.append(
                NextStep(
                    "🚨",
                    f"Resolve {len(urgent)} high-severity quality issue(s)",
                    urgent[0].title,
                    dq_page,
                )
            )
        missing = getattr(profile, "missing_cell_percentage", 0.0)
        if missing > 0 and not urgent:
            steps.append(
                NextStep(
                    "🧩",
                    f"{missing:.1f}% of cells are missing",
                    "Review imputation options.",
                    dq_page,
                )
            )
        dups = getattr(profile, "duplicate_row_count", 0)
        if dups > 0:
            steps.append(
                NextStep("♻️", f"Remove {dups:,} duplicate rows", "Exact duplicates.", dq_page)
            )

    if not findings:
        steps.append(
            NextStep(
                "📈",
                "Explore the data",
                "Generate distributions and relationships.",
                "pages/3_Exploration_Studio.py",
            )
        )
    elif any(getattr(f, "category", "") == "correlation" for f in findings):
        steps.append(
            NextStep(
                "🧪",
                "Test the strongest relationships",
                "Confirm correlations with significance tests and effect sizes.",
                "pages/4_Statistical_Intelligence.py",
            )
        )

    types = column_type_counts(schema)
    if types.get("datetime"):
        steps.append(
            NextStep(
                "🔮", "Forecast a trend", "A date column was detected.", "pages/6_Model_Lab.py"
            )
        )
    elif types.get("numeric", 0) >= 2:
        steps.append(
            NextStep(
                "🔮",
                "Train a predictive model",
                "Pick a target and compare baselines.",
                "pages/6_Model_Lab.py",
            )
        )

    steps.append(
        NextStep("🤖", "Ask the AI Analyst", "Evidence-grounded answers.", "pages/5_AI_Analyst.py")
    )
    if findings:
        steps.append(
            NextStep(
                "📑",
                "Generate a report",
                "Export an executive dossier.",
                "pages/7_Intelligence_Reports.py",
            )
        )
    return steps[:limit]


def build_summary_markdown(
    filename: str,
    rows: int,
    cols: int,
    profile: Any | None,
    display_findings: list[DisplayFinding],
) -> str:
    """Compact Markdown briefing of the overview, for one-click download."""
    lines = [
        f"# Executive Summary — {filename}",
        f"_Generated {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}_",
        "",
        f"- **Rows:** {rows:,}",
        f"- **Columns:** {cols:,}",
    ]
    if profile is not None and hasattr(profile, "quality_score"):
        lines += [
            f"- **Quality score:** {profile.quality_score.overall_score:.0f}/100",
            f"- **Missing cells:** {profile.missing_cell_percentage:.1f}%",
            f"- **Duplicate rows:** {profile.duplicate_row_percentage:.1f}%",
        ]
    lines += ["", "## Key findings"]
    if not display_findings:
        lines.append("_No findings generated yet._")
    for item in display_findings:
        lines.append(f"- **[{item.severity.upper()}] {item.title}** — {item.summary}")
        lines += [f"    - {d}" for d in item.details]
    return "\n".join(lines) + "\n"
