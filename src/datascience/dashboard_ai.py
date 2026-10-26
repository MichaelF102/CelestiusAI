"""Dashboard suggestions, ready-made templates, and plain-language tile creation."""

from __future__ import annotations

import json
import re
import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from src.datascience import dashboard as db

TEMPLATES = {
    "auto": "Auto (best fit for this data)",
    "sales": "Sales performance",
    "customer": "Customers",
    "operations": "Operations",
    "quality": "Data quality",
}
_MEASURE_HINTS = (
    "revenue",
    "sales",
    "amount",
    "spend",
    "price",
    "total",
    "cost",
    "profit",
    "value",
    "income",
)
_CATEGORY_HINTS = (
    "region",
    "country",
    "product",
    "category",
    "segment",
    "plan",
    "channel",
    "status",
    "type",
    "city",
    "state",
)
_BINARY_HINTS = ("churn", "convert", "active", "retained", "default", "fraud", "purchase", "click")


def _looks_like_date(series: pd.Series) -> bool:
    """True for datetime columns and for text columns whose values parse as dates."""
    if str(series.dtype).startswith("datetime"):
        return True
    if series.dtype != object:
        return False
    sample = series.dropna().astype(str).head(60)
    if sample.empty or sample.str.fullmatch(r"-?\d+(\.\d+)?").any():
        return False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return bool(pd.to_datetime(sample, errors="coerce").notna().mean() > 0.9)


class ColumnRoles:
    """Best-guess roles for the columns of a table."""

    def __init__(self, df: pd.DataFrame, schema: Mapping[str, Any] | None = None) -> None:
        self.df = df
        numeric = [c for c in df.select_dtypes("number").columns if df[c].nunique() > 1]
        ids = {c for c, s in (schema or {}).items() if getattr(s, "is_identifier", False)} | {
            c for c in numeric if c.lower().endswith("id") or c.lower() == "id"
        }
        self.dates = [c for c in df.columns if _looks_like_date(df[c])]
        self.binary = [c for c in df.columns if df[c].dropna().nunique() == 2 and c not in ids]
        self.measures = [c for c in numeric if c not in ids and c not in self.binary]
        self.categories = [
            c
            for c in df.columns
            if c not in numeric
            and c not in self.dates
            and c not in ids
            and 2 <= df[c].nunique() <= 30
        ]
        self.measures.sort(key=lambda c: (not any(h in c.lower() for h in _MEASURE_HINTS), c))
        self.categories.sort(
            key=lambda c: (not any(h in c.lower() for h in _CATEGORY_HINTS), df[c].nunique())
        )
        self.binary.sort(key=lambda c: not any(h in c.lower() for h in _BINARY_HINTS))
        self.ids = sorted(ids)


def _t(kind: str, title: str, section: str = "", width: int = 1, **params: Any) -> db.Tile:
    options = params.pop("options", {})
    return db.Tile(
        kind=kind, title=title, params=params, section=section, width=width, options=options
    )


def quality_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per column: type, missing %, distinct count (the source for the quality template)."""
    return pd.DataFrame(
        {
            "column": df.columns.astype(str),
            "dtype": [str(df[c].dtype) for c in df.columns],
            "missing_pct": [round(float(df[c].isna().mean() * 100), 2) for c in df.columns],
            "distinct": [int(df[c].nunique(dropna=True)) for c in df.columns],
        }
    )


def template_spec(
    name: str, df: pd.DataFrame, schema: Mapping[str, Any] | None = None, title: str = ""
) -> db.DashboardSpec:
    """A starting dashboard built from the data's column roles. Raises if the data cannot support it."""
    if name not in TEMPLATES:
        raise db.DashboardError(f"Unknown template '{name}'.")
    r = ColumnRoles(df, schema)
    measure = r.measures[0] if r.measures else None
    measure2 = r.measures[1] if len(r.measures) > 1 else None
    cat = r.categories[0] if r.categories else None
    cat2 = r.categories[1] if len(r.categories) > 1 else None
    date = r.dates[0] if r.dates else None
    spec = db.DashboardSpec(title=title or TEMPLATES[name].split(" (")[0], columns=2)

    if name == "quality":
        spec.tiles = [
            _t("kpi", "Columns", "Overview", column=None, agg="count", options={}),
            _t(
                "bar",
                "Missing % by column",
                "Overview",
                width=2,
                x="column",
                y="missing_pct",
                agg="max",
                options={"sort": "desc", "orientation": "v"},
            ),
            _t(
                "table",
                "Column summary",
                "Details",
                width=2,
                columns=["column", "dtype", "missing_pct", "distinct"],
                rows=50,
                options={
                    "highlight": [{"column": "missing_pct", "op": ">", "value": 20, "color": "red"}]
                },
            ),
        ]
        spec.tiles[0].source = spec.tiles[1].source = spec.tiles[2].source = "quality_summary"
        spec.tiles[0].params = {"column": "missing_pct", "agg": "mean"}
        spec.tiles[0].title = "Average missing %"
        spec.tiles[0].options = {
            "decimals": 1,
            "suffix": "%",
            "thresholds": [{"op": ">", "value": 10, "color": "red"}],
        }
        return spec

    kpis: list[db.Tile] = [
        _t(
            "kpi",
            "Rows",
            "Overview",
            agg="count",
            options={"sparkline": {"date_col": date, "freq": "W"}} if date else {},
        )
    ]
    if name in ("auto", "sales", "operations") and measure:
        opts: dict[str, Any] = {}
        if date:
            opts = {
                "compare": {"kind": "previous_period", "date_col": date, "days": 30},
                "sparkline": {"date_col": date, "freq": "W"},
            }
        kpis.append(
            _t("kpi", f"Total {measure}", "Overview", column=measure, agg="sum", options=opts)
        )
        kpis.append(
            _t(
                "kpi",
                f"Average {measure}",
                "Overview",
                column=measure,
                agg="mean",
                options={"decimals": 2},
            )
        )
    if name in ("customer", "auto") and r.binary:
        b = r.binary[0]
        kpis.append(
            _t("kpi", f"{b} rate", "Overview", column=b, agg="mean", options={"format": "percent"})
        )
    spec.tiles += kpis

    if date and measure:
        spec.tiles.append(
            _t(
                "line",
                f"{measure} over time",
                "Trends",
                width=2,
                x=date,
                y=measure,
                agg="sum",
                freq="W",
                options={"ref_lines": [{"kind": "mean", "label": "average"}]},
            )
        )
    if cat and measure:
        spec.tiles.append(
            _t(
                "bar",
                f"{measure} by {cat}",
                "Breakdown",
                x=cat,
                y=measure,
                agg="sum",
                options={"sort": "desc", "top_n": 10, "other_group": True, "data_labels": True},
            )
        )
    elif cat:
        spec.tiles.append(
            _t(
                "bar",
                f"Records by {cat}",
                "Breakdown",
                x=cat,
                agg="count",
                options={"sort": "desc"},
            )
        )
    if cat:
        spec.tiles.append(
            _t(
                "pie",
                f"Share by {cat}",
                "Breakdown",
                names=cat,
                values=measure,
                agg="sum" if measure else "count",
                options={"donut": True},
            )
        )
    if name in ("customer", "auto") and r.binary and cat:
        b = r.binary[0]
        spec.tiles.append(
            _t(
                "bar",
                f"{b} rate by {cat}",
                "Customers",
                width=1,
                x=cat,
                y=b,
                agg="mean",
                options={"sort": "desc"},
            )
        )
    if measure:
        spec.tiles.append(
            _t("hist", f"Distribution of {measure}", "Distributions", column=measure, bins=30)
        )
    if measure and measure2:
        spec.tiles.append(
            _t(
                "scatter",
                f"{measure2} vs {measure}",
                "Distributions",
                x=measure,
                y=measure2,
                color=cat,
                options={"trendline": True},
            )
        )
    if cat and cat2 and measure:
        spec.tiles.append(
            _t(
                "heatmap",
                f"{measure}: {cat} x {cat2}",
                "Breakdown",
                width=2,
                x=cat2,
                y=cat,
                value=measure,
                agg="mean",
            )
        )
    spec.tiles.append(
        _t("table", "Data preview", "Details", width=2, columns=list(df.columns[:8]), rows=15)
    )
    if cat:
        spec.filters.append(db.Filter(column=cat, kind="categorical", values=[]))
    if date:
        spec.filters.append(db.Filter(column=date, kind="date", preset="all"))
    if not (r.measures or r.categories or r.dates):
        raise db.DashboardError("This data has too few usable columns for that template.")
    return spec


def suggest_spec(
    df: pd.DataFrame,
    schema: Mapping[str, Any] | None = None,
    findings: Sequence[Any] | None = None,
    title: str = "",
) -> db.DashboardSpec:
    """Auto template, plus a note tile that lists the strongest findings already computed."""
    spec = template_spec("auto", df, schema, title or "Suggested dashboard")
    top = [f for f in (findings or []) if getattr(f, "severity", "medium") in ("high", "critical")][
        :3
    ] or list(findings or [])[:3]
    if top:
        lines = "\n".join(f"- {getattr(f, 'title', '')}" for f in top)
        spec.tiles.insert(
            0,
            _t("text", "Key findings", "Overview", width=2, markdown=f"### Key findings\n{lines}"),
        )
    return spec


# --------------------------------------------------------------------------- plain language -> tile
_JSON = re.compile(r"\{.*\}", re.S)


def build_tile_prompt(request: str, df: pd.DataFrame) -> str:
    cols = ", ".join(f"{c} ({df[c].dtype})" for c in df.columns[:60])
    return (
        "Create ONE dashboard tile as a JSON object, nothing else.\n"
        f"Allowed kinds: {', '.join(db.TILE_LABELS)}.\n"
        'Shape: {"kind": ..., "title": ..., "params": {...}}.\n'
        "Params by kind: bar/area {x,y,agg}; line {x,y,agg,freq(D|W|MS)}; scatter {x,y,color}; "
        "hist {column,bins}; pie {names,values,agg}; kpi {column,agg}; table {columns,rows}.\n"
        f"agg is one of {', '.join(db.AGGS)}. Use only these columns: {cols}.\n"
        f"Request: {request}\n"
    )


def nl_to_tile(provider: Any, request: str, df: pd.DataFrame) -> db.Tile:
    """Ask an LLM for a tile and validate it. The caller must show it to the user before adding."""
    result = provider.generate_text(build_tile_prompt(request, df), temperature=0.0)
    if getattr(result, "status", "success") != "success" or not result.raw_text.strip():
        raise db.DashboardError(
            getattr(result, "error_message", None) or "The language model returned nothing."
        )
    match = _JSON.search(result.raw_text)
    try:
        tile = db.Tile.model_validate(json.loads(match.group(0) if match else result.raw_text))
    except (ValueError, TypeError) as err:
        raise db.DashboardError(f"The model did not return a valid tile: {err}") from err
    missing = [c for c in db.tile_columns(tile) if c not in df.columns]
    if missing:
        raise db.DashboardError(f"The suggested tile uses unknown column(s): {', '.join(missing)}.")
    return tile
