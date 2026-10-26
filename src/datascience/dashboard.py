"""Dashboard model: tiles, filters, parameters, calculated fields, figures, and interactions.

Exports (HTML/Excel/PDF), code generation, templates and AI helpers live in the sibling
``dashboard_export``, ``dashboard_code`` and ``dashboard_ai`` modules; thin wrappers at the
bottom keep the original public names working.
"""

from __future__ import annotations

import copy
import json
import uuid
from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pydantic import BaseModel, Field

from src.datascience.transform import TransformError, validate_expression

TileKind = Literal[
    "kpi",
    "gauge",
    "bar",
    "line",
    "area",
    "combo",
    "scatter",
    "hist",
    "pie",
    "box",
    "heatmap",
    "funnel",
    "waterfall",
    "treemap",
    "map",
    "table",
    "text",
]
TILE_LABELS = {
    "kpi": "KPI number",
    "gauge": "Gauge",
    "bar": "Bar chart",
    "line": "Line chart",
    "area": "Area chart",
    "combo": "Combo (bars + line)",
    "scatter": "Scatter plot",
    "hist": "Histogram",
    "pie": "Pie chart",
    "box": "Box plot",
    "heatmap": "Heatmap",
    "funnel": "Funnel",
    "waterfall": "Waterfall",
    "treemap": "Treemap",
    "map": "Map",
    "table": "Data table",
    "text": "Text / notes",
}
CHART_KINDS = tuple(k for k in TILE_LABELS if k not in ("kpi", "gauge", "table", "text"))
AGGS = ("sum", "mean", "median", "min", "max", "count", "nunique")
MAX_POINTS = 5000
DATE_PRESETS = {
    "all": "All dates",
    "last_7": "Last 7 days",
    "last_30": "Last 30 days",
    "last_90": "Last 90 days",
    "this_year": "This year",
}
COLORBLIND = [
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
    "#999999",
]
PALETTES = ("default", "colorblind", "viridis", "blues", "teal", "plasma")
SORTS = ("none", "asc", "desc")
STACKING = ("group", "stack", "percent")
VALUE_FORMATS = ("auto", "integer", "decimal", "currency", "percent")


class DashboardError(ValueError):
    """Invalid tile, filter, or specification (message is user-facing)."""


# --------------------------------------------------------------------------- models
class Filter(BaseModel):
    column: str
    kind: Literal["categorical", "numeric", "date", "text"]
    values: list[Any] | None = None
    low: float | None = None
    high: float | None = None
    start: str | None = None
    end: str | None = None
    preset: str | None = None  # date presets relative to the newest date in the data
    contains: str | None = None  # text search
    single: bool = False  # categorical: pick exactly one value (dropdown)
    tiles: list[str] | None = None  # None = every tile; else only these tile ids


class Tile(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    kind: TileKind
    title: str
    params: dict[str, Any] = Field(default_factory=dict)
    options: dict[str, Any] = Field(default_factory=dict)
    width: int = 1  # grid columns spanned (1-3)
    height: int = 340
    section: str = ""
    source: str = "active"
    apply_global_filters: bool = True
    local_filters: list[Filter] = Field(default_factory=list)


class Theme(BaseModel):
    template: Literal["dark", "light"] = "dark"
    palette: str = "default"
    font_size: int = 12
    accent: str = "#38bdf8"
    tile_border: bool = True


class Parameter(BaseModel):
    name: str
    kind: Literal["number", "choice"] = "number"
    value: Any = None
    options: list[Any] | None = None
    label: str = ""


class CalcField(BaseModel):
    name: str
    expr: str


class DashboardSpec(BaseModel):
    title: str = "My Dashboard"
    description: str = ""
    columns: int = 2
    tiles: list[Tile] = Field(default_factory=list)
    filters: list[Filter] = Field(default_factory=list)
    theme: Theme = Field(default_factory=Theme)
    parameters: list[Parameter] = Field(default_factory=list)
    calculated: list[CalcField] = Field(default_factory=list)

    @property
    def sections(self) -> list[str]:
        """Section names in first-appearance order ('' = unsectioned tiles come first)."""
        seen: list[str] = []
        for t in self.tiles:
            if t.section not in seen:
                seen.append(t.section)
        return seen


def spec_to_json(spec: DashboardSpec) -> str:
    return spec.model_dump_json(indent=2)


def spec_from_json(text: str | bytes) -> DashboardSpec:
    try:
        return DashboardSpec.model_validate(json.loads(text))
    except (ValueError, TypeError) as err:
        raise DashboardError(f"Not a valid dashboard file: {err}") from err


def clone_spec(spec: DashboardSpec) -> DashboardSpec:
    return spec.model_copy(deep=True)


# --------------------------------------------------------------------------- parameters & calculated fields
def param_values(spec: DashboardSpec) -> dict[str, Any]:
    return {p.name: p.value for p in spec.parameters}


def resolve(value: Any, params: Mapping[str, Any]) -> Any:
    """Replace ``{"$param": name}`` markers (recursively) with the parameter's current value."""
    if isinstance(value, dict):
        if set(value) == {"$param"}:
            name = value["$param"]
            if name not in params:
                raise DashboardError(f"Unknown parameter '{name}'.")
            return params[name]
        return {k: resolve(v, params) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve(v, params) for v in value]
    return value


def resolve_tile(tile: Tile, spec: DashboardSpec) -> Tile:
    params = param_values(spec)
    return tile.model_copy(
        update={"params": resolve(tile.params, params), "options": resolve(tile.options, params)}
    )


def apply_calculated(
    df: pd.DataFrame, fields: list[CalcField]
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Add calculated columns; returns (frame, {field: error}) for the ones that failed."""
    out = df.copy()
    errors: dict[str, str] = {}
    for f in fields:
        try:
            if f.name in df.columns:
                raise DashboardError("a column with this name already exists")
            out[f.name] = out.eval(validate_expression(f.expr))
        except (TransformError, DashboardError) as err:
            errors[f.name] = str(err)
        except Exception as err:
            errors[f.name] = f"could not evaluate: {err}"
    return out, errors


# --------------------------------------------------------------------------- filtering
def _preset_range(dates: pd.Series, preset: str) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    clean = dates.dropna()
    if clean.empty or preset == "all":
        return None, None
    end = clean.max()
    if preset == "this_year":
        return pd.Timestamp(year=end.year, month=1, day=1), end
    days = {"last_7": 7, "last_30": 30, "last_90": 90}.get(preset)
    if days is None:
        raise DashboardError(f"Unknown date preset '{preset}'.")
    return end - pd.Timedelta(days=days), end


def filter_applies(f: Filter, tile_id: str | None) -> bool:
    return tile_id is None or f.tiles is None or tile_id in f.tiles


def apply_filters(
    df: pd.DataFrame, filters: list[Filter], tile_id: str | None = None
) -> pd.DataFrame:
    out = df
    for f in filters:
        if f.column not in out.columns or not filter_applies(f, tile_id):
            continue
        col = out[f.column]
        if f.kind == "categorical" and f.values:
            out = out[col.astype(str).isin([str(v) for v in f.values])]
        elif f.kind == "numeric":
            num = pd.to_numeric(col, errors="coerce")
            if f.low is not None:
                out = out[num >= f.low]
            if f.high is not None:
                out = out[pd.to_numeric(out[f.column], errors="coerce") <= f.high]
        elif f.kind == "date":
            dates = pd.to_datetime(col, errors="coerce")
            start = pd.Timestamp(f.start) if f.start else None
            end = pd.Timestamp(f.end) if f.end else None
            if f.preset:
                start, end = _preset_range(dates, f.preset)
            if start is not None:
                out = out[dates.reindex(out.index) >= start]
            if end is not None:
                out = out[pd.to_datetime(out[f.column], errors="coerce") <= end]
        elif f.kind == "text" and f.contains:
            out = out[col.astype(str).str.contains(f.contains, case=False, regex=False, na=False)]
    return out


def tile_data(
    tables: Mapping[str, pd.DataFrame],
    spec: DashboardSpec,
    tile: Tile,
    extra_filters: list[Filter] | None = None,
) -> pd.DataFrame:
    """The filtered table a tile draws from (calculated fields, global, local and cross filters)."""
    if tile.source not in tables:
        raise DashboardError(f"Table '{tile.source}' is not available.")
    data, _ = apply_calculated(tables[tile.source], spec.calculated)
    if tile.apply_global_filters:
        data = apply_filters(data, spec.filters, tile.id)
    data = apply_filters(data, tile.local_filters)
    if extra_filters:
        data = apply_filters(data, extra_filters, tile.id)
    return data


# --------------------------------------------------------------------------- KPIs
def _need(df: pd.DataFrame, *cols: str | None) -> None:
    missing = [c for c in cols if c and c not in df.columns]
    if missing:
        raise DashboardError(f"Column(s) not found: {', '.join(missing)}")


def _agg_series(df: pd.DataFrame, column: str | None, agg: str) -> float | None:
    if agg == "count" and not column:
        return float(len(df))
    _need(df, column)
    series = (
        df[column] if agg in ("count", "nunique") else pd.to_numeric(df[column], errors="coerce")
    )
    if df.empty:
        return None
    value = getattr(series, agg)()
    return None if pd.isna(value) else float(value)


def kpi_value(df: pd.DataFrame, params: dict[str, Any]) -> float | None:
    return _agg_series(df, params.get("column"), params.get("agg", "count"))


def format_value(
    value: float | None,
    fmt: str = "auto",
    prefix: str = "",
    suffix: str = "",
    decimals: int | None = None,
) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    if fmt == "percent":
        text = f"{value * 100:.{1 if decimals is None else decimals}f}%"
    elif fmt == "currency":
        text = f"${value:,.{0 if decimals is None else decimals}f}"
    elif fmt == "integer":
        text = f"{value:,.0f}"
    elif fmt == "decimal":
        text = f"{value:,.{2 if decimals is None else decimals}f}"
    elif abs(value) >= 1_000_000:
        text = f"{value / 1_000_000:.2f}M"
    elif abs(value) >= 10_000:
        text = f"{value / 1_000:.1f}K"
    else:
        text = f"{value:,.0f}" if float(value).is_integer() else f"{value:,.2f}"
    return f"{prefix}{text}{suffix}"


_OPS = {"<": np.less, "<=": np.less_equal, ">": np.greater, ">=": np.greater_equal, "==": np.equal}


def kpi_card(df: pd.DataFrame, tile: Tile) -> dict[str, Any]:
    """Everything a KPI card shows: value, comparison, sparkline, goal progress, colour."""
    p, o = tile.params, tile.options
    agg, col = p.get("agg", "count"), p.get("column")
    value = kpi_value(df, p)
    card: dict[str, Any] = {
        "value": value,
        "text": format_value(
            value,
            o.get("format", "auto"),
            o.get("prefix", ""),
            o.get("suffix", ""),
            o.get("decimals"),
        ),
        "delta": None,
        "delta_text": None,
        "direction": None,
        "progress": None,
        "spark": None,
        "color": None,
        "note": "",
    }
    compare = o.get("compare") or {}
    if compare.get("kind") == "previous_period":
        date_col, days = compare.get("date_col"), int(compare.get("days", 30))
        _need(df, date_col)
        dates = pd.to_datetime(df[date_col], errors="coerce")
        end = dates.max()
        if pd.notna(end):
            cur = df[(dates > end - pd.Timedelta(days=days)) & (dates <= end)]
            prev = df[
                (dates > end - pd.Timedelta(days=2 * days))
                & (dates <= end - pd.Timedelta(days=days))
            ]
            cv, pv = _agg_series(cur, col, agg), _agg_series(prev, col, agg)
            card["value"] = cv
            card["text"] = format_value(
                cv,
                o.get("format", "auto"),
                o.get("prefix", ""),
                o.get("suffix", ""),
                o.get("decimals"),
            )
            card["note"] = f"last {days} days vs the {days} before"
            if cv is not None and pv not in (None, 0):
                card["delta"] = cv - pv
                card["delta_text"] = f"{(cv - pv) / abs(pv):+.1%}"
            elif cv is not None and pv == 0:
                card["delta"] = cv
                card["delta_text"] = "n/a (previous was 0)"
    elif (
        compare.get("kind") == "target"
        and compare.get("target") not in (None, 0)
        and value is not None
    ):
        target = float(compare["target"])
        card["delta"] = value - target
        card["delta_text"] = f"{(value - target) / abs(target):+.1%} vs target"
        card["progress"] = max(0.0, min(1.0, value / target))
        card["note"] = f"target {format_value(target)}"
    goal = o.get("goal")
    if goal and card["progress"] is None and card["value"] is not None:
        card["progress"] = max(0.0, min(1.0, card["value"] / float(goal)))
        card["note"] = card["note"] or f"goal {format_value(float(goal))}"
    spark = o.get("sparkline") or {}
    if spark.get("date_col"):
        _need(df, spark["date_col"])
        dated = df.assign(_d=pd.to_datetime(df[spark["date_col"]], errors="coerce")).dropna(
            subset=["_d"]
        )
        if len(dated):
            freq = spark.get("freq", "W")
            indexed = dated.set_index("_d")
            if agg == "count" and not col:
                series = indexed.resample(freq).size()
            else:
                values = (
                    indexed[col]
                    if agg in ("count", "nunique")
                    else pd.to_numeric(indexed[col], errors="coerce")
                )
                series = values.resample(freq).agg(agg)
            card["spark"] = series.astype(float)
    if card["value"] is not None:
        for rule in o.get("thresholds", []) or []:
            op = _OPS.get(rule.get("op", "<"))
            if op is not None and bool(op(card["value"], float(rule["value"]))):
                card["color"] = rule.get("color", "red")
                break
    if card["delta"] is not None:
        card["direction"] = "up" if card["delta"] > 0 else ("down" if card["delta"] < 0 else "flat")
    return card


# --------------------------------------------------------------------------- tile frames
def _palette(name: str) -> list[str] | None:
    if name == "colorblind":
        return COLORBLIND
    return {
        "viridis": px.colors.sequential.Viridis,
        "blues": px.colors.sequential.Blues,
        "teal": px.colors.sequential.Teal,
        "plasma": px.colors.sequential.Plasma,
    }.get(name)


def _aggregate(
    df: pd.DataFrame, keys: list[str], value: str | None, agg: str
) -> tuple[pd.DataFrame, str]:
    if not value or agg == "count":
        return df.groupby(keys, dropna=False).size().reset_index(name="count"), "count"
    src = df.copy()
    if agg not in ("count", "nunique"):
        src[value] = pd.to_numeric(src[value], errors="coerce")
    return src.groupby(keys, dropna=False)[value].agg(agg).reset_index(), value


def _finish(
    frame: pd.DataFrame, key: str, value: str, o: dict[str, Any], agg: str = "sum"
) -> pd.DataFrame:
    sort = o.get("sort", "none")
    if sort in ("asc", "desc"):
        frame = frame.sort_values(value, ascending=sort == "asc")
    top = o.get("top_n")
    if top and key in frame.columns and frame[key].nunique() > int(top):
        order = frame.groupby(key)[value].sum().sort_values(ascending=False)
        keep = order.head(int(top)).index
        rest = frame[~frame[key].isin(keep)]
        frame = frame[frame[key].isin(keep)]
        if o.get("other_group") and agg in ("sum", "count") and len(rest):
            other = {
                c: (rest[value].sum() if c == value else ("Other" if c == key else None))
                for c in frame.columns
            }
            frame = pd.concat([frame, pd.DataFrame([other])], ignore_index=True)
    return frame


def _time_group(df: pd.DataFrame, x: str, freq: str | None) -> pd.DataFrame:
    data = df.copy()
    xs = (
        data[x]
        if pd.api.types.is_numeric_dtype(data[x])
        else pd.to_datetime(data[x], errors="coerce")
    )
    data[x] = xs
    data = data.dropna(subset=[x])
    return data


def tile_frame(df: pd.DataFrame, tile: Tile) -> pd.DataFrame:
    """The aggregated rows behind a tile (what the chart draws and what exports contain)."""
    try:
        return _tile_frame(df, tile)
    except DashboardError:
        raise
    except (ValueError, KeyError, TypeError, AttributeError) as err:
        raise DashboardError(f"Could not build '{tile.title}': {err}") from err


def _tile_frame(df: pd.DataFrame, tile: Tile) -> pd.DataFrame:
    p, o, kind = tile.params, tile.options, tile.kind
    if kind == "text":
        return pd.DataFrame({"text": [p.get("markdown", "")]})
    if kind == "kpi" or kind == "gauge":
        card = kpi_card(df, tile)
        return pd.DataFrame({"metric": [tile.title], "value": [card["value"]]})
    if kind == "table":
        cols = p.get("columns") or list(df.columns)
        _need(df, *cols)
        return df[cols].head(int(p.get("rows", 20)))
    if df.empty:
        raise DashboardError("No rows match the current filters.")
    if kind in ("line", "combo", "scatter") and len({p.get("x"), p.get("y")}) < 2:
        raise DashboardError("Pick different columns for x and y.")
    agg = p.get("agg", "count")

    if kind in ("bar", "area"):
        _need(df, p["x"], p.get("y"), p.get("color"))
        keys = [p["x"]] + ([p["color"]] if p.get("color") else [])
        data, value = _aggregate(df, keys, p.get("y"), agg)
        if kind == "area":
            data = data.sort_values(p["x"])
        return _finish(data, p["x"], value, o, agg)
    if kind == "line":
        _need(df, p["x"], p.get("y"), p.get("color"))
        data = _time_group(df, p["x"], p.get("freq"))
        keys: list[Any] = [p["color"]] if p.get("color") else []
        if p.get("freq") and pd.api.types.is_datetime64_any_dtype(data[p["x"]]):
            keys = [pd.Grouper(key=p["x"], freq=p["freq"]), *keys]
        else:
            keys = [p["x"], *keys]
        data[p["y"]] = pd.to_numeric(data[p["y"]], errors="coerce")
        out = data.groupby(keys)[p["y"]].agg(agg if agg != "count" else "count").reset_index()
        return out.sort_values(p["x"])
    if kind == "combo":
        _need(df, p["x"], p["y"], p["y2"])
        data = _time_group(df, p["x"], p.get("freq"))
        for c in (p["y"], p["y2"]):
            data[c] = pd.to_numeric(data[c], errors="coerce")
        key: Any = (
            pd.Grouper(key=p["x"], freq=p["freq"])
            if p.get("freq") and pd.api.types.is_datetime64_any_dtype(data[p["x"]])
            else p["x"]
        )
        out = (
            data.groupby(key)
            .agg(**{p["y"]: (p["y"], agg), p["y2"]: (p["y2"], p.get("agg2", "mean"))})
            .reset_index()
        )
        return out.sort_values(p["x"])
    if kind == "scatter":
        _need(df, p["x"], p["y"], p.get("color"), p.get("size"))
        cols = [c for c in (p["x"], p["y"], p.get("color"), p.get("size")) if c]
        data = df[cols].dropna()
        return data.sample(MAX_POINTS, random_state=0) if len(data) > MAX_POINTS else data
    if kind == "hist":
        _need(df, p["column"], p.get("color"))
        return df[[c for c in (p["column"], p.get("color")) if c]]
    if kind == "pie":
        _need(df, p["names"], p.get("values"))
        data, value = _aggregate(df, [p["names"]], p.get("values"), p.get("agg", "sum"))
        data = data.sort_values(value, ascending=False)
        top = int(o.get("top_n", p.get("top_n", 8)))
        if len(data) > top:
            rest = data.iloc[top:][value].sum()
            data = data.head(top)
            if o.get("other_group", True):
                data = pd.concat(
                    [data, pd.DataFrame({p["names"]: ["Other"], value: [rest]})], ignore_index=True
                )
        return data
    if kind == "box":
        _need(df, p["y"], p.get("x"), p.get("color"))
        return df[[c for c in (p.get("x"), p["y"], p.get("color")) if c]]
    if kind == "heatmap":
        _need(df, p["x"], p["y"], p.get("value"))
        if p.get("value") and agg != "count":
            src = df.assign(**{p["value"]: pd.to_numeric(df[p["value"]], errors="coerce")})
            return src.pivot_table(index=p["y"], columns=p["x"], values=p["value"], aggfunc=agg)
        return pd.crosstab(df[p["y"]], df[p["x"]])
    if kind == "funnel":
        _need(df, p["names"], p.get("values"))
        data, value = _aggregate(df, [p["names"]], p.get("values"), agg)
        return data.sort_values(value, ascending=False)
    if kind == "waterfall":
        _need(df, p["x"], p.get("y"))
        data, value = _aggregate(df, [p["x"]], p.get("y"), agg)
        return _finish(data, p["x"], value, o, agg)
    if kind == "treemap":
        path = p.get("path") or [p["names"]]
        _need(df, *path, p.get("values"))
        data = df.copy()
        if p.get("values"):
            data[p["values"]] = pd.to_numeric(data[p["values"]], errors="coerce")
            return (
                data.groupby(path, dropna=False)[p["values"]]
                .agg(agg if agg != "count" else "sum")
                .reset_index()
            )
        return data.groupby(path, dropna=False).size().reset_index(name="count")
    if kind == "map":
        if p.get("mode", "country") == "latlon":
            _need(df, p["lat"], p["lon"], p.get("size"), p.get("color"))
            cols = [c for c in (p["lat"], p["lon"], p.get("size"), p.get("color")) if c]
            data = df[cols].dropna(subset=[p["lat"], p["lon"]])
            return data.sample(MAX_POINTS, random_state=0) if len(data) > MAX_POINTS else data
        _need(df, p["names"], p.get("values"))
        data, _ = _aggregate(df, [p["names"]], p.get("values"), p.get("agg", "sum"))
        return data
    raise DashboardError(f"Unsupported tile type '{kind}'.")


def table_data(df: pd.DataFrame, tile: Tile) -> pd.DataFrame:
    return tile_frame(df, tile)


# --------------------------------------------------------------------------- figures
def _template(theme: Theme | None, fallback: str) -> str:
    if theme is None:
        return fallback
    return "plotly_dark" if theme.template == "dark" else "plotly_white"


def _value_col(frame: pd.DataFrame, key: str) -> str:
    candidates = [c for c in frame.columns if c != key and pd.api.types.is_numeric_dtype(frame[c])]
    return candidates[-1] if candidates else frame.columns[-1]


def _ref_lines(
    fig: go.Figure, frame: pd.DataFrame, value_col: str | None, o: dict[str, Any]
) -> None:
    for ref in o.get("ref_lines", []) or []:
        kind = ref.get("kind", "value")
        if kind == "value":
            y = ref.get("value")
        elif value_col and value_col in frame.columns:
            series = pd.to_numeric(frame[value_col], errors="coerce")
            y = series.mean() if kind == "mean" else series.median()
        else:
            y = None
        if y is not None and not pd.isna(y):
            fig.add_hline(
                y=float(y),
                line_dash="dash",
                annotation_text=ref.get("label") or kind,
                annotation_position="top left",
            )


def build_figure(
    df: pd.DataFrame, tile: Tile, template: str = "plotly_dark", theme: Theme | None = None
) -> go.Figure | None:
    """Plotly figure for a tile; None for KPI/table/text tiles (rendered natively)."""
    try:
        return _build_figure(df, tile, template, theme)
    except DashboardError:
        raise
    except (ValueError, KeyError, TypeError, AttributeError) as err:
        raise DashboardError(f"Could not draw '{tile.title}': {err}") from err


def _build_figure(
    df: pd.DataFrame, tile: Tile, template: str, theme: Theme | None
) -> go.Figure | None:
    """Plotly figure for a tile; None for KPI/table/text tiles (rendered natively)."""
    if tile.kind in ("kpi", "table", "text"):
        return None
    p, o, kind = tile.params, tile.options, tile.kind
    tpl = _template(theme, template)
    palette = _palette(o.get("colors") or (theme.palette if theme else "default"))
    seq = {"color_discrete_sequence": palette} if palette else {}

    if kind == "gauge":
        card = kpi_card(df, tile)
        value = card["value"] or 0
        lo, hi = float(p.get("min", 0)), float(p.get("max", max(value * 1.5, 1)))
        target = (o.get("compare") or {}).get("target") or o.get("goal")
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number" + ("+delta" if target else ""),
                value=value,
                delta={"reference": float(target)} if target else None,
                gauge={
                    "axis": {"range": [lo, hi]},
                    "threshold": {"line": {"color": "red", "width": 3}, "value": float(target)}
                    if target
                    else None,
                },
                title={"text": tile.title},
            )
        )
        fig.update_layout(
            template=tpl,
            height=tile.height,
            margin={"l": 20, "r": 20, "t": 50, "b": 20},
            **({"font": {"size": theme.font_size}} if theme else {}),
        )
        return fig

    frame = tile_frame(df, tile)
    value_col: str | None = None
    if kind in ("bar", "area"):
        x = p["x"]
        value_col = _value_col(frame, x) if not p.get("color") else frame.columns[-1]
        horizontal = o.get("orientation") == "h"
        common = {"color": p.get("color"), "template": tpl, **seq}
        if kind == "bar":
            stacking = o.get("stacked", "group")
            barmode = {"group": "group", "stack": "stack", "percent": "relative"}.get(
                stacking, "group"
            )
            fig = px.bar(
                frame,
                x=value_col if horizontal else x,
                y=x if horizontal else value_col,
                orientation="h" if horizontal else "v",
                barmode=barmode,
                text_auto=bool(o.get("data_labels")),
                **common,
            )
            if stacking == "percent":
                fig.update_layout(barnorm="percent")
        else:
            fig = px.area(
                frame,
                x=x,
                y=value_col,
                groupnorm="percent" if o.get("stacked") == "percent" else None,
                **common,
            )
    elif kind == "line":
        value_col = p["y"]
        fig = px.line(
            frame,
            x=p["x"],
            y=p["y"],
            color=p.get("color"),
            markers=bool(o.get("data_labels")),
            template=tpl,
            **seq,
        )
        if o.get("data_labels"):
            fig.update_traces(
                text=frame[p["y"]].round(1), textposition="top center", mode="lines+markers+text"
            )
    elif kind == "combo":
        value_col = p["y"]
        fig = go.Figure()
        fig.add_bar(
            x=frame[p["x"]], y=frame[p["y"]], name=p["y"], marker_color=(palette or [None])[0]
        )
        fig.add_scatter(
            x=frame[p["x"]],
            y=frame[p["y2"]],
            name=p["y2"],
            yaxis="y2",
            mode="lines+markers",
            marker_color=(palette or [None, None])[1] if palette else None,
        )
        fig.update_layout(
            template=tpl, yaxis2={"overlaying": "y", "side": "right", "title": p["y2"]}
        )
    elif kind == "scatter":
        value_col = p["y"]
        fig = px.scatter(
            frame,
            x=p["x"],
            y=p["y"],
            color=p.get("color"),
            size=p.get("size"),
            trendline="ols" if o.get("trendline") else None,
            template=tpl,
            **seq,
        )
    elif kind == "hist":
        value_col = None
        fig = px.histogram(
            frame,
            x=p["column"],
            nbins=int(p.get("bins", 30)),
            color=p.get("color"),
            template=tpl,
            **seq,
        )
    elif kind == "pie":
        value_col = _value_col(frame, p["names"])
        fig = px.pie(
            frame,
            names=p["names"],
            values=value_col,
            hole=0.4 if o.get("donut") else 0,
            template=tpl,
            **seq,
        )
    elif kind == "box":
        value_col = p["y"]
        fig = px.box(frame, y=p["y"], x=p.get("x"), color=p.get("color"), template=tpl, **seq)
    elif kind == "heatmap":
        fig = px.imshow(
            frame,
            text_auto=".0f" if o.get("data_labels", True) else False,
            aspect="auto",
            color_continuous_scale=_palette(o.get("colors", "default")) or "Viridis",
            template=tpl,
        )
    elif kind == "funnel":
        value_col = _value_col(frame, p["names"])
        fig = px.funnel(frame, x=value_col, y=p["names"], template=tpl, **seq)
    elif kind == "waterfall":
        value_col = _value_col(frame, p["x"])
        measures = ["relative"] * len(frame) + ["total"]
        xs = [*frame[p["x"]].astype(str), "Total"]
        ys = [*frame[value_col], 0]
        fig = go.Figure(go.Waterfall(x=xs, y=ys, measure=measures))
        fig.update_layout(template=tpl)
    elif kind == "treemap":
        path = p.get("path") or [p["names"]]
        value_col = p.get("values") or "count"
        fig = px.treemap(
            frame, path=path, values=value_col, template=tpl, color_discrete_sequence=palette
        )
    elif kind == "map":
        if p.get("mode", "country") == "latlon":
            fig = px.scatter_geo(
                frame,
                lat=p["lat"],
                lon=p["lon"],
                size=p.get("size"),
                color=p.get("color"),
                template=tpl,
                **seq,
            )
        else:
            value_col = _value_col(frame, p["names"])
            fig = px.choropleth(
                frame,
                locations=p["names"],
                locationmode="country names",
                color=value_col,
                template=tpl,
            )
    else:
        raise DashboardError(f"Unsupported tile type '{kind}'.")

    if o.get("log_y"):
        fig.update_yaxes(type="log")
    _ref_lines(fig, frame, value_col, o)
    layout: dict[str, Any] = {
        "title": tile.title,
        "height": tile.height,
        "margin": {"l": 20, "r": 20, "t": 50, "b": 20},
    }
    if o.get("x_title"):
        layout["xaxis_title"] = o["x_title"]
    if o.get("y_title"):
        layout["yaxis_title"] = o["y_title"]
    if theme:
        layout["font"] = {"size": theme.font_size}
    fig.update_layout(**layout)
    return fig


# --------------------------------------------------------------------------- captions
def caption_for(df: pd.DataFrame, tile: Tile) -> str:
    """One deterministic sentence describing what a tile shows, from its computed numbers."""
    try:
        kind = tile.kind
        if kind == "text":
            return ""
        if kind in ("kpi", "gauge"):
            card = kpi_card(df, tile)
            extra = f" ({card['delta_text']})" if card["delta_text"] else ""
            return f"{tile.title}: {card['text']}{extra}."
        frame = tile_frame(df, tile)
        p = tile.params
        if kind == "table":
            return f"Showing {len(frame):,} of {len(df):,} rows."
        if kind in ("bar", "pie", "funnel", "treemap", "map", "waterfall"):
            key = p.get("x") or p.get("names") or (p.get("path") or [None])[-1]
            value = _value_col(frame, key)
            top = frame.sort_values(value, ascending=False)
            hi, lo = top.iloc[0], top.iloc[-1]
            total = frame[value].sum()
            share = f", {hi[value] / total:.0%} of the total" if total else ""
            return f"Highest: {hi[key]} ({hi[value]:,.4g}{share}); lowest: {lo[key]} ({lo[value]:,.4g}); {len(frame)} groups."
        if kind in ("line", "area", "combo"):
            value = p.get("y") or _value_col(frame, p["x"])
            series = pd.to_numeric(frame[value], errors="coerce").dropna()
            if len(series) >= 2 and series.iloc[0] != 0:
                change = (series.iloc[-1] - series.iloc[0]) / abs(series.iloc[0])
                return f"{value} moved from {series.iloc[0]:,.4g} to {series.iloc[-1]:,.4g} ({change:+.1%}); peak {series.max():,.4g}."
            return f"{value}: {len(series)} points."
        if kind == "scatter":
            r = frame[[p["x"], p["y"]]].corr().iloc[0, 1]
            return f"{p['y']} vs {p['x']}: correlation {r:+.2f} across {len(frame):,} points."
        if kind == "hist":
            s = pd.to_numeric(frame[p["column"]], errors="coerce").dropna()
            return f"{p['column']}: mean {s.mean():,.4g}, median {s.median():,.4g}, range {s.min():,.4g} to {s.max():,.4g}."
        if kind == "box":
            s = pd.to_numeric(frame[p["y"]], errors="coerce").dropna()
            return f"{p['y']}: median {s.median():,.4g}, middle half between {s.quantile(0.25):,.4g} and {s.quantile(0.75):,.4g}."
        if kind == "heatmap":
            stacked = frame.stack()
            return f"Largest cell: {stacked.idxmax()} = {stacked.max():,.4g}."
    except Exception:
        return ""
    return ""


# --------------------------------------------------------------------------- validation
def tile_columns(tile: Tile) -> list[str]:
    """Data columns a tile refers to (plain column names only)."""
    cols: list[str] = []
    for key in (
        "x",
        "y",
        "y2",
        "color",
        "size",
        "column",
        "names",
        "values",
        "value",
        "lat",
        "lon",
    ):
        v = tile.params.get(key)
        if isinstance(v, str):
            cols.append(v)
    for key in ("columns", "path", "drill"):
        v = tile.params.get(key)
        if isinstance(v, list):
            cols += [c for c in v if isinstance(c, str)]
    for opt in ("compare", "sparkline"):
        v = tile.options.get(opt) or {}
        if isinstance(v, dict) and isinstance(v.get("date_col"), str):
            cols.append(v["date_col"])
    return list(dict.fromkeys(cols))


def validate_spec(spec: DashboardSpec, tables: Mapping[str, pd.DataFrame]) -> list[str]:
    """Human-readable problems: missing tables/columns, bad calculated fields."""
    problems: list[str] = []
    for tile in spec.tiles:
        if tile.source not in tables:
            problems.append(f"'{tile.title}': table '{tile.source}' is no longer available.")
            continue
        frame, _ = apply_calculated(tables[tile.source], spec.calculated)
        missing = [c for c in tile_columns(tile) if c not in frame.columns]
        if missing:
            problems.append(f"'{tile.title}': missing column(s) {', '.join(missing)}.")
    base = next(iter(tables.values()), None)
    for f in spec.filters:
        if base is not None and all(f.column not in t.columns for t in tables.values()):
            problems.append(f"Filter on '{f.column}': the column no longer exists.")
    if base is not None:
        _, errors = apply_calculated(base, spec.calculated)
        problems += [
            f"Calculated field '{n}': {e}"
            for n, e in errors.items()
            if "already exists" not in e or True
        ]
    return problems


# --------------------------------------------------------------------------- interactions
def selection_to_filter(tile: Tile, selection: Mapping[str, Any] | None) -> Filter | None:
    """Turn a chart click/selection into a categorical filter (bars, slices, funnels, treemaps)."""
    if not selection:
        return None
    points = (
        (selection.get("selection") or selection).get("points", [])
        if isinstance(selection, Mapping)
        else []
    )
    if not points:
        return None
    p = tile.params
    column = p.get("x") or p.get("names") or ((p.get("path") or [None])[0])
    if not column:
        return None
    horizontal = tile.options.get("orientation") == "h" or tile.kind == "funnel"
    values: list[str] = []
    for pt in points:
        raw = (
            pt.get("label")
            if tile.kind in ("pie", "treemap")
            else pt.get("y" if horizontal else "x")
        )
        if raw is None:
            raw = pt.get("label") or pt.get("x") or pt.get("y")
        if raw is not None and str(raw) not in values:
            values.append(str(raw))
    return Filter(column=column, kind="categorical", values=values) if values else None


class DrillState(BaseModel):
    level: int = 0
    picked: dict[str, str] = Field(default_factory=dict)


def drill_apply(tile: Tile, state: DrillState) -> tuple[Tile, list[Filter]]:
    """Tile re-pointed to the current drill level plus filters for the higher levels."""
    path: list[str] = tile.params.get("drill") or []
    if not path:
        return tile, []
    level = min(state.level, len(path) - 1)
    filters = [
        Filter(column=c, kind="categorical", values=[state.picked[c]])
        for c in path[:level]
        if c in state.picked
    ]
    return tile.model_copy(update={"params": {**tile.params, "x": path[level]}}), filters


def drill_down(tile: Tile, state: DrillState, clicked: str) -> DrillState:
    path: list[str] = tile.params.get("drill") or []
    if not path or state.level >= len(path) - 1:
        return state
    picked = {**state.picked, path[state.level]: clicked}
    return DrillState(level=state.level + 1, picked=picked)


def drill_up(state: DrillState) -> DrillState:
    if state.level <= 0:
        return state
    return DrillState(
        level=state.level - 1,
        picked={k: v for i, (k, v) in enumerate(state.picked.items()) if i < state.level - 1},
    )


# --------------------------------------------------------------------------- layout helpers
def layout_rows(tiles: list[Tile], columns: int) -> list[list[Tile]]:
    """Pack tiles into rows no wider than `columns`, honoring each tile's span."""
    rows: list[list[Tile]] = []
    current: list[Tile] = []
    used = 0
    for tile in tiles:
        span = max(1, min(tile.width, columns))
        if used + span > columns and current:
            rows.append(current)
            current, used = [], 0
        current.append(tile)
        used += span
    if current:
        rows.append(current)
    return rows


def move_tile(spec: DashboardSpec, tile_id: str, direction: int) -> DashboardSpec:
    """Move a tile up (-1) or down (+1) within the list; returns a new spec."""
    out = clone_spec(spec)
    idx = next((i for i, t in enumerate(out.tiles) if t.id == tile_id), None)
    if idx is None:
        return out
    new = max(0, min(len(out.tiles) - 1, idx + direction))
    out.tiles.insert(new, out.tiles.pop(idx))
    return out


def duplicate_tile(spec: DashboardSpec, tile_id: str) -> DashboardSpec:
    out = clone_spec(spec)
    for i, t in enumerate(out.tiles):
        if t.id == tile_id:
            twin = copy.deepcopy(t)
            twin.id = uuid.uuid4().hex[:8]
            twin.title = f"{t.title} (copy)"
            out.tiles.insert(i + 1, twin)
            break
    return out


class History:
    """Undo/redo stack of spec JSON snapshots."""

    def __init__(self, limit: int = 30) -> None:
        self.limit = limit
        self.undo_stack: list[str] = []
        self.redo_stack: list[str] = []

    def push(self, spec: DashboardSpec) -> None:
        snap = spec_to_json(spec)
        if self.undo_stack and self.undo_stack[-1] == snap:
            return
        self.undo_stack.append(snap)
        self.undo_stack = self.undo_stack[-self.limit :]
        self.redo_stack.clear()

    def undo(self, current: DashboardSpec) -> DashboardSpec | None:
        if not self.undo_stack:
            return None
        self.redo_stack.append(spec_to_json(current))
        return spec_from_json(self.undo_stack.pop())

    def redo(self, current: DashboardSpec) -> DashboardSpec | None:
        if not self.redo_stack:
            return None
        self.undo_stack.append(spec_to_json(current))
        return spec_from_json(self.redo_stack.pop())


# --------------------------------------------------------------------------- wrappers (siblings)
def export_html(df_or_tables: Any, spec: DashboardSpec) -> str:
    from src.datascience.dashboard_export import export_html as _export

    return _export(df_or_tables, spec)


def dashboard_snippet(spec: DashboardSpec, **load: Any) -> str:
    from src.datascience.dashboard_code import dashboard_snippet as _snippet

    return _snippet(spec, **load)
