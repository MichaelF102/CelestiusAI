"""Feature engineering steps, leakage warnings, and replayable recipes."""

from __future__ import annotations

import json
import re
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from src.core.code_snippets import make_snippet

FeatureOp = Literal[
    "log1p",
    "sqrt",
    "square",
    "bin_quantile",
    "bin_width",
    "onehot",
    "ordinal",
    "frequency",
    "date_parts",
    "interaction",
    "missing_indicator",
    "drop",
]

OP_LABELS: dict[str, str] = {
    "log1p": "Log transform (log1p) - tames right skew",
    "sqrt": "Square root",
    "square": "Square",
    "bin_quantile": "Quantile bins (equal counts)",
    "bin_width": "Equal-width bins",
    "onehot": "One-hot encode categories",
    "ordinal": "Ordinal codes",
    "frequency": "Frequency encoding",
    "date_parts": "Date parts (year, month, weekday...)",
    "interaction": "Interaction of two numeric columns",
    "missing_indicator": "Missing-value indicator",
    "drop": "Drop columns",
}
# Steps whose parameters are learned from the data they are applied to.
LEARNED_OPS = {"bin_quantile", "frequency"}
DATE_PARTS = ("year", "month", "day", "dayofweek", "quarter", "hour", "is_weekend")
INTERACTIONS = ("product", "ratio", "sum", "difference")


class FeatureError(ValueError):
    """Invalid feature specification for the current data."""


class FeatureSpec(BaseModel):
    op: FeatureOp
    columns: list[str]
    params: dict[str, Any] = Field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"{self.op}({', '.join(self.columns)})"


def _safe(name: str) -> str:
    return re.sub(r"\W+", "_", name).strip("_")


def _need_cols(df: pd.DataFrame, spec: FeatureSpec, count: int | None = None) -> None:
    missing = [c for c in spec.columns if c not in df.columns]
    if missing:
        raise FeatureError(f"Column(s) not found: {', '.join(missing)}")
    if count is not None and len(spec.columns) != count:
        raise FeatureError(f"'{spec.op}' needs exactly {count} column(s).")
    if not spec.columns:
        raise FeatureError("Select at least one column.")


def apply_spec(df: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    """Return a copy of `df` with the step applied."""
    out = df.copy()
    _need_cols(out, spec)
    p = spec.params
    op = spec.op

    if op in ("log1p", "sqrt", "square"):
        for c in spec.columns:
            s = pd.to_numeric(out[c], errors="coerce")
            if op == "log1p":
                if (s.dropna() <= -1).any():
                    raise FeatureError(f"'{c}' has values <= -1; log1p is undefined.")
                out[f"{c}_log1p"] = np.log1p(s)
            elif op == "sqrt":
                if (s.dropna() < 0).any():
                    raise FeatureError(f"'{c}' has negative values; sqrt is undefined.")
                out[f"{c}_sqrt"] = np.sqrt(s)
            else:
                out[f"{c}_sq"] = s**2
    elif op == "bin_quantile":
        q = int(p.get("bins", 4))
        for c in spec.columns:
            out[f"{c}_qbin"] = pd.qcut(
                pd.to_numeric(out[c], errors="coerce"), q=q, labels=False, duplicates="drop"
            )
    elif op == "bin_width":
        n = int(p.get("bins", 5))
        for c in spec.columns:
            out[f"{c}_bin"] = pd.cut(pd.to_numeric(out[c], errors="coerce"), bins=n, labels=False)
    elif op == "onehot":
        out = pd.get_dummies(
            out, columns=spec.columns, drop_first=bool(p.get("drop_first", True)), dtype=int
        )
    elif op == "ordinal":
        for c in spec.columns:
            order = p.get("order", {}).get(c)
            cat = pd.Categorical(out[c], categories=order) if order else pd.Categorical(out[c])
            out[f"{c}_ord"] = pd.Series(cat.codes, index=out.index).replace(-1, np.nan)
    elif op == "frequency":
        for c in spec.columns:
            out[f"{c}_freq"] = out[c].map(out[c].value_counts(normalize=True))
    elif op == "date_parts":
        parts = p.get("parts") or ["year", "month", "dayofweek"]
        bad = [x for x in parts if x not in DATE_PARTS]
        if bad:
            raise FeatureError(f"Unknown date parts: {', '.join(bad)}")
        for c in spec.columns:
            dt = pd.to_datetime(out[c], errors="coerce")
            if dt.notna().sum() == 0:
                raise FeatureError(f"'{c}' could not be parsed as dates.")
            for part in parts:
                name = f"{c}_{part}"
                out[name] = (
                    (dt.dt.dayofweek >= 5).astype("Int64")
                    if part == "is_weekend"
                    else getattr(dt.dt, part)
                )
    elif op == "interaction":
        _need_cols(out, spec, 2)
        a, b = (pd.to_numeric(out[c], errors="coerce") for c in spec.columns)
        kind = p.get("kind", "product")
        if kind not in INTERACTIONS:
            raise FeatureError(f"Unknown interaction '{kind}'.")
        name = f"{_safe(spec.columns[0])}_{kind}_{_safe(spec.columns[1])}"
        out[name] = {
            "product": a * b,
            "ratio": a / b.replace(0, np.nan),
            "sum": a + b,
            "difference": a - b,
        }[kind]
    elif op == "missing_indicator":
        for c in spec.columns:
            out[f"{c}_missing"] = out[c].isna().astype(int)
    elif op == "drop":
        out = out.drop(columns=spec.columns)
    return out


def apply_specs(df: pd.DataFrame, specs: list[FeatureSpec]) -> pd.DataFrame:
    out = df
    for spec in specs:
        out = apply_spec(out, spec)
    return out.copy() if out is df else out


# --------------------------------------------------------------------------- leakage
class LeakageWarning(BaseModel):
    level: Literal["high", "medium", "low"]
    column: str
    message: str


def leakage_report(df: pd.DataFrame, target: str, features: list[str]) -> list[LeakageWarning]:
    """Heuristic checks for features that may leak the target or carry no generalizable signal."""
    out: list[LeakageWarning] = []
    if target not in df.columns:
        return out
    y = df[target]
    y_num = pd.to_numeric(y, errors="coerce") if not pd.api.types.is_numeric_dtype(y) else y
    base = _safe(target).lower()
    for col in features:
        if col == target or col not in df.columns:
            continue
        s = df[col]
        if base and base in _safe(col).lower():
            out.append(
                LeakageWarning(
                    level="high",
                    column=col,
                    message=f"Name contains the target '{target}' - it may be derived from it.",
                )
            )
        if pd.api.types.is_numeric_dtype(s) and y_num.notna().sum() > 3:
            pair = pd.concat([s, y_num], axis=1).dropna()
            if len(pair) > 3 and pair.iloc[:, 0].nunique() > 1 and pair.iloc[:, 1].nunique() > 1:
                r = float(pair.corr().iloc[0, 1])
                if abs(r) >= 0.98:
                    out.append(
                        LeakageWarning(
                            level="high",
                            column=col,
                            message=f"Near-perfect correlation with the target (r = {r:.3f}).",
                        )
                    )
        non_null = s.dropna()
        if len(non_null) > 0:
            ratio = non_null.nunique() / len(non_null)
            if ratio > 0.95 and (not pd.api.types.is_float_dtype(s)):
                out.append(
                    LeakageWarning(
                        level="medium",
                        column=col,
                        message=f"Almost every value is unique ({ratio:.0%}) - looks like an identifier.",
                    )
                )
        if pd.api.types.is_datetime64_any_dtype(s):
            out.append(
                LeakageWarning(
                    level="low",
                    column=col,
                    message="Raw datetime column - extract date parts instead, and split by time if forecasting.",
                )
            )
        miss = s.isna().mean()
        if miss > 0.5:
            out.append(
                LeakageWarning(
                    level="low", column=col, message=f"{miss:.0%} of values are missing."
                )
            )
    return out


# --------------------------------------------------------------------------- recipe & code
def specs_to_json(specs: list[FeatureSpec]) -> str:
    return json.dumps([s.model_dump(mode="json") for s in specs], indent=2)


def specs_from_json(text: str | bytes) -> list[FeatureSpec]:
    try:
        return [FeatureSpec.model_validate(item) for item in json.loads(text)]
    except (ValueError, TypeError) as err:
        raise FeatureError(f"Not a valid feature recipe: {err}") from err


def _spec_code(spec: FeatureSpec) -> list[str]:
    p, cols, op = spec.params, spec.columns, spec.op
    lines: list[str] = []
    if op in ("log1p", "sqrt", "square"):
        for c in cols:
            fn = {
                "log1p": f"np.log1p(pd.to_numeric(df[{c!r}], errors='coerce'))",
                "sqrt": f"np.sqrt(pd.to_numeric(df[{c!r}], errors='coerce'))",
                "square": f"pd.to_numeric(df[{c!r}], errors='coerce') ** 2",
            }[op]
            suffix = {"log1p": "log1p", "sqrt": "sqrt", "square": "sq"}[op]
            lines.append(f"df[{c + '_' + suffix!r}] = {fn}")
    elif op == "bin_quantile":
        for c in cols:
            lines.append(
                f"df[{c + '_qbin'!r}] = pd.qcut(pd.to_numeric(df[{c!r}], errors='coerce'), q={int(p.get('bins', 4))}, labels=False, duplicates='drop')"
            )
    elif op == "bin_width":
        for c in cols:
            lines.append(
                f"df[{c + '_bin'!r}] = pd.cut(pd.to_numeric(df[{c!r}], errors='coerce'), bins={int(p.get('bins', 5))}, labels=False)"
            )
    elif op == "onehot":
        lines.append(
            f"df = pd.get_dummies(df, columns={cols!r}, drop_first={bool(p.get('drop_first', True))}, dtype=int)"
        )
    elif op == "ordinal":
        for c in cols:
            order = p.get("order", {}).get(c)
            cat = (
                f"pd.Categorical(df[{c!r}], categories={order!r})"
                if order
                else f"pd.Categorical(df[{c!r}])"
            )
            lines.append(
                f"df[{c + '_ord'!r}] = pd.Series({cat}.codes, index=df.index).replace(-1, np.nan)"
            )
    elif op == "frequency":
        for c in cols:
            lines.append(
                f"df[{c + '_freq'!r}] = df[{c!r}].map(df[{c!r}].value_counts(normalize=True))"
            )
    elif op == "date_parts":
        for c in cols:
            lines.append(f"_dt = pd.to_datetime(df[{c!r}], errors='coerce')")
            for part in p.get("parts") or ["year", "month", "dayofweek"]:
                expr = (
                    "(_dt.dt.dayofweek >= 5).astype('Int64')"
                    if part == "is_weekend"
                    else f"_dt.dt.{part}"
                )
                lines.append(f"df[{c + '_' + part!r}] = {expr}")
    elif op == "interaction":
        a, b = cols
        kind = p.get("kind", "product")
        expr = {
            "product": "_a * _b",
            "ratio": "_a / _b.replace(0, np.nan)",
            "sum": "_a + _b",
            "difference": "_a - _b",
        }[kind]
        name = f"{_safe(a)}_{kind}_{_safe(b)}"
        lines += [
            f"_a, _b = pd.to_numeric(df[{a!r}], errors='coerce'), pd.to_numeric(df[{b!r}], errors='coerce')",
            f"df[{name!r}] = {expr}",
        ]
    elif op == "missing_indicator":
        for c in cols:
            lines.append(f"df[{c + '_missing'!r}] = df[{c!r}].isna().astype(int)")
    elif op == "drop":
        lines.append(f"df = df.drop(columns={cols!r})")
    return lines


def features_snippet(specs: list[FeatureSpec], **load: Any) -> str:
    lines = ["# Feature engineering steps applied in the app, in order", "df = df.copy()", ""]
    if not specs:
        lines.append("# (no steps added yet)")
    for n, spec in enumerate(specs, start=1):
        lines.append(f"# Step {n}: {spec.label}")
        lines += _spec_code(spec)
        lines.append("")
    lines.append("print(df.shape)\ndf.head()")
    return make_snippet("Feature engineering", "\n".join(lines), **load)
