"""Code generators for Data Quality and Data Dictionary components."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from src.core.code_options import lst as _list
from src.core.code_options import make_snippet, resolve
from src.core.code_options import q as _q
from src.core.code_plots import plot_code

IMPUTE_STRATEGIES = ("median", "mean", "mode", "knn", "constant")
OUTLIER_TREATMENTS = ("none", "cap", "remove", "log1p")


# --------------------------------------------------------------------------- schema & consistency
def schema_snippet(**load: Any) -> str:
    body = """# Schema report: what pandas sees vs. what the data looks like
#~ 'object' columns that parse as numbers or dates are usually mis-typed on load.
rows = []
for col in df.columns:
    s = df[col]
    row = {"column": col, "dtype": str(s.dtype), "non_null": int(s.notna().sum()),
           "distinct": int(s.nunique(dropna=True)), "memory_kb": round(s.memory_usage(deep=True) / 1024, 1)}
    if s.dtype == "object":
        sample = s.dropna().astype(str).head(500)
        row["looks_numeric"] = bool(len(sample) and pd.to_numeric(sample, errors="coerce").notna().mean() > 0.9)
        row["looks_date"] = bool(len(sample) and not row["looks_numeric"]
                                 and pd.to_datetime(sample, errors="coerce").notna().mean() > 0.9)
    row["constant"] = s.nunique(dropna=True) <= 1
    row["high_cardinality"] = bool(s.dtype == "object" and len(df) and s.nunique() / len(df) > 0.4)
    rows.append(row)
schema = pd.DataFrame(rows).set_index("column")
display(schema)
print("Total memory (MB):", round(df.memory_usage(deep=True).sum() / 1024**2, 2))
"""
    return make_snippet("Schema report", body, **load)


def consistency_snippet(**load: Any) -> str:
    body = """# Text consistency: whitespace and casing variants inside categorical columns
text_cols = df.select_dtypes("object").columns
report = []
for col in text_cols:
    s = df[col].dropna().astype(str)
    stripped = s.str.strip()
    #~ Variants collapse to the same value once trimmed and lower-cased.
    variants = s.groupby(stripped.str.lower()).nunique()
    report.append({"column": col,
                   "with_extra_whitespace": int((s != stripped).sum()),
                   "case_or_space_variants": int((variants > 1).sum())})
display(pd.DataFrame(report).set_index("column"))

# %%
# Which values are the variants? (first column that has any)
for col in text_cols:
    s = df[col].dropna().astype(str)
    groups = s.groupby(s.str.strip().str.lower()).unique()
    clashes = groups[groups.map(len) > 1]
    if len(clashes):
        print(col); display(clashes.head(10)); break
"""
    return make_snippet("Text consistency", body, **load)


def convert_types_snippet(mapping: Mapping[str, str], **load: Any) -> str:
    """mapping: column -> numeric | datetime | category | boolean | string"""
    lines = ["# Convert column types (unparseable values become missing)", "df_typed = df.copy()"]
    for col, target in mapping.items():
        c = _q(col)
        if target == "numeric":
            lines.append(f"df_typed[{c}] = pd.to_numeric(df_typed[{c}], errors='coerce')")
        elif target == "datetime":
            lines.append(f"df_typed[{c}] = pd.to_datetime(df_typed[{c}], errors='coerce')")
        elif target == "category":
            lines.append(f"df_typed[{c}] = df_typed[{c}].astype('category')")
        elif target == "boolean":
            lines.append(f"df_typed[{c}] = df_typed[{c}].astype(bool)")
        else:
            lines.append(f"df_typed[{c}] = df_typed[{c}].astype('string')")
    lines += [
        "",
        "#? assert len(df_typed) == len(df), 'conversion must not change the row count'",
        "#~ Compare how many values became missing because they could not be parsed.",
        "print((df_typed.isna().sum() - df.isna().sum()).loc[lambda s: s > 0])",
        "df_typed.dtypes",
    ]
    return make_snippet("Type conversion", "\n".join(lines), **load)


# --------------------------------------------------------------------------- scoring
def quality_score_snippet(**load: Any) -> str:
    body = """# Data quality score (same weights as the app)
#~ Completeness and uniqueness use the app's exact formulas; consistency and validity use
#~ simple column-level proxies here (the app counts detected issues by severity).
WEIGHTS = {"completeness": 0.40, "uniqueness": 0.25, "consistency": 0.20, "validity": 0.15}

total_cells = df.shape[0] * df.shape[1]
missing_pct = df.isna().sum().sum() / total_cells * 100
dup_pct = df.duplicated().sum() / len(df) * 100
completeness = max(0, min(100, 100 - missing_pct * 1.5))     # 10% missing -> -15 points
uniqueness = max(0, min(100, 100 - dup_pct * 2.0))           # 5% duplicates -> -10 points

# consistency proxy: -5 per text column with stray whitespace
text_cols = df.select_dtypes("object").columns
untidy = sum((df[c].dropna().astype(str) != df[c].dropna().astype(str).str.strip()).any() for c in text_cols)
consistency = max(0, 100 - 5 * untidy)

# validity proxy: -4 per numeric column with >1% outliers (Tukey fences)
def outlier_share(s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty: return 0.0
    q1, q3 = s.quantile([0.25, 0.75]); iqr = q3 - q1
    return ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).mean()
heavy = sum(outlier_share(df[c]) > 0.01 for c in df.select_dtypes("number").columns)
validity = max(0, 100 - 4 * heavy)

scores = {"completeness": completeness, "uniqueness": uniqueness, "consistency": consistency, "validity": validity}
overall = sum(scores[k] * w for k, w in WEIGHTS.items())
display(pd.Series(scores).round(1).to_frame("score"))
print(f"Overall quality score: {overall:.1f}/100")
"""
    return make_snippet("Quality score", body, **load)


# --------------------------------------------------------------------------- imputation & outliers
def imputation_snippet(
    columns: Sequence[str], strategy: str = "median", fill_value: Any = 0, **load: Any
) -> str:
    if strategy not in IMPUTE_STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'")
    num_rule = {
        "median": 'SimpleImputer(strategy="median")',
        "mean": 'SimpleImputer(strategy="mean")',
        "mode": 'SimpleImputer(strategy="most_frequent")',
        "constant": f'SimpleImputer(strategy="constant", fill_value={_q(fill_value)})',
        "knn": "KNNImputer(n_neighbors=5)",
    }[strategy]
    cat_rule = (
        f'SimpleImputer(strategy="constant", fill_value={_q(str(fill_value))})'
        if strategy == "constant"
        else 'SimpleImputer(strategy="most_frequent")'
    )
    body = f"""from sklearn.impute import SimpleImputer, KNNImputer

cols = {_list(columns)}
df_imputed = df.copy()
num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df_imputed[c])]
cat_cols = [c for c in cols if c not in num_cols]
before = df_imputed[cols].isna().sum()

if num_cols:
    df_imputed[num_cols] = {num_rule}.fit_transform(df_imputed[num_cols])
if cat_cols:
    df_imputed[cat_cols] = {cat_rule}.fit_transform(df_imputed[cat_cols])
#~ Median is robust to outliers; mean is not. KNN borrows values from similar rows (numeric only).
#~ For machine learning, fit the imputer on training rows only to avoid leakage.
#? assert df_imputed[cols].isna().sum().sum() == 0, "some values are still missing"
display(pd.DataFrame({{"missing_before": before, "missing_after": df_imputed[cols].isna().sum()}}))
"""
    return make_snippet(f"Imputation ({strategy})", body, **load)


def outliers_snippet(
    column: str, method: str = "iqr", treatment: str = "none", k: float | None = None, **load: Any
) -> str:
    """Detect outliers on one column; optionally treat them and compare before/after."""
    if treatment not in OUTLIER_TREATMENTS:
        raise ValueError(f"Unknown treatment '{treatment}'")
    o = resolve(load.get("options"))
    if method == "zscore":
        thr = k or 3.0
        detect = f"""z = (s - s.mean()) / s.std(ddof=0)
mask = z.abs() > {thr}                  # |z| > {thr}
lower, upper = s.mean() - {thr} * s.std(ddof=0), s.mean() + {thr} * s.std(ddof=0)"""
    elif method == "modified_zscore":
        thr = k or 3.5
        detect = f"""mad = (s - s.median()).abs().median()
mz = 0.6745 * (s - s.median()) / mad    # modified z-score (robust, MAD based)
mask = mz.abs() > {thr}
lower, upper = s.median() - {thr} * mad / 0.6745, s.median() + {thr} * mad / 0.6745"""
    else:
        mult = k or 1.5
        detect = f"""q1, q3 = s.quantile(0.25), s.quantile(0.75)
iqr = q3 - q1
lower, upper = q1 - {mult} * iqr, q3 + {mult} * iqr
mask = (s < lower) | (s > upper)        # Tukey fences"""
    treat = {
        "none": "t = s.copy()",
        "cap": "t = s.clip(lower=lower, upper=upper)          # winsorize at the fences",
        "remove": "t = s[~mask]                                   # drop the outlying rows",
        "log1p": "t = np.log1p(s - s.min())                     # compress the long right tail",
    }[treatment]
    chart = plot_code(
        "box",
        {"data": "cmp", "y": "value", "x": "version", "title": f"{column}: before vs after"},
        o,
    )
    body = f"""col = {_q(column)}
s = pd.to_numeric(df[col], errors="coerce").dropna()
{detect}

print(f"{{mask.sum()}} outliers ({{mask.mean():.1%}}) in '{{col}}'")
print("Skewness:", round(s.skew(), 3), " Kurtosis:", round(s.kurt(), 3))
display(df.loc[s[mask].index].head(15))

# %%
#~ Treatment: capping keeps every row; removing loses rows; log compresses the scale.
{treat}
cmp = pd.concat([s.rename("value").to_frame().assign(version="before"),
                 t.rename("value").to_frame().assign(version="after")])
display(cmp.groupby("version")["value"].describe())

# %%
{chart}
"""
    return make_snippet(f"Outliers ({method}, {treatment})", body, **load)


# --------------------------------------------------------------------------- validation rules
def derive_rules(df: pd.DataFrame, max_categories: int = 12) -> list[dict[str, Any]]:
    """Validation rules observed in the data: nulls, uniqueness, ranges, allowed values."""
    rules: list[dict[str, Any]] = []
    for col in df.columns:
        s = df[col]
        rule: dict[str, Any] = {
            "column": col,
            "nullable": bool(s.isna().any()),
            "unique": bool(len(s) > 0 and s.notna().all() and s.is_unique),
            "kind": "str",
            "min": None,
            "max": None,
            "allowed": None,
        }
        if pd.api.types.is_bool_dtype(s):
            rule["kind"] = "bool"
        elif pd.api.types.is_integer_dtype(s):
            rule["kind"] = "int"
        elif pd.api.types.is_float_dtype(s):
            rule["kind"] = "float"
        elif pd.api.types.is_datetime64_any_dtype(s):
            rule["kind"] = "datetime"
        if rule["kind"] in ("int", "float") and s.notna().any():
            rule["min"], rule["max"] = float(s.min()), float(s.max())
            if rule["kind"] == "int":
                rule["min"], rule["max"] = int(s.min()), int(s.max())
        elif rule["kind"] == "str" and 0 < s.nunique() <= max_categories:
            rule["allowed"] = sorted(s.dropna().astype(str).unique().tolist())
        rules.append(rule)
    return rules


def validation_snippet(
    rules: Sequence[Mapping[str, Any]], style: str = "pandera", **load: Any
) -> str:
    """style: 'pandera' (declarative schema) or 'asserts' (no extra dependency)."""
    if style == "pandera":
        cols = []
        for r in rules:
            checks = []
            if r["min"] is not None:
                checks.append(f"pa.Check.ge({r['min']!r})")
            if r["max"] is not None:
                checks.append(f"pa.Check.le({r['max']!r})")
            if r["allowed"]:
                checks.append(f"pa.Check.isin({r['allowed']!r})")
            dtype = {
                "int": "int",
                "float": "float",
                "bool": "bool",
                "datetime": '"datetime64[ns]"',
                "str": "str",
            }[r["kind"]]
            nullable = r["nullable"]
            cols.append(
                f"    {r['column']!r}: pa.Column({dtype}, checks=[{', '.join(checks)}], "
                f"nullable={nullable}, unique={r['unique']}, coerce=True),"
            )
        body = (
            "#~ The schema below was learned from the data as it is now: tighten it to your real business rules.\n"
            "schema = pa.DataFrameSchema({\n" + "\n".join(cols) + "\n}, strict=False)\n\n"
            "try:\n    schema.validate(df, lazy=True)       # lazy=True reports every failure at once\n"
            "    print('All validation rules passed')\n"
            "except pa.errors.SchemaErrors as err:\n    display(err.failure_cases)\n"
        )
        return make_snippet(
            "Data validation (pandera)",
            body,
            imports=(
                "import pandas as pd",
                "import numpy as np",
                "try:\n    import pandera.pandas as pa   # pandera >= 0.24\nexcept ImportError:\n    import pandera as pa",
            ),
            **load,
        )
    lines = ["problems = []"]
    for r in rules:
        c = _q(r["column"])
        if not r["nullable"]:
            lines.append(
                f"if df[{c}].isna().any(): problems.append('{r['column']} has missing values')"
            )
        if r["unique"]:
            lines.append(
                f"if not df[{c}].is_unique: problems.append('{r['column']} should be unique')"
            )
        if r["min"] is not None:
            lines.append(
                f"if (pd.to_numeric(df[{c}], errors='coerce') < {r['min']!r}).any(): problems.append('{r['column']} below {r['min']}')"
            )
        if r["max"] is not None:
            lines.append(
                f"if (pd.to_numeric(df[{c}], errors='coerce') > {r['max']!r}).any(): problems.append('{r['column']} above {r['max']}')"
            )
        if r["allowed"]:
            lines.append(
                f"if (~df[{c}].dropna().astype(str).isin({r['allowed']!r})).any(): problems.append('{r['column']} has unexpected values')"
            )
    lines += [
        "",
        "assert not problems, '\\n'.join(problems)   # fail loudly in a pipeline",
        "print('All validation rules passed')",
    ]
    body = (
        "#~ Plain asserts: no extra package needed. Tighten the rules to your real business logic.\n"
        + "\n".join(lines)
    )
    return make_snippet("Data validation (asserts)", body, **load)


# --------------------------------------------------------------------------- dictionary
_ROLE_CAST = {
    "date": "df_doc[{c}] = pd.to_datetime(df_doc[{c}], errors='coerce')",
    "measure": "df_doc[{c}] = pd.to_numeric(df_doc[{c}], errors='coerce')",
    "identifier": "df_doc[{c}] = df_doc[{c}].astype(str)",
    "dimension": "df_doc[{c}] = df_doc[{c}].astype('category')",
}


def dictionary_apply_snippet(dictionary: Mapping[str, Any], **load: Any) -> str:
    """Rename and cast columns according to the documented roles."""
    entries = {k: v for k, v in dictionary.items() if k != "__glossary__"}
    renames = {
        c: e["rename_to"] for c, e in entries.items() if e.get("rename_to") and e["rename_to"] != c
    }
    lines = ["df_doc = df.copy()", ""]
    for col, entry in entries.items():
        role = entry.get("role", "feature")
        desc = str(entry.get("description", "")).replace("\n", " ")
        lines.append(f"# {col} [{role}{', ' + entry['unit'] if entry.get('unit') else ''}]: {desc}")
        if role in _ROLE_CAST:
            lines.append(_ROLE_CAST[role].format(c=_q(col)))
    if renames:
        lines += ["", f"df_doc = df_doc.rename(columns={renames!r})"]
    lines += ["", "#? assert df_doc.shape == df.shape", "display(df_doc.dtypes)"]
    return make_snippet("Apply the data dictionary", "\n".join(lines), **load)


def dictionary_validation_snippet(
    dictionary: Mapping[str, Any], df: pd.DataFrame, style: str = "asserts", **load: Any
) -> str:
    """Validation derived from documented roles plus observed ranges/categories."""
    observed = {r["column"]: r for r in derive_rules(df)}
    rules = []
    for col, entry in ((k, v) for k, v in dictionary.items() if k != "__glossary__"):
        base = dict(
            observed.get(col)
            or {
                "column": col,
                "nullable": True,
                "unique": False,
                "kind": "str",
                "min": None,
                "max": None,
                "allowed": None,
            }
        )
        role = entry.get("role")
        if role == "identifier":
            base.update(nullable=False, unique=True)
        if role != "measure":
            base["min"] = base["max"] = None
        rules.append(base)
    return validation_snippet(rules, style, **load)
