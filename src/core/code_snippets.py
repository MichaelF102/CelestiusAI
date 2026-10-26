"""Generators for standalone Python snippets that reproduce each analysis in Colab/Jupyter.

Every generator returns plain source text split into cells with ``# %%`` markers (VS Code /
Spyder / Colab-friendly). ``build_notebook`` converts that text into a real ``.ipynb`` file.
Snippets only use pandas, numpy, scipy, statsmodels, scikit-learn and plotly, which are
preinstalled on Google Colab, and they never depend on this application.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from src.core.code_dq import outliers_snippet
from src.core.code_options import (  # re-exported: other modules import these from here
    CELL,
    PLACEHOLDER_FILE,
    CodeOptions,
    build_notebook,
    colab_cell,
    combine_snippets,
    load_cell,
    make_snippet,
    resolve,
)
from src.core.code_options import (
    lst as _list,
)
from src.core.code_options import (
    q as _q,
)
from src.core.code_plots import plot_code, raw

__all__ = [
    "CELL",
    "PLACEHOLDER_FILE",
    "CodeOptions",
    "build_notebook",
    "colab_cell",
    "combine_snippets",
    "load_cell",
    "make_snippet",
    "resolve",
    "plot_code",
    "outliers_snippet",
]


def _opts(load: dict[str, Any]) -> CodeOptions:
    return resolve(load.get("options"))


def load_kwargs(dataset: Any) -> dict[str, Any]:
    """Keyword arguments for the snippet generators describing how `dataset` was loaded."""
    report = getattr(dataset, "ingestion_report", None)
    return {
        "filename": getattr(dataset, "filename", PLACEHOLDER_FILE),
        "sheet": getattr(report, "sheet_name_used", None),
        "encoding": getattr(report, "encoding_used", None),
        "delimiter": getattr(report, "delimiter_detected", None),
    }


# --------------------------------------------------------------------------- profiling / quality
def cleaning_snippet(specs: Sequence[Any], **load: Any) -> str:
    """Pandas code that reproduces the applied cleaning transformations, in order."""
    lines = ["# Cleaning steps applied in the app, in order", "df_clean = df.copy()", ""]
    if not specs:
        lines.append("# (no transformations have been applied yet)")
    for n, spec in enumerate(specs, start=1):
        lines.append(f"# Step {n}: {spec.description or spec.action_type}")
        lines.extend(_spec_to_code(spec))
        lines.append("")
    lines.append("print(df_clean.shape)\ndf_clean.head()")
    return make_snippet("Cleaning pipeline", "\n".join(lines), **load)


def _spec_to_code(spec: Any) -> list[str]:
    a, c, p = spec.action_type, spec.column_name, spec.parameters or {}
    col = _q(c)
    if a == "drop_column":
        return [f"df_clean = df_clean.drop(columns=[{col}])"]
    if a == "drop_duplicates":
        return [
            f"df_clean = df_clean.drop_duplicates(subset={_q(p.get('subset'))}, "
            f"keep={_q(p.get('keep', 'first'))})"
        ]
    if a == "drop_missing_rows":
        subset = [c] if c else p.get("subset")
        return [f"df_clean = df_clean.dropna(subset={_q(subset)})"]
    if a == "impute_numeric":
        strategy = p.get("strategy", "median")
        if strategy == "constant":
            fill = f"float({_q(p.get('fill_value', 0.0))})"
        elif strategy in ("median", "mean"):
            fill = f"df_clean[{col}].dropna().astype(float).{strategy}()"
        else:
            fill = "0.0"
        return [f"df_clean[{col}] = df_clean[{col}].fillna({fill})"]
    if a == "impute_categorical":
        if p.get("strategy", "mode") == "mode":
            fill = f"str(df_clean[{col}].mode().iloc[0]) if not df_clean[{col}].mode().empty else 'Missing'"
        else:
            fill = _q(str(p.get("fill_value", "Missing")))
        return [f"df_clean[{col}] = df_clean[{col}].fillna({fill})"]
    if a == "strip_whitespace":
        return [f"df_clean[{col}] = df_clean[{col}].astype(str).str.strip()"]
    if a == "standardize_casing":
        method = {"lower": "lower", "upper": "upper"}.get(p.get("case", "title"), "title")
        return [
            f"_m = df_clean[{col}].notna()",
            f"df_clean.loc[_m, {col}] = df_clean.loc[_m, {col}].astype(str).str.{method}()",
        ]
    if a == "cap_outliers":
        lo, hi = float(p.get("lower_percentile", 0.01)), float(p.get("upper_percentile", 0.99))
        return [
            f"_s = df_clean[{col}].dropna().astype(float)",
            f"df_clean[{col}] = df_clean[{col}].astype(float)",
            f"df_clean.loc[_s.index, {col}] = _s.clip(_s.quantile({lo}), _s.quantile({hi}))",
        ]
    if a == "convert_type":
        t = p.get("target_type", "datetime")
        if t == "numeric":
            return [f"df_clean[{col}] = pd.to_numeric(df_clean[{col}], errors='coerce')"]
        if t == "boolean":
            return [f"df_clean[{col}] = df_clean[{col}].astype(bool)"]
        return [f"df_clean[{col}] = pd.to_datetime(df_clean[{col}], errors='coerce')"]
    return [f"# Unsupported action '{a}' - skipped"]


# --------------------------------------------------------------------------- exploration
def hypothesis_test_snippet(
    kind: str, variables: Sequence[str], alpha: float = 0.05, **load: Any
) -> str:
    """kind: two_group | multi_group | paired | categorical | correlation."""
    v = list(variables)
    if kind == "two_group":
        body = f"""measure, group = {_q(v[0])}, {_q(v[1])}
alpha = {alpha}
groups = [g.dropna() for _, g in df.groupby(group)[measure]]
assert len(groups) == 2, "needs exactly two groups"
a, b = groups

# Assumption checks
from scipy import stats
print("Shapiro p (each group):", [round(stats.shapiro(x.sample(min(len(x), 5000), random_state=0))[1], 4) for x in groups])
print("Levene p (equal variances):", round(stats.levene(a, b)[1], 4))

# %%
t, p = stats.ttest_ind(a, b, equal_var=False)         # Welch t-test
u, p_mw = stats.mannwhitneyu(a, b)                     # non-parametric alternative
pooled = np.sqrt(((len(a) - 1) * a.var() + (len(b) - 1) * b.var()) / (len(a) + len(b) - 2))
d = (a.mean() - b.mean()) / pooled
se = np.sqrt(a.var() / len(a) + b.var() / len(b))
ci = (a.mean() - b.mean() + np.array([-1, 1]) * stats.norm.ppf(1 - alpha / 2) * se)
print(f"Welch t={{t:.3f}}, p={{p:.4g}} | Mann-Whitney p={{p_mw:.4g}}")
print(f"Cohen's d={{d:.3f}}, mean diff CI{{int((1 - alpha) * 100)}}%=({{ci[0]:.3f}}, {{ci[1]:.3f}})")
"""
    elif kind == "multi_group":
        body = f"""from scipy import stats
measure, group = {_q(v[0])}, {_q(v[1])}
groups = [g.dropna() for _, g in df.groupby(group)[measure]]
f, p = stats.f_oneway(*groups)
h, p_kw = stats.kruskal(*groups)
print(f"ANOVA F={{f:.3f}}, p={{p:.4g}} | Kruskal-Wallis H={{h:.3f}}, p={{p_kw:.4g}}")

# %%
import statsmodels.api as sm
import statsmodels.formula.api as smf
model = smf.ols(f"Q('{{measure}}') ~ C(Q('{{group}}'))", data=df).fit()
aov = sm.stats.anova_lm(model, typ=2)
eta_sq = aov["sum_sq"].iloc[0] / aov["sum_sq"].sum()
print(aov); print("eta squared:", round(eta_sq, 4))

# %%
from statsmodels.stats.multicomp import pairwise_tukeyhsd
print(pairwise_tukeyhsd(df[measure].dropna(), df.loc[df[measure].notna(), group]))
"""
    elif kind == "paired":
        body = f"""from scipy import stats
a, b = df[{_q(v[0])}].dropna(), df[{_q(v[1])}].dropna()
n = min(len(a), len(b)); a, b = a.iloc[:n], b.iloc[:n]
t, p = stats.ttest_rel(a, b)
w, p_w = stats.wilcoxon(a - b)
print(f"paired t={{t:.3f}}, p={{p:.4g}} | Wilcoxon p={{p_w:.4g}}")
print("mean difference:", round((a - b).mean(), 4))
"""
    elif kind == "categorical":
        body = f"""from scipy import stats
table = pd.crosstab(df[{_q(v[0])}], df[{_q(v[1])}])
chi2, p, dof, expected = stats.chi2_contingency(table)
print(table)
print(f"chi2={{chi2:.3f}}, dof={{dof}}, p={{p:.4g}}")
print("Min expected count:", expected.min().round(2), "(<5 weakens chi-square)")
if table.shape == (2, 2):
    print("Fisher exact p:", stats.fisher_exact(table)[1])
print("Cramer's V:", round(np.sqrt((chi2 / table.values.sum()) / (min(table.shape) - 1)), 3))
"""
    else:  # correlation
        body = f"""from scipy import stats
x, y = pd.to_numeric(df[{_q(v[0])}], errors="coerce"), pd.to_numeric(df[{_q(v[1])}], errors="coerce")
mask = x.notna() & y.notna()
r, p = stats.pearsonr(x[mask], y[mask])
rho, p_s = stats.spearmanr(x[mask], y[mask])
n = mask.sum()
z = np.arctanh(r); se = 1 / np.sqrt(n - 3)
lo, hi = np.tanh(z + np.array([-1, 1]) * stats.norm.ppf(0.975) * se)
print(f"Pearson r={{r:.3f}} (p={{p:.4g}}, 95% CI {{lo:.3f}}..{{hi:.3f}}) | Spearman rho={{rho:.3f}} (p={{p_s:.4g}}), n={{n}}")
"""
    return make_snippet(f"Hypothesis test ({kind.replace('_', ' ')})", body, **load)


def regression_snippet(outcome: str, predictors: Sequence[str], **load: Any) -> str:
    body = f"""import statsmodels.api as sm

outcome, predictors = {_q(outcome)}, {_list(predictors)}
data = df[[outcome, *predictors]].dropna()
X = pd.get_dummies(data[predictors], drop_first=True).astype(float)
X = sm.add_constant(X)
model = sm.OLS(data[outcome].astype(float), X).fit()
print(model.summary())

# %%
# Diagnostics
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
print("Breusch-Pagan p (homoscedasticity):", round(het_breuschpagan(model.resid, X)[1], 4))
print("Shapiro p (residual normality):", round(stats.shapiro(model.resid.sample(min(len(model.resid), 5000), random_state=0))[1], 4))
vif = pd.Series([variance_inflation_factor(X.values, i) for i in range(1, X.shape[1])], index=X.columns[1:])
display(vif.round(2).to_frame("VIF"))

# %%
import plotly.express as px
px.scatter(x=model.fittedvalues, y=model.resid, labels={{"x": "fitted", "y": "residual"}},
           title="Residuals vs fitted").show()
"""
    return make_snippet(f"OLS regression: {outcome}", body, **load)


# --------------------------------------------------------------------------- predictive
def model_training_snippet(
    target: str,
    features: Sequence[str],
    task: str = "regression",
    test_size: float = 0.2,
    seed: int = 42,
    **load: Any,
) -> str:
    classification = "classification" in task
    if classification:
        models = """    "Baseline (most frequent)": DummyClassifier(strategy="most_frequent"),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=SEED),
    "Hist Gradient Boosting": HistGradientBoostingClassifier(random_state=SEED),"""
        metrics = """pred = pipe.predict(X_test)
    row = {"model": name, "accuracy": accuracy_score(y_test, pred),
           "f1_macro": f1_score(y_test, pred, average="macro")}
    if hasattr(pipe, "predict_proba") and y.nunique() == 2:
        row["roc_auc"] = roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])"""
        imports = (
            "from sklearn.dummy import DummyClassifier",
            "from sklearn.linear_model import LogisticRegression",
            "from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier",
            "from sklearn.metrics import accuracy_score, f1_score, roc_auc_score",
        )
        split_extra = ", stratify=y"
        scoring = "f1_macro"
    else:
        models = """    "Baseline (mean)": DummyRegressor(strategy="mean"),
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=SEED),
    "Hist Gradient Boosting": HistGradientBoostingRegressor(random_state=SEED),"""
        metrics = """pred = pipe.predict(X_test)
    row = {"model": name, "MAE": mean_absolute_error(y_test, pred),
           "RMSE": mean_squared_error(y_test, pred) ** 0.5, "R2": r2_score(y_test, pred)}"""
        imports = (
            "from sklearn.dummy import DummyRegressor",
            "from sklearn.linear_model import LinearRegression, Ridge",
            "from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor",
            "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score",
        )
        split_extra = ""
        scoring = "r2"
    body = f"""SEED = {seed}
target, features = {_q(target)}, {_list(features)}
data = df[[target, *features]].dropna(subset=[target])
X, y = data[features], data[target]

num_cols = X.select_dtypes("number").columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

# Preprocessing is fitted on the training split only (no leakage)
preprocess = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num_cols),
    ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size={test_size}, random_state=SEED{split_extra})

# %%
models = {{
{models}
}}
rows = []
for name, estimator in models.items():
    pipe = Pipeline([("prep", preprocess), ("model", estimator)]).fit(X_train, y_train)
    {metrics}
    rows.append(row)
leaderboard = pd.DataFrame(rows).set_index("model").round(4)
leaderboard

# %%
# Which features matter? (permutation importance on held-out data)
from sklearn.inspection import permutation_importance
best_name = leaderboard.index[leaderboard.iloc[:, -1 if "{scoring}" == "r2" else 1].argmax()]
best = Pipeline([("prep", preprocess), ("model", models[best_name])]).fit(X_train, y_train)
imp = permutation_importance(best, X_test, y_test, n_repeats=5, random_state=SEED, scoring="{scoring}")
pd.Series(imp.importances_mean, index=X_test.columns).sort_values(ascending=False)
"""
    all_imports = (
        "import pandas as pd",
        "import numpy as np",
        "from sklearn.compose import ColumnTransformer",
        "from sklearn.impute import SimpleImputer",
        "from sklearn.model_selection import train_test_split",
        "from sklearn.pipeline import Pipeline",
        "from sklearn.preprocessing import OneHotEncoder, StandardScaler",
        *imports,
    )
    return make_snippet(f"Predictive model: {target}", body, imports=all_imports, **load)


# --------------------------------------------------------------------------- profiling
def profile_snippet(**load: Any) -> str:
    body = """# Schema, missing values and duplicates
#~ dtypes tell you how pandas will treat each column; 'object' usually means text.
print(df.dtypes)

#~ Share of missing cells per column, worst first.
missing = df.isna().sum().to_frame("missing")
missing["missing_pct"] = (missing["missing"] / len(df) * 100).round(2)
display(missing.sort_values("missing", ascending=False))

#~ Exact duplicate rows are copies of an earlier row across ALL columns.
print("Exact duplicate rows:", df.duplicated().sum())
df.describe(include="all").T
"""
    return make_snippet("Dataset profile", body, **load)


def missing_snippet(**load: Any) -> str:
    o = _opts(load)
    chart = plot_code(
        "bar",
        {
            "data": "missing.reset_index()",
            "x": "index",
            "y": "missing_pct",
            "title": "Missing % by column",
        },
        o,
    )
    body = f"""# Missing-value audit
missing = df.isna().sum().to_frame("missing_count")
missing["missing_pct"] = (missing["missing_count"] / len(df) * 100).round(2)
missing = missing.sort_values("missing_count", ascending=False)
display(missing[missing["missing_count"] > 0])

#~ If two columns are often missing together, a single upstream cause is likely.
null_corr = df.isna().astype(int).loc[:, df.isna().any()].corr()
null_corr

# %%
missing = missing.rename(columns={{"missing_pct": "missing_pct"}})
{chart}
"""
    return make_snippet("Missing values", body, **load)


def duplicates_snippet(subset: Sequence[str] | None = None, **load: Any) -> str:
    sub = _list(subset) if subset else "None"
    body = f"""# Exact and key-based duplicates
subset = {sub}   # None = compare entire rows
dup_mask = df.duplicated(subset=subset, keep=False)
print("Rows involved in duplicates:", dup_mask.sum())
df[dup_mask].sort_values(by=list(df.columns) if subset is None else subset).head(20)

# %%
#~ A candidate key is a column that is never empty and never repeats.
candidates = [c for c in df.columns if df[c].notna().all() and df[c].is_unique]
print("Candidate primary keys:", candidates)
#? assert df.drop_duplicates(subset=subset).shape[0] <= df.shape[0]
"""
    return make_snippet("Duplicate analysis", body, **load)


# --------------------------------------------------------------------------- exploration
def distribution_snippet(column: str, numeric: bool = True, **load: Any) -> str:
    o = _opts(load)
    col = _q(column)
    if numeric:
        chart = plot_code(
            "hist",
            {
                "data": "s.to_frame(col)",
                "x": raw("col"),
                "title": f"Distribution of {column}",
                "marginal": "box",
            },
            o,
        )
        body = f"""col = {col}
#? assert col in df.columns, f"{{col}} is not a column"
s = pd.to_numeric(df[col], errors="coerce").dropna()
display(s.describe())
print("Skewness:", round(s.skew(), 3), "| Kurtosis:", round(s.kurt(), 3))

# %%
#~ Normality: a small p-value (< 0.05) means the data departs from a bell curve.
from scipy import stats
sample = s.sample(min(len(s), 5000), random_state=0)
if len(sample) >= 8:
    print("D'Agostino K2 p-value:", round(stats.normaltest(sample)[1], 4))
    print("Shapiro-Wilk p-value:", round(stats.shapiro(sample)[1], 4))

# %%
{chart}
"""
    else:
        chart = plot_code(
            "bar",
            {
                "data": "counts.reset_index()",
                "x": raw("col"),
                "y": "count",
                "title": f"Top categories of {column}",
            },
            o,
        )
        body = f"""col = {col}
#? assert col in df.columns, f"{{col}} is not a column"
counts = df[col].value_counts(dropna=False).head(20)
counts.index = counts.index.astype(str)
counts.index.name = col
counts.name = "count"
display(counts.to_frame().assign(pct=lambda d: (d["count"] / len(df) * 100).round(2)))

# %%
{chart}
"""
    return make_snippet(f"Distribution of {column}", body, **load)


def correlation_snippet(
    columns: Sequence[str] | None = None, method: str = "pearson", **load: Any
) -> str:
    o = _opts(load)
    cols = _list(columns) if columns else "df.select_dtypes('number').columns.tolist()"
    chart = plot_code("heatmap", {"data": "corr", "title": f"{method.capitalize()} correlation"}, o)
    body = f"""cols = {cols}
corr = df[cols].corr(method={_q(method)})
#~ +1 = rise together, -1 = one rises as the other falls, 0 = no linear/monotonic link.

# %%
{chart}

# %%
#~ Strongest pairs, with a p-value: is the relationship distinguishable from chance?
from scipy import stats
rows = []
for i, a in enumerate(cols):
    for b in cols[i + 1:]:
        pair = df[[a, b]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(pair) > 3:
            r, p = (stats.spearmanr if {_q(method)} == "spearman" else stats.pearsonr)(pair[a], pair[b])
            rows.append({{"var_1": a, "var_2": b, "r": r, "p_value": p, "n": len(pair)}})
pairs = pd.DataFrame(rows)
pairs.reindex(pairs["r"].abs().sort_values(ascending=False).index).head(15)
"""
    return make_snippet(f"{method.capitalize()} correlation", body, **load)


def grouped_snippet(group_col: str, measure: str, agg: str = "mean", **load: Any) -> str:
    o = _opts(load)
    chart = plot_code(
        "bar",
        {
            "data": "summary.reset_index()",
            "x": raw("group_col"),
            "y": raw("agg"),
            "title": f"{agg} of {measure} by {group_col}",
        },
        o,
    )
    body = f"""group_col, measure, agg = {_q(group_col)}, {_q(measure)}, {_q(agg)}
#? assert {{group_col, measure}} <= set(df.columns)
summary = (df.groupby(group_col)[measure]
             .agg([agg, "count"])
             .sort_values(agg, ascending=False))
display(summary)
#~ Small groups (low 'count') give unreliable averages: read them with care.

# %%
{chart}
"""
    return make_snippet(f"{agg.capitalize()} of {measure} by {group_col}", body, **load)


def crosstab_snippet(col_a: str, col_b: str, **load: Any) -> str:
    body = f"""a, b = {_q(col_a)}, {_q(col_b)}
table = pd.crosstab(df[a], df[b])
display(table)
display(pd.crosstab(df[a], df[b], normalize="index").round(3))   # row percentages

# %%
from scipy.stats import chi2_contingency
chi2, p, dof, expected = chi2_contingency(table)
n = table.values.sum()
cramers_v = np.sqrt((chi2 / n) / (min(table.shape) - 1))
#~ Cramer's V: 0 = independent, 1 = perfectly associated.
print(f"chi2={{chi2:.2f}}, p={{p:.4g}}, Cramer's V={{cramers_v:.3f}}")
#? assert (expected >= 1).all(), "very small expected counts: chi-square may be unreliable"
"""
    return make_snippet(f"Association: {col_a} x {col_b}", body, **load)


def time_trend_snippet(date_col: str, measure: str, freq: str = "D", **load: Any) -> str:
    o = _opts(load)
    chart = plot_code(
        "line",
        {
            "data": "ts.reset_index()",
            "x": raw("date_col"),
            "y": raw("measure"),
            "title": f"{measure} over time",
        },
        o,
    )
    body = f"""date_col, measure, freq = {_q(date_col)}, {_q(measure)}, {_q(freq)}
ts = (df.assign(**{{date_col: pd.to_datetime(df[date_col], errors="coerce")}})
        .dropna(subset=[date_col])
        .set_index(date_col)[measure]
        .resample(freq).mean())
#? assert ts.notna().any(), "no dated values found"

# %%
{chart}

# %%
#~ A rolling mean smooths noise so the trend is easier to see.
display(ts.rolling(window=7, min_periods=1).mean().tail())
"""
    return make_snippet(f"Trend of {measure}", body, **load)


def violin_snippet(measure: str, group_col: str | None = None, **load: Any) -> str:
    o = _opts(load)
    spec: dict[str, Any] = {"data": "data", "y": measure, "title": f"{measure} distribution"}
    if group_col:
        spec["x"] = group_col
    chart = plot_code("violin", spec, o)
    body = f"""data = df[[{_list([c for c in (group_col, measure) if c])[1:-1]}]].dropna()
data[{_q(measure)}] = pd.to_numeric(data[{_q(measure)}], errors="coerce")
display(data.groupby({_q(group_col)})[{_q(measure)}].describe() if {_q(group_col)} else data[{_q(measure)}].describe())

# %%
{chart}
"""
    return make_snippet(f"Violin: {measure}", body, **load)


def multi_scatter_snippet(
    x: str, y: str, color: str | None = None, size: str | None = None, **load: Any
) -> str:
    o = _opts(load)
    chart = plot_code(
        "scatter",
        {
            "data": "data",
            "x": x,
            "y": y,
            "color": color,
            "size": size,
            "trend": True,
            "title": f"{y} vs {x}",
        },
        o,
    )
    cols = [c for c in (x, y, color, size) if c]
    body = f"""data = df[{_list(cols)}].dropna()
if len(data) > 2500:
    data = data.sample(2500, random_state=42)      # keep the plot responsive

# %%
{chart}
"""
    return make_snippet(f"Scatter: {y} vs {x}", body, **load)


CUSTOM_CHART_KINDS = {
    "Bar Chart (Aggregated / Category)": "bar",
    "Scatter / Bubble Plot": "scatter",
    "Line Chart (Trend / Temporal)": "line",
    "Histogram / Frequency": "hist",
    "Box Plot (Segment Comparison)": "box",
    "Violin Plot (Kernel Density)": "violin",
    "Pie / Donut Chart": "pie",
    "Treemap (Hierarchical Slices)": "treemap",
    "2D Density Heatmap": "density",
}


def custom_chart_snippet(
    chart_type: str,
    x: str,
    y: str | None = None,
    agg: str = "Mean",
    color: str | None = None,
    size: str | None = None,
    facet: str | None = None,
    title: str = "",
    log_y: bool = False,
    trend: bool = False,
    horizontal: bool = False,
    **load: Any,
) -> str:
    """Code for the Custom Chart Studio configuration (all 9 chart types)."""
    o = _opts(load)
    kind = CUSTOM_CHART_KINDS.get(chart_type, chart_type)
    agg_map = {"Sum": "sum", "Mean": "mean", "Median": "median", "Min": "min", "Max": "max"}
    keys = (
        [x]
        + ([color] if color and color != x else [])
        + ([facet] if facet and facet not in (x, color) else [])
    )
    aggregated = kind in ("bar", "line", "pie", "treemap") and agg not in (
        "None (Raw Values)",
        "None",
    )
    prep = ["data = df.copy()"]
    value = y
    if aggregated:
        if agg == "Count" or not y:
            prep.append(
                f"data = data.groupby({_list(keys)}, as_index=False).size().rename(columns={{'size': 'Record_Count'}})"
            )
            value = "Record_Count"
        else:
            prep.append(
                f"data = data.groupby({_list(keys)}, as_index=False)[{_q(y)}].agg({_q(agg_map.get(agg, 'mean'))})"
            )
    elif kind == "scatter":
        needed = [c for c in (x, y, size) if c]
        prep.append(
            f"data = data.dropna(subset={_list(needed)})   # plotting needs complete x/y/size values"
        )
        prep.append("if len(data) > 3000:\n    data = data.sample(3000, random_state=42)")
    prep_code = "\n".join(prep)
    base = {
        "data": "data",
        "color": color,
        "facet": facet,
        "title": title or f"{kind}: {x}",
        "horizontal": horizontal,
    }

    if kind in ("treemap", "density"):
        # plotly-only chart types: other libraries fall back to a close equivalent
        if o.plot_lib == "plotly":
            if kind == "treemap":
                path = [c for c in (x, color) if c]
                chart = (
                    f"fig = px.treemap(data, path={_list(path)}, values={_q(value) if value else None})\n"
                    f"fig.update_layout(template={_q('plotly_dark' if o.template == 'dark' else 'plotly_white')}, "
                    f"title={_q(o.title or base['title'])}, width={o.width}, height={o.height})\nfig.show()"
                )
            else:
                chart = (
                    f"fig = px.density_heatmap(data, x={_q(x)}, y={_q(y)}, marginal_x='histogram', marginal_y='histogram')\n"
                    f"fig.update_layout(template={_q('plotly_dark' if o.template == 'dark' else 'plotly_white')}, "
                    f"title={_q(o.title or base['title'])}, width={o.width}, height={o.height})\nfig.show()"
                )
        elif kind == "density":
            chart = (
                f"fig, ax = plt.subplots(figsize=({o.width / 100:g}, {o.height / 100:g}))\n"
                f"xy = data[[{_q(x)}, {_q(y)}]].apply(pd.to_numeric, errors='coerce').dropna()\n"
                "h = ax.hist2d(xy.iloc[:, 0], xy.iloc[:, 1], bins=30, cmap='viridis')\n"
                f"fig.colorbar(h[3])\nax.set_xlabel({_q(x)}); ax.set_ylabel({_q(y)})\n"
                "plt.tight_layout()\nplt.show()"
            )
        else:
            chart = (
                "# Treemaps need plotly (matplotlib/seaborn have no equivalent): using a bar chart instead\n"
                + plot_code("bar", {**base, "x": x, "y": value or "Record_Count"}, o)
            )
    elif kind == "bar":
        chart = plot_code("bar", {**base, "x": x, "y": value or "Record_Count"}, o)
    elif kind == "line":
        chart = plot_code("line", {**base, "x": x, "y": value or "Record_Count"}, o)
    elif kind == "scatter":
        chart = plot_code("scatter", {**base, "x": x, "y": y, "size": size, "trend": trend}, o)
    elif kind == "hist":
        chart = plot_code("hist", {**base, "x": x}, o)
    elif kind in ("box", "violin"):
        chart = plot_code(kind, {**base, "x": x, "y": y or x} if y else {**base, "y": x}, o)
    elif kind == "pie":
        chart = plot_code("pie", {**base, "names": x, "values": value}, o)
    else:
        raise ValueError(f"Unknown chart type '{chart_type}'")
    if log_y and o.plot_lib == "plotly":
        chart = chart.replace("fig.show()", "fig.update_yaxes(type='log')\nfig.show()")
    elif log_y:
        chart = chart.replace("plt.tight_layout()", "ax.set_yscale('log')\nplt.tight_layout()")
    body = f"""{prep_code}
display(data.head())

# %%
{chart}
"""
    return make_snippet(title or "Custom chart", body, **load)


def interactive_explorer_snippet(**load: Any) -> str:
    """ipywidgets explorer: pick a column and chart type with dropdowns (Colab/Jupyter)."""
    body = """import ipywidgets as widgets
import plotly.express as px
from IPython.display import display as ipy_display

numeric = df.select_dtypes("number").columns.tolist()
categorical = [c for c in df.columns if c not in numeric]

col = widgets.Dropdown(options=list(df.columns), description="Column")
kind = widgets.ToggleButtons(options=["histogram", "box", "bar (top 15)"], description="Chart")
out = widgets.Output()

def redraw(_=None):
    with out:
        out.clear_output(wait=True)
        c = col.value
        if kind.value == "bar (top 15)" or c in categorical:
            counts = df[c].astype(str).value_counts().head(15).reset_index()
            counts.columns = [c, "count"]
            px.bar(counts, x=c, y="count", title=f"Top values of {c}").show()
        elif kind.value == "box":
            px.box(df, y=c, title=f"{c} spread").show()
        else:
            px.histogram(df, x=c, nbins=30, marginal="box", title=f"Distribution of {c}").show()

col.observe(redraw, names="value")
kind.observe(redraw, names="value")
ipy_display(widgets.VBox([col, kind, out]))
redraw()
"""
    return make_snippet(
        "Interactive column explorer",
        body,
        imports=("import pandas as pd", "import numpy as np"),
        **load,
    )
