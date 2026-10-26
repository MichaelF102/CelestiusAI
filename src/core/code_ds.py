"""Standalone code for the data-scientist components (selection, ensembles, classification, ...)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from src.core.code_options import lst as _list
from src.core.code_options import make_snippet
from src.core.code_options import q as _q
from src.datascience import catalog
from src.datascience.tracking import estimator_code

_BASE_IMPORTS = [
    "import pandas as pd",
    "import numpy as np",
    "from sklearn.compose import ColumnTransformer",
    "from sklearn.impute import SimpleImputer",
    "from sklearn.model_selection import train_test_split, cross_validate, KFold, StratifiedKFold",
    "from sklearn.pipeline import Pipeline",
    "from sklearn.preprocessing import OneHotEncoder, StandardScaler",
]


def _is_reg(task: str) -> bool:
    return task == "regression"


def prelude(
    target: str,
    features: Sequence[str],
    task: str,
    seed: int = 42,
    test_size: float = 0.2,
    split: bool = True,
) -> str:
    """Data selection, leakage-safe preprocessing, and (optionally) the train/test split."""
    stratify = "" if _is_reg(task) else ", stratify=y"
    code = f"""SEED = {seed}
target, features = {_q(target)}, {_list(features)}
data = df[[target, *features]].dropna(subset=[target])
X, y = data[features], data[target]
num_cols = X.select_dtypes("number").columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]
# Imputation, scaling and encoding are fitted on training rows only (inside the pipeline).
preprocess = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num_cols),
    ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", max_categories=30, sparse_output=False))]), cat_cols),
])
"""
    if split:
        code += f"X_train, X_test, y_train, y_test = train_test_split(X, y, test_size={test_size}, random_state=SEED{stratify})\n"
    return code


def _est(
    name: str, task: str, seed: int = 42, params: dict[str, Any] | None = None
) -> tuple[list[str], str]:
    return estimator_code(catalog.build(name, task, seed, params or None))


def _imports(extra: Sequence[str]) -> list[str]:
    return list(dict.fromkeys([*_BASE_IMPORTS, *extra]))


# --------------------------------------------------------------------------- model selection
def race_snippet(
    models: Sequence[str],
    target: str,
    features: Sequence[str],
    task: str,
    cv: int = 5,
    scoring: str | None = None,
    seed: int = 42,
    **load: Any,
) -> str:
    scoring = scoring or ("r2" if _is_reg(task) else "f1_macro")
    imports: list[str] = []
    entries = []
    for m in models:
        imp, expr = _est(m, task, seed)
        imports += imp
        entries.append(f"    {_q(m)}: {expr},")
    splitter = "KFold" if _is_reg(task) else "StratifiedKFold"
    body = (
        prelude(target, features, task, seed, split=False)
        + f"""
cv = {splitter}(n_splits={cv}, shuffle=True, random_state=SEED)   # identical folds for every model
models = {{
{chr(10).join(entries)}
}}

# %%
rows, fold_scores = [], {{}}
for name, estimator in models.items():
    res = cross_validate(Pipeline([("prep", preprocess), ("model", estimator)]), X, y, cv=cv,
                         scoring={_q(scoring)}, return_train_score=True)
    fold_scores[name] = res["test_score"]
    rows.append({{"model": name, "cv_mean": res["test_score"].mean(), "cv_std": res["test_score"].std(ddof=1),
                 "train_mean": res["train_score"].mean(), "fit_seconds": res["fit_time"].sum()}})
board = pd.DataFrame(rows).sort_values("cv_mean", ascending=False).reset_index(drop=True)
board["overfit_gap"] = board["train_mean"] - board["cv_mean"]   # large gap = memorizing the training data
display(board)

# %%
# One-standard-error rule: prefer a simpler model if it is within one SE of the best
best = board.iloc[0]
se = best["cv_std"] / np.sqrt({cv})
print(f"Best: {{best['model']}} ({{best['cv_mean']:.4f}}); anything above {{best['cv_mean'] - se:.4f}} is statistically close.")
display(board[board["cv_mean"] >= best["cv_mean"] - se][["model", "cv_mean", "cv_std"]])
"""
    )
    return make_snippet("Model race", body, imports=_imports(imports), **load)


def nested_cv_snippet(
    model: str,
    target: str,
    features: Sequence[str],
    task: str,
    outer: int = 5,
    inner: int = 3,
    n_iter: int = 8,
    seed: int = 42,
    **load: Any,
) -> str:
    imp, expr = _est(model, task, seed)
    space = {f"model__{k}": v for k, v in catalog.param_space(model, task).items()}
    scoring = "r2" if _is_reg(task) else "f1_macro"
    splitter = "KFold" if _is_reg(task) else "StratifiedKFold"
    body = (
        prelude(target, features, task, seed, split=False)
        + f"""
from sklearn.model_selection import RandomizedSearchCV, cross_val_score

pipe = Pipeline([("prep", preprocess), ("model", {expr})])
space = {space!r}
inner_cv = {splitter}(n_splits={inner}, shuffle=True, random_state=SEED + 1)
outer_cv = {splitter}(n_splits={outer}, shuffle=True, random_state=SEED)

# %%
# The search is repeated INSIDE every outer fold, so the outer score is an honest estimate of "tune, then deploy".
search = RandomizedSearchCV(pipe, space, n_iter={n_iter}, cv=inner_cv, scoring={_q(scoring)}, random_state=SEED)
nested = cross_val_score(search, X, y, cv=outer_cv, scoring={_q(scoring)})
print(f"Nested CV: {{nested.mean():.4f}} +/- {{nested.std(ddof=1):.4f}}")
search.fit(X, y)
print(f"Inner search best score (optimistic): {{search.best_score_:.4f}}  <- compare with the nested score")
"""
    )
    return make_snippet(
        f"Nested CV: {model}",
        body,
        imports=_imports([*imp, "from sklearn.model_selection import RandomizedSearchCV"]),
        **load,
    )


# --------------------------------------------------------------------------- ensembles
def ensemble_snippet(
    kind: str,
    base_models: Sequence[str],
    target: str,
    features: Sequence[str],
    task: str,
    meta: str | None = None,
    seed: int = 42,
    cv: int = 5,
    **load: Any,
) -> str:
    reg = _is_reg(task)
    imports: list[str] = []
    members = []
    for i, m in enumerate(base_models):
        imp, expr = _est(m, task, seed)
        imports += imp
        members.append(f"    ({_q(f'm{i}')}, {expr}),   # {m}")
    suffix = "Regressor" if reg else "Classifier"
    if kind == "bagging":
        ens = f"Bagging{suffix}(estimator=members[0][1], n_estimators=25, random_state=SEED, n_jobs=-1)"
        cls = f"Bagging{suffix}"
    elif kind in ("voting_hard", "voting_soft"):
        voting = "" if reg else f", voting={'hard' if kind == 'voting_hard' else 'soft'!r}"
        ens = f"Voting{suffix}(members{voting})"
        cls = f"Voting{suffix}"
    else:
        mimp, mexpr = _est(
            meta or ("Ridge / L2 Logistic" if reg else "Linear / Logistic Regression"), task, seed
        )
        imports += mimp
        ens = f"Stacking{suffix}(members, final_estimator={mexpr}, cv=3)"
        cls = f"Stacking{suffix}"
    splitter = "KFold" if reg else "StratifiedKFold"
    scoring = "r2" if reg else "f1_macro"
    body = (
        prelude(target, features, task, seed, split=False)
        + f"""
members = [
{chr(10).join(members)}
]
cv = {splitter}(n_splits={cv}, shuffle=True, random_state=SEED)
ensemble = {ens}

# %%
scores = {{}}
for name, est in [*members, ({_q(kind)}, ensemble)]:
    s = cross_validate(Pipeline([("prep", preprocess), ("model", est)]), X, y, cv=cv, scoring={_q(scoring)})["test_score"]
    scores[name] = s
    print(f"{{name:>12}}: {{s.mean():.4f}} +/- {{s.std(ddof=1):.4f}}")
best_base = max(scores[n].mean() for n, _ in members)
print(f"Ensemble gain over the best base model: {{scores[{_q(kind)}].mean() - best_base:+.4f}}")
"""
    )
    return make_snippet(
        f"Ensemble: {kind}",
        body,
        imports=_imports([*imports, f"from sklearn.ensemble import {cls}"]),
        **load,
    )


# --------------------------------------------------------------------------- classification
def classification_snippet(
    model: str,
    target: str,
    features: Sequence[str],
    strategy: str = "none",
    cost_fp: float = 1.0,
    cost_fn: float = 5.0,
    seed: int = 42,
    **load: Any,
) -> str:
    imp, expr = _est(model, "binary_classification", seed)
    resample = {
        "none": "",
        "class_weight": "pipe.named_steps['model'].set_params(class_weight='balanced')   # model must support class_weight\n",
        "oversample": "# random oversampling of the minority class (training rows only)\nidx = np.concatenate([rng.choice(np.flatnonzero(y_fit == c), counts.max(), replace=True) for c in classes])\nX_fit, y_fit = X_fit[idx], y_fit[idx]\n",
        "undersample": "idx = np.concatenate([rng.choice(np.flatnonzero(y_fit == c), counts.min(), replace=False) for c in classes])\nX_fit, y_fit = X_fit[idx], y_fit[idx]\n",
    }.get(strategy, "")
    body = (
        prelude(target, features, "binary_classification", seed)
        + f"""
prep = preprocess.fit(X_train)
X_fit, y_fit = prep.transform(X_train), y_train.to_numpy()
rng = np.random.default_rng(SEED)
classes, counts = np.unique(y_fit, return_counts=True)
print("class balance:", dict(zip(classes, counts)))
pipe = Pipeline([("model", {expr})])
{resample}clf = pipe.fit(X_fit, y_fit)
classes = list(clf.classes_); positive = classes[-1]
score = clf.predict_proba(prep.transform(X_test))[:, -1]
is_pos = (y_test == positive).to_numpy()

# %%
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, confusion_matrix, brier_score_loss
print("ROC AUC:", round(roc_auc_score(is_pos, score), 4), "| average precision:", round(average_precision_score(is_pos, score), 4))
fpr, tpr, _ = roc_curve(is_pos, score)
print("KS statistic:", round(float(np.max(tpr - fpr)), 4), "| Brier:", round(brier_score_loss(is_pos, score), 4))
import plotly.express as px
px.line(pd.DataFrame({{"fpr": fpr, "tpr": tpr}}), x="fpr", y="tpr", title="ROC curve").show()

# %%
# Cost-sensitive threshold: a false negative costs {cost_fn}, a false positive {cost_fp}
rows = []
for t in np.linspace(0.01, 0.99, 99):
    pred = score >= t
    tp, fp, fn = (pred & is_pos).sum(), (pred & ~is_pos).sum(), (~pred & is_pos).sum()
    rows.append({{"threshold": t, "cost": fp * {cost_fp} + fn * {cost_fn}, "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1)}})
thresholds = pd.DataFrame(rows)
best_t = thresholds.loc[thresholds["cost"].idxmin()]
print("cheapest threshold:", round(best_t["threshold"], 2), "| cost:", best_t["cost"], "| cost at 0.5:", thresholds.iloc[49]["cost"])
print(confusion_matrix(is_pos, score >= best_t["threshold"]))

# %%
# Calibration: when the model says 70%, does it happen ~70% of the time?
from sklearn.calibration import calibration_curve
frac, mean_pred = calibration_curve(is_pos, score, n_bins=8, strategy="quantile")
display(pd.DataFrame({{"mean_predicted": mean_pred, "fraction_positive": frac}}))
"""
    )
    return make_snippet(f"Classification toolkit ({strategy})", body, imports=_imports(imp), **load)


# --------------------------------------------------------------------------- regression
def regression_diagnostics_snippet(target: str, features: Sequence[str], **load: Any) -> str:
    body = f"""import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.outliers_influence import variance_inflation_factor
import plotly.express as px

target, features = {_q(target)}, {_list(features)}
data = df[[target, *features]].copy()
data[target] = pd.to_numeric(data[target], errors="coerce")
data = data.dropna()
X = pd.get_dummies(data[features], drop_first=True, dtype=float)
X = sm.add_constant(X.loc[:, X.nunique() > 1])
model = sm.OLS(data[target], X).fit()
print(model.summary())

# %%
infl = model.get_influence()
diag = pd.DataFrame({{"fitted": model.fittedvalues, "std_resid": infl.resid_studentized_internal,
                     "leverage": infl.hat_matrix_diag, "cooks_d": infl.cooks_distance[0]}})
print("Breusch-Pagan p:", het_breuschpagan(model.resid, X)[1], "| White p:", het_white(model.resid, X)[1])
print("Shapiro p (residual normality):", stats.shapiro(model.resid.sample(min(len(model.resid), 5000), random_state=0))[1])
display(diag.sort_values("cooks_d", ascending=False).head(10))   # the most influential rows
px.scatter(diag, x="fitted", y="std_resid", title="Residuals vs fitted (look for funnels / curves)").show()

# %%
n = len(diag)
qq = pd.DataFrame({{"theoretical": stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n), "sample": np.sort(diag["std_resid"])}})
px.scatter(qq, x="theoretical", y="sample", title="Normal Q-Q plot").show()
vif = pd.Series([variance_inflation_factor(X.values, i) for i in range(1, X.shape[1])], index=X.columns[1:])
display(vif.sort_values(ascending=False).round(2).to_frame("VIF"))   # > 10 means collinear features
"""
    return make_snippet("OLS regression diagnostics", body, **load)


def glm_snippet(target: str, features: Sequence[str], family: str = "poisson", **load: Any) -> str:
    fam = {
        "gaussian": "Gaussian()",
        "poisson": "Poisson()",
        "gamma": "Gamma(link=sm.families.links.Log())",
        "tweedie": "Tweedie(var_power=1.5, link=sm.families.links.Log())",
        "binomial": "Binomial()",
    }[family]
    y_expr = (
        "(data[target] == sorted(data[target].unique(), key=str)[1]).astype(float)"
        if family == "binomial"
        else "pd.to_numeric(data[target], errors='coerce')"
    )
    body = f"""import statsmodels.api as sm

target, features = {_q(target)}, {_list(features)}
data = df[[target, *features]].dropna()
y = {y_expr}
X = pd.get_dummies(data[features], drop_first=True, dtype=float)
X = sm.add_constant(X.loc[:, X.nunique() > 1])
glm = sm.GLM(y, X, family=sm.families.{fam}).fit()
print(glm.summary())

# %%
out = pd.DataFrame({{"coef": glm.params, "ci_low": glm.conf_int()[0], "ci_high": glm.conf_int()[1], "p_value": glm.pvalues}})
{"out['odds_ratio'] = np.exp(out['coef'])   # multiplicative change in the odds per unit" if family == "binomial" else ("out['rate_ratio'] = np.exp(out['coef'])   # multiplicative change in the expected value per unit" if family != "gaussian" else "")}
display(out.round(4))
print("deviance explained:", round(1 - glm.deviance / glm.null_deviance, 4), "| Pearson chi2 / df:", round(glm.pearson_chi2 / glm.df_resid, 3))
"""
    return make_snippet(f"GLM ({family})", body, **load)


def quantile_snippet(
    target: str, features: Sequence[str], quantiles: Sequence[float] = (0.1, 0.5, 0.9), **load: Any
) -> str:
    body = f"""import statsmodels.api as sm

target, features = {_q(target)}, {_list(features)}
data = df[[target, *features]].dropna()
X = pd.get_dummies(data[features], drop_first=True, dtype=float)
X = sm.add_constant(X.loc[:, X.nunique() > 1])
y = pd.to_numeric(data[target], errors="coerce")
fits = {{q: sm.QuantReg(y, X).fit(q=q, max_iter=2000) for q in {list(quantiles)!r}}}
display(pd.DataFrame({{q: f.params for q, f in fits.items()}}).round(4))   # effects can differ across the distribution
lo, hi = fits[{min(quantiles)!r}].predict(X), fits[{max(quantiles)!r}].predict(X)
print("coverage of the", {min(quantiles)!r}, "-", {max(quantiles)!r}, "band:", ((y >= lo) & (y <= hi)).mean())
"""
    return make_snippet("Quantile regression", body, **load)


def regularization_snippet(
    target: str, features: Sequence[str], kind: str = "lasso", **load: Any
) -> str:
    cv_cls = "LassoCV" if kind == "lasso" else "RidgeCV"
    body = f"""from sklearn.linear_model import {cv_cls}
import plotly.express as px

target, features = {_q(target)}, {_list(features)}
data = df[[target, *features]].dropna()
X = pd.get_dummies(data[features], drop_first=True, dtype=float)
X = X.loc[:, X.nunique() > 1]
Xs = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)
y = pd.to_numeric(data[target], errors="coerce").reset_index(drop=True)

# %%
model = {cv_cls}({"cv=5, random_state=42, max_iter=20000" if kind == "lasso" else "alphas=np.logspace(-3, 4, 40), cv=5"}).fit(Xs, y)
print("cross-validated alpha:", model.alpha_)
coefs = pd.Series(model.coef_, index=X.columns).sort_values(key=abs, ascending=False)
display(coefs.round(4).to_frame("standardized coefficient"))
print("kept:", [c for c, v in coefs.items() if abs(v) > 1e-8])
"""
    return make_snippet(
        f"{kind.title()} path",
        body,
        imports=[
            "import pandas as pd",
            "import numpy as np",
            "from sklearn.preprocessing import StandardScaler",
        ],
        **load,
    )


# --------------------------------------------------------------------------- feature selection
def feature_selection_snippet(target: str, features: Sequence[str], task: str, **load: Any) -> str:
    reg = _is_reg(task)
    mi = "mutual_info_regression" if reg else "mutual_info_classif"
    fs = "f_regression" if reg else "f_classif"
    body = (
        prelude(target, features, task, split=False)
        + f"""
from sklearn.feature_selection import RFE, {mi}, {fs}
from sklearn.linear_model import {"Lasso" if reg else "LogisticRegression"}

M = preprocess.fit_transform(X)
names = preprocess.get_feature_names_out()

# %%
# 1. Filter: statistical association with the target
scores = pd.DataFrame({{"feature": names, "mutual_info": {mi}(M, y, random_state=SEED), "anova_F": np.nan_to_num({fs}(M, y)[0])}})
display(scores.sort_values("mutual_info", ascending=False).head(15))

# %%
# 2. Wrapper: recursive feature elimination (rank 1 = last survivor)
est = {"Lasso(alpha=0.05)" if reg else "LogisticRegression(max_iter=2000)"}
rfe = RFE(est, n_features_to_select=1).fit(M, y)
display(pd.DataFrame({{"feature": names, "rfe_rank": rfe.ranking_}}).sort_values("rfe_rank").head(15))

# %%
# 3. Embedded: which coefficients does an L1 penalty keep?
l1 = {"Lasso(alpha=0.05, max_iter=20000)" if reg else "LogisticRegression(l1_ratio=1.0, solver='saga', C=0.2, max_iter=5000)"}.fit(M, y)
coef = np.abs(l1.coef_).max(axis=0) if np.ndim(l1.coef_) > 1 else np.abs(l1.coef_)
print("kept by L1:", list(names[coef > 1e-8]))
"""
    )
    return make_snippet("Feature selection", body, imports=_imports([]), **load)


# --------------------------------------------------------------------------- explanation
def explainer_snippet(
    model: str,
    target: str,
    features: Sequence[str],
    task: str,
    feature: str | None = None,
    seed: int = 42,
    **load: Any,
) -> str:
    imp, expr = _est(model, task, seed)
    scoring = "r2" if _is_reg(task) else "f1_macro"
    feature = feature or features[0]
    body = (
        prelude(target, features, task, seed)
        + f"""
pipe = Pipeline([("prep", preprocess), ("model", {expr})]).fit(X_train, y_train)

# %%
# Permutation importance on held-out rows: how much does the score drop when a feature is shuffled?
from sklearn.inspection import permutation_importance, partial_dependence
imp = permutation_importance(pipe, X_test, y_test, n_repeats=5, random_state=SEED, scoring={_q(scoring)})
display(pd.Series(imp.importances_mean, index=X_test.columns).sort_values(ascending=False).to_frame("importance"))

# %%
# Partial dependence and ICE curves for one feature
feature = {_q(feature)}
sample = X_test.sample(min(len(X_test), 300), random_state=0)
sample = sample.astype({{c: float for c in sample.select_dtypes("integer").columns}})   # PDP rejects integer columns
pdp = partial_dependence(pipe, sample, [feature], kind="both", grid_resolution=20)
import plotly.graph_objects as go
fig = go.Figure()
for curve in np.asarray(pdp["individual"]).reshape(-1, len(pdp["grid_values"][0]))[:30]:
    fig.add_scatter(x=pdp["grid_values"][0], y=curve, mode="lines", line={{"color": "rgba(120,120,120,0.25)"}}, showlegend=False)
fig.add_scatter(x=pdp["grid_values"][0], y=np.asarray(pdp["average"]).ravel()[:len(pdp["grid_values"][0])], mode="lines", line={{"color": "red", "width": 3}}, name="average (PDP)")
fig.update_layout(title=f"Effect of {{feature}} on the prediction", xaxis_title=feature)
fig.show()

# %%
# Local explanation of one prediction with SHAP (pip install shap)
import shap
row = X_test.iloc[[0]]
predict = (lambda d: pipe.predict_proba(pd.DataFrame(d, columns=X_test.columns).astype(X_test.dtypes.to_dict()))[:, -1]) if hasattr(pipe, "predict_proba") else (lambda d: pipe.predict(pd.DataFrame(d, columns=X_test.columns).astype(X_test.dtypes.to_dict())))
explainer = shap.KernelExplainer(predict, X_train.sample(min(len(X_train), 30), random_state=0).to_numpy(dtype=object))
values = explainer.shap_values(row.to_numpy(dtype=object), nsamples=100)
display(pd.Series(np.ravel(values), index=X_test.columns).sort_values(key=abs, ascending=False).to_frame("shap value"))
"""
    )
    return make_snippet(f"Explain {model}", body, imports=_imports(imp), **load)


def fairness_snippet(
    group_col: str, target: str, features: Sequence[str], model: str, seed: int = 42, **load: Any
) -> str:
    imp, expr = _est(model, "binary_classification", seed)
    body = (
        prelude(
            target,
            [f for f in features if f != group_col] or list(features),
            "binary_classification",
            seed,
        )
        + f"""
pipe = Pipeline([("prep", preprocess), ("model", {expr})]).fit(X_train, y_train)
pred = pipe.predict(X_test)
group = df.loc[X_test.index, {_q(group_col)}].astype(str)
positive = sorted(y.unique(), key=str)[-1]

# %%
rows = []
for g, idx in group.groupby(group).groups.items():
    yt, yp = (y_test.loc[idx] == positive).to_numpy(), (pd.Series(pred, index=X_test.index).loc[idx] == positive).to_numpy()
    tp, fp, fn, tn = (yp & yt).sum(), (yp & ~yt).sum(), (~yp & yt).sum(), (~yp & ~yt).sum()
    rows.append({{"group": g, "n": len(idx), "selection_rate": yp.mean(), "tpr": tp / max(tp + fn, 1), "fpr": fp / max(fp + tn, 1), "accuracy": (yp == yt).mean()}})
report = pd.DataFrame(rows)
display(report)
print("Disparate impact ratio (min/max selection rate):", round(report["selection_rate"].min() / report["selection_rate"].max(), 3), "(< 0.8 is a common warning sign)")
print("Equal-opportunity gap (max-min TPR):", round(report["tpr"].max() - report["tpr"].min(), 3))
"""
    )
    return make_snippet("Fairness by group", body, imports=_imports(imp), **load)


def drift_snippet(features: Sequence[str], **load: Any) -> str:
    body = f"""from scipy import stats

train_df = df                                   # the data the model was trained on
new_df = pd.read_csv("new_data.csv")            # the new file you want to score
features = {_list(features)}

def psi(expected, actual, bins=10):
    edges = np.unique(np.quantile(expected.dropna(), np.linspace(0, 1, bins + 1))); edges[0], edges[-1] = -np.inf, np.inf
    pe = np.clip(np.histogram(expected.dropna(), edges)[0] / expected.notna().sum(), 1e-4, None)
    pa = np.clip(np.histogram(pd.to_numeric(actual, errors="coerce").dropna(), edges)[0] / actual.notna().sum(), 1e-4, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))

rows = []
for col in features:
    if pd.api.types.is_numeric_dtype(train_df[col]):
        rows.append({{"feature": col, "psi": psi(train_df[col], new_df[col]), "ks_p": stats.ks_2samp(train_df[col].dropna(), pd.to_numeric(new_df[col], errors="coerce").dropna()).pvalue}})
    else:
        cats = sorted(set(train_df[col].dropna().astype(str)) | set(new_df[col].dropna().astype(str)))
        pe = train_df[col].astype(str).value_counts(normalize=True).reindex(cats).fillna(0).clip(lower=1e-4)
        pa = new_df[col].astype(str).value_counts(normalize=True).reindex(cats).fillna(0).clip(lower=1e-4)
        rows.append({{"feature": col, "psi": float(((pa - pe) * np.log(pa / pe)).sum()), "ks_p": np.nan}})
drift = pd.DataFrame(rows).sort_values("psi", ascending=False)
drift["status"] = pd.cut(drift["psi"], [-1, 0.1, 0.25, np.inf], labels=["stable", "moderate shift", "major shift"])
drift   # PSI < 0.1 stable, 0.1-0.25 moderate, > 0.25 major
"""
    return make_snippet("Data drift (PSI)", body, include_load=True, **load)


# --------------------------------------------------------------------------- time series ML, causal, survival, text
def tsml_snippet(
    date_col: str,
    value_col: str,
    model: str,
    horizon: int,
    strategy: str = "direct",
    freq: str = "D",
    **load: Any,
) -> str:
    imp, expr = _est(model, "regression")
    body = f"""date_col, value_col, horizon, freq = {_q(date_col)}, {_q(value_col)}, {horizon}, {_q(freq)}
y = (df.assign(**{{date_col: pd.to_datetime(df[date_col], errors="coerce")}}).dropna(subset=[date_col])
       .set_index(date_col)[value_col].sort_index().resample(freq).mean().interpolate(limit_direction="both"))
lags, rolling = (0, 1, 2, 3, 7), (7,)

def features_at(series, step_ahead):
    f = pd.DataFrame({{f"lag_{{k}}": series.shift(k) for k in lags}}, index=series.index)
    for w in rolling:
        f[f"roll_mean_{{w}}"] = series.rolling(w).mean(); f[f"roll_std_{{w}}"] = series.rolling(w).std()
    target_dates = series.index + step_ahead * series.index.freq if series.index.freq else series.index + pd.Timedelta(days=step_ahead)
    f["dow"], f["month"] = target_dates.dayofweek, target_dates.month       # calendar of the day being predicted
    return f

def make_xy(series, step_ahead):
    f = features_at(series, step_ahead)
    d = pd.concat([f, series.shift(-step_ahead).rename("_y")], axis=1).dropna()
    return d.drop(columns="_y"), d["_y"]

def new_model():
    return Pipeline([("scale", StandardScaler()), ("m", {expr})])

# %%
# {strategy} strategy
train = y.iloc[:-horizon]
if {strategy!r} == "direct":
    preds = []
    for h in range(1, horizon + 1):
        X_h, y_h = make_xy(train, h)
        preds.append(new_model().fit(X_h, y_h).predict(features_at(train, h).iloc[[-1]])[0])
else:
    X_1, y_1 = make_xy(train, 1)
    model = new_model().fit(X_1, y_1)
    history, preds = train.copy(), []
    for step in range(horizon):
        nxt = history.index[-1] + (history.index.freq or pd.Timedelta(days=1))
        history.loc[nxt] = np.nan
        value = model.predict(features_at(history.iloc[:-1], 1).iloc[[-1]])[0]
        history.iloc[-1] = value
        preds.append(value)
actual = y.iloc[-horizon:].to_numpy()
naive = np.repeat(train.iloc[-1], horizon)
print("ML MAE:", np.abs(actual - np.array(preds)).mean(), "| naive MAE:", np.abs(actual - naive).mean())
"""
    return make_snippet(f"ML forecast ({model}, {strategy})", body, imports=_imports(imp), **load)


def did_snippet(
    outcome: str, group_col: str, time_col: str, start: Any, treated_value: Any, **load: Any
) -> str:
    body = f"""import statsmodels.formula.api as smf

outcome, group_col, time_col = {_q(outcome)}, {_q(group_col)}, {_q(time_col)}
d = df[[outcome, group_col, time_col]].dropna().copy()
d["treated"] = (d[group_col] == {treated_value!r}).astype(int)
d["post"] = (pd.to_numeric(d[time_col], errors="coerce") >= {start!r}).astype(int) if pd.api.types.is_numeric_dtype(d[time_col]) else (pd.to_datetime(d[time_col]) >= pd.Timestamp({start!r})).astype(int)
fit = smf.ols(f"{{outcome}} ~ treated + post + treated:post", data=d).fit(cov_type="HC1")
print(fit.summary().tables[1])
print("DiD effect:", fit.params["treated:post"], " 95% CI:", fit.conf_int().loc["treated:post"].tolist())

# %%
means = d.groupby(["treated", "post"])[outcome].mean().unstack()
print(means.rename(index={{0: "control", 1: "treated"}}, columns={{0: "before", 1: "after"}}))
print("manual DiD:", (means.loc[1, 1] - means.loc[1, 0]) - (means.loc[0, 1] - means.loc[0, 0]))
"""
    return make_snippet("Difference-in-differences", body, **load)


def matching_snippet(
    treatment: str,
    outcome: str,
    covariates: Sequence[str],
    treated_value: Any,
    caliper_sd: float = 0.2,
    **load: Any,
) -> str:
    body = f"""from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

treatment, outcome, covariates = {_q(treatment)}, {_q(outcome)}, {_list(covariates)}
d = df[[treatment, outcome, *covariates]].dropna().reset_index(drop=True)
d["t"] = (d[treatment] == {treated_value!r}).astype(int)
X = StandardScaler().fit_transform(pd.get_dummies(d[covariates], drop_first=True, dtype=float))
ps = np.clip(LogisticRegression(max_iter=2000).fit(X, d["t"]).predict_proba(X)[:, 1], 1e-4, 1 - 1e-4)
logit = np.log(ps / (1 - ps))
caliper = {caliper_sd} * logit.std()

# %%
treated, control = d.index[d.t == 1].to_numpy(), d.index[d.t == 0].to_numpy()
nn = NearestNeighbors(n_neighbors=min(len(control), 200)).fit(logit[control].reshape(-1, 1))
dist, nbr = nn.kneighbors(logit[treated].reshape(-1, 1))
used, pairs = set(), []
for i in np.random.default_rng(42).permutation(len(treated)):
    for dd, j in zip(dist[i], nbr[i]):
        if control[j] in used: continue
        if dd <= caliper: used.add(control[j]); pairs.append((treated[i], control[j]))
        break
diff = np.array([d.loc[a, outcome] - d.loc[b, outcome] for a, b in pairs])
print(f"matched {{len(pairs)}} of {{len(treated)}} treated | ATT = {{diff.mean():.4f}} (naive difference: {{d[d.t == 1][outcome].mean() - d[d.t == 0][outcome].mean():.4f}})")
boot = np.random.default_rng(0).choice(diff, (500, len(diff))).mean(axis=1)
print("95% bootstrap interval:", np.percentile(boot, [2.5, 97.5]))
"""
    return make_snippet("Propensity-score matching", body, imports=_imports([]), **load)


def uplift_snippet(
    treatment: str,
    outcome: str,
    features: Sequence[str],
    treated_value: Any,
    model: str = "Random Forest",
    seed: int = 42,
    **load: Any,
) -> str:
    imp, expr = _est(model, "binary_classification", seed)
    body = f"""treatment, outcome, features = {_q(treatment)}, {_q(outcome)}, {_list(features)}
d = df[[treatment, outcome, *features]].dropna()
t = (d[treatment] == {treated_value!r}).astype(int)
y = (d[outcome] == sorted(d[outcome].unique(), key=str)[-1]).astype(int)
num_cols = d[features].select_dtypes("number").columns.tolist(); cat_cols = [c for c in features if c not in num_cols]
preprocess = ColumnTransformer([("num", StandardScaler(), num_cols), ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)])
X_tr, X_te, t_tr, t_te, y_tr, y_te = train_test_split(d[features], t, y, test_size=0.3, random_state={seed}, stratify=t)

# %%
def arm_model(mask):
    return Pipeline([("prep", preprocess), ("m", {expr})]).fit(X_tr[mask], y_tr[mask])
m_t, m_c = arm_model(t_tr == 1), arm_model(t_tr == 0)
uplift = m_t.predict_proba(X_te)[:, 1] - m_c.predict_proba(X_te)[:, 1]     # T-learner

# %%
order = np.argsort(-uplift)
yy, tt = y_te.to_numpy()[order], t_te.to_numpy()[order]
n_t, n_c = np.cumsum(tt), np.cumsum(1 - tt)
qini = np.cumsum(yy * tt) - np.where(n_c > 0, np.cumsum(yy * (1 - tt)) * n_t / np.maximum(n_c, 1), 0)
import plotly.express as px
px.line(pd.DataFrame({{"targeted": np.arange(1, len(qini) + 1), "qini": qini, "random": qini[-1] * np.arange(1, len(qini) + 1) / len(qini)}}), x="targeted", y=["qini", "random"], title="Qini curve").show()
top = np.argsort(-uplift)[: len(uplift) // 3]
print("observed uplift in the top third:", y_te.to_numpy()[top][t_te.to_numpy()[top] == 1].mean() - y_te.to_numpy()[top][t_te.to_numpy()[top] == 0].mean())
"""
    return make_snippet("Uplift model (T-learner)", body, imports=_imports(imp), **load)


def survival_snippet(
    duration: str,
    event: str,
    group: str | None = None,
    features: Sequence[str] | None = None,
    **load: Any,
) -> str:
    cox = ""
    if features:
        cox = f"""
# %%
from statsmodels.duration.hazard_regression import PHReg
X = pd.get_dummies(d[{_list(features)}], drop_first=True, dtype=float)
res = PHReg(d[{_q(duration)}].to_numpy(float), X.to_numpy(float), status=d[{_q(event)}].to_numpy(int), ties="efron").fit(disp=0)
ci = res.conf_int()
display(pd.DataFrame({{"hazard_ratio": np.exp(res.params), "hr_low": np.exp(ci[:, 0]), "hr_high": np.exp(ci[:, 1]), "p": res.pvalues}}, index=X.columns).round(4))
"""
    body = f"""from statsmodels.duration.survfunc import SurvfuncRight, survdiff
import plotly.graph_objects as go

duration, event, group = {_q(duration)}, {_q(event)}, {group!r}
d = df[[c for c in dict.fromkeys([duration, event, group, *{list(features or [])!r}]) if c]].dropna()
d = d[d[duration] > 0]
fig = go.Figure()
for name, part in (d.groupby(group) if group else [("all", d)]):
    sf = SurvfuncRight(part[duration].to_numpy(float), part[event].to_numpy(int))
    fig.add_scatter(x=sf.surv_times, y=sf.surv_prob, mode="lines", line_shape="hv", name=str(name))
    below = np.flatnonzero(sf.surv_prob <= 0.5)
    print(name, "median survival:", sf.surv_times[below[0]] if len(below) else "not reached", "| n =", len(part), "| events =", int(part[event].sum()))
fig.update_layout(title="Kaplan-Meier survival curves", xaxis_title=duration, yaxis_title="probability of survival").show()
if group:
    chi2, p = survdiff(d[duration].to_numpy(float), d[event].to_numpy(int), d[group].astype(str).to_numpy())
    print(f"log-rank test: chi2 = {{chi2:.2f}}, p = {{p:.4g}}")
{cox}"""
    return make_snippet("Survival analysis", body, **load)


def text_snippet(
    text_col: str, target: str, model: str = "Logistic Regression", **load: Any
) -> str:
    clf = {
        "Logistic Regression": "LogisticRegression(max_iter=2000, C=3.0)",
        "Naive Bayes": "MultinomialNB(alpha=0.5)",
        "Linear SVM": "CalibratedClassifierCV(LinearSVC(C=1.0), cv=3)",
        "SGD (linear)": "SGDClassifier(loss='log_loss', max_iter=2000)",
    }[model]
    body = f"""from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC

text_col, target = {_q(text_col)}, {_q(target)}
d = df[[text_col, target]].dropna()
d = d[d[text_col].astype(str).str.strip() != ""]
X, y = d[text_col].astype(str), d[target].astype(str)
pipe = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, stop_words="english", sublinear_tf=True)),
                 ("clf", {clf})])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
res = cross_validate(pipe, X, y, cv=cv, scoring=["accuracy", "f1_macro"])
print("accuracy:", res["test_accuracy"].mean().round(4), "| f1_macro:", res["test_f1_macro"].mean().round(4), "| majority baseline:", y.value_counts(normalize=True).max().round(4))

# %%
pipe.fit(X, y)
terms = np.asarray(pipe.named_steps["tfidf"].get_feature_names_out())
clf_step = pipe.named_steps["clf"]
if hasattr(clf_step, "coef_"):
    coefs = clf_step.coef_ if clf_step.coef_.shape[0] > 1 else np.vstack([-clf_step.coef_[0], clf_step.coef_[0]])
    for k, cls in enumerate(clf_step.classes_):
        print(cls, "->", list(terms[np.argsort(-coefs[k])[:10]]))
"""
    return make_snippet(
        f"Text classification ({model})",
        body,
        imports=[
            "import pandas as pd",
            "import numpy as np",
            "from sklearn.pipeline import Pipeline",
            "from sklearn.model_selection import StratifiedKFold, cross_validate",
        ],
        **load,
    )
