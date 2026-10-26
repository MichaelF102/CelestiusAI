"""Regression workbench: diagnostics, GLMs, quantile models, regularization, transforms, errors."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import BayesianRidge, LassoCV, RidgeCV, lasso_path
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    PolynomialFeatures,
    PowerTransformer,
    SplineTransformer,
    StandardScaler,
)
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson

from src.datascience import catalog
from src.datascience.modeling import ModelingError, _prepare, build_preprocessor


def design_matrix(
    df: pd.DataFrame, target: str, features: list[str]
) -> tuple[pd.DataFrame, pd.Series]:
    """Numeric design matrix (dummies, no missing rows) and the matching target."""
    if target in features or not features:
        raise ModelingError("Pick a target and at least one different feature.")
    data = df[[target, *features]].copy()
    data[target] = pd.to_numeric(data[target], errors="coerce")
    data = data.dropna()
    if len(data) < 20:
        raise ModelingError(f"Need at least 20 complete rows (found {len(data)}).")
    X = pd.get_dummies(data[features], drop_first=True, dtype=float)
    X = X.loc[:, X.nunique() > 1]
    if X.empty:
        raise ModelingError("No usable (non-constant) features.")
    return X, data[target]


# --------------------------------------------------------------------------- OLS diagnostics
@dataclass
class OLSDiagnostics:
    coefficients: pd.DataFrame
    r2: float
    adj_r2: float
    f_pvalue: float
    n: int
    fitted: pd.Series
    residuals: pd.Series
    standardized: pd.Series
    leverage: pd.Series
    cooks: pd.Series
    qq: pd.DataFrame
    scale_location: pd.DataFrame
    vif: pd.DataFrame
    tests: dict[str, float]
    influential: pd.DataFrame
    notes: list[str]


def _white_p(resid: pd.Series, exog: pd.DataFrame) -> float:
    """White's test also catches variance that depends on |x| or x^2 (Breusch-Pagan only sees linear)."""
    try:
        return float(het_white(resid, exog)[1])
    except Exception:
        return 1.0


def ols_diagnostics(
    df: pd.DataFrame, target: str, features: list[str], top_n: int = 10
) -> OLSDiagnostics:
    X, y = design_matrix(df, target, features)
    model = sm.OLS(y, sm.add_constant(X)).fit()
    infl = model.get_influence()
    std_resid = pd.Series(infl.resid_studentized_internal, index=y.index)
    leverage = pd.Series(infl.hat_matrix_diag, index=y.index)
    cooks = pd.Series(infl.cooks_distance[0], index=y.index)
    ci = model.conf_int()
    coefs = pd.DataFrame(
        {
            "coef": model.params,
            "std_err": model.bse,
            "t": model.tvalues,
            "p_value": model.pvalues,
            "ci_low": ci[0],
            "ci_high": ci[1],
        }
    )
    n = int(model.nobs)
    order = np.argsort(std_resid.to_numpy())
    qq = pd.DataFrame(
        {
            "theoretical": stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n),
            "sample": std_resid.to_numpy()[order],
        }
    )
    scale = pd.DataFrame(
        {"fitted": model.fittedvalues, "sqrt_abs_std_resid": np.sqrt(np.abs(std_resid))}
    )
    exog = sm.add_constant(X)
    vif = pd.DataFrame(
        {
            "feature": X.columns,
            "vif": (
                [variance_inflation_factor(exog.values, i + 1) for i in range(X.shape[1])]
                if X.shape[1] <= 80
                else [np.nan] * X.shape[1]
            ),
        }
    )
    sample = model.resid.sample(min(n, 5000), random_state=0)
    tests = {
        "breusch_pagan_p": float(het_breuschpagan(model.resid, exog)[1]),
        "white_p": _white_p(model.resid, exog),
        "shapiro_p": float(stats.shapiro(sample)[1]),
        "durbin_watson": float(durbin_watson(model.resid)),
    }
    influential = (
        pd.DataFrame(
            {
                "row": y.index,
                "cooks_d": cooks.to_numpy(),
                "leverage": leverage.to_numpy(),
                "std_residual": std_resid.to_numpy(),
                "actual": y.to_numpy(),
                "fitted": model.fittedvalues.to_numpy(),
            }
        )
        .sort_values("cooks_d", ascending=False)
        .head(top_n)
    )
    notes: list[str] = []
    if min(tests["breusch_pagan_p"], tests["white_p"]) < 0.05:
        notes.append(
            "Residual spread changes with the fitted value (heteroscedasticity): use robust standard errors or transform the target."
        )
    if tests["shapiro_p"] < 0.05:
        notes.append(
            "Residuals depart from a normal shape: p-values and intervals are approximate."
        )
    if not 1.5 < tests["durbin_watson"] < 2.5:
        notes.append(
            "Residuals look autocorrelated (Durbin-Watson outside 1.5 to 2.5): rows may be time-ordered."
        )
    if (vif["vif"] > 10).any():
        notes.append("Some features are strongly collinear (VIF > 10): coefficients are unstable.")
    if (cooks > 4 / n).sum() > 0.10 * n:
        notes.append("Many rows exert unusual influence on the fit.")
    return OLSDiagnostics(
        coefs,
        float(model.rsquared),
        float(model.rsquared_adj),
        float(model.f_pvalue),
        n,
        model.fittedvalues,
        model.resid,
        std_resid,
        leverage,
        cooks,
        qq,
        scale,
        vif,
        tests,
        influential,
        notes,
    )


# --------------------------------------------------------------------------- GLMs
FAMILIES = {
    "gaussian": ("Normal (ordinary regression)", sm.families.Gaussian),
    "poisson": ("Poisson (counts)", sm.families.Poisson),
    # log links, so exp(coefficient) is a multiplicative effect (rate ratio)
    "gamma": (
        "Gamma (positive, skewed amounts)",
        lambda: sm.families.Gamma(link=sm.families.links.Log()),
    ),
    "tweedie": (
        "Tweedie (zeros plus positive amounts)",
        lambda: sm.families.Tweedie(var_power=1.5, link=sm.families.links.Log()),
    ),
    "binomial": ("Binomial (yes/no, odds ratios)", sm.families.Binomial),
}


@dataclass
class GLMResult:
    family: str
    coefficients: pd.DataFrame
    deviance: float
    null_deviance: float
    deviance_explained: float
    aic: float
    dispersion: float
    n: int
    notes: list[str]


def glm_fit(
    df: pd.DataFrame, target: str, features: list[str], family: str = "poisson"
) -> GLMResult:
    if family not in FAMILIES:
        raise ModelingError(f"Unknown family '{family}'.")
    X, y = design_matrix(df, target, features) if family != "binomial" else (None, None)
    if family == "binomial":
        data = df[[target, *features]].dropna()
        levels = sorted(data[target].unique(), key=str)
        if len(levels) != 2:
            raise ModelingError("Binomial needs a target with exactly two values.")
        y = (data[target] == levels[1]).astype(float)
        X = pd.get_dummies(data[features], drop_first=True, dtype=float)
        X = X.loc[:, X.nunique() > 1]
    notes: list[str] = []
    if (
        family == "poisson"
        and (y < 0).any()
        or family in ("poisson", "gamma", "tweedie")
        and (y < 0).any()
    ):
        raise ModelingError(f"{family.title()} needs a non-negative target.")
    if family == "gamma" and (y <= 0).any():
        raise ModelingError("Gamma needs a strictly positive target.")
    fam = FAMILIES[family][1]()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sm.GLM(y, sm.add_constant(X), family=fam).fit()
    ci = model.conf_int()
    table = pd.DataFrame(
        {
            "coef": model.params,
            "std_err": model.bse,
            "z": model.tvalues,
            "p_value": model.pvalues,
            "ci_low": ci[0],
            "ci_high": ci[1],
        }
    )
    if family in ("poisson", "gamma", "tweedie", "binomial"):
        label = "odds_ratio" if family == "binomial" else "rate_ratio"
        table[label] = np.exp(table["coef"])
        table[f"{label}_low"], table[f"{label}_high"] = (
            np.exp(table["ci_low"]),
            np.exp(table["ci_high"]),
        )
    pearson = float(model.pearson_chi2 / model.df_resid) if model.df_resid else float("nan")
    if family == "poisson" and pearson > 1.5:
        notes.append(
            f"Overdispersed counts (Pearson chi2 / df = {pearson:.2f}): consider a negative binomial or Tweedie model."
        )
    explained = 1 - model.deviance / model.null_deviance if model.null_deviance else float("nan")
    return GLMResult(
        family,
        table,
        float(model.deviance),
        float(model.null_deviance),
        float(explained),
        float(model.aic),
        pearson,
        int(model.nobs),
        notes,
    )


def logistic_odds(df: pd.DataFrame, target: str, features: list[str]) -> GLMResult:
    return glm_fit(df, target, features, "binomial")


# --------------------------------------------------------------------------- quantile models
def quantile_regression(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
) -> dict[str, Any]:
    """Linear quantile regression: how effects differ across the distribution of the target."""
    X, y = design_matrix(df, target, features)
    exog = sm.add_constant(X)
    coefs, preds = {}, {}
    for q in quantiles:
        res = sm.QuantReg(y, exog).fit(q=q, max_iter=2000)
        coefs[q] = res.params
        preds[q] = res.predict(exog)
    pred_df = pd.DataFrame(preds)
    lo, hi = pred_df[min(quantiles)], pred_df[max(quantiles)]
    coverage = float(((y >= lo) & (y <= hi)).mean())
    return {
        "coefficients": pd.DataFrame(coefs),
        "predictions": pred_df,
        "coverage": coverage,
        "nominal": max(quantiles) - min(quantiles),
        "actual": y,
    }


def gb_quantile_bands(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    lower: float = 0.1,
    upper: float = 0.9,
    seed: int = 42,
) -> dict[str, Any]:
    """Non-linear prediction bands: boosted quantile models + conformal calibration.

    Raw quantile models usually under-cover on small data. Conformalized quantile regression
    widens the band by the amount needed on a calibration split so the held-out coverage lands
    near the nominal level.
    """
    X, y = _prepare(df, target, features, 20000, seed)
    X_rest, X_te, y_rest, y_te = train_test_split(X, y, test_size=0.25, random_state=seed)
    X_tr, X_cal, y_tr, y_cal = train_test_split(X_rest, y_rest, test_size=0.3, random_state=seed)
    models = {}
    for name, alpha in (("lower", lower), ("median", 0.5), ("upper", upper)):
        pipe = Pipeline(
            [
                ("prep", build_preprocessor(X_tr)),
                (
                    "m",
                    GradientBoostingRegressor(
                        loss="quantile",
                        alpha=alpha,
                        n_estimators=150,
                        max_depth=3,
                        learning_rate=0.05,
                        min_samples_leaf=10,
                        random_state=seed,
                    ),
                ),
            ]
        )
        models[name] = pipe.fit(X_tr, y_tr)
    cal_lo, cal_hi = models["lower"].predict(X_cal), models["upper"].predict(X_cal)
    scores = np.maximum(cal_lo - y_cal.to_numpy(), y_cal.to_numpy() - cal_hi)
    n_cal = len(scores)
    level = min(1.0, (upper - lower) * (1 + 1 / n_cal))
    adjust = float(np.quantile(scores, level))
    out = {name: m.predict(X_te) for name, m in models.items()}
    raw = pd.DataFrame(out, index=y_te.index)
    band = raw.copy()
    band["lower"] = raw["lower"] - adjust
    band["upper"] = raw["upper"] + adjust
    band["actual"] = y_te
    raw_cov = float(((y_te >= raw["lower"]) & (y_te <= raw["upper"])).mean())
    coverage = float(((band["actual"] >= band["lower"]) & (band["actual"] <= band["upper"])).mean())
    return {
        "band": band,
        "coverage": coverage,
        "raw_coverage": raw_cov,
        "adjustment": adjust,
        "nominal": upper - lower,
        "mean_width": float((band["upper"] - band["lower"]).mean()),
        "models": models,
    }


# --------------------------------------------------------------------------- regularization
def regularization_path(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    kind: str = "lasso",
    n_alphas: int = 40,
    seed: int = 42,
) -> dict[str, Any]:
    """Coefficient shrinkage as the penalty grows, plus the cross-validated best alpha."""
    X, y = design_matrix(df, target, features)
    Xs = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns, index=X.index)
    cv = KFold(5, shuffle=True, random_state=seed)
    if kind == "lasso":
        centered = y - y.mean()
        alpha_max = float(np.max(np.abs(Xs.to_numpy().T @ centered.to_numpy())) / len(centered))
        grid = np.logspace(np.log10(alpha_max), np.log10(alpha_max * 1e-3), n_alphas)
        alphas, coefs, _ = lasso_path(Xs, centered, alphas=grid)
        best = LassoCV(cv=cv, random_state=seed, max_iter=20000).fit(Xs, y)
        best_alpha, final = float(best.alpha_), pd.Series(best.coef_, index=X.columns)
    elif kind == "ridge":
        alphas = np.logspace(-3, 4, n_alphas)[::-1]
        from sklearn.linear_model import Ridge

        coefs = np.column_stack([Ridge(alpha=a).fit(Xs, y).coef_ for a in alphas])
        best = RidgeCV(alphas=alphas, cv=cv).fit(Xs, y)
        best_alpha, final = float(best.alpha_), pd.Series(best.coef_, index=X.columns)
    else:
        raise ModelingError("kind must be 'lasso' or 'ridge'.")
    path = pd.DataFrame(coefs.T, columns=X.columns)
    path.insert(0, "alpha", alphas)
    return {
        "path": path,
        "best_alpha": best_alpha,
        "final_coefficients": final,
        "kept": [c for c in X.columns if abs(final[c]) > 1e-8],
        "kind": kind,
    }


# --------------------------------------------------------------------------- polynomial / spline
def curvature_study(
    df: pd.DataFrame,
    target: str,
    feature: str,
    degrees: tuple[int, ...] = (1, 2, 3, 4, 5),
    cv: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Does a single feature need curvature? CV R2 for polynomial degrees and a spline."""
    X, y = design_matrix(df, target, [feature])
    X = X[[c for c in X.columns if c == feature]] if feature in X.columns else X
    splitter = KFold(cv, shuffle=True, random_state=seed)
    from sklearn.linear_model import Ridge

    rows = []
    for d in degrees:
        pipe = Pipeline(
            [
                ("poly", PolynomialFeatures(d, include_bias=False)),
                ("scale", StandardScaler()),
                ("ridge", Ridge(alpha=1e-3)),
            ]
        )
        s = cross_val_score(pipe, X, y, cv=splitter, scoring="r2")
        rows.append(
            {
                "model": f"polynomial degree {d}",
                "cv_r2": float(s.mean()),
                "cv_std": float(s.std(ddof=1)),
            }
        )
    spline = Pipeline(
        [("spl", SplineTransformer(n_knots=6, degree=3)), ("ridge", Ridge(alpha=1e-2))]
    )
    s = cross_val_score(spline, X, y, cv=splitter, scoring="r2")
    rows.append(
        {
            "model": "cubic spline (6 knots)",
            "cv_r2": float(s.mean()),
            "cv_std": float(s.std(ddof=1)),
        }
    )
    out = pd.DataFrame(rows)
    out["best"] = out["cv_r2"] == out["cv_r2"].max()
    return out


# --------------------------------------------------------------------------- target transforms
def transformed_model(model: str, transform: str, seed: int = 42) -> Any:
    est = catalog.build(model, "regression", seed)
    if transform == "none":
        return est
    funcs = {"log1p": (np.log1p, np.expm1), "sqrt": (np.sqrt, np.square)}
    if transform in funcs:
        f, inv = funcs[transform]
        return TransformedTargetRegressor(regressor=est, func=f, inverse_func=inv)
    if transform == "yeo-johnson":
        return TransformedTargetRegressor(
            regressor=est, transformer=PowerTransformer(method="yeo-johnson")
        )
    raise ModelingError(f"Unknown transform '{transform}'.")


def compare_target_transforms(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    model: str = "Ridge / L2 Logistic",
    transforms: tuple[str, ...] = ("none", "log1p", "sqrt", "yeo-johnson"),
    cv: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Scores are computed on the ORIGINAL scale after back-transforming predictions."""
    X, y = _prepare(df, target, features, 20000, seed)
    splitter = KFold(cv, shuffle=True, random_state=seed)
    rows = []
    for tr in transforms:
        if tr in ("log1p", "sqrt") and (y < 0).any():
            rows.append({"transform": tr, "note": "needs a non-negative target"})
            continue
        r2s, maes, stable = [], [], True
        for a, b in splitter.split(X, y):
            pipe = Pipeline(
                [("prep", build_preprocessor(X)), ("m", transformed_model(model, tr, seed))]
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pipe.fit(X.iloc[a], y.iloc[a])
            pred = pipe.predict(X.iloc[b])
            if not np.isfinite(pred).all():
                stable = False
                break
            r2s.append(r2_score(y.iloc[b], pred))
            maes.append(mean_absolute_error(y.iloc[b], pred))
        if not stable:
            rows.append({"transform": tr, "note": "the inverse transform produced invalid values"})
            continue
        rows.append(
            {
                "transform": tr,
                "cv_r2": float(np.mean(r2s)),
                "cv_mae": float(np.mean(maes)),
                "target_skew_after": float(pd.Series(_apply(tr, y)).skew()),
                "note": "",
            }
        )
    return pd.DataFrame(rows)


def _apply(name: str, y: pd.Series) -> np.ndarray:
    if name == "log1p":
        return np.log1p(y.clip(lower=0))
    if name == "sqrt":
        return np.sqrt(y.clip(lower=0))
    if name == "yeo-johnson":
        return PowerTransformer(method="yeo-johnson").fit_transform(y.to_frame()).ravel()
    return y.to_numpy()


# --------------------------------------------------------------------------- error analysis
def error_analysis(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    model: str = "Random Forest",
    bins: int = 5,
    seed: int = 42,
    factor: float = 1.5,
) -> dict[str, Any]:
    """Where is the model wrong? Error by feature range and category, with weak segments flagged."""
    X, y = _prepare(df, target, features, 20000, seed)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=seed)
    pipe = Pipeline(
        [("prep", build_preprocessor(X_tr)), ("m", catalog.build(model, "regression", seed))]
    ).fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    res = pd.DataFrame({"actual": y_te, "pred": pred})
    res["error"] = res["pred"] - res["actual"]
    res["abs_error"] = res["error"].abs()
    overall = float(res["abs_error"].mean())
    rows = []
    for col in X_te.columns:
        s = X_te[col]
        if pd.api.types.is_numeric_dtype(s) and s.nunique() > bins:
            groups = pd.qcut(s, q=bins, duplicates="drop").astype(str)
        else:
            groups = s.astype(str)
        for seg, part in res.groupby(groups.reindex(res.index)):
            if len(part) < 5:
                continue
            rows.append(
                {
                    "feature": col,
                    "segment": str(seg),
                    "rows": len(part),
                    "mae": float(part["abs_error"].mean()),
                    "bias": float(part["error"].mean()),
                    "mae_vs_overall": float(part["abs_error"].mean() / overall)
                    if overall
                    else np.nan,
                }
            )
    segments = pd.DataFrame(rows)
    weak = (
        segments[segments["mae_vs_overall"] > factor].sort_values("mae_vs_overall", ascending=False)
        if len(segments)
        else segments
    )
    return {
        "overall_mae": overall,
        "overall_bias": float(res["error"].mean()),
        "r2": float(r2_score(res["actual"], res["pred"])),
        "segments": segments,
        "weak_segments": weak,
        "predictions": res,
    }


# --------------------------------------------------------------------------- advanced statistics
def bayesian_regression(df: pd.DataFrame, target: str, features: list[str]) -> dict[str, Any]:
    """Bayesian ridge: coefficient posteriors (mean, sd, 95% credible interval) and predictive sd."""
    X, y = design_matrix(df, target, features)
    Xs = StandardScaler().fit(X)
    model = BayesianRidge().fit(Xs.transform(X), y)
    sd = np.sqrt(np.diag(model.sigma_))
    table = pd.DataFrame(
        {
            "mean": model.coef_,
            "sd": sd,
            "ci_low": model.coef_ - 1.96 * sd,
            "ci_high": model.coef_ + 1.96 * sd,
        },
        index=X.columns,
    )
    table["probability_nonzero"] = 2 * stats.norm.cdf(np.abs(table["mean"]) / table["sd"]) - 1
    _, pred_sd = model.predict(Xs.transform(X), return_std=True)
    return {
        "coefficients": table,
        "noise_sd": float(np.sqrt(1 / model.alpha_)),
        "mean_predictive_sd": float(pred_sd.mean()),
        "note": "Coefficients are per standard deviation of each feature.",
    }


def bootstrap_coefficients(
    df: pd.DataFrame, target: str, features: list[str], n_boot: int = 300, seed: int = 42
) -> pd.DataFrame:
    """Percentile bootstrap intervals for OLS coefficients (no normality assumption)."""
    X, y = design_matrix(df, target, features)
    exog = sm.add_constant(X).to_numpy()
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        beta, *_ = np.linalg.lstsq(exog[idx], y.to_numpy()[idx], rcond=None)
        draws.append(beta)
    arr = np.asarray(draws)
    names = ["const", *X.columns]
    return pd.DataFrame(
        {
            "coef": np.linalg.lstsq(exog, y.to_numpy(), rcond=None)[0],
            "boot_low": np.percentile(arr, 2.5, axis=0),
            "boot_high": np.percentile(arr, 97.5, axis=0),
            "boot_sd": arr.std(axis=0, ddof=1),
        },
        index=names,
    )


def regression_power(f2: float, predictors: int, n: int, alpha: float = 0.05) -> float:
    """Power of the overall F test for effect size f2 = R2 / (1 - R2)."""
    dfd = n - predictors - 1
    if dfd < 1:
        return 0.0
    crit = stats.f.ppf(1 - alpha, predictors, dfd)
    return float(1 - stats.ncf.cdf(crit, predictors, dfd, f2 * n))


def regression_sample_size(
    f2: float, predictors: int, power: float = 0.8, alpha: float = 0.05, max_n: int = 100000
) -> int:
    if f2 <= 0:
        raise ModelingError("Effect size f2 must be positive.")
    lo, hi = predictors + 3, max_n
    if regression_power(f2, predictors, hi, alpha) < power:
        raise ModelingError("Even the maximum sample size cannot reach that power.")
    while lo < hi:
        mid = (lo + hi) // 2
        if regression_power(f2, predictors, mid, alpha) >= power:
            hi = mid
        else:
            lo = mid + 1
    return lo


def mixed_effects(df: pd.DataFrame, target: str, features: list[str], group: str) -> dict[str, Any]:
    """Random-intercept model for grouped data (e.g. customers within stores)."""
    import statsmodels.formula.api as smf

    data = df[[target, *features, group]].dropna()
    if data[group].nunique() < 3:
        raise ModelingError("Need at least 3 groups for a mixed model.")
    data = data.rename(columns={target: "_y", group: "_g"})
    renames = {c: f"f{i}" for i, c in enumerate(features)}
    data = data.rename(columns=renames)
    cat = [renames[c] for c in features if not pd.api.types.is_numeric_dtype(df[c])]
    rhs = " + ".join(f"C({renames[c]})" if renames[c] in cat else renames[c] for c in features)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = smf.mixedlm(f"_y ~ {rhs}", data, groups=data["_g"]).fit()
    fixed = pd.DataFrame(
        {"coef": fit.fe_params, "std_err": fit.bse_fe, "p_value": fit.pvalues[fit.fe_params.index]}
    )
    var_g, var_e = float(fit.cov_re.iloc[0, 0]), float(fit.scale)
    return {
        "fixed_effects": fixed,
        "group_variance": var_g,
        "residual_variance": var_e,
        "icc": var_g / (var_g + var_e),
        "n_groups": int(data["_g"].nunique()),
    }
