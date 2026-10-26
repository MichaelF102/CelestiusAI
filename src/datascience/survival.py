"""Survival analysis: Kaplan-Meier curves, log-rank test, and Cox proportional hazards."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.duration.survfunc import SurvfuncRight, survdiff


class SurvivalError(ValueError):
    """Invalid survival input (message is user-facing)."""


def _prep(df: pd.DataFrame, duration: str, event: str) -> pd.DataFrame:
    data = df[[duration, event]].copy()
    data[duration] = pd.to_numeric(data[duration], errors="coerce")
    levels = data[event].dropna().unique()
    if set(levels) - {0, 1}:
        raise SurvivalError(
            "The event column must be 0/1 (1 = the event happened, 0 = still observed/censored)."
        )
    data[event] = data[event].astype(int)
    data = data.dropna()
    if (data[duration] <= 0).any():
        data = data[data[duration] > 0]
    if len(data) < 10 or data[event].sum() < 3:
        raise SurvivalError("Need at least 10 rows and 3 events.")
    return data


def _km(time: np.ndarray, status: np.ndarray) -> dict[str, Any]:
    sf = SurvfuncRight(time, status)
    t, s, se = np.asarray(sf.surv_times), np.asarray(sf.surv_prob), np.asarray(sf.surv_prob_se)
    # log(-log) transform keeps the interval inside [0, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        se_ll = np.where((s > 0) & (s < 1), se / (s * np.abs(np.log(s))), 0.0)
        theta = np.where((s > 0) & (s < 1), np.log(-np.log(s)), 0.0)
        low = np.where((s > 0) & (s < 1), np.exp(-np.exp(theta + 1.96 * se_ll)), s)
        high = np.where((s > 0) & (s < 1), np.exp(-np.exp(theta - 1.96 * se_ll)), s)
    at_risk = np.array([(time >= x).sum() for x in t])
    below = np.flatnonzero(s <= 0.5)
    return {
        "time": t,
        "survival": s,
        "ci_low": low,
        "ci_high": high,
        "at_risk": at_risk,
        "median": float(t[below[0]]) if len(below) else float("nan"),
        "n": len(time),
        "events": int(status.sum()),
    }


def kaplan_meier(
    df: pd.DataFrame, duration: str, event: str, group: str | None = None
) -> dict[str, Any]:
    """Survival curves (overall or per group) with median survival and a log-rank test."""
    cols = [duration, event] + ([group] if group else [])
    base = _prep(df[cols], duration, event)
    if group:
        base[group] = df.loc[base.index, group].astype(str)
    curves, summary = [], []
    groups = [(None, base)] if not group else list(base.groupby(group))
    for name, part in groups:
        if len(part) < 5:
            continue
        res = _km(part[duration].to_numpy(float), part[event].to_numpy(int))
        label = "all" if name is None else str(name)
        curves.append(
            pd.DataFrame(
                {
                    "group": label,
                    "time": res["time"],
                    "survival": res["survival"],
                    "ci_low": res["ci_low"],
                    "ci_high": res["ci_high"],
                    "at_risk": res["at_risk"],
                }
            )
        )
        summary.append(
            {
                "group": label,
                "n": res["n"],
                "events": res["events"],
                "median_survival": res["median"],
                "event_rate": res["events"] / res["n"],
            }
        )
    logrank = None
    if group and len(summary) >= 2:
        keep = base[base[group].isin([s["group"] for s in summary])]
        chi2, p = survdiff(
            keep[duration].to_numpy(float), keep[event].to_numpy(int), keep[group].to_numpy()
        )
        logrank = {"chi2": float(chi2), "p_value": float(p)}
    return {
        "curves": pd.concat(curves, ignore_index=True),
        "summary": pd.DataFrame(summary),
        "logrank": logrank,
    }


def concordance(time: np.ndarray, event: np.ndarray, risk: np.ndarray) -> float:
    """Harrell's C: among comparable pairs, how often does the higher-risk subject fail first?"""
    t_i, t_j = time[:, None], time[None, :]
    comparable = (t_i < t_j) & (event[:, None] == 1)
    conc = (risk[:, None] > risk[None, :]) & comparable
    ties = (risk[:, None] == risk[None, :]) & comparable
    total = comparable.sum()
    return float((conc.sum() + 0.5 * ties.sum()) / total) if total else float("nan")


def cox_ph(df: pd.DataFrame, duration: str, event: str, features: list[str]) -> dict[str, Any]:
    """Cox proportional hazards: hazard ratios, concordance, and a proportional-hazards check."""
    if not features:
        raise SurvivalError("Pick at least one feature.")
    base = _prep(df[[duration, event]], duration, event)
    X = pd.get_dummies(df.loc[base.index, features], drop_first=True, dtype=float)
    X = X.loc[:, X.nunique() > 1].dropna()
    base = base.loc[X.index]
    if X.empty:
        raise SurvivalError("No usable features.")
    scale = X.std(ddof=0).replace(0, 1)
    model = PHReg(
        base[duration].to_numpy(float),
        X.to_numpy(float),
        status=base[event].to_numpy(int),
        ties="efron",
    )
    res = model.fit(disp=0)
    ci = res.conf_int()
    table = pd.DataFrame(
        {
            "coef": res.params,
            "hazard_ratio": np.exp(res.params),
            "hr_low": np.exp(ci[:, 0]),
            "hr_high": np.exp(ci[:, 1]),
            "p_value": res.pvalues,
        },
        index=X.columns,
    )
    table["hr_per_sd"] = np.exp(res.params * scale.to_numpy())
    risk = X.to_numpy(float) @ res.params
    c_index = concordance(base[duration].to_numpy(float), base[event].to_numpy(int), risk)
    # Schoenfeld residual check: correlation with event time (small p = hazard ratio changes over time)
    resid = np.asarray(res.schoenfeld_residuals)
    mask = ~np.isnan(resid).any(axis=1)
    ph_rows = []
    if mask.sum() > 5:
        # Grambsch-Therneau: scale the residuals by the coefficient covariance before testing
        scaled = resid[mask] @ np.asarray(res.cov_params()) * mask.sum() + res.params
        times = base[duration].to_numpy(float)[mask]
        g_time = stats.rankdata(times) / len(times)  # monotone time transform, as in cox.zph
        for k, col in enumerate(X.columns):
            r, p = stats.pearsonr(scaled[:, k], g_time)
            ph_rows.append(
                {
                    "feature": col,
                    "correlation_with_time": float(r),
                    "p_value": float(p),
                    "violated": bool(p < 0.05),
                }
            )
    ph = pd.DataFrame(ph_rows)
    notes = []
    if len(ph) and ph["violated"].any():
        notes.append(
            "Proportional hazards looks violated for: "
            + ", ".join(ph.loc[ph["violated"], "feature"])
            + ". The hazard ratio for these changes over time; consider stratifying or time-varying effects."
        )
    if c_index < 0.6:
        notes.append(
            f"Concordance {c_index:.2f} is close to random (0.5): these features barely rank who fails first."
        )
    return {
        "table": table,
        "concordance": c_index,
        "n": int(len(base)),
        "events": int(base[event].sum()),
        "ph_check": ph,
        "notes": notes,
        "result": res,
        "columns": list(X.columns),
        "durations": base[duration].to_numpy(float),
    }


def predict_survival(
    cox: dict[str, Any], profile: dict[str, float], times: np.ndarray | None = None
) -> pd.DataFrame:
    """Survival curve for one covariate profile (baseline hazard x exp(linear predictor))."""
    res = cox["result"]
    x = np.array([float(profile.get(c, 0.0)) for c in cox["columns"]])
    baseline = res.baseline_cumulative_hazard[0]  # (times, cumulative hazard, survival)
    bt, bch = baseline[0], baseline[1]
    bt, bch = np.asarray(bt), np.asarray(bch)
    grid = np.asarray(times) if times is not None else bt
    idx = np.searchsorted(bt, grid, side="right") - 1
    H0 = np.where(idx >= 0, bch[np.clip(idx, 0, len(bch) - 1)], 0.0)
    return pd.DataFrame({"time": grid, "survival": np.exp(-H0 * np.exp(float(x @ res.params)))})
