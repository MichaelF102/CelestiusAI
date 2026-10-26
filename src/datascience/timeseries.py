"""Time-series preparation, diagnostics, rolling-origin backtesting, and forecasting."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf

from src.core.code_snippets import make_snippet

FREQ_PERIOD = {
    "D": 7,
    "W": 52,
    "M": 12,
    "MS": 12,
    "ME": 12,
    "Q": 4,
    "QS": 4,
    "QE": 4,
    "H": 24,
    "h": 24,
}
FREQ_LABELS = {"D": "Daily", "W": "Weekly", "MS": "Monthly", "QS": "Quarterly", "h": "Hourly"}
MODELS = ("Naive", "Seasonal naive", "Drift", "Holt-Winters (ETS)", "ARIMA(1,1,1)")
MIN_POINTS = 12


class TimeSeriesError(ValueError):
    """Invalid series or settings (message is user-facing)."""


def prepare_series(
    df: pd.DataFrame, date_col: str, value_col: str, freq: str = "D", agg: str = "mean"
) -> pd.Series:
    """Aggregate to a regular frequency; gaps are linearly interpolated."""
    dates = pd.to_datetime(df[date_col], errors="coerce")
    values = pd.to_numeric(df[value_col], errors="coerce")
    frame = pd.DataFrame({"d": dates, "v": values}).dropna()
    if len(frame) < MIN_POINTS:
        raise TimeSeriesError(
            f"Need at least {MIN_POINTS} dated observations (found {len(frame)})."
        )
    series = frame.set_index("d")["v"].sort_index().resample(freq).agg(agg)
    series = series.interpolate(limit_direction="both")
    if len(series) < MIN_POINTS:
        raise TimeSeriesError(
            "After resampling there are too few periods - choose a finer frequency."
        )
    series.name = value_col
    return series


def default_period(freq: str) -> int:
    return FREQ_PERIOD.get(freq, 1)


def decompose(series: pd.Series, period: int, model: str = "additive") -> pd.DataFrame:
    if period < 2 or len(series) < 2 * period:
        raise TimeSeriesError(
            f"Decomposition needs at least 2 full seasons ({2 * period} points) of data."
        )
    if model == "multiplicative" and (series <= 0).any():
        raise TimeSeriesError("Multiplicative decomposition requires strictly positive values.")
    res = seasonal_decompose(series, model=model, period=period)
    return pd.DataFrame(
        {
            "observed": res.observed,
            "trend": res.trend,
            "seasonal": res.seasonal,
            "residual": res.resid,
        }
    )


def stationarity(series: pd.Series) -> dict[str, Any]:
    """ADF (H0: unit root) and KPSS (H0: stationary) with a combined verdict."""
    clean = series.dropna()
    if clean.nunique() < 3:
        raise TimeSeriesError(
            "The series is (almost) constant; stationarity tests are meaningless."
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adf_stat, adf_p, *_ = adfuller(clean, autolag="AIC")
        kpss_stat, kpss_p, *_ = kpss(clean, regression="c", nlags="auto")
    adf_stationary, kpss_stationary = adf_p < 0.05, kpss_p >= 0.05
    if adf_stationary and kpss_stationary:
        verdict = "Stationary: both tests agree."
    elif not adf_stationary and not kpss_stationary:
        verdict = "Non-stationary: both tests agree. Try differencing or detrending."
    elif adf_stationary:
        verdict = "Mixed: trend-stationary or borderline. Check a plot and consider detrending."
    else:
        verdict = "Mixed: possible difference-stationary series. Try differencing."
    return {
        "adf_stat": float(adf_stat),
        "adf_p": float(adf_p),
        "kpss_stat": float(kpss_stat),
        "kpss_p": float(kpss_p),
        "verdict": verdict,
    }


def acf_pacf(series: pd.Series, nlags: int = 30) -> dict[str, Any]:
    clean = series.dropna()
    nlags = max(1, min(nlags, len(clean) // 2 - 1))
    return {
        "acf": acf(clean, nlags=nlags, fft=True),
        "pacf": pacf(clean, nlags=nlags),
        "band": float(1.96 / np.sqrt(len(clean))),
    }


# --------------------------------------------------------------------------- forecasting models
def _fit_forecast(
    name: str, train: pd.Series, horizon: int, period: int
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return (point forecast, residual std or None)."""
    y = train.to_numpy(dtype=float)
    if name == "Naive":
        return np.repeat(y[-1], horizon), float(np.std(np.diff(y))) * np.ones(horizon)
    if name == "Seasonal naive":
        if period < 2 or len(y) < period:
            raise TimeSeriesError("Seasonal naive needs at least one full season.")
        reps = int(np.ceil(horizon / period))
        return np.tile(y[-period:], reps)[:horizon], None
    if name == "Drift":
        slope = (y[-1] - y[0]) / max(len(y) - 1, 1)
        return y[-1] + slope * np.arange(1, horizon + 1), float(np.std(np.diff(y))) * np.ones(
            horizon
        )
    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore", category=(ConvergenceWarning, RuntimeWarning, FutureWarning, UserWarning)
        )
        if name == "Holt-Winters (ETS)":
            seasonal = "add" if period >= 2 and len(y) >= 2 * period else None
            model = ExponentialSmoothing(
                y,
                trend="add",
                damped_trend=True,
                seasonal=seasonal,
                seasonal_periods=period if seasonal else None,
            ).fit()
            return np.asarray(model.forecast(horizon)), float(np.std(model.resid)) * np.ones(
                horizon
            )
        if name == "ARIMA(1,1,1)":
            model = ARIMA(y, order=(1, 1, 1)).fit()
            fc = model.get_forecast(horizon)
            return np.asarray(fc.predicted_mean), np.asarray(fc.se_mean)
    raise TimeSeriesError(f"Unknown model '{name}'.")


def smape(actual: np.ndarray, pred: np.ndarray) -> float:
    denom = (np.abs(actual) + np.abs(pred)) / 2
    return float(
        np.mean(np.where(denom == 0, 0, np.abs(actual - pred) / np.where(denom == 0, 1, denom)))
    )


def backtest(
    series: pd.Series, horizon: int, period: int, models: list[str] | None = None, folds: int = 3
) -> pd.DataFrame:
    """Rolling-origin evaluation: average MAE/RMSE/sMAPE/MASE across `folds` forecast origins."""
    if horizon < 1:
        raise TimeSeriesError("Horizon must be at least 1.")
    n = len(series)
    min_train = max(MIN_POINTS, 2 * period if period > 1 else MIN_POINTS)
    if n < min_train + horizon:
        raise TimeSeriesError(
            f"Series too short: need at least {min_train + horizon} points for a horizon of {horizon}."
        )
    folds = max(1, min(folds, (n - min_train) // horizon))
    origins = [n - horizon * (folds - k) for k in range(folds)]
    rows = []
    for name in models or list(MODELS):
        errs: list[tuple[float, float, float, float]] = []
        failed = None
        for origin in origins:
            train, test = (
                series.iloc[:origin],
                series.iloc[origin : origin + horizon].to_numpy(dtype=float),
            )
            try:
                pred, _ = _fit_forecast(name, train, len(test), period)
            except (TimeSeriesError, ValueError, np.linalg.LinAlgError) as err:
                failed = str(err)
                break
            scale = np.mean(np.abs(np.diff(train.to_numpy(dtype=float)))) or np.nan
            err = test - pred
            errs.append(
                (
                    float(np.mean(np.abs(err))),
                    float(np.sqrt(np.mean(err**2))),
                    smape(test, pred),
                    float(np.mean(np.abs(err)) / scale),
                )
            )
        if failed or not errs:
            rows.append(
                {
                    "model": name,
                    "MAE": np.nan,
                    "RMSE": np.nan,
                    "sMAPE": np.nan,
                    "MASE": np.nan,
                    "note": failed or "failed",
                }
            )
            continue
        mae, rmse, sm, mase = np.mean(errs, axis=0)
        rows.append(
            {"model": name, "MAE": mae, "RMSE": rmse, "sMAPE": sm, "MASE": mase, "note": ""}
        )
    return pd.DataFrame(rows).sort_values("RMSE", na_position="last").reset_index(drop=True)


def forecast(
    series: pd.Series, model: str, horizon: int, period: int, level: float = 0.95
) -> pd.DataFrame:
    pred, spread = _fit_forecast(model, series, horizon, period)
    freq = series.index.freq or pd.infer_freq(series.index)
    future = pd.date_range(series.index[-1], periods=horizon + 1, freq=freq)[1:]
    out = pd.DataFrame({"forecast": pred}, index=future)
    if spread is None:  # seasonal naive: use in-sample seasonal error
        diffs = series.to_numpy(dtype=float)
        spread = np.full(
            horizon, float(np.std(diffs[period:] - diffs[:-period])) if period < len(diffs) else 0.0
        )
    else:
        spread = np.asarray(spread, dtype=float)
        if model in ("Naive", "Drift"):
            spread = spread * np.sqrt(np.arange(1, horizon + 1))
    z = (
        1.96
        if abs(level - 0.95) < 1e-9
        else float(__import__("scipy.stats", fromlist=["norm"]).norm.ppf(0.5 + level / 2))
    )
    out["lower"], out["upper"] = pred - z * spread, pred + z * spread
    return out


# --------------------------------------------------------------------------- code
def timeseries_snippet(
    date_col: str,
    value_col: str,
    freq: str,
    agg: str,
    period: int,
    horizon: int,
    model: str,
    **load: Any,
) -> str:
    body = f"""date_col, value_col, freq, period, horizon = {date_col!r}, {value_col!r}, {freq!r}, {period}, {horizon}
series = (df.assign(**{{date_col: pd.to_datetime(df[date_col], errors="coerce")}})
            .dropna(subset=[date_col]).set_index(date_col)[value_col]
            .sort_index().resample(freq).{agg}().interpolate(limit_direction="both"))

# %%
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
decomp = seasonal_decompose(series, model="additive", period=period)
import plotly.graph_objects as go
from plotly.subplots import make_subplots
fig = make_subplots(rows=4, cols=1, shared_xaxes=True, subplot_titles=["observed", "trend", "seasonal", "residual"])
for i, part in enumerate(["observed", "trend", "seasonal", "resid"], start=1):
    fig.add_trace(go.Scatter(x=series.index, y=getattr(decomp, part), name=part), row=i, col=1)
fig.update_layout(height=700, showlegend=False)
fig.show()
print("ADF p-value (small = stationary):", round(adfuller(series.dropna())[1], 4))
print("KPSS p-value (large = stationary):", round(kpss(series.dropna(), regression="c", nlags="auto")[1], 4))

# %%
# Rolling-origin backtest: compare naive baselines against ETS and ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA

def forecasters(train):
    seasonal = "add" if len(train) >= 2 * period else None
    return {{
        "Naive": np.repeat(train.iloc[-1], horizon),
        "Seasonal naive": np.tile(train.iloc[-period:].to_numpy(), int(np.ceil(horizon / period)))[:horizon],
        "ETS": ExponentialSmoothing(train, trend="add", damped_trend=True, seasonal=seasonal,
                                    seasonal_periods=period if seasonal else None).fit().forecast(horizon).to_numpy(),
        "ARIMA(1,1,1)": ARIMA(train, order=(1, 1, 1)).fit().forecast(horizon).to_numpy(),
    }}

rows = []
for origin in [len(series) - horizon * k for k in (3, 2, 1)]:
    train, test = series.iloc[:origin], series.iloc[origin:origin + horizon]
    for name, pred in forecasters(train).items():
        rows.append({{"model": name, "RMSE": float(np.sqrt(np.mean((test.to_numpy() - pred) ** 2)))}})
pd.DataFrame(rows).groupby("model")["RMSE"].mean().sort_values()
"""
    return make_snippet("Time-series analysis", body, **load)
