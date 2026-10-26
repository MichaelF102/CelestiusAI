"""Machine-learning forecasting: lag/rolling features, direct and recursive multi-step forecasts."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.datascience import catalog
from src.datascience.timeseries import TimeSeriesError, _fit_forecast, smape

DEFAULT_LAGS = (0, 1, 2, 3, 7)
DEFAULT_ROLLING = (7,)
STRATEGIES = {
    "direct": "Direct (one model per step ahead)",
    "recursive": "Recursive (one model, feed predictions back)",
}


def _step(index: pd.DatetimeIndex) -> pd.Timedelta | pd.DateOffset:
    freq = index.freq or pd.infer_freq(index)
    if freq is None:
        raise TimeSeriesError("The series needs a regular frequency.")
    return pd.tseries.frequencies.to_offset(freq)


def _calendar(index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dow": index.dayofweek,
            "month": index.month,
            "is_weekend": (index.dayofweek >= 5).astype(int),
            "day_of_month": index.day,
        },
        index=index,
    )


def feature_frame(y: pd.Series, lags: tuple[int, ...], rolling: tuple[int, ...]) -> pd.DataFrame:
    """Features known at time t: the value now and in the past, rolling mean/std (no future leakage)."""
    f = pd.DataFrame(index=y.index)
    for k in lags:
        f[f"lag_{k}"] = y.shift(k)
    for w in rolling:
        f[f"roll_mean_{w}"] = y.rolling(w).mean()
        f[f"roll_std_{w}"] = y.rolling(w).std()
    return f


def supervised(
    y: pd.Series,
    horizon_step: int,
    lags: tuple[int, ...] = DEFAULT_LAGS,
    rolling: tuple[int, ...] = DEFAULT_ROLLING,
    calendar: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """(X, target) where the target is the value `horizon_step` periods after each row's timestamp."""
    f = feature_frame(y, lags, rolling)
    if calendar:
        target_index = y.index + horizon_step * _step(y.index)
        cal = _calendar(pd.DatetimeIndex(target_index))
        cal.index = y.index
        f = pd.concat([f, cal], axis=1)
    target = y.shift(-horizon_step)
    data = pd.concat([f, target.rename("_target")], axis=1).dropna()
    return data.drop(columns="_target"), data["_target"]


def _model(name: str, seed: int) -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("m", catalog.build(name, "regression", seed))])


class MLForecaster:
    """Direct or recursive multi-step forecaster over a catalog regression model."""

    def __init__(
        self,
        model: str = "Ridge / L2 Logistic",
        strategy: str = "direct",
        horizon: int = 7,
        lags: tuple[int, ...] = DEFAULT_LAGS,
        rolling: tuple[int, ...] = DEFAULT_ROLLING,
        calendar: bool = True,
        seed: int = 42,
    ) -> None:
        if strategy not in STRATEGIES:
            raise TimeSeriesError(f"Unknown strategy '{strategy}'.")
        self.model, self.strategy, self.horizon = model, strategy, horizon
        self.lags, self.rolling, self.calendar, self.seed = lags, rolling, calendar, seed

    def fit(self, y: pd.Series) -> MLForecaster:
        steps = range(1, self.horizon + 1) if self.strategy == "direct" else [1]
        self.models_: dict[int, Pipeline] = {}
        for h in steps:
            X, t = supervised(y, h, self.lags, self.rolling, self.calendar)
            if len(X) < 20:
                raise TimeSeriesError(
                    f"Only {len(X)} usable rows for step {h}: the series is too short for these lags."
                )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.models_[h] = _model(self.model, self.seed).fit(X, t)
        self.history_ = y.copy()
        return self

    def predict(self) -> pd.Series:
        y = self.history_
        step = _step(y.index)
        future = pd.date_range(y.index[-1], periods=self.horizon + 1, freq=step)[1:]
        if self.strategy == "direct":
            last = {}
            for h, mdl in self.models_.items():
                feats = feature_frame(y, self.lags, self.rolling).iloc[[-1]]
                if self.calendar:
                    cal = _calendar(pd.DatetimeIndex([y.index[-1] + h * step]))
                    cal.index = feats.index
                    feats = pd.concat([feats, cal], axis=1)
                last[h] = float(mdl.predict(feats)[0])
            return pd.Series([last[h] for h in range(1, self.horizon + 1)], index=future)
        series = y.copy()
        mdl = self.models_[1]
        preds = []
        for ts in future:
            feats = feature_frame(series, self.lags, self.rolling).iloc[[-1]]
            if self.calendar:
                cal = _calendar(pd.DatetimeIndex([ts]))
                cal.index = feats.index
                feats = pd.concat([feats, cal], axis=1)
            value = float(mdl.predict(feats)[0])
            preds.append(value)
            series = pd.concat([series, pd.Series([value], index=[ts])])
        return pd.Series(preds, index=future)


def rolling_evaluation(
    y: pd.Series,
    models: list[str],
    horizon: int,
    strategies: tuple[str, ...] = ("direct", "recursive"),
    period: int = 7,
    folds: int = 3,
    lags: tuple[int, ...] = DEFAULT_LAGS,
    rolling: tuple[int, ...] = DEFAULT_ROLLING,
    seed: int = 42,
) -> pd.DataFrame:
    """Rolling-origin comparison of ML forecasters against naive baselines."""
    n = len(y)
    min_train = max(60, 3 * max(lags + rolling + (period,)))
    if n < min_train + horizon:
        raise TimeSeriesError(
            f"Series too short: need at least {min_train + horizon} points for a horizon of {horizon}."
        )
    folds = max(1, min(folds, (n - min_train) // horizon))
    origins = [n - horizon * (folds - k) for k in range(folds)]
    contenders: dict[str, dict[str, Any]] = {
        "Naive": {"kind": "naive"},
        "Seasonal naive": {"kind": "snaive"},
    }
    for m in models:
        for s in strategies:
            contenders[f"{m} [{s}]"] = {"kind": "ml", "model": m, "strategy": s}
    rows = []
    for name, spec in contenders.items():
        errs = []
        failed = None
        for origin in origins:
            train, test = y.iloc[:origin], y.iloc[origin : origin + horizon]
            try:
                if spec["kind"] == "naive":
                    pred = _fit_forecast("Naive", train, len(test), period)[0]
                elif spec["kind"] == "snaive":
                    pred = _fit_forecast("Seasonal naive", train, len(test), period)[0]
                else:
                    f = MLForecaster(
                        spec["model"], spec["strategy"], len(test), lags, rolling, True, seed
                    ).fit(train)
                    pred = f.predict().to_numpy()
            except (TimeSeriesError, ValueError) as err:
                failed = str(err)
                break
            actual = test.to_numpy(dtype=float)
            scale = np.mean(np.abs(np.diff(train.to_numpy(dtype=float)))) or np.nan
            e = actual - pred
            errs.append(
                (
                    np.mean(np.abs(e)),
                    np.sqrt(np.mean(e**2)),
                    smape(actual, pred),
                    np.mean(np.abs(e)) / scale,
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
        else:
            mae, rmse, sm, mase = np.mean(errs, axis=0)
            rows.append(
                {"model": name, "MAE": mae, "RMSE": rmse, "sMAPE": sm, "MASE": mase, "note": ""}
            )
    return pd.DataFrame(rows).sort_values("RMSE", na_position="last").reset_index(drop=True)


def forecast_ml(
    y: pd.Series,
    model: str,
    horizon: int,
    strategy: str = "direct",
    level: float = 0.9,
    lags: tuple[int, ...] = DEFAULT_LAGS,
    rolling: tuple[int, ...] = DEFAULT_ROLLING,
    seed: int = 42,
) -> pd.DataFrame:
    """Point forecast plus an interval from the model's own rolling out-of-sample errors."""
    forecaster = MLForecaster(model, strategy, horizon, lags, rolling, True, seed).fit(y)
    point = forecaster.predict()
    n = len(y)
    errors: list[float] = []
    for origin in range(max(60, n - 4 * horizon), n - horizon + 1, max(1, horizon)):
        try:
            sub = MLForecaster(model, strategy, horizon, lags, rolling, True, seed).fit(
                y.iloc[:origin]
            )
            errors.extend(
                (
                    y.iloc[origin : origin + horizon].to_numpy()
                    - sub.predict().to_numpy()[: len(y.iloc[origin : origin + horizon])]
                ).tolist()
            )
        except TimeSeriesError:
            continue
    spread = float(np.quantile(np.abs(errors), level)) if errors else float(y.diff().abs().mean())
    out = pd.DataFrame({"forecast": point})
    out["lower"], out["upper"] = point - spread, point + spread
    return out
