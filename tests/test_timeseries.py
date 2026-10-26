"""Tests for time-series preparation, diagnostics, backtesting and forecasting."""

import ast

import numpy as np
import pandas as pd
import pytest
from src.datascience import timeseries as ts


@pytest.fixture
def df():
    idx = pd.date_range("2022-01-01", periods=400, freq="D")
    rng = np.random.default_rng(0)
    value = (
        100
        + 0.2 * np.arange(400)
        + 10 * np.sin(2 * np.pi * np.arange(400) / 7)
        + rng.normal(0, 1, 400)
    )
    out = pd.DataFrame({"date": idx.astype(str), "sales": value})
    return out.drop(index=[10, 11, 12])  # gaps must be interpolated


def test_prepare_series_regular_and_validated(df):
    s = ts.prepare_series(df, "date", "sales", "D")
    assert s.index.freq is not None and len(s) == 400 and not s.isna().any()
    weekly = ts.prepare_series(df, "date", "sales", "W", "sum")
    assert len(weekly) < 70
    with pytest.raises(ts.TimeSeriesError):
        ts.prepare_series(df.head(5), "date", "sales")
    with pytest.raises(ts.TimeSeriesError):
        ts.prepare_series(df.assign(date="nonsense"), "date", "sales")


def test_decompose_recovers_weekly_seasonality(df):
    s = ts.prepare_series(df, "date", "sales")
    parts = ts.decompose(s, 7)
    seasonal_amp = parts["seasonal"].max() - parts["seasonal"].min()
    assert 15 < seasonal_amp < 25  # injected amplitude is 10 (peak to peak 20)
    with pytest.raises(ts.TimeSeriesError):
        ts.decompose(s.head(10), 7)
    with pytest.raises(ts.TimeSeriesError):
        ts.decompose(s - 1000, 7, "multiplicative")


def test_stationarity_verdicts():
    rng = np.random.default_rng(1)
    noise = pd.Series(rng.normal(size=400))
    walk = pd.Series(np.cumsum(rng.normal(size=400)))
    assert ts.stationarity(noise)["verdict"].startswith("Stationary")
    assert ts.stationarity(walk)["verdict"].startswith(("Non-stationary", "Mixed"))
    with pytest.raises(ts.TimeSeriesError):
        ts.stationarity(pd.Series([1.0] * 50))


def test_acf_shapes(df):
    s = ts.prepare_series(df, "date", "sales")
    res = ts.acf_pacf(s, 20)
    assert len(res["acf"]) == 21 and res["acf"][0] == pytest.approx(1.0)
    assert abs(res["acf"][7]) > res["band"]  # weekly pattern shows up at lag 7


def test_backtest_ranks_seasonal_models_above_naive(df):
    s = ts.prepare_series(df, "date", "sales")
    table = ts.backtest(s, horizon=14, period=7, folds=3)
    assert set(table["model"]) == set(ts.MODELS)
    rmse = table.set_index("model")["RMSE"]
    assert rmse["Seasonal naive"] < rmse["Naive"]
    assert rmse["Holt-Winters (ETS)"] < rmse["Naive"]
    assert table["RMSE"].dropna().is_monotonic_increasing
    with pytest.raises(ts.TimeSeriesError):
        ts.backtest(s.head(20), horizon=14, period=7)


def test_forecast_shape_dates_and_intervals(df):
    s = ts.prepare_series(df, "date", "sales")
    for model in ts.MODELS:
        fc = ts.forecast(s, model, 10, 7)
        assert len(fc) == 10 and fc.index[0] == s.index[-1] + pd.Timedelta(days=1)
        assert (fc["lower"] <= fc["forecast"]).all() and (fc["forecast"] <= fc["upper"]).all()
    # trend continues upward for ETS
    assert ts.forecast(s, "Holt-Winters (ETS)", 7, 7)["forecast"].mean() > s.tail(7).mean() - 5


def test_snippet_runs(df, tmp_path, monkeypatch):
    import plotly.graph_objects as go

    monkeypatch.setattr(go.Figure, "show", lambda self, *a, **k: None)
    csv = tmp_path / "t.csv"
    df.to_csv(csv, index=False)
    code = ts.timeseries_snippet("date", "sales", "D", "mean", 7, 7, "ETS", filename=str(csv))
    ast.parse(code)
    exec(compile(code, "<ts>", "exec"), {})  # noqa: S102
