"""Section: Time-Series Lab - decomposition, stationarity, backtests, and forecasts."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.core.code_snippets import load_kwargs
from src.core.ui_components import render_code_button, require_dataset
from src.datascience import timeseries as ts

SUBTITLE = (
    "Understand trend and seasonality, test stationarity, and compare forecasting methods fairly."
)


def render() -> None:
    """Render this section inside its parent page."""

    dataset = require_dataset("Time-Series Lab")
    df = dataset.working_df
    num_cols = df.select_dtypes("number").columns.tolist()
    if not num_cols:
        st.warning("A numeric column is needed to analyze over time.")
        return

    date_guess = [
        c
        for c in df.columns
        if "date" in c.lower() or "time" in c.lower() or str(df[c].dtype).startswith("datetime")
    ]
    c1, c2, c3, c4 = st.columns(4)
    date_col = c1.selectbox(
        "Date column", date_guess + [c for c in df.columns if c not in date_guess]
    )
    value_col = c2.selectbox("Value column", num_cols)
    freq = c3.selectbox("Frequency", list(ts.FREQ_LABELS), format_func=ts.FREQ_LABELS.get, index=2)
    agg = c4.selectbox("Combine rows by", ["mean", "sum", "median", "max", "min"])

    try:
        series = ts.prepare_series(df, date_col, value_col, freq, agg)
    except ts.TimeSeriesError as err:
        st.info(str(err))
        return

    period = st.number_input(
        "Season length (periods)",
        1,
        max(1, len(series) // 2),
        min(ts.default_period(freq), max(1, len(series) // 2)),
        help="7 for daily data with weekly patterns, 12 for monthly data with yearly patterns.",
    )
    st.caption(
        f"{len(series):,} periods from {series.index.min().date()} to {series.index.max().date()}"
    )
    fig = go.Figure(go.Scatter(x=series.index, y=series.values, mode="lines"))
    fig.update_layout(
        template="plotly_dark",
        height=280,
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        yaxis_title=value_col,
    )
    st.plotly_chart(fig, use_container_width=True)

    tab_dec, tab_stat, tab_bt, tab_fc = st.tabs(
        ["🧬 Decomposition", "📏 Stationarity & ACF", "🏁 Backtest", "🔮 Forecast"]
    )

    with tab_dec:
        model = st.radio("Model", ["additive", "multiplicative"], horizontal=True)
        try:
            parts = ts.decompose(series, int(period), model)
            fig = make_subplots(
                rows=4, cols=1, shared_xaxes=True, subplot_titles=list(parts.columns)
            )
            for i, col in enumerate(parts.columns, start=1):
                fig.add_trace(go.Scatter(x=parts.index, y=parts[col], name=col), row=i, col=1)
            fig.update_layout(template="plotly_dark", height=650, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        except ts.TimeSeriesError as err:
            st.info(str(err))

    with tab_stat:
        try:
            res = ts.stationarity(series)
            a, b = st.columns(2)
            a.metric(
                "ADF p-value", f"{res['adf_p']:.4f}", help="Small (<0.05) suggests stationary."
            )
            b.metric(
                "KPSS p-value", f"{res['kpss_p']:.4f}", help="Large (≥0.05) suggests stationary."
            )
            st.info(res["verdict"])
            diff = st.checkbox("Look at the first difference instead")
            target = series.diff().dropna() if diff else series
            lags = ts.acf_pacf(target, min(40, len(target) // 2 - 1))
            fig = make_subplots(rows=1, cols=2, subplot_titles=["ACF", "PACF"])
            for i, key in enumerate(("acf", "pacf"), start=1):
                fig.add_trace(go.Bar(x=list(range(len(lags[key]))), y=lags[key]), row=1, col=i)
                fig.add_hline(y=lags["band"], line_dash="dot", row=1, col=i)
                fig.add_hline(y=-lags["band"], line_dash="dot", row=1, col=i)
            fig.update_layout(template="plotly_dark", height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        except ts.TimeSeriesError as err:
            st.info(str(err))

    with tab_bt:
        horizon = st.slider(
            "Forecast horizon (periods)",
            1,
            max(2, min(90, len(series) // 4)),
            min(14, max(1, len(series) // 6)),
        )
        folds = st.slider("Backtest folds", 1, 5, 3)
        chosen = st.multiselect("Models", list(ts.MODELS), default=list(ts.MODELS))
        if st.button("Run backtest", type="primary", disabled=not chosen):
            with st.spinner("Fitting models on rolling windows..."):
                try:
                    st.session_state["ts_backtest"] = ts.backtest(
                        series, horizon, int(period), chosen, folds
                    )
                except ts.TimeSeriesError as err:
                    st.error(str(err))
        table = st.session_state.get("ts_backtest")
        if table is not None:
            st.dataframe(
                table.style.format(
                    {"MAE": "{:.3f}", "RMSE": "{:.3f}", "sMAPE": "{:.1%}", "MASE": "{:.2f}"},
                    na_rep="-",
                ),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                "MASE < 1 beats a naive one-step forecast. Lower is better for every metric; a model that cannot beat 'Naive' is not adding value."
            )
            if len(table) and pd.notna(table.iloc[0]["RMSE"]):
                st.success(f"Best by RMSE: **{table.iloc[0]['model']}**")

    with tab_fc:
        f1, f2 = st.columns(2)
        method = f1.selectbox("Model", list(ts.MODELS), index=3)
        fc_h = f2.slider(
            "Periods ahead",
            1,
            max(2, min(120, len(series) // 2)),
            min(14, max(1, len(series) // 6)),
        )
        try:
            fc = ts.forecast(series, method, fc_h, int(period))
            tail = series.tail(min(len(series), fc_h * 4))
            fig = go.Figure()
            fig.add_scatter(x=tail.index, y=tail.values, name="history")
            fig.add_scatter(x=fc.index, y=fc["forecast"], name="forecast")
            fig.add_scatter(
                x=list(fc.index) + list(fc.index[::-1]),
                y=list(fc["upper"]) + list(fc["lower"][::-1]),
                fill="toself",
                line={"width": 0},
                name="95% range",
                opacity=0.25,
            )
            fig.update_layout(
                template="plotly_dark", height=340, margin={"l": 20, "r": 20, "t": 20, "b": 20}
            )
            st.plotly_chart(fig, use_container_width=True)
            st.download_button("⬇️ Download forecast (CSV)", fc.to_csv(), "forecast.csv", "text/csv")
            render_code_button(
                "Time-series analysis",
                ts.timeseries_snippet(
                    date_col,
                    value_col,
                    freq,
                    agg,
                    int(period),
                    fc_h,
                    method,
                    **load_kwargs(dataset),
                ),
                key="code_ts",
            )
        except (ts.TimeSeriesError, ValueError) as err:
            st.info(str(err))
    return
