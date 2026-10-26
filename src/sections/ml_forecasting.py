"""Section: ML Forecasting - tree and linear models on lag features, compared with naive baselines."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.core.code_ds import tsml_snippet
from src.core.code_snippets import load_kwargs
from src.core.ds_ui import model_multiselect
from src.core.ui_components import (
    render_code_button,
    require_dataset,
)
from src.datascience import timeseries as ts
from src.datascience import tsml

SUBTITLE = "Forecast with machine-learning models on lag and calendar features, and prove they beat the naive baselines."


def render() -> None:
    """Render this section inside its parent page."""

    dataset = require_dataset("ML Forecasting")
    df = dataset.working_df
    LOAD = load_kwargs(dataset)
    num_cols = df.select_dtypes("number").columns.tolist()
    if not num_cols:
        st.warning("A numeric column is needed.")
        return

    date_guess = [
        c
        for c in df.columns
        if "date" in c.lower() or "time" in c.lower() or str(df[c].dtype).startswith("datetime")
    ]
    c1, c2, c3, c4 = st.columns(4)
    date_col = c1.selectbox(
        "Date column", date_guess + [c for c in df.columns if c not in date_guess], key="mf_date"
    )
    value_col = c2.selectbox("Value", num_cols, key="mf_value")
    freq = c3.selectbox(
        "Frequency", list(ts.FREQ_LABELS), format_func=ts.FREQ_LABELS.get, index=0, key="mf_freq"
    )
    agg = c4.selectbox("Combine rows by", ["mean", "sum", "median"], key="mf_agg")
    try:
        series = ts.prepare_series(df, date_col, value_col, freq, agg)
    except ts.TimeSeriesError as err:
        st.info(str(err))
        return
    st.caption(
        f"{len(series):,} periods ({series.index.min().date()} to {series.index.max().date()})"
    )

    models = model_multiselect(
        "regression",
        "mf_models",
        ["Ridge / L2 Logistic", "Random Forest", "Hist Gradient Boosting"],
        "Models",
        include_baseline=False,
    )
    h1, h2, h3, h4 = st.columns(4)
    horizon = h1.slider(
        "Horizon (periods)",
        1,
        max(2, min(60, len(series) // 5)),
        min(14, max(1, len(series) // 8)),
        key="mf_h",
    )
    strategies = h2.multiselect(
        "Strategy",
        list(tsml.STRATEGIES),
        default=["direct", "recursive"],
        format_func=tsml.STRATEGIES.get,
        key="mf_strat",
    )
    period = h3.number_input(
        "Season length",
        1,
        max(1, len(series) // 3),
        min(ts.default_period(freq), max(1, len(series) // 3)),
        key="mf_period",
    )
    folds = h4.slider("Backtest folds", 1, 5, 3, key="mf_folds")

    tab_eval, tab_fc = st.tabs(["🏁 Backtest", "🔮 Forecast"])
    with tab_eval:
        st.caption(
            "Every model is re-fitted at several past forecast origins and scored on the periods that followed. 'Naive' repeats the last value; 'Seasonal naive' repeats the last season."
        )
        if st.button(
            "Run the backtest", type="primary", disabled=not models or not strategies, key="mf_run"
        ):
            with st.spinner("Fitting models at each forecast origin..."):
                try:
                    st.session_state["mf_table"] = tsml.rolling_evaluation(
                        series, models, horizon, tuple(strategies), int(period), folds
                    )
                except ts.TimeSeriesError as err:
                    st.error(str(err))
        table = st.session_state.get("mf_table")
        if table is not None:
            st.dataframe(
                table.style.format(
                    {"MAE": "{:.3f}", "RMSE": "{:.3f}", "sMAPE": "{:.1%}", "MASE": "{:.2f}"},
                    na_rep="-",
                ),
                hide_index=True,
                use_container_width=True,
            )
            ok = table.dropna(subset=["RMSE"])
            if len(ok):
                best = ok.iloc[0]
                naive = ok[ok["model"] == "Naive"]
                if best["model"] in ("Naive", "Seasonal naive"):
                    st.warning(
                        f"A naive baseline ({best['model']}) is best: the ML models add nothing here. Check for trend or seasonality they cannot learn, or get more history."
                    )
                elif len(naive):
                    st.success(
                        f"**{best['model']}** cuts the error {1 - best['RMSE'] / naive.iloc[0]['RMSE']:.0%} below the naive forecast (RMSE {best['RMSE']:.3f} vs {naive.iloc[0]['RMSE']:.3f})."
                    )
                st.caption("MASE below 1 beats a one-step naive forecast.")
    with tab_fc:
        ml_rows = (
            [
                m
                for m in (st.session_state.get("mf_table") or pd.DataFrame({"model": []}))["model"]
                if "[" in m
            ]
            if st.session_state.get("mf_table") is not None
            else []
        )
        options = ml_rows or [f"{m} [{s}]" for m in models for s in strategies]
        choice = st.selectbox("Model and strategy", options, key="mf_choice") if options else None
        if choice:
            model_name, strat = choice.rsplit(" [", 1)
            strat = strat.rstrip("]")
            try:
                fc = tsml.forecast_ml(series, model_name, horizon, strat)
                tail = series.tail(min(len(series), horizon * 5))
                fig = go.Figure()
                fig.add_scatter(x=tail.index, y=tail.values, name="history")
                fig.add_scatter(x=fc.index, y=fc["forecast"], name="forecast")
                fig.add_scatter(
                    x=list(fc.index) + list(fc.index[::-1]),
                    y=list(fc["upper"]) + list(fc["lower"][::-1]),
                    fill="toself",
                    line={"width": 0},
                    fillcolor="rgba(56,189,248,0.25)",
                    name="90% range",
                )
                fig.update_layout(
                    template="plotly_dark", height=340, margin={"l": 20, "r": 20, "t": 30, "b": 20}
                )
                st.plotly_chart(fig, use_container_width=True)
                st.download_button(
                    "⬇️ Forecast (CSV)", fc.to_csv(), "ml_forecast.csv", "text/csv", key="mf_dl"
                )
                render_code_button(
                    f"ML forecast ({choice})",
                    lambda o: tsml_snippet(
                        date_col, value_col, model_name, horizon, strat, freq, options=o, **LOAD
                    ),
                    key="code_mf",
                )
            except ts.TimeSeriesError as err:
                st.info(str(err))
        else:
            st.info("Select models and strategies above.")
    return
