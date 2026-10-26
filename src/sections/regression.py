"""Section: Regression Workbench - diagnostics, GLMs, quantiles, regularization, curvature, errors."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.core.code_ds import (
    glm_snippet,
    quantile_snippet,
    regression_diagnostics_snippet,
    regularization_snippet,
)
from src.core.code_snippets import load_kwargs
from src.core.ds_ui import pick_target_features
from src.core.ui_components import (
    render_code_button,
    require_dataset,
)
from src.datascience import catalog
from src.datascience import regression as rg
from src.datascience.modeling import ModelingError

SUBTITLE = (
    "Diagnose a regression, pick the right family, quantify uncertainty, and find where it fails."
)


def render() -> None:
    """Render this section inside its parent page."""

    dataset = require_dataset("Regression Workbench")
    df = dataset.working_df
    LOAD = load_kwargs(dataset)
    target, features, _ = pick_target_features(df, "rg", show_task=False)
    if not features:
        return

    def guarded(fn, *args, **kw):
        try:
            return fn(*args, **kw)
        except ModelingError as err:
            st.error(str(err))
        except Exception as err:
            st.error(f"Could not fit this model: {err}")
        return None

    tabs = st.tabs(
        [
            "🩺 Diagnostics",
            "🧮 GLMs",
            "📏 Quantiles & bands",
            "🪢 Regularization",
            "〰️ Curvature",
            "🔁 Target transforms",
            "🔍 Error analysis",
            "🎓 Advanced statistics",
        ]
    )
    t_diag, t_glm, t_quant, t_reg, t_curve, t_trans, t_err, t_adv = tabs

    # ------------------------------------------------------------------ diagnostics
    with t_diag:
        d = guarded(rg.ols_diagnostics, df, target, features)
        if d is not None:
            a, b, c, e = st.columns(4)
            a.metric("R²", f"{d.r2:.3f}")
            b.metric("Adjusted R²", f"{d.adj_r2:.3f}")
            c.metric("F-test p", f"{d.f_pvalue:.2g}")
            e.metric("Rows", f"{d.n:,}")
            for note in d.notes:
                st.warning(note)
            if not d.notes:
                st.success("No assumption problems flagged.")
            st.dataframe(d.coefficients.style.format("{:.4f}"), use_container_width=True)
            p1, p2 = st.columns(2)
            fig = px.scatter(
                x=d.fitted,
                y=d.standardized,
                template="plotly_dark",
                labels={"x": "fitted", "y": "standardized residual"},
                title="Residuals vs fitted (want a flat, even band)",
                opacity=0.6,
            )
            fig.add_hline(y=0, line_dash="dot")
            p1.plotly_chart(fig, use_container_width=True)
            qq = px.scatter(
                d.qq,
                x="theoretical",
                y="sample",
                template="plotly_dark",
                title="Normal Q-Q (want the diagonal)",
                opacity=0.6,
            )
            lim = [d.qq["theoretical"].min(), d.qq["theoretical"].max()]
            qq.add_scatter(x=lim, y=lim, mode="lines", line={"dash": "dot"}, showlegend=False)
            p2.plotly_chart(qq, use_container_width=True)
            p3, p4 = st.columns(2)
            p3.plotly_chart(
                px.scatter(
                    d.scale_location,
                    x="fitted",
                    y="sqrt_abs_std_resid",
                    template="plotly_dark",
                    title="Scale-location (a rising trend = heteroscedasticity)",
                    opacity=0.6,
                ),
                use_container_width=True,
            )
            infl = px.scatter(
                x=d.leverage,
                y=d.standardized,
                size=d.cooks.clip(lower=1e-6),
                template="plotly_dark",
                labels={"x": "leverage", "y": "standardized residual"},
                title="Leverage vs residual (bubble = Cook's distance)",
                opacity=0.6,
            )
            p4.plotly_chart(infl, use_container_width=True)
            s1, s2 = st.columns(2)
            s1.markdown("**Tests**")
            s1.dataframe(
                pd.DataFrame(
                    {
                        "test": [
                            "Breusch-Pagan (equal variance)",
                            "White (equal variance)",
                            "Shapiro-Wilk (normal residuals)",
                            "Durbin-Watson (autocorrelation)",
                        ],
                        "value": [
                            d.tests["breusch_pagan_p"],
                            d.tests["white_p"],
                            d.tests["shapiro_p"],
                            d.tests["durbin_watson"],
                        ],
                    }
                ).style.format({"value": "{:.4g}"}),
                hide_index=True,
                use_container_width=True,
            )
            s2.markdown("**Collinearity (VIF > 10 is a problem)**")
            s2.dataframe(
                d.vif.style.format({"vif": "{:.1f}"}), hide_index=True, use_container_width=True
            )
            st.markdown("**Most influential rows**")
            st.dataframe(d.influential, hide_index=True, use_container_width=True)
            render_code_button(
                "OLS diagnostics",
                lambda o: regression_diagnostics_snippet(target, features, options=o, **LOAD),
                key="code_rg_diag",
            )

    # ------------------------------------------------------------------ GLMs
    with t_glm:
        st.caption(
            "Pick the distribution that matches the target: counts, positive skewed amounts, zeros plus amounts, or yes/no."
        )
        family = st.selectbox(
            "Family", list(rg.FAMILIES), format_func=lambda k: rg.FAMILIES[k][0], key="rg_family"
        )
        res = guarded(rg.glm_fit, df, target, features, family)
        if res is not None:
            a, b, c = st.columns(3)
            a.metric("Deviance explained", f"{res.deviance_explained:.1%}")
            b.metric(
                "AIC",
                f"{res.aic:,.1f}",
                help="Lower is better; compare only across models on the same rows.",
            )
            c.metric(
                "Dispersion",
                f"{res.dispersion:.2f}",
                help="Close to 1 is right for Poisson/Binomial.",
            )
            for note in res.notes:
                st.warning(note)
            st.dataframe(res.coefficients.style.format("{:.4f}"), use_container_width=True)
            ratio = (
                "odds_ratio"
                if family == "binomial"
                else ("rate_ratio" if "rate_ratio" in res.coefficients else None)
            )
            if ratio:
                t = res.coefficients.drop(index="const", errors="ignore")
                fig = go.Figure(
                    go.Scatter(
                        x=t[ratio],
                        y=t.index,
                        mode="markers",
                        error_x={
                            "type": "data",
                            "symmetric": False,
                            "array": t[f"{ratio}_high"] - t[ratio],
                            "arrayminus": t[ratio] - t[f"{ratio}_low"],
                        },
                    )
                )
                fig.add_vline(x=1, line_dash="dot")
                fig.update_layout(
                    template="plotly_dark",
                    title=f"{ratio.replace('_', ' ').title()} with 95% intervals (1 = no effect)",
                    height=max(250, 40 * len(t)),
                )
                st.plotly_chart(fig, use_container_width=True)
            render_code_button(
                f"GLM ({family})",
                lambda o: glm_snippet(target, features, family, options=o, **LOAD),
                key="code_rg_glm",
            )

    # ------------------------------------------------------------------ quantiles
    with t_quant:
        lo, hi = st.slider("Prediction band", 0.05, 0.95, (0.1, 0.9), 0.05, key="rg_band")
        q = guarded(rg.quantile_regression, df, target, features, (lo, 0.5, hi))
        if q is not None:
            st.markdown(
                "**Linear quantile regression: effects at different points of the distribution**"
            )
            st.dataframe(q["coefficients"].style.format("{:.4f}"), use_container_width=True)
            st.caption(
                f"The linear {lo:.0%}-{hi:.0%} band contains {q['coverage']:.0%} of the training rows (nominal {q['nominal']:.0%})."
            )
            render_code_button(
                "Quantile regression",
                lambda o: quantile_snippet(target, features, (lo, 0.5, hi), options=o, **LOAD),
                key="code_rg_quant",
            )
        if st.button("Non-linear bands (boosted + conformal)", key="rg_gbq_btn"):
            with st.spinner("Training quantile models and calibrating the band..."):
                st.session_state["rg_gbq"] = guarded(
                    rg.gb_quantile_bands, df, target, features, lo, hi
                )
        gbq = st.session_state.get("rg_gbq")
        if gbq:
            a, b, c = st.columns(3)
            a.metric(
                "Held-out coverage",
                f"{gbq['coverage']:.0%}",
                help=f"Nominal {gbq['nominal']:.0%}. Raw model coverage was {gbq['raw_coverage']:.0%} before calibration.",
            )
            b.metric("Mean band width", f"{gbq['mean_width']:,.2f}")
            c.metric("Conformal widening", f"±{gbq['adjustment']:,.2f}")
            band = gbq["band"].sort_values("median").reset_index(drop=True)
            fig = go.Figure()
            fig.add_scatter(x=band.index, y=band["upper"], line={"width": 0}, showlegend=False)
            fig.add_scatter(
                x=band.index,
                y=band["lower"],
                fill="tonexty",
                line={"width": 0},
                name="band",
                fillcolor="rgba(56,189,248,0.25)",
            )
            fig.add_scatter(x=band.index, y=band["median"], name="median prediction")
            fig.add_scatter(
                x=band.index, y=band["actual"], mode="markers", name="actual", marker={"size": 4}
            )
            fig.update_layout(
                template="plotly_dark",
                title="Held-out rows sorted by predicted median",
                xaxis_title="row (sorted)",
            )
            st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------ regularization
    with t_reg:
        kind = st.radio("Penalty", ["lasso", "ridge"], horizontal=True, key="rg_pen")
        if st.button("Compute the path", key="rg_reg_btn"):
            st.session_state["rg_reg"] = (
                kind,
                target,
                guarded(rg.regularization_path, df, target, features, kind),
            )
        stored_reg = st.session_state.get("rg_reg")
        r = (
            stored_reg[2]
            if stored_reg and stored_reg[0] == kind and stored_reg[1] == target
            else None
        )
        if r is not None:
            long = r["path"].melt(id_vars="alpha", var_name="feature", value_name="coefficient")
            fig = px.line(
                long,
                x="alpha",
                y="coefficient",
                color="feature",
                log_x=True,
                template="plotly_dark",
                title=f"{kind.title()} path (standardized coefficients)",
            )
            fig.add_vline(x=r["best_alpha"], line_dash="dot", annotation_text="CV best")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(
                f"Cross-validated alpha: **{r['best_alpha']:.4g}**."
                + (
                    f" Lasso keeps {len(r['kept'])} of {len(r['final_coefficients'])} terms: {', '.join(r['kept'])}."
                    if kind == "lasso"
                    else ""
                )
            )
            st.dataframe(
                r["final_coefficients"].round(4).to_frame("standardized coefficient"),
                use_container_width=True,
            )
            render_code_button(
                f"{kind.title()} path",
                lambda o: regularization_snippet(target, features, kind, options=o, **LOAD),
                key="code_rg_reg",
            )

    # ------------------------------------------------------------------ curvature
    with t_curve:
        num = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
        if not num:
            st.info("Needs a numeric feature.")
        else:
            feat = st.selectbox("Feature", num, key="rg_curve_feat")
            if st.button("Compare polynomial degrees and a spline", key="rg_curve_btn"):
                st.session_state["rg_curve"] = (
                    feat,
                    target,
                    guarded(rg.curvature_study, df, target, feat),
                )
            stored_c = st.session_state.get("rg_curve")
            t = stored_c[2] if stored_c and stored_c[0] == feat and stored_c[1] == target else None
            if t is not None:
                st.dataframe(
                    t.style.format({"cv_r2": "{:.4f}", "cv_std": "{:.4f}"}),
                    hide_index=True,
                    use_container_width=True,
                )
                best = t.loc[t["best"], "model"].iloc[0]
                lin = t.loc[t["model"] == "polynomial degree 1", "cv_r2"].iloc[0]
                gain = t["cv_r2"].max() - lin
                st.info(
                    f"Best: **{best}** (+{gain:.3f} R² over a straight line)."
                    + (
                        " Add a squared or spline term for this feature."
                        if gain > 0.02
                        else " A straight line is enough."
                    )
                )
                sub = df[[feat, target]].dropna().sample(min(len(df), 1500), random_state=0)
                st.plotly_chart(
                    px.scatter(
                        sub,
                        x=feat,
                        y=target,
                        trendline="lowess",
                        template="plotly_dark",
                        opacity=0.5,
                        title="Data with a smooth (LOWESS) trend",
                    ),
                    use_container_width=True,
                )

    # ------------------------------------------------------------------ transforms
    with t_trans:
        base_model = st.selectbox(
            "Model",
            [s.name for s in catalog.available_models("regression", include_baseline=False)],
            index=0,
            format_func=lambda n: catalog.display_name(catalog.get(n), "regression"),
            key="rg_tr_model",
        )
        if st.button("Compare transforms", key="rg_tr_btn"):
            with st.spinner("Cross-validating each transform..."):
                st.session_state["rg_tr"] = (
                    base_model,
                    target,
                    guarded(rg.compare_target_transforms, df, target, features, base_model),
                )
        stored_t = st.session_state.get("rg_tr")
        t = (
            stored_t[2]
            if stored_t and stored_t[0] == base_model and stored_t[1] == target
            else None
        )
        if t is not None:
            st.dataframe(
                t.style.format(
                    {"cv_r2": "{:.4f}", "cv_mae": "{:.3f}", "target_skew_after": "{:.2f}"},
                    na_rep="-",
                ),
                hide_index=True,
                use_container_width=True,
            )
            st.caption(
                "Scores are on the original scale after back-transforming predictions, so they are directly comparable."
            )
            ok = t.dropna(subset=["cv_mae"])
            if len(ok):
                st.success(f"Lowest error: **{ok.loc[ok['cv_mae'].idxmin(), 'transform']}**.")

    # ------------------------------------------------------------------ error analysis
    with t_err:
        em = st.selectbox(
            "Model",
            [s.name for s in catalog.available_models("regression", include_baseline=False)],
            index=catalog.available_models("regression", include_baseline=False).index(
                catalog.get("Random Forest")
            ),
            format_func=lambda n: catalog.display_name(catalog.get(n), "regression"),
            key="rg_err_model",
        )
        if st.button("Analyze errors", key="rg_err_btn"):
            with st.spinner("Fitting and slicing the errors..."):
                st.session_state["rg_err"] = (
                    em,
                    guarded(rg.error_analysis, df, target, features, em),
                )
        stored = st.session_state.get("rg_err")
        if stored and stored[1] and stored[0] == em:
            r = stored[1]
            a, b, c = st.columns(3)
            a.metric("Held-out R²", f"{r['r2']:.3f}")
            b.metric("Mean absolute error", f"{r['overall_mae']:,.2f}")
            c.metric("Bias (mean error)", f"{r['overall_bias']:+,.2f}")
            weak = r["weak_segments"]
            if len(weak):
                st.warning(f"{len(weak)} segment(s) have error more than 1.5× the overall level:")
                st.dataframe(
                    weak.style.format(
                        {"mae": "{:.2f}", "bias": "{:+.2f}", "mae_vs_overall": "{:.2f}"}
                    ),
                    hide_index=True,
                    use_container_width=True,
                )
            else:
                st.success("Errors are spread evenly across the segments checked.")
            seg = r["segments"]
            st.plotly_chart(
                px.bar(
                    seg,
                    x="segment",
                    y="mae_vs_overall",
                    color="feature",
                    template="plotly_dark",
                    title="Error relative to overall (1 = average)",
                ),
                use_container_width=True,
            )
            res_df = r["predictions"]
            st.plotly_chart(
                px.scatter(
                    res_df,
                    x="actual",
                    y="pred",
                    template="plotly_dark",
                    opacity=0.5,
                    title="Predicted vs actual (held-out)",
                ).add_scatter(
                    x=[res_df.actual.min(), res_df.actual.max()],
                    y=[res_df.actual.min(), res_df.actual.max()],
                    mode="lines",
                    line={"dash": "dot"},
                    showlegend=False,
                ),
                use_container_width=True,
            )

    # ------------------------------------------------------------------ advanced
    with t_adv:
        st.markdown("##### Bayesian regression")
        bay = guarded(rg.bayesian_regression, df, target, features)
        if bay is not None:
            st.dataframe(bay["coefficients"].style.format("{:.4f}"), use_container_width=True)
            st.caption(
                f"{bay['note']} Noise sd ≈ {bay['noise_sd']:.3f}; mean predictive sd {bay['mean_predictive_sd']:.3f}."
            )
        st.markdown("##### Bootstrap confidence intervals (no normality assumption)")
        if st.button("Run bootstrap (300 resamples)", key="rg_boot_btn"):
            st.session_state["rg_boot"] = guarded(rg.bootstrap_coefficients, df, target, features)
        if st.session_state.get("rg_boot") is not None:
            st.dataframe(
                st.session_state["rg_boot"].style.format("{:.4f}"), use_container_width=True
            )
        st.markdown("##### Power and sample size")
        p1, p2, p3 = st.columns(3)
        f2 = p1.number_input(
            "Effect size f² (0.02 small, 0.15 medium, 0.35 large)",
            0.001,
            5.0,
            0.15,
            0.01,
            key="rg_f2",
        )
        k = p2.number_input("Predictors", 1, 100, max(1, len(features)), key="rg_k")
        power = p3.select_slider("Target power", [0.7, 0.8, 0.9, 0.95], value=0.8, key="rg_pow")
        guarded_n = guarded(rg.regression_sample_size, f2, int(k), power)
        if guarded_n:
            st.metric(
                "Rows needed", f"{guarded_n:,}", delta=f"you have {len(df):,}", delta_color="off"
            )
            st.caption(
                f"With your {len(df):,} rows the F-test has {rg.regression_power(f2, int(k), len(df)):.0%} power for this effect size."
            )
        st.markdown("##### Mixed-effects model (grouped data)")
        group = st.selectbox(
            "Group column (e.g. store, customer)",
            [c for c in df.columns if c not in (target, *features)] or ["(none)"],
            key="rg_group",
        )
        if group != "(none)" and st.button("Fit mixed model", key="rg_mixed_btn"):
            st.session_state["rg_mixed"] = guarded(rg.mixed_effects, df, target, features, group)
        mixed = st.session_state.get("rg_mixed")
        if mixed:
            a, b, c = st.columns(3)
            a.metric("Groups", mixed["n_groups"])
            b.metric(
                "ICC",
                f"{mixed['icc']:.2f}",
                help="Share of variance explained by the grouping itself.",
            )
            c.metric("Group variance", f"{mixed['group_variance']:.3f}")
            st.dataframe(mixed["fixed_effects"].style.format("{:.4f}"), use_container_width=True)
    return
