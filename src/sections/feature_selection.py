"""Section: Feature Selection - filter, wrapper, embedded, stability, minimal set, collinearity."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.core.code_ds import feature_selection_snippet
from src.core.code_snippets import load_kwargs
from src.core.ds_ui import pick_target_features
from src.core.session_state import KEY_SELECTED_FEATURES
from src.core.ui_components import (
    render_code_button,
    require_dataset,
)
from src.datascience import feature_selection as fs
from src.datascience.modeling import ModelingError

SUBTITLE = (
    "Find the features that matter with several independent methods, then keep a small, stable set."
)


def render() -> None:
    """Render this section inside its parent page."""

    dataset = require_dataset("Feature Selection")
    df = dataset.working_df
    LOAD = load_kwargs(dataset)
    target, features, task = pick_target_features(df, "fs")
    if len(features) < 2:
        st.info("Pick at least two features.")
        return
    store = st.session_state.setdefault("fs_results", {})
    if store.get("_target") != target:
        store.clear()
        store["_target"] = target

    def run(name, fn, *a, **kw):
        try:
            with st.spinner(f"Running {name}..."):
                store[name] = fn(*a, **kw)
        except ModelingError as err:
            st.error(str(err))
        except Exception as err:
            st.error(f"{name} failed: {err}")

    tabs = st.tabs(
        [
            "🔬 Filter",
            "🔁 Wrapper (RFE)",
            "🧲 Embedded",
            "🎲 Stability",
            "✂️ Minimal set",
            "🔗 Collinearity",
            "📉 PCA",
            "✅ Summary",
        ]
    )
    t_filter, t_wrap, t_emb, t_stab, t_min, t_col, t_pca, t_sum = tabs

    with t_filter:
        method = st.radio(
            "Method",
            ["mutual_info", "anova"],
            horizontal=True,
            format_func={
                "mutual_info": "Mutual information (any relationship)",
                "anova": "ANOVA F (linear relationship)",
            }.get,
            key="fs_filter_method",
        )
        if st.button("Score features", type="primary", key="fs_filter_btn"):
            run("filter", fs.filter_scores, df, target, features, method)
        if "filter" in store:
            t = store["filter"]
            st.plotly_chart(
                px.bar(
                    t.iloc[::-1],
                    x="score",
                    y="feature",
                    orientation="h",
                    template="plotly_dark",
                    title="Association with the target",
                ),
                use_container_width=True,
            )
            st.dataframe(
                t.style.format({"score": "{:.4f}", "p_value": "{:.2g}"}, na_rep="-"),
                hide_index=True,
                use_container_width=True,
            )

    with t_wrap:
        st.caption(
            "Recursive feature elimination repeatedly drops the weakest feature of a model. Rank 1 is the last survivor."
        )
        rm = st.selectbox(
            "Model used for ranking",
            ["Linear / Logistic Regression", "Random Forest", "Decision Tree"],
            key="fs_rfe_model",
        )
        if st.button("Run RFE", key="fs_rfe_btn"):
            run("rfe", fs.rfe_ranking, df, target, features, rm)
        if "rfe" in store:
            st.dataframe(store["rfe"], hide_index=True, use_container_width=True)

    with t_emb:
        em = st.radio(
            "Method",
            ["lasso", "tree"],
            horizontal=True,
            format_func={"lasso": "L1 penalty (Lasso)", "tree": "Random-forest importance"}.get,
            key="fs_emb_method",
        )
        if st.button("Run", key="fs_emb_btn"):
            run("embedded", fs.embedded_importance, df, target, features, em)
        if "embedded" in store:
            t = store["embedded"]
            st.plotly_chart(
                px.bar(
                    t.iloc[::-1],
                    x="score",
                    y="feature",
                    color="selected",
                    orientation="h",
                    template="plotly_dark",
                ),
                use_container_width=True,
            )
            st.caption(f"Selected: {', '.join(t.loc[t['selected'], 'feature']) or 'none'}")

    with t_stab:
        st.caption(
            "A feature that is selected on most random subsamples is trustworthy; one that comes and goes is noise."
        )
        rounds = st.slider("Subsamples", 10, 60, 25, key="fs_rounds")
        if st.button("Run stability selection", key="fs_stab_btn"):
            run("stability", fs.stability_selection, df, target, features, rounds)
        if "stability" in store:
            t = store["stability"]
            fig = px.bar(
                t.iloc[::-1],
                x="selection_frequency",
                y="feature",
                orientation="h",
                color="stable",
                template="plotly_dark",
                range_x=[0, 1],
            )
            fig.add_vline(x=0.8, line_dash="dot")
            st.plotly_chart(fig, use_container_width=True)

    with t_min:
        st.caption(
            "Add features in RFE order and watch the cross-validated score. The recommended set is the smallest one within one standard error of the best."
        )
        mm = st.selectbox(
            "Model", ["Linear / Logistic Regression", "Random Forest"], key="fs_min_model"
        )
        if st.button("Find the minimal set", key="fs_min_btn"):
            run("minimal", fs.minimal_feature_set, df, target, features, mm)
        if "minimal" in store:
            r = store["minimal"]
            t = r["table"]
            fig = go.Figure(
                go.Scatter(
                    x=t["n_features"],
                    y=t["cv_mean"],
                    mode="lines+markers",
                    error_y={"type": "data", "array": t["cv_se"]},
                )
            )
            fig.add_vline(x=r["recommended_k"], line_dash="dot", annotation_text="recommended")
            fig.update_layout(
                template="plotly_dark",
                xaxis_title="number of features",
                yaxis_title="cross-validated score",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.success(
                f"Keep **{r['recommended_k']}** feature(s): {', '.join(r['recommended'])}. The best score used {r['best_k']}."
            )

    with t_col:
        num = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
        if len(num) < 2:
            st.info("Needs two numeric features.")
        else:
            try:
                v = fs.vif_table(df, num)
                st.dataframe(
                    v.style.format({"vif": "{:.1f}"}), hide_index=True, use_container_width=True
                )
                st.caption(
                    "VIF above 5 is moderate and above 10 is severe: the feature is largely predictable from the others."
                )
                thr = st.slider(
                    "Group features correlated above", 0.5, 0.99, 0.8, 0.01, key="fs_corr_thr"
                )
                groups = fs.correlated_clusters(df, num, thr)
                for g in [g for g in groups if len(g) > 1]:
                    st.markdown(
                        "- Correlated group: " + ", ".join(f"`{c}`" for c in g) + " (keep one)"
                    )
                if all(len(g) == 1 for g in groups):
                    st.success("No strongly correlated groups at this threshold.")
                st.plotly_chart(
                    px.imshow(
                        df[num].corr(),
                        text_auto=".2f",
                        zmin=-1,
                        zmax=1,
                        color_continuous_scale="RdBu_r",
                        template="plotly_dark",
                    ),
                    use_container_width=True,
                )
            except ModelingError as err:
                st.info(str(err))

    with t_pca:
        var = st.slider("Variance to keep", 0.5, 0.99, 0.9, 0.01, key="fs_pca_var")
        if st.button("Compare with and without PCA", key="fs_pca_btn"):
            run("pca", fs.pca_compression, df, target, features, var)
        if "pca" in store:
            r = store["pca"]
            a, b, c = st.columns(3)
            a.metric("Columns after encoding", r["original_features"])
            b.metric("PCA components kept", r["components"])
            c.metric(
                "Score change",
                f"{r['difference']:+.4f}",
                help=f"{r['score_original']:.4f} → {r['score_pca']:.4f}",
            )
            st.caption("PCA helps when many correlated columns overfit; it costs interpretability.")

    with t_sum:
        rows = pd.DataFrame({"feature": features})
        if "filter" in store:
            rows = rows.merge(
                store["filter"][["feature", "rank"]].rename(columns={"rank": "filter_rank"}),
                on="feature",
                how="left",
            )
        if "rfe" in store:
            rows = rows.merge(store["rfe"][["feature", "rfe_rank"]], on="feature", how="left")
        if "embedded" in store:
            rows = rows.merge(
                store["embedded"][["feature", "selected"]].rename(
                    columns={"selected": "embedded_selected"}
                ),
                on="feature",
                how="left",
            )
        if "stability" in store:
            rows = rows.merge(
                store["stability"][["feature", "selection_frequency"]], on="feature", how="left"
            )
        vote_cols = []
        if "filter_rank" in rows:
            rows["vote_filter"] = rows["filter_rank"] <= max(1, len(features) // 2)
            vote_cols.append("vote_filter")
        if "rfe_rank" in rows:
            rows["vote_rfe"] = rows["rfe_rank"] <= max(1, len(features) // 2)
            vote_cols.append("vote_rfe")
        if "embedded_selected" in rows:
            rows["vote_embedded"] = rows["embedded_selected"].fillna(False).astype(bool)
            vote_cols.append("vote_embedded")
        if "selection_frequency" in rows:
            rows["vote_stable"] = rows["selection_frequency"].fillna(0) >= 0.8
            vote_cols.append("vote_stable")
        if vote_cols:
            rows["votes"] = rows[vote_cols].sum(axis=1)
            rows = rows.sort_values("votes", ascending=False)
            st.dataframe(rows, hide_index=True, use_container_width=True)
            if len(vote_cols) > 1:
                need = st.slider(
                    "Keep features with at least this many votes",
                    1,
                    len(vote_cols),
                    len(vote_cols) // 2 + 1,
                    key="fs_votes",
                )
            else:
                need = 1
                st.caption("Run more methods on the other tabs to vote across them.")
            chosen = list(rows.loc[rows["votes"] >= need, "feature"])
            st.markdown(f"**{len(chosen)} feature(s):** {', '.join(chosen) or 'none'}")
            if st.button(
                "Use this feature set on the other modeling pages",
                disabled=not chosen,
                key="fs_apply",
            ):
                st.session_state[KEY_SELECTED_FEATURES] = {"target": target, "features": chosen}
                st.success(
                    "Saved. Model Selection, Ensembles, and the other pickers will start from this set."
                )
            st.download_button(
                "Download the list", "\n".join(chosen), "selected_features.txt", key="fs_dl"
            )
        else:
            st.info("Run one or more methods in the other tabs; the votes appear here.")
        render_code_button(
            "Feature selection",
            lambda o: feature_selection_snippet(target, features, task, options=o, **LOAD),
            key="code_fs",
        )
    return
