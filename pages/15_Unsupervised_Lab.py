"""Page 15: Unsupervised Lab - clustering, PCA, and anomaly detection."""

import pandas as pd
import plotly.express as px
import streamlit as st
from src.core.code_snippets import load_kwargs
from src.core.library import add_table, get_tables, unique_name
from src.core.ui_components import page_setup, render_code_button, require_dataset
from src.datascience import unsupervised as un

page_setup(
    "Unsupervised Lab",
    "🔭",
    "Find segments, reduce dimensions, and spot unusual records without a target.",
)
dataset = require_dataset("Unsupervised Lab")
df = dataset.working_df
num_cols = [c for c in df.select_dtypes("number").columns if df[c].nunique() > 1]
if len(num_cols) < 2:
    st.warning("This lab needs at least two numeric columns.")
    st.stop()

cols = st.multiselect("Numeric columns to use", num_cols, default=num_cols[:8])
scale = st.checkbox("Standardize columns (recommended)", value=True)
try:
    X, index, used = un.prepare_matrix(df, cols, scale)
except un.UnsupervisedError as err:
    st.info(str(err))
    st.stop()
st.caption(f"{len(X):,} rows × {len(used)} usable columns")

tab_cluster, tab_pca, tab_anom = st.tabs(["🧩 Clusters", "📉 PCA", "🚨 Anomalies"])

with tab_cluster:
    max_k = max(3, min(10, len(X) - 1))
    k_lo, k_hi = st.slider("Try these cluster counts", 2, max_k, (2, min(8, max_k)))
    scan = un.kmeans_scan(X, range(k_lo, k_hi + 1))
    best_k = int(scan.loc[scan["silhouette"].idxmax(), "k"])
    s1, s2 = st.columns(2)
    s1.plotly_chart(
        px.line(
            scan,
            x="k",
            y="inertia",
            markers=True,
            template="plotly_dark",
            title="Elbow (lower is tighter)",
        ),
        use_container_width=True,
    )
    s2.plotly_chart(
        px.line(
            scan,
            x="k",
            y="silhouette",
            markers=True,
            template="plotly_dark",
            title="Silhouette (higher is better)",
        ),
        use_container_width=True,
    )
    k = st.number_input(
        "Number of clusters", 2, max_k, best_k, help=f"Best silhouette at k = {best_k}"
    )
    labels = un.fit_kmeans(X, int(k))
    cats = [
        c
        for c in df.columns
        if c not in df.select_dtypes("number").columns and df[c].nunique() <= 20
    ][:3]
    profile = un.cluster_profiles(df, index, labels, used, cats)
    profile.insert(
        2, "what stands out", [un.describe_cluster(r, used) for _, r in profile.iterrows()]
    )
    st.dataframe(
        profile.style.format({"share": "{:.1%}"}, precision=2),
        use_container_width=True,
        hide_index=True,
    )
    st.caption("Index > 1 means the cluster is above the overall average for that column.")
    scores = un.run_pca(X, used, 2)["scores"]
    fig = px.scatter(
        x=scores[:, 0],
        y=scores[:, 1],
        color=labels.astype(str),
        labels={"x": "PC1", "y": "PC2", "color": "cluster"},
        template="plotly_dark",
        title="Clusters in 2-D (PCA view)",
    )
    st.plotly_chart(fig, use_container_width=True)
    a, b = st.columns(2)
    with a:
        render_code_button(
            "K-means clustering",
            un.clustering_snippet(used, int(k), **load_kwargs(dataset)),
            key="code_km",
        )
    if b.button("Save cluster labels as a table"):
        out = df.loc[index].assign(cluster=labels)
        name = unique_name(get_tables(st.session_state), "clustered")
        add_table(st.session_state, name, out)
        st.success(f"Saved as `{name}` (use it in the Transformation Studio or SQL).")

with tab_pca:
    res = un.run_pca(X, used)
    st.metric(
        "Components to explain 90% of variance",
        res["components_for_90pct"],
        help=f"of {len(used)} original columns",
    )
    scree = pd.DataFrame(
        {
            "component": [f"PC{i + 1}" for i in range(len(res["explained"]))],
            "explained": res["explained"],
            "cumulative": res["cumulative"],
        }
    )
    fig = px.bar(
        scree,
        x="component",
        y="explained",
        template="plotly_dark",
        title="Variance explained per component",
    )
    fig.add_scatter(
        x=scree["component"], y=scree["cumulative"], mode="lines+markers", name="cumulative"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("##### Loadings (which columns drive each component)")
    st.plotly_chart(
        px.imshow(
            res["loadings"].iloc[:, :5],
            zmin=-1,
            zmax=1,
            color_continuous_scale="RdBu_r",
            text_auto=".2f",
            aspect="auto",
            template="plotly_dark",
        ),
        use_container_width=True,
    )
    render_code_button("PCA", un.pca_snippet(used, **load_kwargs(dataset)), key="code_pca")

with tab_anom:
    contamination = st.slider("Expected share of anomalies", 0.01, 0.20, 0.05, 0.01)
    flags, score = un.detect_anomalies(X, contamination)
    st.metric("Flagged records", f"{int(flags.sum()):,}", help=f"{flags.mean():.1%} of rows")
    out = df.loc[index].assign(anomaly_score=score, is_anomaly=flags)
    st.dataframe(
        out.sort_values("anomaly_score", ascending=False).head(25), use_container_width=True
    )
    pcs = un.run_pca(X, used, 2)["scores"]
    st.plotly_chart(
        px.scatter(
            x=pcs[:, 0],
            y=pcs[:, 1],
            color=flags.astype(str),
            color_discrete_map={"True": "#ef4444", "False": "#38bdf8"},
            labels={"x": "PC1", "y": "PC2", "color": "anomaly"},
            template="plotly_dark",
        ),
        use_container_width=True,
    )
    render_code_button(
        "Anomaly detection",
        un.anomaly_snippet(used, contamination, **load_kwargs(dataset)),
        key="code_anom",
    )
