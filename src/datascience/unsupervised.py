"""Clustering, PCA, and anomaly detection with interpretable summaries."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from src.core.code_snippets import _list, make_snippet

MIN_ROWS = 10
SILHOUETTE_SAMPLE = 5000


class UnsupervisedError(ValueError):
    """Invalid input for an unsupervised analysis."""


def prepare_matrix(
    df: pd.DataFrame, columns: list[str], scale: bool = True
) -> tuple[np.ndarray, pd.Index, list[str]]:
    """Numeric matrix (median-imputed, optionally standardized) and the row index kept."""
    if len(columns) < 2:
        raise UnsupervisedError("Select at least two numeric columns.")
    data = df[columns].apply(pd.to_numeric, errors="coerce")
    data = data.dropna(axis=1, how="all")
    data = data.loc[:, data.nunique(dropna=True) > 1]
    if data.shape[1] < 2:
        raise UnsupervisedError("Fewer than two usable (non-constant, numeric) columns.")
    data = data.dropna(how="all")
    if len(data) < MIN_ROWS:
        raise UnsupervisedError(f"Need at least {MIN_ROWS} rows.")
    X = SimpleImputer(strategy="median").fit_transform(data)
    if scale:
        X = StandardScaler().fit_transform(X)
    return X, data.index, list(data.columns)


def kmeans_scan(X: np.ndarray, k_range: range, seed: int = 42) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(seed)
    sample = rng.choice(len(X), min(len(X), SILHOUETTE_SAMPLE), replace=False)
    for k in k_range:
        if k < 2 or k >= len(X):
            continue
        model = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
        rows.append(
            {
                "k": k,
                "inertia": float(model.inertia_),
                "silhouette": float(silhouette_score(X[sample], model.labels_[sample])),
                "davies_bouldin": float(davies_bouldin_score(X, model.labels_)),
            }
        )
    if not rows:
        raise UnsupervisedError("No valid cluster counts to try.")
    return pd.DataFrame(rows)


def fit_kmeans(X: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)


def cluster_profiles(
    df: pd.DataFrame,
    index: pd.Index,
    labels: np.ndarray,
    numeric: list[str],
    categorical: list[str] | None = None,
) -> pd.DataFrame:
    """One row per cluster: size, share, and each numeric mean as an index vs the overall mean."""
    frame = df.loc[index].copy()
    frame["cluster"] = labels
    overall = frame[numeric].apply(pd.to_numeric, errors="coerce").mean()
    rows = []
    for cluster, group in frame.groupby("cluster"):
        row: dict[str, Any] = {
            "cluster": int(cluster),
            "size": len(group),
            "share": len(group) / len(frame),
        }
        means = group[numeric].apply(pd.to_numeric, errors="coerce").mean()
        for col in numeric:
            row[f"{col} (mean)"] = float(means[col])
            row[f"{col} (index)"] = (
                float(means[col] / overall[col]) if overall[col] else float("nan")
            )
        for col in categorical or []:
            top = group[col].astype(str).value_counts(normalize=True)
            row[f"{col} (top)"] = f"{top.index[0]} ({top.iloc[0]:.0%})" if len(top) else ""
        rows.append(row)
    return pd.DataFrame(rows)


def describe_cluster(profile_row: pd.Series, numeric: list[str], top: int = 2) -> str:
    """Plain-English label from the most over/under-indexed columns."""
    idx = {c: profile_row[f"{c} (index)"] for c in numeric if pd.notna(profile_row[f"{c} (index)"])}
    parts = []
    for col, val in sorted(idx.items(), key=lambda kv: abs(kv[1] - 1), reverse=True)[:top]:
        parts.append(f"{'high' if val > 1 else 'low'} {col} ({val:.1f}x avg)")
    return ", ".join(parts) or "close to average"


def run_pca(X: np.ndarray, names: list[str], n_components: int | None = None) -> dict[str, Any]:
    n = min(n_components or min(X.shape), min(X.shape))
    pca = PCA(n_components=n).fit(X)
    loadings = pd.DataFrame(
        pca.components_.T, index=names, columns=[f"PC{i + 1}" for i in range(n)]
    )
    return {
        "explained": pca.explained_variance_ratio_,
        "cumulative": np.cumsum(pca.explained_variance_ratio_),
        "loadings": loadings,
        "scores": pca.transform(X),
        "components_for_90pct": int(
            np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.9) + 1
        ),
    }


def detect_anomalies(
    X: np.ndarray, contamination: float = 0.05, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (is_anomaly bool array, anomaly score where higher = more unusual)."""
    if not 0 < contamination < 0.5:
        raise UnsupervisedError("Contamination must be between 0 and 0.5.")
    model = IsolationForest(contamination=contamination, random_state=seed).fit(X)
    return model.predict(X) == -1, -model.score_samples(X)


def clustering_snippet(columns: list[str], k: int, seed: int = 42, **load: Any) -> str:
    body = f"""from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

cols = {_list(columns)}
data = df[cols].apply(pd.to_numeric, errors="coerce").dropna(how="all")
X = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(data))

# Choose k: look for an elbow in inertia and a high silhouette
scan = pd.DataFrame([
    {{"k": k, "inertia": KMeans(k, n_init=10, random_state={seed}).fit(X).inertia_,
      "silhouette": silhouette_score(X, KMeans(k, n_init=10, random_state={seed}).fit_predict(X))}}
    for k in range(2, 9)])
display(scan)

# %%
k = {k}
data["cluster"] = KMeans(k, n_init=10, random_state={seed}).fit_predict(X)
profile = data.groupby("cluster")[cols].mean()
profile["size"] = data["cluster"].value_counts().sort_index()
display((profile[cols] / data[cols].mean()).round(2))   # >1 means above the overall average
profile
"""
    return make_snippet("K-means clustering", body, **load)


def pca_snippet(columns: list[str], **load: Any) -> str:
    body = f"""from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

cols = {_list(columns)}
data = df[cols].apply(pd.to_numeric, errors="coerce").dropna(how="all")
X = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(data))
pca = PCA().fit(X)
print("explained variance:", pca.explained_variance_ratio_.round(3))
print("cumulative:", np.cumsum(pca.explained_variance_ratio_).round(3))
pd.DataFrame(pca.components_.T, index=data.columns, columns=[f"PC{{i+1}}" for i in range(len(data.columns))]).round(3)
"""
    return make_snippet("PCA", body, **load)


def anomaly_snippet(
    columns: list[str], contamination: float = 0.05, seed: int = 42, **load: Any
) -> str:
    body = f"""from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

cols = {_list(columns)}
data = df[cols].apply(pd.to_numeric, errors="coerce").dropna(how="all")
X = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(data))
model = IsolationForest(contamination={contamination}, random_state={seed}).fit(X)
data["anomaly_score"] = -model.score_samples(X)
data["is_anomaly"] = model.predict(X) == -1
data.sort_values("anomaly_score", ascending=False).head(15)
"""
    return make_snippet("Anomaly detection (Isolation Forest)", body, **load)
