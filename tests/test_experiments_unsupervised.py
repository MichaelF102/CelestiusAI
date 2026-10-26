"""Tests for experiments (power, A/B, SRM, peeking) and unsupervised tools."""

import ast

import numpy as np
import pandas as pd
import pytest
from src.datascience import experiments as e
from src.datascience import unsupervised as u


# ------------------------------------------------------------------ experiments
def test_sample_size_known_values():
    # classic: 10% -> 12% at 80% power, alpha .05 needs ~3,800 per group
    n = e.sample_size_proportions(0.10, 0.12)
    assert 3500 < n < 4100
    assert e.sample_size_proportions(0.10, 0.12, power=0.9) > n
    assert e.sample_size_proportions(0.10, 0.11) > n  # smaller effect -> larger n
    assert 0.78 < e.power_for_proportions(0.10, 0.12, n) < 0.82
    assert e.sample_size_means(delta=5, sd=20) == pytest.approx(252, abs=3)
    with pytest.raises(e.ExperimentError):
        e.sample_size_proportions(0.1, 0.1)
    with pytest.raises(e.ExperimentError):
        e.sample_size_means(0, 1)


def test_conversion_analysis_detects_real_effect_and_null():
    real = e.analyze_conversion(500, 5000, 600, 5000)
    assert real["significant"] and real["abs_lift"] == pytest.approx(0.02)
    assert real["ci_low"] > 0 and real["prob_treatment_better"] > 0.99
    null = e.analyze_conversion(500, 5000, 505, 5000)
    assert not null["significant"] and 0.3 < null["prob_treatment_better"] < 0.7
    assert null["ci_low"] < 0 < null["ci_high"]
    with pytest.raises(e.ExperimentError):
        e.analyze_conversion(10, 5, 1, 5)


def test_continuous_analysis():
    rng = np.random.default_rng(0)
    a, b = rng.normal(10, 2, 400), rng.normal(10.8, 2, 400)
    res = e.analyze_continuous(a, b)
    assert res["significant"] and 0.2 < res["cohens_d"] < 0.6
    assert res["ci_low"] < res["abs_lift"] < res["ci_high"] and res["prob_treatment_better"] > 0.95
    with pytest.raises(e.ExperimentError):
        e.analyze_continuous([1], [2, 3])


def test_analyze_frame_picks_binary_or_continuous():
    df = pd.DataFrame({"g": ["a"] * 50 + ["b"] * 50, "conv": [0, 1] * 50, "rev": np.arange(100.0)})
    assert e.analyze_frame(df, "g", "conv", "a", "b")["type"] == "conversion"
    assert e.analyze_frame(df, "g", "rev", "a", "b")["type"] == "continuous"
    with pytest.raises(e.ExperimentError):
        e.analyze_frame(df, "g", "rev", "a", "zzz")


def test_srm():
    assert not e.srm_check([5000, 5040])["mismatch"]
    assert e.srm_check([5000, 5600])["mismatch"]
    assert not e.srm_check([7000, 3000], [0.7, 0.3])["mismatch"]


def test_peeking_inflates_false_positives_and_alpha_adjustment():
    sim = e.simulate_peeking(looks=8, sims=1500)
    assert sim["peeking_false_positive_rate"] > 2 * sim["single_look_false_positive_rate"]
    assert 0.03 < sim["single_look_false_positive_rate"] < 0.07
    assert e.adjusted_alpha(0.05, 5) == 0.01
    assert e.adjusted_alpha(0.05, 5, "pocock") == 0.0158
    assert e.adjusted_alpha(0.05, 1) == 0.05


@pytest.mark.parametrize("binary", [True, False])
def test_ab_snippet_runs(binary, tmp_path):
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "g": rng.choice(["A", "B"], 400),
            "m": rng.integers(0, 2, 400) if binary else rng.normal(size=400),
        }
    )
    csv = tmp_path / "ab.csv"
    df.to_csv(csv, index=False)
    code = e.ab_test_snippet("g", "m", "A", "B", binary, filename=str(csv))
    ast.parse(code)
    exec(compile(code, "<ab>", "exec"), {})  # noqa: S102
    exec(compile(e.power_snippet(0.1, 0.12), "<pw>", "exec"), {})  # noqa: S102


# ------------------------------------------------------------------ unsupervised
@pytest.fixture
def blobs():
    rng = np.random.default_rng(0)
    centers = [(0, 0, 0), (6, 6, 0), (0, 6, 6)]
    data = np.vstack([rng.normal(c, 0.6, (80, 3)) for c in centers])
    df = pd.DataFrame(data, columns=["a", "b", "c"])
    df["seg"] = np.repeat(["x", "y", "z"], 80)
    df.loc[5, "a"] = np.nan
    return df


def test_kmeans_scan_finds_three_clusters_and_profiles(blobs):
    X, idx, names = u.prepare_matrix(blobs, ["a", "b", "c"])
    scan = u.kmeans_scan(X, range(2, 7))
    assert int(scan.loc[scan["silhouette"].idxmax(), "k"]) == 3
    labels = u.fit_kmeans(X, 3)
    prof = u.cluster_profiles(blobs, idx, labels, names, ["seg"])
    assert sorted(prof["size"]) == [80, 80, 80]
    assert all(prof["seg (top)"].str.contains("100%"))
    assert "high" in u.describe_cluster(prof.iloc[0], names) or "low" in u.describe_cluster(
        prof.iloc[0], names
    )


def test_pca_and_anomalies(blobs):
    X, _, names = u.prepare_matrix(blobs, ["a", "b", "c"])
    res = u.run_pca(X, names)
    assert res["cumulative"][-1] == pytest.approx(1.0) and res["loadings"].shape == (3, 3)
    X2 = np.vstack([X, [[40, 40, 40]]])
    flags, score = u.detect_anomalies(X2, 0.02)
    assert flags[-1] and score[-1] == score.max()
    with pytest.raises(u.UnsupervisedError):
        u.detect_anomalies(X, 0.9)


def test_input_validation():
    with pytest.raises(u.UnsupervisedError):
        u.prepare_matrix(pd.DataFrame({"a": range(20)}), ["a"])
    with pytest.raises(u.UnsupervisedError):
        u.prepare_matrix(pd.DataFrame({"a": [1] * 20, "b": [2] * 20}), ["a", "b"])
    with pytest.raises(u.UnsupervisedError):
        u.prepare_matrix(pd.DataFrame({"a": range(5), "b": range(5)}), ["a", "b"])


def test_unsupervised_snippets_run(blobs, tmp_path):
    csv = tmp_path / "b.csv"
    blobs.to_csv(csv, index=False)
    for code in (
        u.clustering_snippet(["a", "b", "c"], 3, filename=str(csv)),
        u.pca_snippet(["a", "b", "c"], filename=str(csv)),
        u.anomaly_snippet(["a", "b", "c"], 0.05, filename=str(csv)),
    ):
        exec(compile(code, "<u>", "exec"), {})  # noqa: S102
