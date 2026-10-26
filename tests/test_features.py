"""Tests for feature engineering steps, leakage checks and code generation."""

import numpy as np
import pandas as pd
import pytest
from src.core.code_snippets import make_snippet  # noqa: F401  (ensures import path works)
from src.datascience import features as f


@pytest.fixture
def df():
    rng = np.random.default_rng(3)
    n = 120
    out = pd.DataFrame(
        {
            "id": range(n),
            "price": rng.gamma(2.0, 50.0, n),
            "qty": rng.integers(1, 9, n),
            "cat": rng.choice(["a", "b", "c"], n),
            "when": pd.date_range("2024-01-01", periods=n, freq="D").astype(str),
            "target": rng.normal(size=n),
        }
    )
    out.loc[::10, "price"] = np.nan
    out["target_copy"] = out["target"] * 2
    return out


def test_numeric_transforms_and_guards(df):
    out = f.apply_spec(df, f.FeatureSpec(op="log1p", columns=["price", "qty"]))
    assert {"price_log1p", "qty_log1p"} <= set(out.columns)
    assert len(df.columns) == 7, "input must not be mutated"
    with pytest.raises(f.FeatureError):
        f.apply_spec(df.assign(price=-5.0), f.FeatureSpec(op="log1p", columns=["price"]))
    with pytest.raises(f.FeatureError):
        f.apply_spec(df, f.FeatureSpec(op="sqrt", columns=["nope"]))


def test_encodings_bins_dates_interactions(df):
    out = f.apply_specs(
        df,
        [
            f.FeatureSpec(op="onehot", columns=["cat"]),
            f.FeatureSpec(
                op="date_parts", columns=["when"], params={"parts": ["month", "is_weekend"]}
            ),
            f.FeatureSpec(op="interaction", columns=["price", "qty"], params={"kind": "ratio"}),
            f.FeatureSpec(op="bin_quantile", columns=["qty"], params={"bins": 3}),
            f.FeatureSpec(op="missing_indicator", columns=["price"]),
            f.FeatureSpec(op="frequency", columns=["qty"]),
        ],
    )
    assert {"cat_b", "cat_c"} <= set(out.columns) and "cat" not in out
    assert out["when_month"].min() == 1 and set(out["when_is_weekend"].dropna()) <= {0, 1}
    assert out["price_missing"].sum() == df["price"].isna().sum()
    assert out["price_ratio_qty"].notna().any()
    assert 0 < out["qty_freq"].min() <= 1


def test_leakage_report_flags_derived_and_identifier_columns(df):
    warnings = f.leakage_report(df, "target", ["id", "price", "target_copy"])
    by_col = {w.column: w for w in warnings}
    assert by_col["target_copy"].level == "high"
    assert by_col["id"].level == "medium"
    assert "price" not in by_col


def test_recipe_roundtrip_and_invalid(df):
    specs = [f.FeatureSpec(op="square", columns=["qty"]), f.FeatureSpec(op="drop", columns=["id"])]
    assert f.specs_from_json(f.specs_to_json(specs)) == specs
    with pytest.raises(f.FeatureError):
        f.specs_from_json('[{"op": "nonsense", "columns": []}]')


def test_generated_code_matches_app_output(df, tmp_path):
    specs = [
        f.FeatureSpec(op="log1p", columns=["qty"]),
        f.FeatureSpec(op="onehot", columns=["cat"]),
        f.FeatureSpec(
            op="date_parts", columns=["when"], params={"parts": ["year", "dayofweek", "is_weekend"]}
        ),
        f.FeatureSpec(op="interaction", columns=["price", "qty"], params={"kind": "product"}),
        f.FeatureSpec(op="bin_width", columns=["price"], params={"bins": 4}),
        f.FeatureSpec(op="ordinal", columns=["cat"])
        if False
        else f.FeatureSpec(op="missing_indicator", columns=["price"]),
        f.FeatureSpec(op="drop", columns=["id"]),
    ]
    csv = tmp_path / "d.csv"
    df.to_csv(csv, index=False)
    ns: dict = {}
    exec(compile(f.features_snippet(specs, filename=str(csv)), "<f>", "exec"), ns)  # noqa: S102
    expected = f.apply_specs(pd.read_csv(csv), specs)
    pd.testing.assert_frame_equal(ns["df"], expected, check_dtype=False)
