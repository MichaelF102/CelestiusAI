"""Tests for automatic schema inference, semantic type detection, and heuristics."""

import pandas as pd
from src.ingestion.schema import infer_dataset_schema, normalize_column_name


def test_normalize_column_name():
    """Verify conversion of human-readable column names into clean snake_case."""
    assert normalize_column_name("Customer ID") == "customer_id"
    assert normalize_column_name("Total Revenue ($)") == "total_revenue"
    assert normalize_column_name("  unit-cost  ") == "unit_cost"
    assert normalize_column_name("123") == "123"


def test_schema_inference_numeric_and_identifier():
    """Verify distinction between continuous quantitative metrics and entity keys."""
    df = pd.DataFrame(
        {
            "order_id": [1001, 1002, 1003, 1004, 1005],  # Identifier pattern + high uniqueness
            "sales_amount": [12.5, 45.0, 99.99, 15.0, 8.5],  # Quantitative numeric
            "is_member": [True, False, True, True, False],  # Native boolean
        }
    )
    schemas = infer_dataset_schema(df)

    assert schemas["order_id"].inferred_type == "identifier"
    assert schemas["order_id"].is_identifier is True

    assert schemas["sales_amount"].inferred_type == "numeric"
    assert schemas["sales_amount"].is_identifier is False

    assert schemas["is_member"].inferred_type == "boolean"


def test_schema_inference_datetime_candidate():
    """Verify text dates are identified as datetime candidates."""
    df = pd.DataFrame(
        {
            "transaction_date": [
                "2024-01-15",
                "2024-02-20",
                "2024-03-25",
                "2024-04-10",
            ],
            "category": ["A", "B", "A", "B"],
        }
    )
    schemas = infer_dataset_schema(df)

    assert schemas["transaction_date"].inferred_type == "datetime"
    assert schemas["category"].inferred_type == "categorical"


def test_schema_inference_constant_and_near_constant():
    """Verify constant (cardinality 1) and near-constant detection."""
    df = pd.DataFrame(
        {
            "always_one": [1] * 20,
            "mostly_a": ["A"] * 19 + ["B"],  # 95% 'A' across 20 rows
            "balanced": ["X", "Y"] * 10,
        }
    )
    schemas = infer_dataset_schema(df)

    assert schemas["always_one"].is_constant is True
    assert schemas["always_one"].is_near_constant is True

    assert schemas["mostly_a"].is_constant is False
    assert schemas["mostly_a"].is_near_constant is True

    assert schemas["balanced"].is_near_constant is False


def test_schema_inference_potential_target():
    """Verify detection of prospective prediction target columns."""
    df = pd.DataFrame(
        {
            "churn_status": [0, 1, 0, 0, 1] * 10,
            "product_price": [10.0, 20.0, 30.0, 40.0, 50.0] * 10,
            "random_col": [1, 2, 3, 4, 5] * 10,
        }
    )
    schemas = infer_dataset_schema(df)

    assert schemas["churn_status"].is_potential_target is True
    assert schemas["product_price"].is_potential_target is True


def test_schema_inference_100_percent_null():
    """Verify handling of completely null columns."""
    df = pd.DataFrame({"all_missing": [None, None, None]})
    schemas = infer_dataset_schema(df)

    assert schemas["all_missing"].null_percentage == 100.0
    assert schemas["all_missing"].non_null_count == 0
    assert schemas["all_missing"].is_constant is True
