import pytest
import pandas as pd
from src.validation import (
    inspect_dataset,
    infer_column_roles,
    validate_for_churn_prediction,
    validate_for_rfm,
    validate_for_clustering
)


def test_inspect_dataset():
    df = pd.DataFrame({
        "id": ["1", "2", "3", "3"],
        "age": [25, 30, 30, 30],
        "category": ["A", "B", "A", "A"]
    })
    res = inspect_dataset(df)
    assert res["row_count"] == 4
    assert res["col_count"] == 3
    assert res["duplicate_count"] == 1
    assert res["quality_score"] > 0
    assert "numeric_cols" in res
    assert "age" in res["numeric_cols"]


def test_infer_column_roles():
    df = pd.DataFrame({
        "customerID": ["C1", "C2", "C3"],
        "tenure": [1, 2, 3],
        "MonthlyCharges": [50.0, 60.0, 70.0],
        "Churn": ["No", "Yes", "No"]
    })
    roles = infer_column_roles(df)
    assert roles["id_col"] == "customerID"
    assert roles["target_col"] == "Churn"
    assert roles["tenure_col"] == "tenure"
    assert roles["monetary_col"] == "MonthlyCharges"


def test_validate_for_churn():
    df = pd.DataFrame({
        "target": ["Yes"] * 20 + ["No"] * 20
    })
    is_valid, msg, mapping = validate_for_churn_prediction(df, "target")
    assert is_valid is True
    assert mapping["Yes"] == 1
    assert mapping["No"] == 0

    # Invalid with 3 classes
    df_invalid = pd.DataFrame({"target": ["A", "B", "C"] * 15})
    is_valid, msg, _ = validate_for_churn_prediction(df_invalid, "target")
    assert is_valid is False


def test_validate_for_rfm():
    df = pd.DataFrame({
        "cust": ["A", "B"],
        "date": ["2026-01-01", "2026-01-02"],
        "amount": [10.5, 20.0]
    })
    is_valid, _ = validate_for_rfm(df, "cust", "date", "amount")
    assert is_valid is True

    # Missing column
    is_valid, _ = validate_for_rfm(df, "cust", "non_existent", "amount")
    assert is_valid is False


def test_validate_for_clustering():
    df = pd.DataFrame({
        "f1": [1, 2, 3, 4],
        "f2": [10.0, 20.0, 30.0, 40.0],
        "constant": [1, 1, 1, 1]
    })
    is_valid, _ = validate_for_clustering(df, ["f1", "f2"])
    assert is_valid is True

    # Zero variance column
    is_valid, _ = validate_for_clustering(df, ["f1", "constant"])
    assert is_valid is False
