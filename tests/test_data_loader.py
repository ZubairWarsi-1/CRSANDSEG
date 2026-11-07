import pytest
import pandas as pd
from src.data_loader import (
    load_dataset,
    generate_synthetic_telco_data,
    generate_synthetic_retail_transactions,
    get_sample_telco_data,
    get_sample_retail_data
)


def test_generate_synthetic_telco():
    df = generate_synthetic_telco_data(n_samples=50, random_state=42)
    assert len(df) == 50
    assert "customerID" in df.columns
    assert "Churn" in df.columns
    assert "MonthlyCharges" in df.columns
    assert set(df["Churn"].unique()).issubset({"Yes", "No"})


def test_generate_synthetic_retail():
    df = generate_synthetic_retail_transactions(n_customers=20, n_transactions=100, random_state=42)
    assert len(df) == 100
    assert "CustomerID" in df.columns
    assert "TransactionDate" in df.columns
    assert "TotalAmount" in df.columns
    assert df["CustomerID"].nunique() <= 20


def test_load_dataset_from_csv(tmp_path):
    test_csv = tmp_path / "test.csv"
    test_df = pd.DataFrame({
        " CustomerID ": ["C1", "C2", "C3"],
        "Tenure": [12, 24, 36],
        " TotalCharges ": [" 100.5 ", " 200.0 ", " 350.2 "]
    })
    test_df.to_csv(test_csv, index=False)

    loaded = load_dataset(str(test_csv))
    # Column headers should be stripped
    assert "CustomerID" in loaded.columns
    assert "TotalCharges" in loaded.columns
    # Whitespace numeric should be coerced to float
    assert pd.api.types.is_numeric_dtype(loaded["TotalCharges"])

