import pytest
import pandas as pd
from src.data_loader import generate_synthetic_telco_data
from src.churn_prediction import train_churn_model, predict_churn, explain_customer_risk


def test_train_churn_model():
    df = generate_synthetic_telco_data(n_samples=100, random_state=42)
    meta = train_churn_model(
        df,
        target_col="Churn",
        model_type="Logistic Regression",
        id_cols=["customerID"],
        test_size=0.2
    )
    assert meta["pipeline"] is not None
    assert len(meta["y_test_pred"]) == len(meta["y_test"])
    assert not meta["feature_importance_df"].empty

    # Test batch prediction
    pred_df = predict_churn(meta["pipeline"], df, id_cols=["customerID"], target_col="Churn")
    assert "Churn_Probability" in pred_df.columns
    assert "Risk_Category" in pred_df.columns
    assert set(pred_df["Risk_Category"].unique()).issubset({"Low", "Medium", "High"})

    # Test individual customer explanation
    first_cust = pred_df.iloc[0]
    explanation = explain_customer_risk(meta["pipeline"], first_cust, meta["feature_names"])
    assert "top_risk_drivers" in explanation
    assert "top_protective_factors" in explanation

