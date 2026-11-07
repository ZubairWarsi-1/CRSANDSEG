import pytest
import pandas as pd
from src.recommendations import generate_customer_recommendation, generate_retention_plan


def test_customer_recommendations():
    high_risk_cust = pd.Series({
        "Churn_Probability": 0.85,
        "Risk_Category": "High",
        "Segment": "Champions",
        "MonthlyCharges": 95.0,
        "tenure": 24
    })
    rec = generate_customer_recommendation(high_risk_cust)
    assert rec["Recommended_Action"] == "Personalized Win-Back Offer"
    assert rec["Action_Status"] == "Pending"
    assert rec["Priority_Score"] > 50

    new_cust = pd.Series({
        "Churn_Probability": 0.70,
        "Risk_Category": "High",
        "Segment": "Regular",
        "MonthlyCharges": 50.0,
        "tenure": 2
    })
    rec_new = generate_customer_recommendation(new_cust)
    assert rec_new["Recommended_Action"] == "Onboarding Guidance"


def test_generate_retention_plan():
    df = pd.DataFrame([
        {"CustomerID": "C1", "Churn_Probability": 0.9, "Risk_Category": "High", "MonthlyCharges": 100.0, "tenure": 12},
        {"CustomerID": "C2", "Churn_Probability": 0.1, "Risk_Category": "Low", "MonthlyCharges": 30.0, "tenure": 50}
    ])
    plan = generate_retention_plan(df)
    assert len(plan) == 2
    assert "Recommended_Action" in plan.columns
    assert "Priority_Score" in plan.columns
    # C1 must be ranked above C2 by priority score
    assert plan.iloc[0]["CustomerID"] == "C1"

