"""
Customer360 - Retention Recommendation Engine
Generates explainable, rule-based retention actions and priority scoring for customers.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


RECOMMENDATION_TYPES = [
    "Loyalty Rewards",
    "Personalized Win-Back Offer",
    "Onboarding Guidance",
    "Re-engagement Campaign",
    "Customer Support Follow-Up",
    "Standard Nurturing"
]

ACTION_STATUSES = ["Pending", "In Progress", "Completed"]


def generate_customer_recommendation(
    row: pd.Series,
    spending_col: Optional[str] = None,
    tenure_col: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate a tailored retention strategy and explainable rationale for a single customer record.
    """
    churn_prob = float(row.get("Churn_Probability", 0.5 if str(row.get("Churn", "")).lower() == "yes" else 0.2))
    risk_cat = str(row.get("Risk_Category", "High" if churn_prob >= 0.65 else ("Medium" if churn_prob >= 0.35 else "Low")))
    segment = str(row.get("Segment", row.get("ClusterName", "Regular")))

    spending = 0.0
    if spending_col and spending_col in row and pd.notna(row[spending_col]):
        try:
            spending = float(row[spending_col])
        except Exception:
            spending = 0.0
    elif "MonthlyCharges" in row and pd.notna(row["MonthlyCharges"]):
        try:
            spending = float(row["MonthlyCharges"])
        except Exception:
            pass
    elif "Monetary" in row and pd.notna(row["Monetary"]):
        try:
            spending = float(row["Monetary"])
        except Exception:
            pass

    tenure = None
    if tenure_col and tenure_col in row and pd.notna(row[tenure_col]):
        try:
            tenure = float(row[tenure_col])
        except Exception:
            pass
    elif "tenure" in row and pd.notna(row["tenure"]):
        try:
            tenure = float(row["tenure"])
        except Exception:
            pass

    # Priority Score: combines churn probability and financial impact
    if spending > 0:
        priority_score = round(churn_prob * min(100.0, (spending / 1.5)), 1)
    else:
        priority_score = round(churn_prob * 100.0, 1)

    # Decision Logic
    action = "Standard Nurturing"
    reason = "Customer behavior is steady with low current churn risk."

    # 1. High Risk & High Value -> Win-back offer
    if risk_cat == "High":
        if spending >= 70 or "Champion" in segment or "Loyal" in segment:
            action = "Personalized Win-Back Offer"
            reason = f"High churn risk ({round(churn_prob*100, 1)}%) with premium customer value (${spending:,.2f}). Immediate executive discount or contract upgrade required."
        elif tenure is not None and tenure <= 6:
            action = "Onboarding Guidance"
            reason = f"Customer has new tenure ({int(tenure)} months) and elevated churn risk ({round(churn_prob*100, 1)}%). Requires proactive onboarding support and satisfaction check."
        else:
            action = "Customer Support Follow-Up"
            reason = f"High predicted attrition risk ({round(churn_prob*100, 1)}%). Proactive contact by support team recommended to resolve technical or service bottlenecks."

    # 2. Medium Risk
    elif risk_cat == "Medium":
        if "At-Risk" in segment or "Lost" in segment:
            action = "Re-engagement Campaign"
            reason = "Customer engagement has dipped below baseline. Re-engage with an exclusive feature preview or discount coupon."
        elif "Champion" in segment or "Loyal" in segment or spending >= 65:
            action = "Loyalty Rewards"
            reason = f"Valued customer with moderate churn probability ({round(churn_prob*100, 1)}%). Lock in loyalty with VIP anniversary perks or points bonus."
        elif tenure is not None and tenure <= 6:
            action = "Onboarding Guidance"
            reason = f"Early tenure customer ({int(tenure)} months). Provide setup assistance and product tutorials to reinforce adoption."
        else:
            action = "Customer Support Follow-Up"
            reason = f"Moderate churn risk ({round(churn_prob*100, 1)}%). Send a satisfaction survey to identify points of friction."

    # 3. Low Risk
    else:
        if "Champion" in segment or "Loyal" in segment or spending >= 80:
            action = "Loyalty Rewards"
            reason = f"Premier customer with strong loyalty and low churn risk ({round(churn_prob*100, 1)}%). Nurture with VIP recognition and referral incentives."
        elif "At-Risk" in segment or "Inactive" in segment:
            action = "Re-engagement Campaign"
            reason = "Customer has low churn model score but has shown declining transaction frequency in segmentation."
        else:
            action = "Standard Nurturing"
            reason = f"Healthy customer relationship with low churn probability ({round(churn_prob*100, 1)}%). Maintain standard lifecycle communications."

    return {
        "Recommended_Action": action,
        "Recommendation_Reason": reason,
        "Priority_Score": priority_score,
        "Action_Status": "Pending"
    }


def generate_retention_plan(
    df: pd.DataFrame,
    spending_col: Optional[str] = None,
    tenure_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Generate the full retention plan dataframe for all customers in the dataset.
    """
    recs = []
    for _, row in df.iterrows():
        rec = generate_customer_recommendation(row, spending_col=spending_col, tenure_col=tenure_col)
        recs.append(rec)

    rec_df = pd.DataFrame(recs)
    result = pd.concat([df.reset_index(drop=True), rec_df], axis=1)

    # Sort primarily by Priority Score descending
    result = result.sort_values(by="Priority_Score", ascending=False).reset_index(drop=True)
    return result

