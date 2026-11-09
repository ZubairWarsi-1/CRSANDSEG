"""
Customer360 - Retention Recommendation Engine
Generates explainable, rule-based retention actions, transparent priority scores,
risk-adjusted revenue exposure, and database-backed persistent intervention tracking.
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from src.database import upsert_intervention, get_all_interventions


RECOMMENDATION_TYPES = [
    "Loyalty Rewards",
    "Personalized Win-Back Offer",
    "Onboarding Guidance",
    "Re-engagement Campaign",
    "Customer Support Follow-Up",
    "Standard Nurturing"
]

ACTION_STATUSES = ["Pending", "In Progress", "Completed"]


class PriorityBreakdownResult(dict):
    """Dictionary that also behaves as a 2-tuple (priority_score, breakdown_str) when unpacked."""
    def __iter__(self):
        yield self["priority_score"]
        yield self["breakdown"]


def calculate_priority_score_and_breakdown(
    arg1: Any,
    arg2: Optional[Any] = None,
    arg3: Optional[Any] = None,
    spending_col: Optional[str] = None,
    tenure_col: Optional[str] = None
) -> PriorityBreakdownResult:
    """
    Transparent, explainable multi-factor customer priority score (0 - 100):
      - Risk Component (0 - 50 pts): Churn Probability * 50
      - Financial Component (0 - 35 pts): (Monthly Spending / 120) * 35
      - Tenure Urgency (0 - 15 pts): (1 - Tenure / 72) * 15
    Supports both row-based and parameter-based invocation.
    """
    if isinstance(arg1, (pd.Series, dict)):
        row = arg1
        churn_prob = float(row.get("Churn_Probability", 0.5 if str(row.get("Churn", "")).lower() == "yes" else 0.2))
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
    else:
        churn_prob = float(arg1)
        spending = float(arg2) if arg2 is not None else 0.0
        tenure = float(arg3) if arg3 is not None else None

    risk_pts = round(churn_prob * 50.0, 1)
    val_pts = round(min(35.0, max(0.0, (spending / 120.0) * 35.0)), 1)
    
    ten_pts = 0.0
    if tenure is not None and pd.notna(tenure):
        ten_pts = round(max(0.0, (1.0 - min(float(tenure), 72.0) / 72.0) * 15.0), 1)

    total_score = round(min(100.0, risk_pts + val_pts + ten_pts), 1)
    breakdown_str = f"{risk_pts} pts (Risk) + {val_pts} pts (Value) + {ten_pts} pts (Tenure)"

    return PriorityBreakdownResult({
        "priority_score": total_score,
        "risk_points": risk_pts,
        "value_points": val_pts,
        "tenure_points": ten_pts,
        "breakdown": breakdown_str,
        "breakdown_str": breakdown_str
    })



def generate_customer_recommendation(
    row: pd.Series,
    spending_col: Optional[str] = None,
    tenure_col: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate a tailored retention strategy, explainable rationale,
    revenue-at-risk, and transparent priority score for a single customer record.
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

    # Expected Monthly Revenue Exposure = Spending * Churn Probability
    revenue_at_risk = round(spending * churn_prob, 2)

    # Transparent Priority Score
    priority_score, priority_breakdown = calculate_priority_score_and_breakdown(churn_prob, spending, tenure)

    # Decision Logic
    action = "Standard Nurturing"
    reason = "Customer behavior is steady with low current churn risk."

    # 1. High Risk (>= 0.65)
    if risk_cat == "High":
        if spending >= 70 or "Champion" in segment or "Loyal" in segment or "Premier" in segment:
            action = "Personalized Win-Back Offer"
            reason = f"High churn risk ({round(churn_prob*100, 1)}%) with premium monthly spend (${spending:,.2f}). Immediate executive discount or contract upgrade required."
        elif tenure is not None and tenure <= 6:
            action = "Onboarding Guidance"
            reason = f"New customer ({int(tenure)} months tenure) with elevated churn risk ({round(churn_prob*100, 1)}%). Requires proactive onboarding support and satisfaction check."
        else:
            action = "Customer Support Follow-Up"
            reason = f"High predicted attrition risk ({round(churn_prob*100, 1)}%). Support team outreach recommended to resolve service or billing friction."

    # 2. Medium Risk (0.35 - 0.65)
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

    # 3. Low Risk (< 0.35)
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
        "Priority_Breakdown": priority_breakdown,
        "Revenue_at_Risk": revenue_at_risk,
        "Action_Status": "Pending"
    }


def generate_retention_plan(
    df: pd.DataFrame,
    spending_col: Optional[str] = None,
    tenure_col: Optional[str] = None,
    id_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Generate the full retention plan dataframe for all customers in the dataset.
    Syncs and persists statuses in SQLite database.
    """
    recs = []
    for _, row in df.iterrows():
        rec = generate_customer_recommendation(row, spending_col=spending_col, tenure_col=tenure_col)
        recs.append(rec)

    rec_df = pd.DataFrame(recs)
    result = pd.concat([df.reset_index(drop=True), rec_df], axis=1)

    # Sync with SQLite database for persistence across sessions/restarts
    try:
        db_interventions = get_all_interventions()
        resolved_id_col = id_col or ("customerID" if "customerID" in result.columns else None)

        if resolved_id_col and resolved_id_col in result.columns:
            for idx, row in result.iterrows():
                cid = str(row[resolved_id_col])
                if cid in db_interventions:
                    # Update with saved status
                    result.at[idx, "Action_Status"] = db_interventions[cid]["action_status"]
                else:
                    # Persist initial record
                    spend_val = float(row.get("MonthlyCharges", row.get(spending_col or "", 0.0)) or 0.0)
                    prob_val = float(row.get("Churn_Probability", 0.0) or 0.0)
                    upsert_intervention(
                        customer_id=cid,
                        action_status="Pending",
                        recommended_action=row.get("Recommended_Action"),
                        recommendation_reason=row.get("Recommendation_Reason"),
                        priority_score=float(row.get("Priority_Score", 0.0)),
                        churn_probability=prob_val,
                        risk_category=str(row.get("Risk_Category", "Low")),
                        monthly_charges=spend_val
                    )
    except Exception:
        pass

    # Sort primarily by Priority Score descending
    result = result.sort_values(by="Priority_Score", ascending=False).reset_index(drop=True)
    return result


def preview_communication_template(
    arg1: Any = "Email",
    arg2: Optional[Any] = None,
    arg3: Optional[Any] = None,
    channel: str = "Email",
    action_type: str = "Personalized Win-Back Offer",
    customer_id: str = "CUST-0001",
    customer_name: Optional[str] = None,
    custom_offer: Optional[str] = None,
    discount_pct: Optional[int] = None
) -> Dict[str, str]:
    """
    Generate a safe communication preview for Email or SMS.
    Supports row-based or explicit parameter-based calling.
    Does not send real messages; serves as a governance and review layer.
    """
    if isinstance(arg1, (pd.Series, dict)):
        row = arg1
        chan = channel or "Email"
        disc = discount_pct or 15
        act_type = str(row.get("Recommended_Action", action_type))
        cid = str(row.get("customerID", row.get("id", row.get("CustomerID", "CUST-0001"))))
        offer = custom_offer or f"{disc}% monthly discount for 12 months with complimentary priority tech support"
        name = customer_name or f"Customer {cid}"
    else:
        chan = str(arg1) if arg1 else channel
        act_type = str(arg2) if arg2 else action_type
        cid = str(arg3) if arg3 else customer_id
        disc = discount_pct or 15
        offer = custom_offer or f"{disc}% monthly discount for 12 months with priority technical support"
        name = customer_name or f"Customer {cid}"

    if str(chan).lower() == "sms":
        if act_type == "Personalized Win-Back Offer":
            text = f"Hi {name}, we value your business! Renew today and receive: {offer}. Reply YES to claim or call 1-800-CUSTOMER."
        elif act_type == "Onboarding Guidance":
            text = f"Hello {name}, welcome! Need help setting up your services? Schedule your free 1-on-1 onboarding session here: https://support.example.com/onboard"
        elif act_type == "Loyalty Rewards":
            text = f"Thank you for being a loyal customer, {name}! Enjoy your exclusive VIP loyalty perk: {offer}."
        else:
            text = f"Hi {name}, we're checking in to ensure your services are running smoothly. Let us know if you need assistance!"

        return {
            "channel": "SMS",
            "recipient": cid,
            "character_count": str(len(text)),
            "subject": "SMS Notification",
            "body": text
        }
    else:  # Email
        subject = f"Special Customer360 Offer for {name}"
        if act_type == "Personalized Win-Back Offer":
            body = (
                f"Dear {name},\n\n"
                f"We noticed you may be reviewing your subscription options. As one of our most valued customers, "
                f"we would like to extend a special personalized retention package:\n\n"
                f"  ★ {offer}\n\n"
                f"Our priority care team is standing by to ensure your service exceeds your expectations.\n\n"
                f"Warm regards,\nCustomer360 Retention Team"
            )
        elif act_type == "Onboarding Guidance":
            body = (
                f"Dear {name},\n\n"
                f"Welcome to our service community! We want to make sure your first few months are effortless. "
                f"Our onboarding specialists are available to guide you through setup and optimize your features.\n\n"
                f"Visit our customer resource center or reply directly to this email.\n\n"
                f"Warm regards,\nCustomer Success Team"
            )
        else:
            body = (
                f"Dear {name},\n\n"
                f"Thank you for choosing our platform. In recognition of your continued partnership, "
                f"we have applied special rewards to your account: {offer}.\n\n"
                f"Warm regards,\nCustomer Care Team"
            )

        return {
            "channel": "Email",
            "recipient": cid,
            "character_count": str(len(body)),
            "subject": subject,
            "body": body
        }

