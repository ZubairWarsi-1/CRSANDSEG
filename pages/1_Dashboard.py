"""
Customer360 — Page 1: Dashboard Overview
Executive KPI Cards, Segment Distributions, and Dynamic Business Insights.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.ui_components import (
    apply_custom_css,
    render_page_header,
    render_sidebar,
    render_kpi,
    get_plotly_layout,
    init_session_state,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_SUCCESS,
    COLOR_DANGER,
    COLOR_WARNING
)
from src.validation import infer_column_roles

st.set_page_config(page_title="Dashboard Overview — Customer360", page_icon="📊", layout="wide")
apply_custom_css()
init_session_state()
render_sidebar()

render_page_header(
    title="Dashboard Overview",
    subtitle="High-level customer health metrics, portfolio churn risk, and automated business insights.",
    badge="Executive View"
)

df = st.session_state.get("current_dataset")

if df is None or df.empty:
    st.error("No dataset currently loaded. Please go to the **Data Upload** page or select a sample dataset from the sidebar.")
    st.stop()

# Detect roles
roles = infer_column_roles(df)
target_col = roles["target_col"]
monetary_col = roles["monetary_col"]
tenure_col = roles["tenure_col"]
id_col = roles["id_col"]

# Calculate KPIs
total_customers = len(df)

# Active vs Inactive / Churn
active_count = "N/A"
inactive_count = "N/A"
churn_rate_str = "N/A"
churn_rate_val = None

if target_col:
    # Look for Yes/No or 1/0
    counts = df[target_col].astype(str).str.strip().str.lower().value_counts()
    churn_cnt = 0
    retain_cnt = 0
    for k, v in counts.items():
        if k in ["yes", "1", "true", "churned", "churn"]:
            churn_cnt += v
        elif k in ["no", "0", "false", "retained", "active"]:
            retain_cnt += v
    
    if churn_cnt + retain_cnt > 0:
        churn_rate_val = (churn_cnt / (churn_cnt + retain_cnt)) * 100
        churn_rate_str = f"{churn_rate_val:.1f}%"
        active_count = f"{retain_cnt:,}"
        inactive_count = f"{churn_cnt:,}"

# Average Spending
avg_spend_str = "N/A"
total_rev_str = "N/A"
if monetary_col and pd.api.types.is_numeric_dtype(df[monetary_col]):
    avg_spend = df[monetary_col].dropna().mean()
    total_rev = df[monetary_col].dropna().sum()
    avg_spend_str = f"${avg_spend:,.2f}"
    total_rev_str = f"${total_rev:,.2f}"

# High Risk Customers
high_risk_str = "N/A"
pred_df = st.session_state.get("predictions_df")
if pred_df is not None and "Risk_Category" in pred_df.columns:
    high_risk_cnt = (pred_df["Risk_Category"] == "High").sum()
    high_risk_str = f"{high_risk_cnt:,} ({(high_risk_cnt/len(pred_df))*100:.1f}%)"
elif churn_rate_val is not None:
    high_risk_str = f"{inactive_count} ({churn_rate_str})"

# Render Top KPIs
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

with kpi1:
    render_kpi("Total Customers", f"{total_customers:,}", f"Active ID: {id_col or 'Index'}")

with kpi2:
    render_kpi("Active Customers", active_count, "Retained accounts")

with kpi3:
    render_kpi("Average Spending", avg_spend_str, f"Field: {monetary_col or 'Unavailable'}")

with kpi4:
    render_kpi("Historical Churn Rate", churn_rate_str, f"Target: {target_col or 'Unavailable'}")

with kpi5:
    render_kpi("High-Risk Accounts", high_risk_str, "Needs intervention")

st.markdown("<br>", unsafe_allow_html=True)

# Main Visualizations Row
chart_col1, chart_col2 = st.columns([1, 1])

with chart_col1:
    st.markdown("#### 🍩 Customer Churn / Status Breakdown")
    if target_col:
        dist_df = df[target_col].value_counts().reset_index()
        dist_df.columns = ["Status", "Count"]
        fig_pie = px.pie(
            dist_df,
            names="Status",
            values="Count",
            hole=0.45,
            color="Status",
            color_discrete_map={"Yes": COLOR_DANGER, "No": COLOR_SUCCESS, "1": COLOR_DANGER, "0": COLOR_SUCCESS}
        )
        fig_pie.update_layout(get_plotly_layout(height=350))
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("Target status column not detected in current dataset.")

with chart_col2:
    st.markdown("#### 📊 Customer Tenure Distribution")
    if tenure_col and pd.api.types.is_numeric_dtype(df[tenure_col]):
        fig_hist = px.histogram(
            df,
            x=tenure_col,
            nbins=30,
            color_discrete_sequence=[COLOR_PRIMARY],
            labels={tenure_col: f"Tenure ({tenure_col})"}
        )
        fig_hist.update_layout(get_plotly_layout(height=350))
        fig_hist.update_yaxes(title_text="Customer Count")
        st.plotly_chart(fig_hist, use_container_width=True)
    elif "TransactionDate" in df.columns:
        # Time series of transactions
        ts_df = df.copy()
        ts_df["Month"] = pd.to_datetime(ts_df["TransactionDate"]).dt.to_period("M").astype(str)
        monthly_tx = ts_df.groupby("Month").size().reset_index(name="Transactions")
        fig_ts = px.line(monthly_tx, x="Month", y="Transactions", markers=True, color_discrete_sequence=[COLOR_PRIMARY])
        fig_ts.update_layout(get_plotly_layout(height=350))
        st.plotly_chart(fig_ts, use_container_width=True)
    else:
        st.info("Tenure or transaction dates unavailable for time trend display.")

# Second Visualization Row: Spending vs Churn / Contracts
vis_col1, vis_col2 = st.columns([1, 1])

with vis_col1:
    st.markdown("#### 💳 Monthly Charges by Churn Status")
    if target_col and monetary_col and pd.api.types.is_numeric_dtype(df[monetary_col]):
        fig_box = px.box(
            df,
            x=target_col,
            y=monetary_col,
            color=target_col,
            color_discrete_map={"Yes": COLOR_DANGER, "No": COLOR_SUCCESS, "1": COLOR_DANGER, "0": COLOR_SUCCESS},
            points="outliers",
            labels={target_col: "Churn Status", monetary_col: f"Spending ({monetary_col})"}
        )
        fig_box.update_layout(get_plotly_layout(height=350))
        st.plotly_chart(fig_box, use_container_width=True)
    elif monetary_col and pd.api.types.is_numeric_dtype(df[monetary_col]):
        fig_box = px.histogram(df, x=monetary_col, nbins=30, color_discrete_sequence=[COLOR_SECONDARY])
        fig_box.update_layout(get_plotly_layout(height=350))
        st.plotly_chart(fig_box, use_container_width=True)
    else:
        st.info("Monetary spending data unavailable for charge analysis.")

with vis_col2:
    st.markdown("#### 📑 Contract Type Impact on Churn")
    contract_col = next((c for c in df.columns if "contract" in c.lower()), None)
    if contract_col and target_col:
        contract_churn = df.groupby([contract_col, target_col]).size().reset_index(name="Count")
        fig_bar = px.bar(
            contract_churn,
            x=contract_col,
            y="Count",
            color=target_col,
            barmode="group",
            color_discrete_map={"Yes": COLOR_DANGER, "No": COLOR_SUCCESS, "1": COLOR_DANGER, "0": COLOR_SUCCESS}
        )
        fig_bar.update_layout(get_plotly_layout(height=350))
        st.plotly_chart(fig_bar, use_container_width=True)
    elif "TotalAmount" in df.columns and "CustomerID" in df.columns:
        top_buyers = df.groupby("CustomerID")["TotalAmount"].sum().nlargest(10).reset_index()
        fig_top = px.bar(top_buyers, x="CustomerID", y="TotalAmount", color_discrete_sequence=[COLOR_PRIMARY])
        fig_top.update_layout(get_plotly_layout(height=350))
        st.plotly_chart(fig_top, use_container_width=True)
    else:
        st.info("Contract structure or customer group column unavailable.")

# Automated Business Insights Section
st.markdown("### 💡 Automated Business Insights (Derived from Data)")

insights = []

# Insight 1: Contract Analysis
if contract_col and target_col:
    m2m_df = df[df[contract_col].astype(str).str.contains("Month", case=False, na=False)]
    two_yr_df = df[df[contract_col].astype(str).str.contains("Two", case=False, na=False)]
    
    def get_rate(subset):
        if len(subset) == 0:
            return 0
        churns = subset[target_col].astype(str).str.strip().str.lower().isin(["yes", "1", "true"]).sum()
        return (churns / len(subset)) * 100

    m2m_rate = get_rate(m2m_df)
    two_yr_rate = get_rate(two_yr_df)
    if m2m_rate > 0 and two_yr_rate > 0:
        multiplier = round(m2m_rate / max(1, two_yr_rate), 1)
        insights.append(
            f"**Contract Vulnerability:** Month-to-month contracts experience a **{m2m_rate:.1f}%** churn rate, compared to just **{two_yr_rate:.1f}%** for two-year contracts (**{multiplier}x** higher risk)."
        )

# Insight 2: High Spending Churn
if monetary_col and target_col and pd.api.types.is_numeric_dtype(df[monetary_col]):
    churned_spend = df[df[target_col].astype(str).str.lower().isin(["yes", "1"])][monetary_col].dropna().mean()
    retained_spend = df[df[target_col].astype(str).str.lower().isin(["no", "0"])][monetary_col].dropna().mean()
    if pd.notna(churned_spend) and pd.notna(retained_spend):
        insights.append(
            f"**Spending Correlation:** Churning customers average **${churned_spend:.2f}** per month versus **${retained_spend:.2f}** for retained customers, showing attrition is concentrated among higher-billing accounts."
        )

# Insight 3: Tenure Protection
if tenure_col and target_col and pd.api.types.is_numeric_dtype(df[tenure_col]):
    early_tenure = df[df[tenure_col] <= 12]
    early_churn_rate = (early_tenure[target_col].astype(str).str.lower().isin(["yes", "1"]).sum() / max(1, len(early_tenure))) * 100
    insights.append(
        f"**First-Year Drop-Off:** Customers in their first 12 months experience a **{early_churn_rate:.1f}%** churn rate, highlighting onboarding as the primary retention battleground."
    )

# Insight 4: Internet service or retail
internet_col = next((c for c in df.columns if "internet" in c.lower()), None)
tech_col = next((c for c in df.columns if "techsupport" in c.lower()), None)
if internet_col and tech_col and target_col:
    fiber_no_support = df[(df[internet_col].astype(str).str.contains("Fiber", case=False)) & (df[tech_col].astype(str).str.lower() == "no")]
    if len(fiber_no_support) > 0:
        fns_churn = (fiber_no_support[target_col].astype(str).str.lower().isin(["yes", "1"]).sum() / len(fiber_no_support)) * 100
        insights.append(
            f"**Service Bottleneck:** Fiber Optic subscribers without Tech Support have an alarming **{fns_churn:.1f}%** churn rate. Bundling support could immediately protect this customer cohort."
        )

if not insights:
    insights.append("Analyze dataset columns in **Customer Segmentation** and **Churn Prediction** to generate custom behavioral patterns.")

for ins in insights:
    st.info(f"📌 {ins}")
