"""
Customer360 — Page 1: Dashboard Overview
Executive KPI Cards, Revenue at Risk Exposure, Risk Tier Breakdown, and Structured Business Insights.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

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
from src.churn_prediction import predict_churn

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

# Automatically generate predictions if pipeline exists and predictions_df not ready
pred_df = st.session_state.get("predictions_df")
if pred_df is None or len(pred_df) != len(df):
    model_paths = [
        os.path.join("models", "churn_logistic_regression_pipeline.joblib"),
        os.path.join("models", "churn_random_forest_pipeline.joblib")
    ]
    for m_path in model_paths:
        if os.path.exists(m_path) and target_col:
            try:
                pipeline = joblib.load(m_path)
                id_cols = [id_col] if id_col else []
                pred_df = predict_churn(pipeline, df, id_cols=id_cols, target_col=target_col)
                st.session_state["predictions_df"] = pred_df
                break
            except Exception:
                pass

# Merge prediction fields into working view if available
view_df = df.copy()
if pred_df is not None and "Churn_Probability" in pred_df.columns:
    if id_col and id_col in view_df.columns and id_col in pred_df.columns:
        cols_to_add = [c for c in ["Churn_Probability", "Predicted_Churn", "Risk_Category"] if c in pred_df.columns and c not in view_df.columns]
        if cols_to_add:
            view_df = view_df.merge(pred_df[[id_col] + cols_to_add], on=id_col, how="left")
    elif len(view_df) == len(pred_df):
        for col_name in ["Churn_Probability", "Predicted_Churn", "Risk_Category"]:
            if col_name in pred_df.columns:
                view_df[col_name] = pred_df[col_name].values

# -------------------------------------------------------------
# Interactive Dashboard Filters
# -------------------------------------------------------------
with st.expander("🔎 Interactive Portfolio Filters", expanded=True):
    f_cols = st.columns(5)
    
    # 1. Contract Filter
    contract_col = next((c for c in view_df.columns if "contract" in c.lower()), None)
    selected_contracts = []
    with f_cols[0]:
        if contract_col:
            avail_contracts = sorted(view_df[contract_col].dropna().unique().tolist())
            selected_contracts = st.multiselect("Contract Type", options=avail_contracts, default=avail_contracts)
        else:
            st.selectbox("Contract Type", ["All Contracts (N/A)"], disabled=True)
            
    # 2. Tenure Range Slider
    selected_tenure_range = None
    with f_cols[1]:
        if tenure_col and pd.api.types.is_numeric_dtype(view_df[tenure_col]):
            t_min = int(view_df[tenure_col].min())
            t_max = int(view_df[tenure_col].max())
            if t_min < t_max:
                selected_tenure_range = st.slider("Tenure Range (Months)", min_value=t_min, max_value=t_max, value=(t_min, t_max))
            else:
                st.text_input("Tenure", f"{t_min} months", disabled=True)
        else:
            st.text_input("Tenure Range", "N/A", disabled=True)
            
    # 3. Internet Service Filter
    internet_col = next((c for c in view_df.columns if "internet" in c.lower()), None)
    selected_internet = []
    with f_cols[2]:
        if internet_col:
            avail_net = sorted(view_df[internet_col].dropna().unique().tolist())
            selected_internet = st.multiselect("Internet Service", options=avail_net, default=avail_net)
        else:
            st.selectbox("Internet Service", ["All Services (N/A)"], disabled=True)
            
    # 4. Historical Churn Status
    selected_churn_status = "All"
    with f_cols[3]:
        if target_col:
            selected_churn_status = st.selectbox("Historical Churn", ["All", "Retained / Active", "Churned"])
        else:
            st.selectbox("Historical Churn", ["All (No Target)"], disabled=True)
            
    # 5. Risk Category Filter
    selected_risk_cat = "All"
    with f_cols[4]:
        if "Risk_Category" in view_df.columns:
            selected_risk_cat = st.selectbox("Predicted Risk Tier", ["All", "High Risk (≥65%)", "Medium Risk (35-65%)", "Low Risk (<35%)"])
        else:
            st.selectbox("Predicted Risk Tier", ["All (Run Model)"], disabled=True)

# Apply filters
filtered_df = view_df.copy()

if contract_col and selected_contracts:
    filtered_df = filtered_df[filtered_df[contract_col].isin(selected_contracts)]

if tenure_col and selected_tenure_range:
    filtered_df = filtered_df[
        (filtered_df[tenure_col] >= selected_tenure_range[0]) & 
        (filtered_df[tenure_col] <= selected_tenure_range[1])
    ]

if internet_col and selected_internet:
    filtered_df = filtered_df[filtered_df[internet_col].isin(selected_internet)]

if target_col and selected_churn_status != "All":
    is_churn_mask = filtered_df[target_col].astype(str).str.strip().str.lower().isin(["yes", "1", "true", "churned"])
    if selected_churn_status == "Churned":
        filtered_df = filtered_df[is_churn_mask]
    elif selected_churn_status == "Retained / Active":
        filtered_df = filtered_df[~is_churn_mask]

if "Risk_Category" in filtered_df.columns and selected_risk_cat != "All":
    cat_target = selected_risk_cat.split(" ")[0]
    filtered_df = filtered_df[filtered_df["Risk_Category"] == cat_target]

if len(filtered_df) == 0:
    st.warning("⚠️ No customer records match the selected filter criteria. Showing unfiltered dataset instead.")
    filtered_df = view_df.copy()

# -------------------------------------------------------------
# Metric Calculations
# -------------------------------------------------------------
total_customers = len(filtered_df)
portfolio_total = len(view_df)

# Active vs Inactive / Historical Churn
active_count = "N/A"
inactive_count = "N/A"
churn_rate_str = "N/A"
churn_rate_val = None

if target_col and target_col in filtered_df.columns:
    counts = filtered_df[target_col].astype(str).str.strip().str.lower().value_counts()
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
total_monthly_spend = 0.0
if monetary_col and monetary_col in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df[monetary_col]):
    avg_spend = filtered_df[monetary_col].dropna().mean()
    total_monthly_spend = filtered_df[monetary_col].dropna().sum()
    avg_spend_str = f"${avg_spend:,.2f}"

# High Risk Accounts & Revenue at Risk
high_risk_str = "N/A"
revenue_at_risk_str = "N/A"
rev_exposure_total = 0.0

if "Risk_Category" in filtered_df.columns and "Churn_Probability" in filtered_df.columns:
    high_risk_cnt = (filtered_df["Risk_Category"] == "High").sum()
    high_risk_pct = (high_risk_cnt / total_customers) * 100 if total_customers > 0 else 0
    high_risk_str = f"{high_risk_cnt:,} ({high_risk_pct:.1f}%)"
    
    if monetary_col and monetary_col in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df[monetary_col]):
        # Expected Monthly Revenue Exposure = Sum(MonthlyCharges * Churn_Probability)
        rev_exposure_total = (filtered_df[monetary_col].fillna(0) * filtered_df["Churn_Probability"].fillna(0)).sum()
        revenue_at_risk_str = f"${rev_exposure_total:,.2f}"
elif churn_rate_val is not None:
    high_risk_str = f"{inactive_count} ({churn_rate_str})"

# -------------------------------------------------------------
# Top KPI Cards Row
# -------------------------------------------------------------
k1, k2, k3, k4, k5, k6 = st.columns(6)

with k1:
    render_kpi("Total Customers", f"{total_customers:,}", f"Filtered ({total_customers/portfolio_total*100:.0f}% of {portfolio_total:,})")

with k2:
    render_kpi("Active Customers", active_count, "Retained historical accounts")

with k3:
    render_kpi("Avg Monthly Spend", avg_spend_str, f"Field: {monetary_col or 'N/A'}")

with k4:
    render_kpi("Historical Churn", churn_rate_str, "Observed target rate")

with k5:
    render_kpi("High-Risk Accounts", high_risk_str, "Predicted prob ≥ 65%")

with k6:
    render_kpi("Revenue at Risk", revenue_at_risk_str, "Expected Monthly Exposure")

st.markdown("<br>", unsafe_allow_html=True)

# -------------------------------------------------------------
# Section: Risk Breakdown & Revenue Exposure
# -------------------------------------------------------------
risk_col1, risk_col2 = st.columns([1, 1])

with risk_col1:
    st.markdown("#### 🎯 Churn Risk Category Breakdown")
    if "Risk_Category" in filtered_df.columns:
        cat_counts = filtered_df["Risk_Category"].value_counts().reindex(["High", "Medium", "Low"]).fillna(0).astype(int)
        cat_pcts = (cat_counts / total_customers * 100).round(1)
        
        # Display breakdown metric pills
        c_h, c_m, c_l = st.columns(3)
        with c_h:
            st.metric("🔴 High Risk (≥65%)", f"{cat_counts['High']:,}", f"{cat_pcts['High']:.1f}% of cohort", delta_color="inverse")
        with c_m:
            st.metric("🟡 Medium Risk (35-65%)", f"{cat_counts['Medium']:,}", f"{cat_pcts['Medium']:.1f}% of cohort", delta_color="off")
        with c_l:
            st.metric("🟢 Low Risk (<35%)", f"{cat_counts['Low']:,}", f"{cat_pcts['Low']:.1f}% of cohort")

        # Donut Chart
        risk_fig = px.pie(
            names=cat_counts.index,
            values=cat_counts.values,
            hole=0.5,
            color=cat_counts.index,
            color_discrete_map={"High": COLOR_DANGER, "Medium": COLOR_WARNING, "Low": COLOR_SUCCESS}
        )
        risk_fig.update_layout(get_plotly_layout(height=280))
        st.plotly_chart(risk_fig, use_container_width=True)
        st.caption("ℹ️ **Threshold Rationale:** High Risk (≥0.65) targets immediate intervention; Medium Risk (0.35–0.64) targets relationship nurture; Low Risk (<0.35) maintains standard service.")
    else:
        st.info("Predicted churn risk tiers will display here once the Churn Prediction module is run.")

with risk_col2:
    st.markdown("#### 💰 Revenue at Risk Exposure Breakdown")
    if "Risk_Category" in filtered_df.columns and monetary_col in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df[monetary_col]):
        temp_exp = filtered_df.copy()
        temp_exp["Revenue_Exposure"] = temp_exp[monetary_col].fillna(0) * temp_exp["Churn_Probability"].fillna(0)
        
        exp_by_cat = temp_exp.groupby("Risk_Category")["Revenue_Exposure"].sum().reindex(["High", "Medium", "Low"]).fillna(0).reset_index()
        
        fig_rev = px.bar(
            exp_by_cat,
            x="Risk_Category",
            y="Revenue_Exposure",
            text_auto="$,.2f",
            color="Risk_Category",
            color_discrete_map={"High": COLOR_DANGER, "Medium": COLOR_WARNING, "Low": COLOR_SUCCESS},
            labels={"Risk_Category": "Risk Category", "Revenue_Exposure": "Expected Monthly Exposure ($)"}
        )
        fig_rev.update_layout(get_plotly_layout(height=280))
        fig_rev.update_yaxes(title_text="Monthly Exposure ($)")
        st.plotly_chart(fig_rev, use_container_width=True)
        
        st.caption(
            f"**Definition:** Total Monthly Portfolio Spend = **${total_monthly_spend:,.2f}** | "
            f"Expected Monthly Attrition Exposure = **${rev_exposure_total:,.2f}** "
            f"({(rev_exposure_total / max(1.0, total_monthly_spend))*100:.1f}% exposure)."
        )
    else:
        st.info("Revenue exposure breakdown requires spending and predicted churn probabilities.")

st.markdown("<br>", unsafe_allow_html=True)

# -------------------------------------------------------------
# Section: Contract Churn & Distribution Visualizations
# -------------------------------------------------------------
c_row1, c_row2 = st.columns([1, 1])

with c_row1:
    st.markdown("#### 📑 Contract Churn Analysis (Correct Group Denominators)")
    if contract_col and target_col and contract_col in filtered_df.columns and target_col in filtered_df.columns:
        # Calculate Churn % using group denominator: (Group Churn / Group Total) * 100
        contract_summary = []
        for contract_name, group in filtered_df.groupby(contract_col):
            tot = len(group)
            churns = group[target_col].astype(str).str.strip().str.lower().isin(["yes", "1", "true"]).sum()
            retains = tot - churns
            churn_pct = (churns / tot) * 100 if tot > 0 else 0
            contract_summary.append({
                "Contract Type": str(contract_name),
                "Total Customers": tot,
                "Retained": retains,
                "Churned": churns,
                "Churn Rate (%)": round(churn_pct, 2)
            })
        
        c_sum_df = pd.DataFrame(contract_summary)
        
        # Plot dual-axis or grouped bar
        fig_cbar = px.bar(
            c_sum_df,
            x="Contract Type",
            y="Churn Rate (%)",
            text="Churn Rate (%)",
            color="Contract Type",
            color_discrete_sequence=[COLOR_PRIMARY, COLOR_SECONDARY, COLOR_WARNING],
            labels={"Churn Rate (%)": "Churn Rate Within Group (%)"}
        )
        fig_cbar.update_traces(texttemplate="%{y:.1f}%", textposition="outside")
        fig_cbar.update_layout(get_plotly_layout(height=320), yaxis=dict(range=[0, max(c_sum_df["Churn Rate (%)"].max() * 1.25, 20)]))
        st.plotly_chart(fig_cbar, use_container_width=True)
        
        # Display data table with counts and percentages
        st.dataframe(c_sum_df, use_container_width=True, hide_index=True)
    else:
        st.info("Contract structure and churn target required for contract breakdown.")

with c_row2:
    st.markdown("#### ⏳ Customer Tenure Distribution & Churn Overlap")
    if tenure_col and target_col and tenure_col in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df[tenure_col]):
        fig_thist = px.histogram(
            filtered_df,
            x=tenure_col,
            color=target_col,
            barmode="overlay",
            nbins=30,
            opacity=0.75,
            color_discrete_map={"Yes": COLOR_DANGER, "No": COLOR_SUCCESS, "1": COLOR_DANGER, "0": COLOR_SUCCESS},
            labels={tenure_col: "Tenure (Months)", target_col: "Historical Churn"}
        )
        fig_thist.update_layout(get_plotly_layout(height=380))
        fig_thist.update_yaxes(title_text="Customer Count")
        st.plotly_chart(fig_thist, use_container_width=True)
    elif tenure_col and tenure_col in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df[tenure_col]):
        fig_thist = px.histogram(filtered_df, x=tenure_col, nbins=30, color_discrete_sequence=[COLOR_PRIMARY])
        fig_thist.update_layout(get_plotly_layout(height=380))
        st.plotly_chart(fig_thist, use_container_width=True)
    else:
        st.info("Tenure feature not available for tenure distribution.")

# -------------------------------------------------------------
# Section: Automated Structured Business Insights
# -------------------------------------------------------------
st.markdown("### 💡 Automated Business Insights (Derived from Empirical Data)")
st.caption("ℹ️ **Methodology:** Generated dynamically from observed frequencies and risk distributions. Highlights correlational patterns without asserting direct causation.")

insights = []

# Insight 1: Contract Vulnerability
if contract_col and target_col and contract_col in filtered_df.columns and target_col in filtered_df.columns:
    m2m_df = filtered_df[filtered_df[contract_col].astype(str).str.contains("Month", case=False, na=False)]
    long_df = filtered_df[filtered_df[contract_col].astype(str).str.contains("Two|One", case=False, na=False)]
    
    if len(m2m_df) > 0 and len(long_df) > 0:
        m2m_churn = (m2m_df[target_col].astype(str).str.lower().isin(["yes", "1"]).sum() / len(m2m_df)) * 100
        long_churn = (long_df[target_col].astype(str).str.lower().isin(["yes", "1"]).sum() / len(long_df)) * 100
        ratio = round(m2m_churn / max(0.1, long_churn), 1)
        insights.append({
            "title": "Contract Structure & Commitment",
            "observation": f"Month-to-month contracts exhibit an observed churn rate of {m2m_churn:.1f}% compared to {long_churn:.1f}% for long-term contract holders.",
            "metric": f"{ratio}x higher attrition rate in Month-to-Month accounts ({m2m_churn:.1f}% vs {long_churn:.1f}%).",
            "interpretation": "Lack of contractual commitment amplifies customer switching behavior during early friction points.",
            "action": "Introduce annual migration incentives, such as discounted first-quarter rates or feature bundles, to transition month-to-month clients."
        })

# Insight 2: High Spending Exposure
if monetary_col and target_col and monetary_col in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df[monetary_col]):
    churned_spend = filtered_df[filtered_df[target_col].astype(str).str.lower().isin(["yes", "1"])][monetary_col].dropna().mean()
    retained_spend = filtered_df[filtered_df[target_col].astype(str).str.lower().isin(["no", "0"])][monetary_col].dropna().mean()
    if pd.notna(churned_spend) and pd.notna(retained_spend) and retained_spend > 0:
        diff_pct = ((churned_spend - retained_spend) / retained_spend) * 100
        insights.append({
            "title": "Revenue Concentration in Churn",
            "observation": f"Churned customers had an average monthly charge of ${churned_spend:.2f}, while retained customers averaged ${retained_spend:.2f}.",
            "metric": f"${churned_spend:.2f} churned spend vs ${retained_spend:.2f} retained spend ({diff_pct:+.1f}% difference).",
            "interpretation": "Attrition is disproportionately concentrated in higher-spending service tiers, causing amplified revenue drag.",
            "action": "Deploy high-touch account management and proactive value audits for subscribers paying above $70/month."
        })

# Insight 3: Early Tenure Drop-Off
if tenure_col and target_col and tenure_col in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df[tenure_col]):
    early_tenure = filtered_df[filtered_df[tenure_col] <= 12]
    mature_tenure = filtered_df[filtered_df[tenure_col] > 24]
    if len(early_tenure) > 0 and len(mature_tenure) > 0:
        early_churn = (early_tenure[target_col].astype(str).str.lower().isin(["yes", "1"]).sum() / len(early_tenure)) * 100
        mature_churn = (mature_tenure[target_col].astype(str).str.lower().isin(["yes", "1"]).sum() / len(mature_tenure)) * 100
        insights.append({
            "title": "Onboarding & First-Year Hazard",
            "observation": f"Customers within their first 12 months show a {early_churn:.1f}% churn rate, whereas accounts past 24 months drop to {mature_churn:.1f}%.",
            "metric": f"{early_churn:.1f}% early-tenure attrition vs {mature_churn:.1f}% mature-tenure attrition.",
            "interpretation": "The primary customer vulnerability occurs during the initial 90 to 180 days before loyalty habits and switching costs solidify.",
            "action": "Reinforce the 30-day onboarding sequence, deliver customer satisfaction check-ins at day 60, and reward tenure milestones."
        })

# Insight 4: Fiber Optic & Tech Support Dependency
tech_col = next((c for c in filtered_df.columns if "techsupport" in c.lower()), None)
if internet_col and tech_col and target_col:
    fiber_no_sup = filtered_df[(filtered_df[internet_col].astype(str).str.contains("Fiber", case=False)) & 
                               (filtered_df[tech_col].astype(str).str.lower().isin(["no", "false", "0"]))]
    if len(fiber_no_sup) > 0:
        fns_churn = (fiber_no_sup[target_col].astype(str).str.lower().isin(["yes", "1"]).sum() / len(fiber_no_sup)) * 100
        insights.append({
            "title": "Service Cohort Friction: Fiber Without Support",
            "observation": f"Subscribers with Fiber Optic service but lacking dedicated Tech Support experience a {fns_churn:.1f}% churn rate.",
            "metric": f"{len(fiber_no_sup):,} customers in cohort with {fns_churn:.1f}% observed churn.",
            "interpretation": "High-bandwidth Fiber users likely experience unresolved technical friction that drives dissatisfaction when unassisted.",
            "action": "Bundle complimentary 3-month Tech Support with all Fiber Optic packages to lower early abandonment."
        })

if not insights:
    st.info("Explore Customer Segmentation and Churn Prediction modules to generate customized cross-sectional insights.")
else:
    for idx, ins in enumerate(insights, 1):
        with st.container():
            st.markdown(
                f"""
                <div class="content-card" style="margin-bottom: 0.8rem; border-left: 4px solid {COLOR_PRIMARY};">
                    <h5 style="margin: 0 0 0.4rem 0; color: #1E293B;">📌 Insight {idx}: {ins['title']}</h5>
                    <p style="margin: 0.2rem 0; font-size: 0.88rem; color: #334155;"><strong>Observation:</strong> {ins['observation']}</p>
                    <p style="margin: 0.2rem 0; font-size: 0.88rem; color: #2563EB;"><strong>Supporting Metric:</strong> {ins['metric']}</p>
                    <p style="margin: 0.2rem 0; font-size: 0.88rem; color: #475569;"><strong>Business Interpretation:</strong> {ins['interpretation']}</p>
                    <p style="margin: 0.2rem 0; font-size: 0.88rem; color: #059669;"><strong>Recommended Action:</strong> {ins['action']}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
