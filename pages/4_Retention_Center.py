"""
Customer360 — Page 4: Retention Center
Value-at-Risk Prioritization, Automated Action Recommendations, and Session Action Tracker.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

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
from src.recommendations import generate_retention_plan, ACTION_STATUSES
from src.validation import infer_column_roles

st.set_page_config(page_title="Retention Center — Customer360", page_icon="🛡️", layout="wide")
apply_custom_css()
init_session_state()
render_sidebar()

render_page_header(
    title="Customer Retention Action Center",
    subtitle="Prioritize vulnerable revenue, dispatch tailored retention workflows, and track intervention progress.",
    badge="Action Management"
)

st.caption("ℹ️ **Governance Notice:** Recommendation rules provide advisory guidance. External automated communication integrations (SMS/Email) are decoupled.")

df = st.session_state.get("current_dataset")
pred_df = st.session_state.get("predictions_df")

# Use predictions df if available, else current dataset
active_df = pred_df if pred_df is not None else df

if active_df is None or active_df.empty:
    st.error("No customer records found. Please load a dataset first.")
    st.stop()

roles = infer_column_roles(active_df)
id_col = roles["id_col"] or "customerID"
spending_col = roles["monetary_col"]
tenure_col = roles["tenure_col"]

# Generate or retrieve retention plan
with st.spinner("Compiling customer retention priorities..."):
    retention_plan = generate_retention_plan(active_df, spending_col=spending_col, tenure_col=tenure_col)

# Apply session state status overrides
actions_state = st.session_state.get("retention_actions", {})
if id_col in retention_plan.columns:
    for idx, row in retention_plan.iterrows():
        c_id = str(row[id_col])
        if c_id in actions_state:
            retention_plan.at[idx, "Action_Status"] = actions_state[c_id]

# Calculate High-Level Metrics
total_accounts = len(retention_plan)
high_priority_cnt = (retention_plan["Priority_Score"] >= 40.0).sum()
completed_actions = sum(1 for v in actions_state.values() if v == "Completed")
in_progress_actions = sum(1 for v in actions_state.values() if v == "In Progress")

r1, r2, r3, r4 = st.columns(4)
with r1:
    render_kpi("Total Accounts", f"{total_accounts:,}", "In retention scope")
with r2:
    render_kpi("High Urgency Accounts", f"{high_priority_cnt:,}", "Priority Score ≥ 40")
with r3:
    render_kpi("In-Progress Interventions", f"{in_progress_actions:,}", "Active workflows")
with r4:
    render_kpi("Completed Interventions", f"{completed_actions:,}", "Resolved this session")

st.markdown("<br>", unsafe_allow_html=True)

# Visualizations: Strategy Distribution & Status
c_v1, c_v2 = st.columns([1, 1])

with c_v1:
    st.markdown("#### 🎯 Recommended Strategy Distribution")
    rec_counts = retention_plan["Recommended_Action"].value_counts().reset_index()
    rec_counts.columns = ["Action", "Count"]
    fig_rec = px.pie(
        rec_counts,
        names="Action",
        values="Count",
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig_rec.update_layout(get_plotly_layout(height=350))
    st.plotly_chart(fig_rec, use_container_width=True)

with c_v2:
    st.markdown("#### ⚡ Value at Risk (Monthly Charges by Risk Category)")
    risk_col = "Risk_Category" if "Risk_Category" in retention_plan.columns else None
    if risk_col and spending_col and pd.api.types.is_numeric_dtype(retention_plan[spending_col]):
        var_df = retention_plan.groupby(risk_col)[spending_col].sum().reset_index()
        fig_var = px.bar(
            var_df,
            x=risk_col,
            y=spending_col,
            color=risk_col,
            color_discrete_map={"High": COLOR_DANGER, "Medium": COLOR_WARNING, "Low": COLOR_SUCCESS},
            labels={spending_col: "Total Portfolio Spend ($)", risk_col: "Risk Tier"}
        )
        fig_var.update_layout(get_plotly_layout(height=350))
        st.plotly_chart(fig_var, use_container_width=True)
    else:
        st.info("Spending data or risk tier unavailable for portfolio valuation.")

# Filter and Interactive Customer Queue
st.markdown("### 📋 Prioritized Intervention Queue")

col_flt1, col_flt2, col_flt3 = st.columns(3)
with col_flt1:
    strat_filter = st.selectbox("Filter Strategy", ["All"] + list(retention_plan["Recommended_Action"].unique()))
with col_flt2:
    status_filter = st.selectbox("Filter Status", ["All"] + ACTION_STATUSES)
with col_flt3:
    search_query = st.text_input("Search Customer ID", placeholder="e.g. 7590-VHVEG")

filtered_plan = retention_plan.copy()
if strat_filter != "All":
    filtered_plan = filtered_plan[filtered_plan["Recommended_Action"] == strat_filter]
if status_filter != "All":
    filtered_plan = filtered_plan[filtered_plan["Action_Status"] == status_filter]
if search_query and id_col in filtered_plan.columns:
    filtered_plan = filtered_plan[filtered_plan[id_col].astype(str).str.contains(search_query, case=False)]

# Table Display
queue_cols = [id_col] if id_col in filtered_plan.columns else []
queue_cols += ["Priority_Score", "Recommended_Action", "Recommendation_Reason", "Action_Status"]
if "Risk_Category" in filtered_plan.columns:
    queue_cols.insert(1, "Risk_Category")
if spending_col and spending_col in filtered_plan.columns:
    queue_cols.insert(2, spending_col)

st.dataframe(filtered_plan[queue_cols].head(50), use_container_width=True)

# Interactive Status Updater
st.markdown("---")
st.markdown("### ✍️ Update Account Intervention Status")

if id_col in retention_plan.columns:
    queue_ids = filtered_plan[id_col].astype(str).head(30).tolist()
    if queue_ids:
        up_col1, up_col2, up_col3 = st.columns([2, 1, 1])
        with up_col1:
            target_account = st.selectbox("Select Customer to Update", queue_ids)
        with up_col2:
            current_status = actions_state.get(target_account, "Pending")
            new_status = st.selectbox("Update Status", ACTION_STATUSES, index=ACTION_STATUSES.index(current_status))
        with up_col3:
            st.write("")
            st.write("")
            if st.button("Save Status Update", type="primary"):
                actions_state[target_account] = new_status
                st.session_state["retention_actions"] = actions_state
                st.success(f"Customer `{target_account}` updated to **{new_status}**!")
                st.rerun()
    else:
        st.info("No customers match the current filter selection.")
