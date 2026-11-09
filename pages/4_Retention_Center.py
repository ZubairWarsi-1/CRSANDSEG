"""
Customer360 — Page 4: Retention Center
Value-at-Risk Prioritization, Transparent Priority Scoring, Persistent SQLite Queue, Contact History, Notes, and Communication Previews.
"""

from datetime import datetime
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
    COLOR_WARNING,
    COLOR_ACCENT
)
from src.recommendations import (
    generate_retention_plan,
    calculate_priority_score_and_breakdown,
    preview_communication_template,
    ACTION_STATUSES
)
from src.database import (
    upsert_intervention,
    get_all_interventions,
    log_contact_event,
    get_contact_history,
    add_retention_note,
    get_retention_notes,
    record_retention_outcome,
    get_retention_outcomes,
    get_observed_retention_rate
)
from src.validation import infer_column_roles

st.set_page_config(page_title="Retention Center — Customer360", page_icon="🛡️", layout="wide")
apply_custom_css()
init_session_state()
render_sidebar()

render_page_header(
    title="Customer Retention Action Center",
    subtitle="Prioritize vulnerable revenue, dispatch tailored retention workflows, track persistent interventions, and measure observed outcomes.",
    badge="Action Management"
)

df = st.session_state.get("current_dataset")
pred_df = st.session_state.get("predictions_df")

# Use predictions df if available, else current dataset
active_df = pred_df if pred_df is not None else df

if active_df is None or active_df.empty:
    st.error("No customer records found. Please load or upload a dataset first.")
    st.stop()

roles = infer_column_roles(active_df)
id_col = roles["id_col"] or "customerID"
spending_col = roles["monetary_col"]
tenure_col = roles["tenure_col"]

# -------------------------------------------------------------
# Priority Score Formula & Methodology Explanation
# -------------------------------------------------------------
with st.expander("📐 Priority Score Methodology & Formula Breakdown", expanded=False):
    st.markdown(
        r"""
        #### Multi-Factor Priority Score Formula:
        $$\text{Priority Score} = \text{Risk Points} (0\text{--}50) + \text{Value Points} (0\text{--}35) + \text{Tenure Points} (0\text{--}15)$$
        
        - **1. Risk Points (50% max):** $\text{Predicted Churn Probability} \times 50$. Higher predicted attrition likelihood earns higher priority.
        - **2. Value Points (35% max):** Normalized monthly spending rank $\times 35$. Maximizes intervention on accounts contributing large recurring revenue.
        - **3. Tenure Urgency Points (15% max):** $\left(1 - \frac{\min(\text{Tenure}, 72)}{72}\right) \times 15$. Early-stage accounts receive higher urgency due to onboarding hazard.
        
        **Revenue at Risk Definition:**
        $$\text{Revenue at Risk} = \text{Monthly Charges} \times \text{Predicted Churn Probability}$$
        *Note: Represents expected monthly statistical exposure, not guaranteed immediate loss.*
        """
    )

# Compile retention plan
with st.spinner("Compiling customer retention priorities and synchronizing with SQLite database..."):
    retention_plan = generate_retention_plan(active_df, spending_col=spending_col, tenure_col=tenure_col)

# Synchronize with SQLite persistent database
db_interventions = get_all_interventions()
db_map = {row["customer_id"]: row for _, row in db_interventions.iterrows()} if not db_interventions.empty else {}

# Merge persistent database fields into retention plan
retention_plan["Action_Status"] = "Pending"
retention_plan["Assigned_To"] = "Unassigned"
retention_plan["DB_Notes"] = ""

if id_col in retention_plan.columns:
    for idx, row in retention_plan.iterrows():
        c_id = str(row[id_col])
        if c_id in db_map:
            db_row = db_map[c_id]
            retention_plan.at[idx, "Action_Status"] = db_row.get("action_status", "Pending")
            retention_plan.at[idx, "Assigned_To"] = db_row.get("assigned_to", "Unassigned")
            retention_plan.at[idx, "DB_Notes"] = db_row.get("notes", "")

# Compute summary metrics
total_accounts = len(retention_plan)
high_priority_cnt = (retention_plan["Priority_Score"] >= 40.0).sum()
in_progress_cnt = (retention_plan["Action_Status"] == "In Progress").sum()
completed_cnt = (retention_plan["Action_Status"] == "Completed").sum()

total_rev_at_risk = 0.0
if "Revenue_at_Risk" in retention_plan.columns:
    total_rev_at_risk = retention_plan["Revenue_at_Risk"].sum()

# Observed retention rate from database outcomes
outcome_stats = get_observed_retention_rate()
observed_rate_str = f"{outcome_stats['retention_rate']:.1f}%" if outcome_stats["total_evaluated"] > 0 else "N/A"

# -------------------------------------------------------------
# Top KPI Cards Row
# -------------------------------------------------------------
r1, r2, r3, r4, r5 = st.columns(5)
with r1:
    render_kpi("Queue Accounts", f"{total_accounts:,}", "In retention scope")
with r2:
    render_kpi("High Urgency (≥40)", f"{high_priority_cnt:,}", f"{high_priority_cnt/total_accounts*100:.1f}% of cohort")
with r3:
    render_kpi("Revenue at Risk", f"${total_rev_at_risk:,.2f}", "Expected Monthly Exposure")
with r4:
    render_kpi("Active Workflows", f"{in_progress_cnt + completed_cnt:,}", f"{completed_cnt} Completed / {in_progress_cnt} Active")
with r5:
    render_kpi("Observed Retention", observed_rate_str, f"Evaluated: {outcome_stats['total_evaluated']}")

st.markdown("<br>", unsafe_allow_html=True)

# -------------------------------------------------------------
# Portfolio Visualizations Row
# -------------------------------------------------------------
c_v1, c_v2 = st.columns([1, 1])

with c_v1:
    st.markdown("#### 🎯 Prescriptive Retention Strategy Distribution")
    rec_counts = retention_plan["Recommended_Action"].value_counts().reset_index()
    rec_counts.columns = ["Action", "Count"]
    fig_rec = px.pie(
        rec_counts,
        names="Action",
        values="Count",
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig_rec.update_layout(get_plotly_layout(height=320))
    st.plotly_chart(fig_rec, use_container_width=True)

with c_v2:
    st.markdown("#### ⚡ Revenue at Risk Exposure by Risk Category")
    risk_col = "Risk_Category" if "Risk_Category" in retention_plan.columns else None
    if risk_col and "Revenue_at_Risk" in retention_plan.columns:
        var_df = retention_plan.groupby(risk_col)["Revenue_at_Risk"].sum().reindex(["High", "Medium", "Low"]).fillna(0).reset_index()
        fig_var = px.bar(
            var_df,
            x=risk_col,
            y="Revenue_at_Risk",
            color=risk_col,
            text_auto="$,.2f",
            color_discrete_map={"High": COLOR_DANGER, "Medium": COLOR_WARNING, "Low": COLOR_SUCCESS},
            labels={"Revenue_at_Risk": "Expected Monthly Exposure ($)", risk_col: "Risk Tier"}
        )
        fig_var.update_layout(get_plotly_layout(height=320))
        st.plotly_chart(fig_var, use_container_width=True)
    else:
        st.info("Risk categories or spending data not present.")

st.markdown("<br>", unsafe_allow_html=True)

# -------------------------------------------------------------
# Prioritized Intervention Queue & Table
# -------------------------------------------------------------
st.markdown("### 📋 Prioritized Intervention Queue (SQLite Synced)")

col_flt1, col_flt2, col_flt3 = st.columns(3)
with col_flt1:
    strat_filter = st.selectbox("Filter Strategy", ["All"] + sorted(retention_plan["Recommended_Action"].unique().tolist()))
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

queue_cols = [id_col] if id_col in filtered_plan.columns else []
queue_cols += ["Risk_Category", spending_col, "Revenue_at_Risk", "Priority_Score", "Recommended_Action", "Action_Status", "Assigned_To"]
valid_display_cols = [c for c in queue_cols if c in filtered_plan.columns]

st.dataframe(filtered_plan[valid_display_cols].head(50), use_container_width=True)

# -------------------------------------------------------------
# Detailed Customer Action Hub (SQLite Management)
# -------------------------------------------------------------
st.markdown("---")
st.markdown("### 🛠️ Customer Intervention Management Hub")
st.caption("Manage workflows, record contact logs, view notes, evaluate actual retention outcomes, and preview communications in SQLite.")

if id_col in filtered_plan.columns and not filtered_plan.empty:
    customer_list = filtered_plan[id_col].astype(str).head(50).tolist()
    selected_customer = st.selectbox("Select Target Customer for Intervention Management", customer_list)
    cust_record = retention_plan[retention_plan[id_col].astype(str) == selected_customer].iloc[0]

    # Calculate and display transparent points breakdown
    score_breakdown = calculate_priority_score_and_breakdown(cust_record, spending_col=spending_col, tenure_col=tenure_col)
    
    # Customer Highlight Header
    st.markdown(
        f"""
        <div class="content-card" style="margin-bottom: 1rem; border-left: 4px solid {COLOR_PRIMARY};">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h4 style="margin: 0; color: #1E293B;">Customer ID: <code>{selected_customer}</code></h4>
                <span class="badge badge-{'danger' if cust_record.get('Risk_Category')=='High' else 'warning'}">
                    Risk: {cust_record.get('Risk_Category', 'N/A')} ({cust_record.get('Churn_Probability', 0)*100:.1f}%)
                </span>
            </div>
            <p style="margin: 0.3rem 0; font-size: 0.88rem; color: #475569;">
                <strong>Monthly Spend:</strong> ${cust_record.get(spending_col, 0):.2f} | 
                <strong>Tenure:</strong> {cust_record.get(tenure_col, 'N/A')} months | 
                <strong>Revenue at Risk:</strong> ${cust_record.get('Revenue_at_Risk', 0):.2f} | 
                <strong>Current Status:</strong> <span class="badge badge-info">{cust_record.get('Action_Status', 'Pending')}</span>
            </p>
            <div style="background: #F1F5F9; padding: 0.5rem 0.8rem; border-radius: 6px; font-size: 0.84rem; color: #334155; margin-top: 0.4rem;">
                <strong>Priority Score Breakdown:</strong> Risk Points ({score_breakdown['risk_points']}) + 
                Value Points ({score_breakdown['value_points']}) + 
                Tenure Points ({score_breakdown['tenure_points']}) = 
                <strong style="color: #1E3A8A;">{score_breakdown['priority_score']} Total Score</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Management Tabs
    tab_status, tab_contact, tab_notes, tab_outcome, tab_comm = st.tabs([
        "✍️ Update Status & Assignment",
        "📞 Contact History & Logging",
        "📝 Retention Notes",
        "🎯 Observed Outcome Tracker",
        "✉️ Communication Preview"
    ])

    # TAB 1: STATUS & ASSIGNMENT
    with tab_status:
        st.markdown("##### Persistent Intervention State (SQLite)")
        with st.form("status_update_form"):
            s_c1, s_c2 = st.columns(2)
            with s_c1:
                cur_stat = cust_record.get("Action_Status", "Pending")
                stat_idx = ACTION_STATUSES.index(cur_stat) if cur_stat in ACTION_STATUSES else 0
                up_status = st.selectbox("Action Status", ACTION_STATUSES, index=stat_idx)
            with s_c2:
                up_assigned = st.text_input("Assigned Retention Agent / Team", value=str(cust_record.get("Assigned_To", "Customer Success Team")))
            
            up_note = st.text_area("Status Notes", value=str(cust_record.get("DB_Notes", "")))
            save_stat_btn = st.form_submit_button("💾 Save Changes to SQLite Database")

        if save_stat_btn:
            upsert_intervention(
                customer_id=selected_customer,
                recommended_action=str(cust_record.get("Recommended_Action", "")),
                action_status=up_status,
                assigned_to=up_assigned,
                notes=up_note,
                priority_score=float(cust_record.get("Priority_Score", 0.0)),
                revenue_at_risk=float(cust_record.get("Revenue_at_Risk", 0.0))
            )
            st.success(f"Intervention updated successfully in SQLite for customer `{selected_customer}`!")
            st.rerun()

    # TAB 2: CONTACT HISTORY & LOGGING
    with tab_contact:
        st.markdown("##### Log New Customer Contact Event")
        with st.form("contact_log_form"):
            cc1, cc2, cc3 = st.columns(3)
            with cc1:
                c_date = st.date_input("Contact Date", value=datetime.today())
            with cc2:
                c_channel = st.selectbox("Contact Channel", ["Email", "Phone", "SMS", "In-App Message", "Meeting"])
            with cc3:
                c_outcome = st.selectbox("Contact Outcome", ["Reached", "Voicemail", "No Answer", "Declined", "Accepted Offer", "Follow-up Scheduled"])
            
            c_agent = st.text_input("Representative Name", value="Retention Specialist")
            c_notes = st.text_input("Conversation Summary / Notes", placeholder="Customer was receptive to the 1-year contract discount...")
            c_submit = st.form_submit_button("📞 Log Contact Event to SQLite")

        if c_submit:
            log_contact_event(
                customer_id=selected_customer,
                contact_date=str(c_date),
                channel=c_channel,
                outcome=c_outcome,
                notes=c_notes,
                agent_name=c_agent
            )
            st.success(f"Contact event recorded for customer `{selected_customer}`!")
            st.rerun()

        # Display Existing History
        st.markdown("##### Historical Contact Timeline")
        hist_df = get_contact_history(selected_customer)
        if not hist_df.empty:
            st.dataframe(hist_df[["contact_date", "channel", "outcome", "agent_name", "notes"]], use_container_width=True, hide_index=True)
        else:
            st.info("No prior contact events recorded for this customer in SQLite.")

    # TAB 3: RETENTION NOTES
    with tab_notes:
        st.markdown("##### Add Collaborative Retention Note")
        with st.form("add_note_form"):
            author = st.text_input("Author Name", value="Account Manager")
            note_content = st.text_area("Note Content", placeholder="Customer mentioned dissatisfaction with billing cycle...")
            note_submit = st.form_submit_button("📝 Add Note")

        if note_submit and note_content.strip():
            add_retention_note(selected_customer, note_content.strip(), author=author)
            st.success("Note stored in SQLite database!")
            st.rerun()

        st.markdown("##### Existing Customer Notes")
        notes_df = get_retention_notes(selected_customer)
        if not notes_df.empty:
            for _, n_row in notes_df.iterrows():
                st.markdown(
                    f"""
                    <div style="background: #F8FAFC; border-left: 3px solid #64748B; padding: 0.5rem 0.8rem; margin-bottom: 0.5rem; border-radius: 4px;">
                        <small style="color: #94A3B8;">{n_row['created_at']} — <strong>{n_row['author']}</strong></small>
                        <p style="margin: 0.2rem 0; font-size: 0.88rem; color: #1E293B;">{n_row['note_text']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.info("No notes logged for this customer yet.")

    # TAB 4: OBSERVED OUTCOME TRACKER
    with tab_outcome:
        st.markdown("##### Record Actual Observed Retention Outcome")
        st.caption("Measure whether intervention prevented customer defection.")
        with st.form("outcome_form"):
            oc1, oc2 = st.columns(2)
            with oc1:
                final_out = st.selectbox("Observed Customer Outcome", ["Retained", "Churned", "Upgraded", "Downgraded"])
                days_int = st.number_input("Days Since Intervention", min_value=0, max_value=365, value=30)
            with oc2:
                saved_rev = st.number_input("Monthly Revenue Saved / Retained ($)", min_value=0.0, value=float(cust_record.get(spending_col, 0.0)))
                out_notes = st.text_input("Outcome Notes", placeholder="Accepted 1-year contract migration.")
            
            out_submit = st.form_submit_button("🎯 Record Observed Outcome")

        if out_submit:
            record_retention_outcome(
                customer_id=selected_customer,
                final_outcome=final_out,
                days_since_intervention=int(days_int),
                revenue_saved=float(saved_rev),
                notes=out_notes
            )
            # Automatically update intervention status to Completed
            upsert_intervention(
                customer_id=selected_customer,
                recommended_action=str(cust_record.get("Recommended_Action", "")),
                action_status="Completed",
                assigned_to=str(cust_record.get("Assigned_To", "Customer Success Team")),
                notes=f"Outcome recorded: {final_out}. {out_notes}",
                priority_score=float(cust_record.get("Priority_Score", 0.0)),
                revenue_at_risk=float(cust_record.get("Revenue_at_Risk", 0.0))
            )
            st.success(f"Observed outcome recorded as **{final_out}** in SQLite!")
            st.rerun()

    # TAB 5: COMMUNICATION PREVIEW
    with tab_comm:
        st.markdown("##### Tailored Communication Template Dispatch")
        st.caption("Preview realistic communication copy containing customer personalization variables before dispatching.")

        cm_c1, cm_c2 = st.columns([1, 1])
        with cm_c1:
            comm_channel = st.radio("Dispatch Channel", ["Email", "SMS"], horizontal=True)
            comm_discount = st.slider("Target Incentive Discount (%)", min_value=5, max_value=35, value=15, step=5)
        
        tpl = preview_communication_template(cust_record, channel=comm_channel, discount_pct=comm_discount)

        with cm_c2:
            st.markdown(
                f"""
                <div style="background: #FFFFFF; border: 1px solid #CBD5E1; border-radius: 8px; padding: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                    <div style="font-size: 0.8rem; font-weight: 600; color: #64748B; text-transform: uppercase;">
                        {tpl['channel']} Preview • {tpl['recipient']}
                    </div>
                    <div style="font-weight: 700; color: #0F172A; margin: 0.4rem 0;">
                        {tpl['subject']}
                    </div>
                    <div style="background: #F8FAFC; border: 1px dashed #CBD5E1; padding: 0.8rem; border-radius: 6px; font-family: monospace; font-size: 0.85rem; color: #1E293B; white-space: pre-wrap;">
{tpl['body']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        if st.button("🚀 Confirm & Log Communication Dispatch in SQLite", type="primary"):
            log_contact_event(
                customer_id=selected_customer,
                contact_date=str(datetime.today().date()),
                channel=comm_channel,
                outcome="Sent / Dispatched",
                notes=f"Dispatched {comm_channel} campaign ({tpl['subject']}) with {comm_discount}% incentive offer.",
                agent_name="Customer360 System"
            )
            upsert_intervention(
                customer_id=selected_customer,
                recommended_action=str(cust_record.get("Recommended_Action", "")),
                action_status="In Progress",
                assigned_to=str(cust_record.get("Assigned_To", "Customer Success Team")),
                notes=f"Dispatched {comm_channel} retention template on {datetime.today().strftime('%Y-%m-%d')}.",
                priority_score=float(cust_record.get("Priority_Score", 0.0)),
                revenue_at_risk=float(cust_record.get("Revenue_at_Risk", 0.0))
            )
            st.success(f"Communication dispatched and logged in SQLite for customer `{selected_customer}`!")
            st.rerun()

else:
    st.info("No matching customers in current filtered queue.")
