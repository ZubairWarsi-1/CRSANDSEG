"""
Customer360 — Page 7: Reports and Data Exports
Multi-Format Deliverables (Styled Excel .xlsx, Publication-Grade PDF, One-Page Briefing PDF,
and 6 Standardized Modular CSV Deliverables) with SQLite Report History.
"""

from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np

from src.ui_components import (
    apply_custom_css,
    render_page_header,
    render_sidebar,
    render_kpi,
    init_session_state,
    COLOR_PRIMARY,
    COLOR_SUCCESS,
    COLOR_SECONDARY
)
from src.reporting import (
    export_dataframe_to_csv,
    export_multi_sheet_excel,
    generate_pdf_report,
    generate_one_page_executive_summary_pdf,
    generate_data_quality_markdown,
    get_data_quality_df
)
from src.validation import inspect_dataset, infer_column_roles
from src.model_evaluation import compare_models_advanced, evaluate_classification_model
from src.recommendations import generate_retention_plan
from src.database import get_all_interventions, get_report_history, log_report_generation

st.set_page_config(page_title="Reports & Exports — Customer360", page_icon="📄", layout="wide")
apply_custom_css()
init_session_state()
render_sidebar()

render_page_header(
    title="Analytical Reports & Intelligence Exports",
    subtitle="Export publication-ready deliverables: Styled Multi-Sheet Excel workbooks, Executive PDF Briefings, and 6 standardized CSV data feeds.",
    badge="Exports"
)

df = st.session_state.get("current_dataset")

if df is None or not isinstance(df, pd.DataFrame) or df.empty:
    st.error("No dataset available. Please upload a dataset in Data Upload or select a demo.")
    st.stop()

ds_name = st.session_state.get("dataset_name", "dataset").replace(".csv", "").replace(" ", "_").lower()
is_syn = st.session_state.get("is_synthetic", False)
ds_source = "Synthetic Test Data" if is_syn else st.session_state.get("dataset_source_class", "Benchmark Sample Dataset")
roles = infer_column_roles(df)
spending_col = roles.get("monetary_col")
tenure_col = roles.get("tenure_col")
target_col = roles.get("target_col")
id_col = roles.get("id_col") or "customerID"

pred_df = st.session_state.get("predictions_df")
kmeans_res = st.session_state.get("kmeans_data")
rfm_df = st.session_state.get("rfm_data")
trained_models = st.session_state.get("trained_models", {})

# Prepare common retention plan
active_source = pred_df if pred_df is not None else df
ret_plan = generate_retention_plan(active_source, spending_col=spending_col, tenure_col=tenure_col)

# Sync with SQLite interventions
db_interventions = get_all_interventions()
if not db_interventions.empty and id_col in ret_plan.columns:
    db_map = {str(row["customer_id"]): row for _, row in db_interventions.iterrows()}
    for idx, row in ret_plan.iterrows():
        c_id = str(row[id_col])
        if c_id in db_map:
            ret_plan.at[idx, "Action_Status"] = db_map[c_id].get("action_status", "Pending")

# Calculate portfolio totals
total_customers = len(df)
avg_spend = float(pd.to_numeric(df[spending_col], errors="coerce").dropna().mean()) if (spending_col and spending_col in df.columns) else 0.0
rev_at_risk = float(ret_plan["Revenue_at_Risk"].sum()) if "Revenue_at_Risk" in ret_plan.columns else 0.0
high_risk_cnt = int((pred_df["Risk_Category"] == "High").sum()) if (pred_df is not None and "Risk_Category" in pred_df.columns) else 0

# Benchmark comparison DF
if trained_models:
    comp_df = compare_models_advanced(list(trained_models.values()))
else:
    comp_df = pd.DataFrame({"Status": ["No supervised models trained yet"]})

# -------------------------------------------------------------
# Section 1: Executive Multi-Format Downloads
# -------------------------------------------------------------
st.markdown("### 🏆 Executive Multi-Format Reporting Suite")
st.caption("Consolidated analytical deliverables designed for executive leadership, operational managers, and cross-functional teams.")

ex_col1, ex_col2, ex_col3 = st.columns(3)

# 1. Multi-Sheet Styled Excel (.xlsx)
with ex_col1:
    st.markdown(
        """
        <div class="content-card" style="border-top: 4px solid #10B981; min-height: 240px;">
            <h4 style="margin: 0; color: #1E293B;">📊 Multi-Sheet Excel (.xlsx)</h4>
            <p style="font-size: 0.84rem; color: #64748B; margin: 0.4rem 0;">
                Comprehensive styled workbook containing 5 dedicated worksheets with corporate styling, column width formatting, and borders:
            </p>
            <ul style="font-size: 0.8rem; color: #475569; padding-left: 1rem; margin: 0.2rem 0;">
                <li>Executive KPIs & Portfolio Summary</li>
                <li>Customer Segmentation Profiles</li>
                <li>High-Risk Churn Predictions</li>
                <li>Prioritized Intervention Action Plan</li>
                <li>Model Benchmark Comparison Leaderboard</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    kpi_sheet_df = pd.DataFrame([
        {"Metric": "Dataset Name", "Value": str(ds_name)},
        {"Metric": "Data Classification", "Value": str(ds_source)},
        {"Metric": "Total Portfolio Customers", "Value": total_customers},
        {"Metric": "Average Monthly Spending", "Value": f"${avg_spend:,.2f}"},
        {"Metric": "High-Risk Customer Count", "Value": high_risk_cnt},
        {"Metric": "Expected Monthly Revenue Exposure", "Value": f"${rev_at_risk:,.2f}"},
        {"Metric": "Report Generated", "Value": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    ])

    excel_sheets = {
        "Executive Summary": kpi_sheet_df,
        "Retention Action Plan": ret_plan.head(500),
        "High-Risk Churn Targets": pred_df[pred_df["Risk_Category"] == "High"].head(500) if (pred_df is not None and "Risk_Category" in pred_df.columns) else pd.DataFrame(),
        "Customer Segments": (kmeans_res["cluster_profiles"] if kmeans_res is not None else (rfm_df if rfm_df is not None else pd.DataFrame())),
        "Model Leaderboard": comp_df
    }

    try:
        excel_bytes = export_multi_sheet_excel(excel_sheets)
        st.download_button(
            label="📥 Download Multi-Sheet Excel (.xlsx)",
            data=excel_bytes,
            file_name=f"Customer360_Executive_Report_{ds_name}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            on_click=lambda: log_report_generation("Executive Multi-Sheet Excel", "XLSX", len(excel_bytes), f"Customers: {total_customers}, Sheets: 5")
        )
    except Exception as e:
        st.error(f"Excel generation error: {str(e)}")

# 2. Executive Comprehensive PDF Report
with ex_col2:
    st.markdown(
        """
        <div class="content-card" style="border-top: 4px solid #2563EB; min-height: 240px;">
            <h4 style="margin: 0; color: #1E293B;">📑 Executive Audit PDF (.pdf)</h4>
            <p style="font-size: 0.84rem; color: #64748B; margin: 0.4rem 0;">
                Publication-grade analytical dossier formatted with ReportLab:
            </p>
            <ul style="font-size: 0.8rem; color: #475569; padding-left: 1rem; margin: 0.2rem 0;">
                <li>Executive Portfolio Health Summary</li>
                <li>Expected Monthly Revenue at Risk Exposure</li>
                <li>Data Quality Score Breakdown</li>
                <li>Behavioral Segmentation Findings</li>
                <li>Top 10 High-Urgency Accounts Table</li>
                <li>Model Performance Benchmark & Recommendations</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    pdf_data_payload = {
        "dataset_name": ds_name,
        "dataset_source": ds_source,
        "row_count": total_customers,
        "col_count": len(df.columns),
        "quality_score": inspect_dataset(df).get("quality_score", 90),
        "selected_model": st.session_state.get("active_model_name", "Random Forest"),
        "active_customers": f"{(total_customers - high_risk_cnt):,}",
        "avg_spending": avg_spend,
        "churn_rate": f"{(high_risk_cnt / max(1, total_customers))*100:.1f}%",
        "high_risk_count": f"{high_risk_cnt:,}",
        "revenue_at_risk": rev_at_risk,
        "models_summary": comp_df,
        "clusters_summary": (kmeans_res["cluster_profiles"] if kmeans_res is not None else None),
        "top_customers": ret_plan.head(10)
    }

    try:
        full_pdf_bytes = generate_pdf_report(pdf_data_payload)
        st.download_button(
            label="📥 Download Executive PDF Dossier (.pdf)",
            data=full_pdf_bytes,
            file_name=f"Customer360_Executive_Audit_{ds_name}.pdf",
            mime="application/pdf",
            use_container_width=True,
            on_click=lambda: log_report_generation("Executive Audit Dossier", "PDF", len(full_pdf_bytes), f"Customers: {total_customers}, Models: {len(trained_models)}")
        )
    except Exception as e:
        st.error(f"PDF generation error: {str(e)}")

# 3. One-Page Executive Summary PDF
with ex_col3:
    st.markdown(
        """
        <div class="content-card" style="border-top: 4px solid #6366F1; min-height: 240px;">
            <h4 style="margin: 0; color: #1E293B;">📄 One-Page Executive Briefing (.pdf)</h4>
            <p style="font-size: 0.84rem; color: #64748B; margin: 0.4rem 0;">
                High-impact 1-page executive brief tailored for C-suite and VP review:
            </p>
            <ul style="font-size: 0.8rem; color: #475569; padding-left: 1rem; margin: 0.2rem 0;">
                <li>Single-page standalone briefing layout</li>
                <li>Key financial revenue-at-risk exposure</li>
                <li>Top 3 empirical attrition drivers</li>
                <li>Actionable 30/60/90-day retention roadmap</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    try:
        one_page_bytes = generate_one_page_executive_summary_pdf(pdf_data_payload)
        st.download_button(
            label="📥 Download 1-Page Executive Brief (.pdf)",
            data=one_page_bytes,
            file_name=f"Customer360_1Page_Briefing_{ds_name}.pdf",
            mime="application/pdf",
            use_container_width=True,
            on_click=lambda: log_report_generation("One-Page Leadership Briefing", "PDF", len(one_page_bytes), f"Customers: {total_customers}")
        )
    except Exception as e:
        st.error(f"One-page PDF generation error: {str(e)}")

st.markdown("<br>", unsafe_allow_html=True)

# -------------------------------------------------------------
# Section 2: Six Granular Modular CSV Deliverables (UTF-8-SIG)
# -------------------------------------------------------------
st.markdown("### 📦 Granular Data Deliverables (6 Modular CSV Exports)")
st.caption("Standardized CSV exports encoded in UTF-8-SIG for direct compatibility with Microsoft Excel, CRM, and ERP pipelines.")

c1, c2 = st.columns(2)

# =============================================================
# Export 1: Churn Predictions Export
# =============================================================
with c1:
    st.markdown("##### 🎯 1. Churn Predictions Export")
    if pred_df is not None:
        exp1_df = pd.DataFrame()
        exp1_df["Customer ID"] = pred_df[id_col] if id_col in pred_df.columns else pred_df.iloc[:, 0]
        exp1_df["Actual churn status"] = pred_df[target_col] if (target_col and target_col in pred_df.columns) else "N/A"
        exp1_df["Predicted churn probability"] = pred_df["Churn_Probability"].round(4) if "Churn_Probability" in pred_df.columns else 0.0
        exp1_df["Predicted churn class"] = pred_df["Predicted_Churn"] if "Predicted_Churn" in pred_df.columns else "N/A"
        exp1_df["Risk category"] = pred_df["Risk_Category"] if "Risk_Category" in pred_df.columns else "N/A"
        exp1_df["Model name"] = st.session_state.get("active_model_name", "Random Forest")
        exp1_df["Prediction timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        csv_preds = export_dataframe_to_csv(exp1_df)
        st.download_button(
            f"⬇️ Download Predictions CSV ({len(exp1_df):,} rows)",
            data=csv_preds,
            file_name=f"churn_predictions_{ds_name}.csv",
            mime="text/csv",
            key="dl_preds"
        )
    else:
        st.warning("⚠️ Please run **Churn Prediction** before exporting.")
        st.button("⬇️ Download Predictions (CSV)", disabled=True, key="dis_preds")

# =============================================================
# Export 2: High-Risk Customer Prioritization List
# =============================================================
with c2:
    st.markdown("##### 🚨 2. High-Risk Customer Prioritization List")
    if pred_df is not None and "Churn_Probability" in pred_df.columns:
        hr_cutoff = st.slider(
            "High-Risk Churn Probability Cutoff",
            min_value=0.50,
            max_value=0.95,
            value=0.70,
            step=0.05,
            key="hr_slider_rep",
            help="Filter customers whose predicted probability of churn meets or exceeds this threshold."
        )
        hr_sub = pred_df[pred_df["Churn_Probability"] >= hr_cutoff].sort_values(by="Churn_Probability", ascending=False).reset_index(drop=True)
        
        exp2_df = pd.DataFrame()
        exp2_df["Customer ID"] = hr_sub[id_col] if id_col in hr_sub.columns else hr_sub.iloc[:, 0]
        exp2_df["Churn probability"] = hr_sub["Churn_Probability"].round(4)
        exp2_df["Risk category"] = hr_sub.get("Risk_Category", "High")
        exp2_df["Monthly charges"] = hr_sub[spending_col].round(2) if (spending_col and spending_col in hr_sub.columns) else "N/A"
        exp2_df["Tenure"] = hr_sub[tenure_col] if (tenure_col and tenure_col in hr_sub.columns) else "N/A"
        exp2_df["Priority rank"] = range(1, len(hr_sub) + 1)
        
        # Action map
        action_map = dict(zip(ret_plan[id_col].astype(str), ret_plan["Retention_Action"])) if (ret_plan is not None and id_col in ret_plan.columns and "Retention_Action" in ret_plan.columns) else {}
        exp2_df["Recommended next action"] = exp2_df["Customer ID"].astype(str).map(action_map).fillna("Immediate proactive retention outreach")

        csv_hr = export_dataframe_to_csv(exp2_df)
        st.download_button(
            f"⬇️ Download High-Risk List ({len(exp2_df):,} Accounts @ ≥{hr_cutoff:.2f})",
            data=csv_hr,
            file_name=f"high_risk_customers_{ds_name}.csv",
            mime="text/csv",
            key="dl_hr"
        )
    else:
        st.warning("⚠️ Please run **Churn Prediction** before exporting high-risk targets.")
        st.button("⬇️ Download High-Risk List (CSV)", disabled=True, key="dis_hr")

st.markdown("<br>", unsafe_allow_html=True)
c3, c4 = st.columns(2)

# =============================================================
# Export 3: Customer Segmentation Results
# =============================================================
with c3:
    st.markdown("##### 🔍 3. Customer Segmentation Results")
    if kmeans_res is not None:
        km_df = kmeans_res["dataframe"]
        exp3_df = pd.DataFrame()
        exp3_df["Customer ID"] = km_df[id_col] if id_col in km_df.columns else km_df.iloc[:, 0]
        exp3_df["Cluster ID"] = km_df["Cluster"]
        exp3_df["Cluster name"] = km_df["ClusterName"]
        for f in kmeans_res.get("feature_cols", []):
            if f in km_df.columns:
                exp3_df[f] = km_df[f]
        strat_map = dict(zip(kmeans_res["cluster_profiles"]["Cluster"], kmeans_res["cluster_profiles"]["RecommendedStrategy"]))
        exp3_df["Segment recommendation"] = exp3_df["Cluster ID"].map(strat_map).fillna("Automated Lifecycle Nurturing")

        csv_km = export_dataframe_to_csv(exp3_df)
        st.download_button(
            f"⬇️ Download K-Means Clusters CSV ({len(exp3_df):,} accounts)",
            data=csv_km,
            file_name=f"customer_segments_{ds_name}.csv",
            mime="text/csv",
            key="dl_km"
        )
    elif rfm_df is not None:
        exp3_df = pd.DataFrame()
        rfm_id = roles.get("id_col") if roles.get("id_col") in rfm_df.columns else rfm_df.columns[0]
        exp3_df["Customer ID"] = rfm_df[rfm_id]
        exp3_df["Cluster ID"] = rfm_df["RFM_Score"]
        exp3_df["Cluster name"] = rfm_df["Segment"]
        exp3_df["Recency (Days)"] = rfm_df["Recency"]
        exp3_df["Frequency (Orders)"] = rfm_df["Frequency"]
        exp3_df["Monetary (Total $)"] = rfm_df["Monetary"]
        exp3_df["Segment recommendation"] = rfm_df["Segment"].map({
            "Champions": "VIP Loyalty Rewards & Dedicated Support",
            "Loyal Customers": "Cross-Sell Premium Tier Services",
            "Potential Loyalists": "Adoption Guidance & Feature Discovery",
            "At-Risk Customers": "Proactive Win-Back Incentive",
            "Lost / Inactive Customers": "Reactivation Campaign"
        }).fillna("Standard Lifecycle Engagement")

        csv_rfm = export_dataframe_to_csv(exp3_df)
        st.download_button(
            f"⬇️ Download RFM Segments CSV ({len(exp3_df):,} customers)",
            data=csv_rfm,
            file_name=f"customer_segments_{ds_name}.csv",
            mime="text/csv",
            key="dl_rfm"
        )
    else:
        st.warning("⚠️ Please run **Customer Segmentation (K-Means or RFM)** before exporting.")
        st.button("⬇️ Download Segmentation Results (CSV)", disabled=True, key="dis_seg")

# =============================================================
# Export 4: Prescriptive Retention Action Plan
# =============================================================
with c4:
    st.markdown("##### 🛡️ 4. Prescriptive Retention Action Plan")
    if ret_plan is not None and not ret_plan.empty:
        exp4_df = pd.DataFrame()
        exp4_df["Customer ID"] = ret_plan[id_col] if id_col in ret_plan.columns else ret_plan.iloc[:, 0]
        exp4_df["Churn risk score / probability"] = ret_plan["Churn_Probability"].round(4) if "Churn_Probability" in ret_plan.columns else "N/A"
        exp4_df["Risk category"] = ret_plan["Risk_Category"] if "Risk_Category" in ret_plan.columns else "N/A"
        exp4_df["Estimated revenue exposure ($)"] = ret_plan["Revenue_at_Risk"].round(2) if "Revenue_at_Risk" in ret_plan.columns else 0.0
        exp4_df["Recommended retention action"] = ret_plan["Retention_Action"] if "Retention_Action" in ret_plan.columns else ret_plan.get("Recommended_Action", "Proactive check-in")
        exp4_df["Priority"] = ret_plan["Priority_Level"] if "Priority_Level" in ret_plan.columns else ret_plan.get("Risk_Category", "Medium")
        exp4_df["Intervention status"] = ret_plan["Action_Status"] if "Action_Status" in ret_plan.columns else "Pending"

        csv_ret = export_dataframe_to_csv(exp4_df)
        st.download_button(
            f"⬇️ Download Retention Plan CSV ({len(exp4_df):,} accounts)",
            data=csv_ret,
            file_name=f"retention_action_plan_{ds_name}.csv",
            mime="text/csv",
            key="dl_ret"
        )
    else:
        st.warning("⚠️ Retention plan data unavailable.")
        st.button("⬇️ Download Retention Plan (CSV)", disabled=True, key="dis_ret")

st.markdown("<br>", unsafe_allow_html=True)
c5, c6 = st.columns(2)

# =============================================================
# Export 5: Supervised Model Evaluation Summary
# =============================================================
with c5:
    st.markdown("##### 📈 5. Supervised Model Evaluation Summary")
    if trained_models:
        eval_rows = []
        ts_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for m_name, meta in trained_models.items():
            t_eval = evaluate_classification_model(
                meta["y_test"], meta["y_test_pred"], meta["y_test_prob"], dataset_label="Held-Out Test (20%)"
            )
            eval_rows.append({
                "Model name": m_name,
                "Dataset split": "Held-Out Test (20% Stratified)",
                "Accuracy": t_eval["accuracy"],
                "Precision": t_eval["precision"],
                "Recall": t_eval["recall"],
                "F1-score": t_eval["f1_score"],
                "ROC-AUC": t_eval["roc_auc"],
                "PR-AUC": t_eval["pr_auc"],
                "Evaluation timestamp": ts_now
            })
        exp5_df = pd.DataFrame(eval_rows)
        csv_bench = export_dataframe_to_csv(exp5_df)
        st.download_button(
            f"⬇️ Download Model Evaluation Summary CSV ({len(exp5_df)} models)",
            data=csv_bench,
            file_name=f"model_evaluation_summary_{ds_name}.csv",
            mime="text/csv",
            key="dl_bench"
        )
    else:
        st.warning("⚠️ Please train models in **Churn Prediction** before exporting.")
        st.button("⬇️ Download Model Evaluation Summary (CSV)", disabled=True, key="dis_bench")

# =============================================================
# Export 6: Full Data Quality Audit Report
# =============================================================
with c6:
    st.markdown("##### 📋 6. Full Data Quality Audit Report")
    inspection = inspect_dataset(df)
    cols_df = get_data_quality_df(inspection)
    csv_qual = export_dataframe_to_csv(cols_df)
    st.download_button(
        f"⬇️ Download Quality Audit CSV ({len(cols_df)} columns)",
        data=csv_qual,
        file_name=f"data_quality_report_{ds_name}.csv",
        mime="text/csv",
        key="dl_qual"
    )

# -------------------------------------------------------------
# Section 3: SQLite Report Generation History
# -------------------------------------------------------------
st.markdown("---")
st.markdown("### 📜 Report Generation History (SQLite Database)")
rep_hist_df = get_report_history()

if not rep_hist_df.empty:
    st.dataframe(
        rep_hist_df[["generated_at", "report_type", "format", "file_size_bytes", "parameters"]].rename(
            columns={
                "generated_at": "Timestamp",
                "report_type": "Report Type",
                "format": "Format",
                "file_size_bytes": "File Size (Bytes)",
                "parameters": "Parameters / Scope"
            }
        ),
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("No reports logged yet in SQLite history. Downloads above will register here.")
