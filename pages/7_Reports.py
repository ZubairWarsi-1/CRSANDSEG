"""
Customer360 — Page 7: Reports and Data Exports
Multi-Format Deliverables (Styled Excel .xlsx, Publication-Grade PDF, One-Page Briefing PDF, and CSVs) with SQLite Report History.
"""

from datetime import datetime
import streamlit as st
import pandas as pd

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
from src.model_evaluation import compare_models, compare_models_advanced
from src.recommendations import generate_retention_plan
from src.database import get_all_interventions, get_report_history, log_report_generation

st.set_page_config(page_title="Reports & Exports — Customer360", page_icon="📄", layout="wide")
apply_custom_css()
init_session_state()
render_sidebar()

render_page_header(
    title="Analytical Reports & Intelligence Exports",
    subtitle="Export publication-ready deliverables: Styled Multi-Sheet Excel workbooks, Executive PDF Briefings, and granular CSV logs.",
    badge="Exports"
)

df = st.session_state.get("current_dataset")

if df is None or df.empty:
    st.error("No dataset available. Please upload or load a dataset.")
    st.stop()

ds_name = st.session_state.get("dataset_name", "dataset").replace(".csv", "").replace(" ", "_").lower()
is_syn = st.session_state.get("is_synthetic", False)
ds_source = "Synthetic Test Data" if is_syn else st.session_state.get("dataset_source_class", "Benchmark Sample Dataset")
roles = infer_column_roles(df)
spending_col = roles["monetary_col"]
tenure_col = roles["tenure_col"]
target_col = roles["target_col"]
id_col = roles["id_col"] or "customerID"

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
    db_map = {row["customer_id"]: row for _, row in db_interventions.iterrows()}
    for idx, row in ret_plan.iterrows():
        c_id = str(row[id_col])
        if c_id in db_map:
            ret_plan.at[idx, "Action_Status"] = db_map[c_id].get("action_status", "Pending")

# Calculate portfolio totals
total_customers = len(df)
avg_spend = float(df[spending_col].dropna().mean()) if (spending_col and pd.api.types.is_numeric_dtype(df[spending_col])) else 0.0
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

    # Build Excel sheets dictionary
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
# Section 2: Individual Granular CSV Deliverables
# -------------------------------------------------------------
st.markdown("### 📦 Granular Data Deliverables (CSV Exports)")
st.caption("Download individual processed datasets for import into downstream CRM, ERP, or marketing automation platforms.")

c1, c2 = st.columns(2)

with c1:
    st.markdown("##### 🎯 1. Churn Predictions & Risk Scores")
    if pred_df is not None:
        csv_preds = export_dataframe_to_csv(pred_df)
        st.download_button(
            "⬇️ Download Predictions (CSV)",
            data=csv_preds,
            file_name=f"churn_predictions_{ds_name}.csv",
            mime="text/csv",
            key="dl_preds"
        )
    else:
        st.info("Train a model on the **Churn Prediction** page to generate predictions.")

with c2:
    st.markdown("##### 🚨 2. High-Risk Customer Prioritization List")
    if pred_df is not None and "Risk_Category" in pred_df.columns:
        high_risk_subset = pred_df[pred_df["Risk_Category"] == "High"]
        csv_hr = export_dataframe_to_csv(high_risk_subset)
        st.download_button(
            f"⬇️ Download High-Risk List ({len(high_risk_subset):,} Accounts)",
            data=csv_hr,
            file_name=f"high_risk_customers_{ds_name}.csv",
            mime="text/csv",
            key="dl_hr"
        )
    else:
        st.info("High-risk filter will be available once churn predictions are run.")

c3, c4 = st.columns(2)

with c3:
    st.markdown("##### 🔍 3. Customer Segmentation Results")
    if rfm_df is not None:
        csv_rfm = export_dataframe_to_csv(rfm_df)
        st.download_button(
            "⬇️ Download RFM Segmentation Results (CSV)",
            data=csv_rfm,
            file_name=f"rfm_segmentation_{ds_name}.csv",
            mime="text/csv",
            key="dl_rfm"
        )
    elif kmeans_res is not None:
        csv_km = export_dataframe_to_csv(kmeans_res["dataframe"])
        st.download_button(
            "⬇️ Download K-Means Clusters (CSV)",
            data=csv_km,
            file_name=f"kmeans_clusters_{ds_name}.csv",
            mime="text/csv",
            key="dl_km"
        )
    else:
        st.info("Execute RFM or K-Means on the **Customer Segmentation** page to unlock exports.")

with c4:
    st.markdown("##### 🛡️ 4. Prescriptive Retention Action Plan")
    csv_ret = export_dataframe_to_csv(ret_plan)
    st.download_button(
        "⬇️ Download Retention Plan (CSV)",
        data=csv_ret,
        file_name=f"retention_plan_{ds_name}.csv",
        mime="text/csv",
        key="dl_ret"
    )

c5, c6 = st.columns(2)

with c5:
    st.markdown("##### 📈 5. Supervised Model Evaluation Summary")
    if trained_models:
        csv_bench = export_dataframe_to_csv(comp_df)
        st.download_button(
            "⬇️ Download Model Evaluation Summary (CSV)",
            data=csv_bench,
            file_name=f"model_evaluation_{ds_name}.csv",
            mime="text/csv",
            key="dl_bench"
        )
    else:
        st.info("Train models in **Churn Prediction** to export statistical benchmarks.")

with c6:
    st.markdown("##### 📋 6. Full Data Quality Audit")
    inspection = inspect_dataset(df)
    cols_df = get_data_quality_df(inspection)
    csv_qual = export_dataframe_to_csv(cols_df)
    st.download_button(
        "⬇️ Download Quality Audit (CSV)",
        data=csv_qual,
        file_name=f"data_quality_audit_{ds_name}.csv",
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
