"""
Customer360 — Page 7: Reports and Data Exports
Comprehensive CSV and Markdown Downloads of Analytics, Predictions, and Retention Plans.
"""

import streamlit as st
import pandas as pd

from src.ui_components import (
    apply_custom_css,
    render_page_header,
    render_sidebar,
    render_kpi,
    init_session_state
)
from src.reporting import export_dataframe_to_csv, generate_data_quality_markdown, get_data_quality_df
from src.validation import inspect_dataset, infer_column_roles
from src.model_evaluation import compare_models
from src.recommendations import generate_retention_plan

st.set_page_config(page_title="Reports & Exports — Customer360", page_icon="📄", layout="wide")
apply_custom_css()
init_session_state()
render_sidebar()

render_page_header(
    title="Analytical Reports & Intelligence Exports",
    subtitle="Export high-fidelity CSV deliverables and executive summaries reflecting completed analysis pipelines.",
    badge="Exports"
)

df = st.session_state.get("current_dataset")

if df is None or df.empty:
    st.error("No dataset available. Please upload or load a dataset.")
    st.stop()

ds_name = st.session_state.get("dataset_name", "dataset").replace(".csv", "").replace(" ", "_").lower()
roles = infer_column_roles(df)
spending_col = roles["monetary_col"]
tenure_col = roles["tenure_col"]

st.markdown("### 📦 Available Intelligence Deliverables")

# Card Grid for Reports
c1, c2 = st.columns(2)

# Deliverable 1: Churn Predictions
with c1:
    st.markdown(
        """
        <div class="content-card">
            <h4>🎯 1. Churn Predictions & Risk Scores</h4>
            <p style="color: #64748B; font-size: 0.85rem;">
                Full dataset annotated with model-generated churn probabilities, predicted outcomes, and Risk Categories (Low, Medium, High).
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    pred_df = st.session_state.get("predictions_df")
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

# Deliverable 2: High-Risk Customer Target List
with c2:
    st.markdown(
        """
        <div class="content-card">
            <h4>🚨 2. High-Risk Customer Prioritization List</h4>
            <p style="color: #64748B; font-size: 0.85rem;">
                Filtered subset isolating urgent accounts with high churn probabilities (Risk_Category == 'High') for immediate triage.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
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

st.markdown("<br>", unsafe_allow_html=True)
c3, c4 = st.columns(2)

# Deliverable 3: Customer Segmentation Results
with c3:
    st.markdown(
        """
        <div class="content-card">
            <h4>🔍 3. Customer Segmentation Profiles</h4>
            <p style="color: #64748B; font-size: 0.85rem;">
                Export customer records with assigned RFM behavioral segments or unsupervised K-Means clusters.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    rfm_df = st.session_state.get("rfm_data")
    kmeans_res = st.session_state.get("kmeans_data")

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

# Deliverable 4: Retention Action Plan
with c4:
    st.markdown(
        """
        <div class="content-card">
            <h4>🛡️ 4. Prescriptive Retention Action Plan</h4>
            <p style="color: #64748B; font-size: 0.85rem;">
                Tactical interventions, recommendation rationales, Value-at-Risk priority scores, and current session tracking statuses.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    active_source = pred_df if pred_df is not None else df
    ret_plan = generate_retention_plan(active_source, spending_col=spending_col, tenure_col=tenure_col)
    
    # Apply session updates
    actions_state = st.session_state.get("retention_actions", {})
    id_col = roles["id_col"] or "customerID"
    if id_col in ret_plan.columns:
        for idx, row in ret_plan.iterrows():
            c_id = str(row[id_col])
            if c_id in actions_state:
                ret_plan.at[idx, "Action_Status"] = actions_state[c_id]

    csv_ret = export_dataframe_to_csv(ret_plan)
    st.download_button(
        "⬇️ Download Retention Plan (CSV)",
        data=csv_ret,
        file_name=f"retention_plan_{ds_name}.csv",
        mime="text/csv",
        key="dl_ret"
    )

st.markdown("<br>", unsafe_allow_html=True)
c5, c6 = st.columns(2)

# Deliverable 5: Model Evaluation Benchmark
with c5:
    st.markdown(
        """
        <div class="content-card">
            <h4>📈 5. Supervised Model Evaluation Summary</h4>
            <p style="color: #64748B; font-size: 0.85rem;">
                Hold-out test metrics (Accuracy, Precision, Recall, F1, ROC-AUC) comparing trained algorithms side-by-side.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    trained_models = st.session_state.get("trained_models", {})
    if trained_models:
        bench_df = compare_models(list(trained_models.values()))
        csv_bench = export_dataframe_to_csv(bench_df)
        st.download_button(
            "⬇️ Download Model Evaluation Summary (CSV)",
            data=csv_bench,
            file_name=f"model_evaluation_{ds_name}.csv",
            mime="text/csv",
            key="dl_bench"
        )
    else:
        st.info("Train models to export statistical performance benchmarks.")

# Deliverable 6: Data Quality Audit Report
with c6:
    st.markdown(
        """
        <div class="content-card">
            <h4>📋 6. Full Data Governance & Quality Audit</h4>
            <p style="color: #64748B; font-size: 0.85rem;">
                Complete structural profile, missingness frequencies, duplicate counts, and schema inspection.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
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

st.markdown("---")
st.markdown("### 👁️ Live Data Quality Summary")
md_audit = generate_data_quality_markdown(inspection)
st.markdown(md_audit)
