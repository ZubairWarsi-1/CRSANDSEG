"""
Customer360 — Page 5: Data Ingestion & Quality Audit
Upload CSV Data, Inspect Data Types, Audit Missingness, and Map Analytical Roles.
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
from src.data_loader import load_dataset
from src.validation import inspect_dataset, infer_column_roles
from src.reporting import generate_data_quality_markdown, get_data_quality_df, export_dataframe_to_csv

st.set_page_config(page_title="Data Upload & Audit — Customer360", page_icon="📁", layout="wide")
apply_custom_css()
init_session_state()
render_sidebar()

render_page_header(
    title="Data Ingestion & Quality Audit",
    subtitle="Upload customer datasets, inspect schema health, detect anomalies, and configure analytical roles.",
    badge="Data Governance"
)

# File Uploader
uploaded_file = st.file_uploader(
    "Choose a Customer CSV File",
    type=["csv"],
    help="Upload customer subscription records or transactional order logs in CSV format."
)

if uploaded_file is not None:
    try:
        with st.spinner("Ingesting and auditing CSV file..."):
            new_df = load_dataset(uploaded_file)
            dataset_source = uploaded_file.name
    except Exception as e:
        st.error(f"❌ Failed to parse uploaded file: {str(e)}")
        st.stop()
else:
    new_df = st.session_state.get("current_dataset")
    dataset_source = st.session_state.get("dataset_name", "Preloaded Sample")

if new_df is None or new_df.empty:
    st.info("Please upload a CSV file or select a demo dataset from the sidebar.")
    st.stop()

# Run full quality inspection
inspection = inspect_dataset(new_df)
roles = infer_column_roles(new_df)

# Quality Score Cards
q1, q2, q3, q4, q5 = st.columns(5)
with q1:
    render_kpi("Total Records", f"{inspection['row_count']:,}", f"{inspection['col_count']} Features")
with q2:
    render_kpi("Data Quality Score", f"{inspection['quality_score']}/100", f"Grade: {inspection['grade']}")
with q3:
    render_kpi("Missing Values", f"{inspection['total_missing_cells']:,}", "Null data cells")
with q4:
    render_kpi("Duplicate Rows", f"{inspection['duplicate_count']:,}", f"{inspection['duplicate_pct']}% of records")
with q5:
    render_kpi("Memory Size", f"{inspection['memory_mb']} MB", "Deep footprint")

st.markdown("<br>", unsafe_allow_html=True)

# Dataset Activation
if uploaded_file is not None:
    act_col1, act_col2 = st.columns([3, 1])
    with act_col1:
        st.success(f"File `{dataset_source}` validated successfully!")
    with act_col2:
        if st.button("Set as Active Dataset", type="primary"):
            st.session_state["current_dataset"] = new_df
            st.session_state["dataset_name"] = dataset_source
            st.session_state["is_synthetic"] = False
            st.session_state["predictions_df"] = None
            st.session_state["trained_models"] = {}
            st.session_state["rfm_data"] = None
            st.session_state["kmeans_data"] = None
            st.success("Active platform dataset updated!")
            st.rerun()

# Data Preview
st.markdown("### 📋 Dataset Preview (First 10 Rows)")
st.dataframe(new_df.head(10), use_container_width=True)

# Schema & Missingness Table
st.markdown("### 🔍 Column Schema & Quality Diagnostics")
cols_df = get_data_quality_df(inspection)
if not cols_df.empty:
    st.dataframe(cols_df, use_container_width=True)

# Analytical Column Mapping Tool
st.markdown("### 🗺️ Analytical Role Mapping")
st.caption("Confirm or override auto-inferred column roles used by downstream machine learning and segmentation algorithms.")

col_m1, col_m2, col_m3, col_m4 = st.columns(4)

with col_m1:
    customer_id_col = st.selectbox(
        "Customer Identifier",
        options=["None"] + list(new_df.columns),
        index=(list(new_df.columns).index(roles["id_col"]) + 1) if roles["id_col"] in new_df.columns else 0
    )

with col_m2:
    churn_target_col = st.selectbox(
        "Historical Churn Target",
        options=["None"] + list(new_df.columns),
        index=(list(new_df.columns).index(roles["target_col"]) + 1) if roles["target_col"] in new_df.columns else 0
    )

with col_m3:
    spending_col = st.selectbox(
        "Spending / Monetary",
        options=["None"] + list(new_df.columns),
        index=(list(new_df.columns).index(roles["monetary_col"]) + 1) if roles["monetary_col"] in new_df.columns else 0
    )

with col_m4:
    tenure_col = st.selectbox(
        "Tenure / Account Age",
        options=["None"] + list(new_df.columns),
        index=(list(new_df.columns).index(roles["tenure_col"]) + 1) if roles["tenure_col"] in new_df.columns else 0
    )

# Export Quality Audit
st.markdown("---")
st.markdown("### 📥 Download Quality Audit Reports")

rep_col1, rep_col2 = st.columns(2)

with rep_col1:
    csv_bytes = export_dataframe_to_csv(cols_df)
    st.download_button(
        "📄 Download Column Schema Audit (CSV)",
        data=csv_bytes,
        file_name=f"data_quality_audit_{dataset_source.replace('.csv','')}.csv",
        mime="text/csv"
    )

with rep_col2:
    md_text = generate_data_quality_markdown(inspection)
    st.download_button(
        "📝 Download Executive Quality Report (Markdown)",
        data=md_text.encode("utf-8"),
        file_name=f"data_quality_report_{dataset_source.replace('.csv','')}.md",
        mime="text/markdown"
    )
