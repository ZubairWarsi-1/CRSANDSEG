"""
Customer360 — Page 5: Data Ingestion, Cleaning & Quality Audit
Upload CSV Data, Transparent Quality Deductions, User-Controlled Data Cleaning, Analytical Readiness Badges, and SQLite Upload History.
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
    COLOR_WARNING,
    COLOR_DANGER
)
from src.data_loader import load_dataset
from src.validation import (
    inspect_dataset,
    infer_column_roles,
    calculate_transparent_quality_score,
    clean_dataset,
    check_analytical_readiness
)
from src.database import log_dataset_upload, get_upload_history
from src.reporting import get_data_quality_df, export_dataframe_to_csv

st.set_page_config(page_title="Data Upload & Audit — Customer360", page_icon="📁", layout="wide")
apply_custom_css()
init_session_state()
render_sidebar()

render_page_header(
    title="Data Ingestion & Quality Audit",
    subtitle="Upload customer datasets, inspect data hygiene, apply automated cleaning, and assess analytical readiness.",
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
            is_synthetic_flag = False
            
            # Log to SQLite upload history
            try:
                score_meta = calculate_transparent_quality_score(new_df)
                log_dataset_upload(
                    filename=dataset_source,
                    row_count=len(new_df),
                    col_count=len(new_df.columns),
                    quality_score=float(score_meta["quality_score"]),
                    dataset_type="user_upload",
                    is_synthetic=False
                )
            except Exception:
                pass
    except Exception as e:
        st.error(f"❌ Failed to parse uploaded file: {str(e)}")
        st.stop()
else:
    new_df = st.session_state.get("current_dataset")
    dataset_source = st.session_state.get("dataset_name", "Preloaded Sample")
    is_synthetic_flag = st.session_state.get("is_synthetic", False)

if new_df is None or new_df.empty:
    st.info("Please upload a CSV file or select a demo dataset from the sidebar.")
    st.stop()

# Run full quality inspection and infer roles
inspection = inspect_dataset(new_df)
roles = infer_column_roles(new_df)
quality_audit = calculate_transparent_quality_score(new_df)

# Top KPI Metric Cards
q1, q2, q3, q4, q5 = st.columns(5)
with q1:
    render_kpi("Total Records", f"{len(new_df):,}", f"{len(new_df.columns)} Features")
with q2:
    render_kpi("Quality Score", f"{quality_audit['quality_score']}/100", f"Grade: {quality_audit['grade']}")
with q3:
    render_kpi("Missing Cells", f"{inspection['total_missing_cells']:,}", f"{inspection['missing_pct']}% of cells")
with q4:
    render_kpi("Duplicate Rows", f"{inspection['duplicate_count']:,}", f"{inspection['duplicate_pct']}% duplicates")
with q5:
    render_kpi("Memory Size", f"{inspection['memory_mb']} MB", "Deep footprint")

st.markdown("<br>", unsafe_allow_html=True)

# Dataset Activation Button if user uploaded new file
if uploaded_file is not None:
    act_col1, act_col2 = st.columns([3, 1])
    with act_col1:
        st.success(f"File `{dataset_source}` validated successfully!")
    with act_col2:
        if st.button("Set as Active Platform Dataset", type="primary"):
            st.session_state["current_dataset"] = new_df
            st.session_state["dataset_name"] = dataset_source
            st.session_state["is_synthetic"] = False
            st.session_state["dataset_source_class"] = "User Uploaded Dataset"
            st.session_state["predictions_df"] = None
            st.session_state["trained_models"] = {}
            st.session_state["rfm_data"] = None
            st.session_state["kmeans_data"] = None
            st.success("Active platform dataset updated!")
            st.rerun()

# -------------------------------------------------------------
# Section 1: Analytical Readiness Assessment Badges
# -------------------------------------------------------------
st.markdown("### 🚦 Analytical Module Readiness Assessment")
readiness = check_analytical_readiness(new_df, roles)

r_col1, r_col2, r_col3 = st.columns(3)

with r_col1:
    r_churn = readiness["churn_prediction"]
    badge_type = "success" if r_churn["status"] == "Ready" else ("warning" if r_churn["status"] == "Needs Preparation" else "danger")
    st.markdown(
        f"""
        <div class="content-card" style="border-top: 4px solid {'#10B981' if badge_type=='success' else ('#F59E0B' if badge_type=='warning' else '#EF4444')};">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h5 style="margin: 0;">🎯 Churn Prediction</h5>
                <span class="badge badge-{badge_type}">{r_churn['status']}</span>
            </div>
            <p style="font-size: 0.85rem; color: #475569; margin: 0.4rem 0 0 0;">{r_churn['message']}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with r_col2:
    r_km = readiness["kmeans_clustering"]
    badge_km = "success" if r_km["status"] == "Ready" else "warning"
    st.markdown(
        f"""
        <div class="content-card" style="border-top: 4px solid {'#10B981' if badge_km=='success' else '#F59E0B'};">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h5 style="margin: 0;">🧩 K-Means Clustering</h5>
                <span class="badge badge-{badge_km}">{r_km['status']}</span>
            </div>
            <p style="font-size: 0.85rem; color: #475569; margin: 0.4rem 0 0 0;">{r_km['message']}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with r_col3:
    r_rfm = readiness["rfm_analysis"]
    badge_rfm = "success" if r_rfm["status"] == "Ready" else "info"
    st.markdown(
        f"""
        <div class="content-card" style="border-top: 4px solid {'#10B981' if badge_rfm=='success' else '#64748B'};">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h5 style="margin: 0;">🛒 Transactional RFM</h5>
                <span class="badge badge-{badge_rfm}">{r_rfm['status']}</span>
            </div>
            <p style="font-size: 0.85rem; color: #475569; margin: 0.4rem 0 0 0;">{r_rfm['message']}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# -------------------------------------------------------------
# Section 2: Data Quality Score Deductions & Transparency
# -------------------------------------------------------------
st.markdown("### 💯 Transparent Data Quality Score Breakdown")
st.caption("A baseline score of 100 is penalized strictly for data defects, structural anomalies, and mathematical violations.")

dq_col1, dq_col2 = st.columns([1, 1])

with dq_col1:
    st.markdown(
        f"""
        <div class="content-card" style="text-align: center; padding: 1.5rem;">
            <div style="font-size: 0.9rem; color: #64748B; font-weight: 600;">COMPOSITE QUALITY SCORE</div>
            <div style="font-size: 3rem; font-weight: 800; color: #1E3A8A; line-height: 1.1;">{quality_audit['quality_score']}<span style="font-size: 1.5rem; color: #94A3B8;">/100</span></div>
            <div style="margin-top: 0.5rem;"><span class="badge badge-{'success' if quality_audit['quality_score']>=80 else ('warning' if quality_audit['quality_score']>=65 else 'danger')}">Grade {quality_audit['grade']}</span></div>
            <p style="font-size: 0.82rem; color: #64748B; margin-top: 0.8rem;">{len(quality_audit['deductions'])} audit deduction items recorded.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with dq_col2:
    st.markdown("##### Deductions Applied")
    if quality_audit["deductions"]:
        ded_df = pd.DataFrame(quality_audit["deductions"])
        st.dataframe(ded_df[["penalty", "reason"]].rename(columns={"penalty": "Penalty (Pts)", "reason": "Audit Reason"}), use_container_width=True, hide_index=True)
    else:
        st.success("🎉 Zero quality deductions applied! Perfect schema hygiene.")

st.markdown("<br>", unsafe_allow_html=True)

# -------------------------------------------------------------
# Section 3: Interactive Data Cleaning Actions
# -------------------------------------------------------------
st.markdown("### 🧹 User-Controlled Data Cleaning Engine")
st.caption("Clean and prepare the current dataset without overwriting the raw data until confirmed.")

with st.expander("⚙️ Configure Data Cleaning Options", expanded=True):
    cl_col1, cl_col2 = st.columns(2)
    with cl_col1:
        opt_dups = st.checkbox("Remove Duplicate Rows", value=True, help="Removes exact duplicate rows across all features.")
        opt_types = st.checkbox("Correct Numeric Types (Fix Whitespace Strings)", value=True, help="Converts object columns like TotalCharges with spaces to float numbers.")
        opt_impute = st.checkbox("Impute Missing Values (Median for numbers, Mode for categories)", value=True)
    with cl_col2:
        opt_norm = st.checkbox("Normalize Categories (Trim whitespace)", value=True)
        opt_outliers = st.checkbox("Cap Extreme Outliers (IQR Method: 3.0×IQR threshold)", value=False, help="Clips extreme values to prevent metric skewing.")
    
    clean_btn = st.button("🚀 Execute Data Cleaning", type="primary")

if clean_btn:
    with st.spinner("Executing cleaning pipeline..."):
        cleaned_df, clean_summary = clean_dataset(
            new_df,
            remove_duplicates=opt_dups,
            impute_missing=opt_impute,
            correct_types=opt_types,
            cap_outliers=opt_outliers,
            normalize_categories=opt_norm
        )
        st.session_state["cleaned_dataset"] = cleaned_df
        st.session_state["clean_summary"] = clean_summary

if "cleaned_dataset" in st.session_state and "clean_summary" in st.session_state:
    cleaned_df = st.session_state["cleaned_dataset"]
    c_sum = st.session_state["clean_summary"]

    st.success("✅ Dataset cleaning executed successfully!")

    # Before and After Summary Table
    st.markdown("#### 📊 Cleaning Summary: Before vs After")
    sum_table = pd.DataFrame([
        {"Metric": "Row Count", "Before": f"{c_sum['initial_rows']:,}", "After": f"{c_sum['final_rows']:,}", "Net Change": f"-{c_sum['duplicates_removed']:,}"},
        {"Metric": "Missing Data Cells", "Before": f"{c_sum['initial_missing']:,}", "After": f"{c_sum['final_missing']:,}", "Net Change": f"-{c_sum['missing_imputed']:,}"},
        {"Metric": "Data Types Corrected", "Before": "—", "After": f"{c_sum['types_corrected']}", "Net Change": f"+{c_sum['types_corrected']} converted to float"},
        {"Metric": "Extreme Outliers Capped", "Before": "—", "After": f"{c_sum['outliers_capped']}", "Net Change": f"{c_sum['outliers_capped']} values clipped"},
        {"Metric": "Categories Normalized", "Before": "—", "After": f"{c_sum['categories_normalized']}", "Net Change": "Trimmed whitespace"}
    ])
    st.dataframe(sum_table, use_container_width=True, hide_index=True)

    # Actions: Download or Activate
    act_btn1, act_btn2 = st.columns(2)
    with act_btn1:
        cleaned_csv = export_dataframe_to_csv(cleaned_df)
        st.download_button(
            label="📥 Download Cleaned CSV Dataset",
            data=cleaned_csv,
            file_name=f"cleaned_{dataset_source}.csv",
            mime="text/csv",
            use_container_width=True
        )
    with act_btn2:
        if st.button("⭐ Set Cleaned Data as Active Platform Dataset", type="primary", use_container_width=True):
            st.session_state["current_dataset"] = cleaned_df
            st.session_state["dataset_name"] = f"Cleaned_{dataset_source}"
            st.session_state["predictions_df"] = None
            st.session_state["trained_models"] = {}
            st.session_state["rfm_data"] = None
            st.session_state["kmeans_data"] = None
            st.success("Cleaned dataset is now active across all platform modules!")
            st.rerun()

st.markdown("<br>", unsafe_allow_html=True)

# -------------------------------------------------------------
# Section 4: Schema & Column Quality Diagnostics Table
# -------------------------------------------------------------
st.markdown("### 📋 Column Schema Diagnostics")
cols_df = get_data_quality_df(inspection)
if not cols_df.empty:
    st.dataframe(cols_df, use_container_width=True)

# -------------------------------------------------------------
# Section 5: SQLite Upload History
# -------------------------------------------------------------
st.markdown("---")
st.markdown("### 📜 Dataset Ingestion History (SQLite Database)")
upload_history_df = get_upload_history()

if not upload_history_df.empty:
    st.dataframe(
        upload_history_df[["uploaded_at", "filename", "row_count", "col_count", "quality_score", "dataset_type"]].rename(
            columns={
                "uploaded_at": "Timestamp",
                "filename": "Filename",
                "row_count": "Rows",
                "col_count": "Columns",
                "quality_score": "Quality Score",
                "dataset_type": "Source Type"
            }
        ),
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("No prior uploads logged in SQLite history.")
