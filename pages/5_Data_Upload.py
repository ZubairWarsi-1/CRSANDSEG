"""
Customer360 — Page 5: Data Ingestion, Cleaning & Quality Audit
Upload CSV Data, Transparent Quality Deductions, Interactive Role Mapping,
User-Controlled Data Cleaning, Analytical Readiness Badges, and SQLite Upload History.
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
    subtitle="Upload customer datasets, inspect data hygiene, map schema roles, apply automated cleaning, and assess analytical readiness.",
    badge="Data Governance"
)

# -------------------------------------------------------------
# Step 1 & 2: File Upload and Validation
# -------------------------------------------------------------
uploaded_file = st.file_uploader(
    "Choose a Customer CSV File",
    type=["csv"],
    help="Upload customer subscription records or transactional order logs in CSV format (Max 200MB)."
)

new_df = None
dataset_source = None
is_synthetic_flag = False

if uploaded_file is not None:
    # Validate extension and non-empty size
    if not uploaded_file.name.lower().endswith(".csv"):
        st.error("❌ Unsupported file format. Please upload a standard CSV (.csv) file.")
        st.stop()
    
    if uploaded_file.size == 0:
        st.error("❌ The uploaded file is empty (0 bytes). Please upload a valid CSV dataset.")
        st.stop()

    try:
        with st.spinner("Ingesting and auditing CSV file..."):
            new_df = load_dataset(uploaded_file, clean_whitespace_numeric=False)
            dataset_source = uploaded_file.name
            is_synthetic_flag = False
            
            # Log to SQLite upload history
            try:
                score_meta = calculate_transparent_quality_score(new_df)
                log_dataset_upload(
                    filename=dataset_source,
                    row_count=len(new_df),
                    col_count=len(new_df.columns),
                    quality_score=float(score_meta.get("quality_score", 0.0)),
                    dataset_type="user_upload",
                    is_synthetic=False
                )
            except Exception:
                pass
    except Exception as e:
        st.error(f"❌ Failed to parse uploaded file: {str(e)}")
        st.info("Tip: Ensure the file is encoded in UTF-8, Latin-1, or CP1252 with standard comma separation.")
        st.stop()
else:
    new_df = st.session_state.get("current_dataset")
    dataset_source = st.session_state.get("dataset_name", "Preloaded Sample")
    is_synthetic_flag = st.session_state.get("is_synthetic", False)

if new_df is None or not isinstance(new_df, pd.DataFrame) or new_df.empty or len(new_df.columns) == 0:
    st.info("👋 No dataset currently loaded. Please upload a CSV file above or select a demo dataset from the sidebar navigation.")
    st.stop()

# -------------------------------------------------------------
# Run Full Quality Inspection and Infer Column Roles
# -------------------------------------------------------------
inspection = inspect_dataset(new_df)
roles = infer_column_roles(new_df)
quality_audit = calculate_transparent_quality_score(new_df)

# Top KPI Metric Cards (using safe .get() to prevent KeyErrors)
q1, q2, q3, q4, q5 = st.columns(5)
with q1:
    render_kpi("Total Records", f"{inspection.get('total_records', len(new_df)):,}", f"{inspection.get('total_features', len(new_df.columns))} Features")
with q2:
    q_score = quality_audit.get("quality_score", 0.0)
    q_grade = quality_audit.get("grade", "N/A")
    render_kpi("Quality Score", f"{q_score}/100", f"Grade: {q_grade}")
with q3:
    m_cells = inspection.get("total_missing_cells", 0)
    m_pct = inspection.get("missing_pct", 0.0)
    render_kpi("Missing Cells", f"{m_cells:,}", f"{m_pct}% of total cells")
with q4:
    d_rows = inspection.get("duplicate_count", inspection.get("duplicate_records", 0))
    d_pct = inspection.get("duplicate_pct", 0.0)
    render_kpi("Duplicate Rows", f"{d_rows:,}", f"{d_pct}% duplicates")
with q5:
    mem = inspection.get("memory_mb", 0.0)
    render_kpi("Memory Footprint", f"{mem} MB", "Deep footprint")

st.markdown("<br>", unsafe_allow_html=True)

# Dataset Activation Button if user uploaded new file
if uploaded_file is not None:
    act_col1, act_col2 = st.columns([3, 1])
    with act_col1:
        st.success(f"File `{dataset_source}` validated successfully ({len(new_df):,} rows × {len(new_df.columns)} columns).")
    with act_col2:
        if st.button("Set as Active Platform Dataset", type="primary", use_container_width=True):
            st.session_state["current_dataset"] = new_df
            st.session_state["dataset_name"] = dataset_source
            st.session_state["is_synthetic"] = False
            st.session_state["dataset_source_class"] = "User Uploaded Dataset"
            st.session_state["predictions_df"] = None
            st.session_state["trained_models"] = {}
            st.session_state["rfm_data"] = None
            st.session_state["kmeans_data"] = None
            st.session_state["column_roles"] = roles
            st.success("Active platform dataset updated!")
            st.rerun()

# -------------------------------------------------------------
# Step 3: Raw Data Preview
# -------------------------------------------------------------
with st.expander(f"👁️ Raw Data Preview: First 10 Rows (Shape: {new_df.shape[0]:,} rows × {new_df.shape[1]} columns)", expanded=False):
    st.dataframe(new_df.head(10), use_container_width=True)

# -------------------------------------------------------------
# Step 4, 5, 6, 7 & 13: Analytical Readiness Assessment Badges
# -------------------------------------------------------------
st.markdown("### 🚦 Analytical Module Readiness Assessment")
readiness = check_analytical_readiness(new_df, roles)

r_col1, r_col2, r_col3 = st.columns(3)

with r_col1:
    r_churn = readiness.get("churn_prediction", {"status": "Needs Preparation", "message": "Evaluating prerequisites..."})
    badge_type = "success" if r_churn.get("status") == "Ready" else ("warning" if r_churn.get("status") == "Needs Preparation" else "danger")
    st.markdown(
        f"""
        <div class="content-card" style="border-top: 4px solid {'#10B981' if badge_type=='success' else ('#F59E0B' if badge_type=='warning' else '#EF4444')};">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h5 style="margin: 0;">🎯 Churn Prediction</h5>
                <span class="badge badge-{badge_type}">{r_churn.get('status', 'Unknown')}</span>
            </div>
            <p style="font-size: 0.85rem; color: #475569; margin: 0.4rem 0 0 0;">{r_churn.get('message', '')}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with r_col2:
    r_km = readiness.get("kmeans_clustering", {"status": "Needs Preparation", "message": "Evaluating prerequisites..."})
    badge_km = "success" if r_km.get("status") == "Ready" else "warning"
    st.markdown(
        f"""
        <div class="content-card" style="border-top: 4px solid {'#10B981' if badge_km=='success' else '#F59E0B'};">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h5 style="margin: 0;">🧩 K-Means Clustering</h5>
                <span class="badge badge-{badge_km}">{r_km.get('status', 'Unknown')}</span>
            </div>
            <p style="font-size: 0.85rem; color: #475569; margin: 0.4rem 0 0 0;">{r_km.get('message', '')}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with r_col3:
    r_rfm = readiness.get("rfm_analysis", {"status": "Incompatible", "message": "Evaluating prerequisites..."})
    badge_rfm = "success" if r_rfm.get("status") == "Ready" else "info"
    st.markdown(
        f"""
        <div class="content-card" style="border-top: 4px solid {'#10B981' if badge_rfm=='success' else '#64748B'};">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h5 style="margin: 0;">🛒 Transactional RFM</h5>
                <span class="badge badge-{badge_rfm}">{r_rfm.get('status', 'Unknown')}</span>
            </div>
            <p style="font-size: 0.85rem; color: #475569; margin: 0.4rem 0 0 0;">{r_rfm.get('message', '')}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# -------------------------------------------------------------
# Step 8: Interactive Column Role Mapping
# -------------------------------------------------------------
st.markdown("### 🏷️ Interactive Column Role Mapping")
st.caption("Review or reassign the operational roles detected for each column. Confirmed mappings persist across all platform pages.")

existing_roles = st.session_state.get("column_roles", roles)
cols_list = ["(None)"] + list(new_df.columns)

with st.expander("⚙️ Configure & Save Column Role Mapping", expanded=True):
    m_col1, m_col2, m_col3 = st.columns(3)
    
    with m_col1:
        # Customer ID
        curr_id = existing_roles.get("id_col") if existing_roles.get("id_col") in new_df.columns else "(None)"
        id_idx = cols_list.index(curr_id) if curr_id in cols_list else 0
        selected_id = st.selectbox("Customer Identifier Column", cols_list, index=id_idx, help="Unique identifier for each customer.")

        # Churn Target
        curr_target = existing_roles.get("target_col") if existing_roles.get("target_col") in new_df.columns else "(None)"
        target_idx = cols_list.index(curr_target) if curr_target in cols_list else 0
        selected_target = st.selectbox("Churn Target Column (Yes/No, 1/0)", cols_list, index=target_idx, help="Binary label indicating historical customer churn.")

    with m_col2:
        # Tenure
        curr_tenure = existing_roles.get("tenure_col") if existing_roles.get("tenure_col") in new_df.columns else "(None)"
        tenure_idx = cols_list.index(curr_tenure) if curr_tenure in cols_list else 0
        selected_tenure = st.selectbox("Tenure Column (Months)", cols_list, index=tenure_idx, help="Customer relationship duration in months.")

        # Monthly Charges
        curr_mc = existing_roles.get("monetary_col") if existing_roles.get("monetary_col") in new_df.columns else "(None)"
        mc_idx = cols_list.index(curr_mc) if curr_mc in cols_list else 0
        selected_mc = st.selectbox("Monthly Charges Column ($)", cols_list, index=mc_idx, help="Current recurring monthly spending.")

    with m_col3:
        # Total Charges
        detected_tc = None
        for c in new_df.columns:
            if "total" in c.lower() and ("charge" in c.lower() or "amount" in c.lower() or "spend" in c.lower()):
                detected_tc = c
                break
        curr_tc = existing_roles.get("total_charges_col", detected_tc) if existing_roles.get("total_charges_col", detected_tc) in new_df.columns else "(None)"
        tc_idx = cols_list.index(curr_tc) if curr_tc in cols_list else 0
        selected_tc = st.selectbox("Total Charges Column ($)", cols_list, index=tc_idx, help="Cumulative historical spend.")

        # Contract Type
        curr_contract = existing_roles.get("contract_col") if existing_roles.get("contract_col") in new_df.columns else "(None)"
        con_idx = cols_list.index(curr_contract) if curr_contract in cols_list else 0
        selected_contract = st.selectbox("Contract Type Column", cols_list, index=con_idx, help="Contract duration (e.g. Month-to-month, One year).")

    if st.button("💾 Save Column Role Mapping", type="secondary"):
        saved_roles = {
            "id_col": None if selected_id == "(None)" else selected_id,
            "target_col": None if selected_target == "(None)" else selected_target,
            "tenure_col": None if selected_tenure == "(None)" else selected_tenure,
            "monetary_col": None if selected_mc == "(None)" else selected_mc,
            "total_charges_col": None if selected_tc == "(None)" else selected_tc,
            "contract_col": None if selected_contract == "(None)" else selected_contract
        }
        st.session_state["column_roles"] = saved_roles
        st.success("Column roles saved successfully and synced with analytical models!")
        st.rerun()

st.markdown("<br>", unsafe_allow_html=True)

# -------------------------------------------------------------
# Step 9: Transparent Quality Score & Deductions
# -------------------------------------------------------------
st.markdown("### 💯 Transparent Data Quality Score Breakdown")
st.caption("A baseline score of 100 is penalized strictly for data defects, structural anomalies, and mathematical violations.")

dq_col1, dq_col2 = st.columns([1, 1])

with dq_col1:
    st.markdown(
        f"""
        <div class="content-card" style="text-align: center; padding: 1.5rem;">
            <div style="font-size: 0.9rem; color: #64748B; font-weight: 600;">COMPOSITE QUALITY SCORE</div>
            <div style="font-size: 3rem; font-weight: 800; color: #1E3A8A; line-height: 1.1;">{quality_audit.get('quality_score', 0)}<span style="font-size: 1.5rem; color: #94A3B8;">/100</span></div>
            <div style="margin-top: 0.5rem;"><span class="badge badge-{'success' if quality_audit.get('quality_score', 0)>=90 else ('primary' if quality_audit.get('quality_score', 0)>=80 else ('warning' if quality_audit.get('quality_score', 0)>=70 else 'danger'))}">Grade {quality_audit.get('grade', 'Poor')}</span></div>
            <p style="font-size: 0.82rem; color: #64748B; margin-top: 0.8rem;">{len(quality_audit.get('deductions', []))} audit deduction items recorded.</p>
            <p style="font-size: 0.8rem; color: #334155; font-style: italic;">{quality_audit.get('grade_description', '')}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with dq_col2:
    st.markdown("##### Itemized Deductions Applied")
    deductions_list = quality_audit.get("deductions", [])
    if deductions_list:
        ded_df = pd.DataFrame(deductions_list)
        st.dataframe(ded_df[["penalty", "reason"]].rename(columns={"penalty": "Deduction Item", "reason": "Audit Reason & Impact"}), use_container_width=True, hide_index=True)
    else:
        st.success("🎉 Zero quality deductions applied! Perfect schema hygiene and feature completeness.")

st.markdown("<br>", unsafe_allow_html=True)

# -------------------------------------------------------------
# Step 9 (cont): Automated Cleaning Recommendations
# -------------------------------------------------------------
st.markdown("### 💡 Automated Cleaning Recommendations")
recs = []
if inspection.get("total_missing_cells", 0) > 0:
    recs.append(f"• **Impute Missing Values:** {inspection.get('total_missing_cells', 0):,} missing cells detected ({inspection.get('missing_pct', 0.0)}% of dataset). Recommend median imputation for numerical features and mode for categoricals.")
if inspection.get("duplicate_count", 0) > 0:
    recs.append(f"• **Remove Duplicate Records:** {inspection.get('duplicate_count', 0):,} exact duplicate rows detected ({inspection.get('duplicate_pct', 0.0)}%). Removing duplicates prevents false statistical weighting.")
if inspection.get("inconsistent_type_cols"):
    incon_names = ", ".join([c["column"] for c in inspection.get("inconsistent_type_cols", [])])
    recs.append(f"• **Convert Inconsistent Types:** Columns ({incon_names}) contain numeric amounts stored as whitespace text or dollar strings. Convert to proper float numbers.")
if inspection.get("impossible_tenure_count", 0) > 0:
    recs.append(f"• **Cap Impossible Tenure:** {inspection.get('impossible_tenure_count', 0)} rows have tenure values outside realistic bounds (<0 or >120 months). Cap tenure to [0, 120].")
if inspection.get("negative_value_cols"):
    neg_cols = ", ".join([c["column"] for c in inspection.get("negative_value_cols", [])])
    recs.append(f"• **Correct Negative Financial Values:** Columns ({neg_cols}) contain negative charges. Clip negative values to zero.")
if not recs:
    recs.append("• **No Critical Defects Detected:** Data is well-structured and ready for analytical processing.")

for r in recs:
    st.markdown(r)

st.markdown("<br>", unsafe_allow_html=True)

# -------------------------------------------------------------
# Step 10: User-Controlled Data Cleaning Engine
# -------------------------------------------------------------
st.markdown("### 🧹 User-Controlled Data Cleaning Engine")
st.caption("Clean and prepare the current dataset without overwriting the raw data until confirmed.")

with st.expander("⚙️ Configure Data Cleaning Options", expanded=True):
    cl_col1, cl_col2 = st.columns(2)
    with cl_col1:
        opt_dups = st.checkbox("Remove Duplicate Rows", value=True, help="Removes exact duplicate rows across all features.")
        opt_types = st.checkbox("Correct Numeric Types (Fix Whitespace & Currency Strings)", value=True, help="Converts object columns like TotalCharges with spaces to float numbers.")
        opt_impute = st.checkbox("Impute Missing Values (Median for numbers, Mode for categories)", value=True)
    with cl_col2:
        opt_norm = st.checkbox("Normalize Categories (Trim whitespace and casing)", value=True)
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

# -------------------------------------------------------------
# Step 11 & 12: Cleaned Data Preview, Comparison & Download
# -------------------------------------------------------------
if "cleaned_dataset" in st.session_state and "clean_summary" in st.session_state:
    cleaned_df = st.session_state["cleaned_dataset"]
    c_sum = st.session_state["clean_summary"]

    st.success("✅ Dataset cleaning executed successfully!")

    # Before and After Summary Table
    st.markdown("#### 📊 Cleaning Summary: Before vs After")
    sum_table = pd.DataFrame([
        {"Metric": "Row Count", "Before": f"{c_sum.get('initial_rows', 0):,}", "After": f"{c_sum.get('final_rows', 0):,}", "Net Change": f"-{c_sum.get('duplicates_removed', 0):,}"},
        {"Metric": "Missing Data Cells", "Before": f"{c_sum.get('initial_missing', 0):,}", "After": f"{c_sum.get('final_missing', 0):,}", "Net Change": f"-{c_sum.get('missing_imputed', 0):,}"},
        {"Metric": "Data Types Corrected", "Before": "—", "After": f"{c_sum.get('types_corrected', 0)}", "Net Change": f"+{c_sum.get('types_corrected', 0)} converted to numeric"},
        {"Metric": "Extreme Outliers Capped", "Before": "—", "After": f"{c_sum.get('outliers_capped', 0)}", "Net Change": f"{c_sum.get('outliers_capped', 0)} values clipped"},
        {"Metric": "Categories Normalized", "Before": "—", "After": f"{c_sum.get('categories_normalized', 0)}", "Net Change": "Trimmed whitespace and casing"}
    ])
    st.dataframe(sum_table, use_container_width=True, hide_index=True)

    # Actions: Download Cleaned CSV (UTF-8-SIG) or Activate
    act_btn1, act_btn2 = st.columns(2)
    with act_btn1:
        cleaned_csv = export_dataframe_to_csv(cleaned_df)
        st.download_button(
            label="📥 Download Cleaned CSV Dataset",
            data=cleaned_csv,
            file_name=f"cleaned_{dataset_source}.csv" if not str(dataset_source).endswith(".csv") else f"cleaned_{dataset_source}",
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
    st.dataframe(cols_df, use_container_width=True, hide_index=True)

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
