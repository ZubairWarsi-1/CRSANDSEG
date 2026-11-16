"""
Customer360 — Customer Segmentation & Retention Intelligence
Main Entry Point and Platform Overview
"""

import streamlit as st
import pandas as pd
from src.ui_components import (
    apply_custom_css,
    render_page_header,
    render_sidebar,
    init_session_state
)

st.set_page_config(
    page_title="Customer360 — Customer Intelligence Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply global styles
apply_custom_css()
init_session_state()
render_sidebar()

# Page Banner
render_page_header(
    title="Customer360 — Customer Intelligence Platform",
    subtitle="End-to-End Customer Segmentation, Predictive Churn Intelligence & Automated Retention Strategy",
    badge="Enterprise ML v1.0"
)

# Active dataset alert banner
df = st.session_state["current_dataset"]
ds_name = st.session_state["dataset_name"]
is_syn = st.session_state["is_synthetic"]

if is_syn:
    st.warning("⚠️ **Notice:** You are currently viewing a synthetic test dataset (**Synthetic Test Data**). In production, connect real transactional or subscription data.")
else:
    source_class = st.session_state.get("dataset_source_class", "Benchmark Sample Dataset")
    st.success(f"✅ **Active Dataset:** Loaded `{ds_name}` (*{source_class}*) with **{len(df):,}** records and **{len(df.columns)}** features.")

# Executive Overview Columns
st.markdown("### 🏛️ Platform Architecture & Analytical Capabilities")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div class="content-card">
            <h4>🔍 1. Customer Segmentation</h4>
            <p style="color: #64748B; font-size: 0.9rem;">
                Dual-method segmentation engine:
            </p>
            <ul style="color: #475569; font-size: 0.85rem; padding-left: 1.2rem;">
                <li><strong>RFM Analysis:</strong> Quantifies Recency, Frequency, and Monetary value for transactional order data.</li>
                <li><strong>K-Means Clustering:</strong> Discovers unsupervised natural behavioral cohorts with Silhouette and Elbow validation.</li>
                <li><strong>Interpretable Profiles:</strong> Automatically generates human-understandable segment descriptors.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <div class="content-card">
            <h4>🎯 2. Churn Prediction</h4>
            <p style="color: #64748B; font-size: 0.9rem;">
                Supervised machine learning pipeline:
            </p>
            <ul style="color: #475569; font-size: 0.85rem; padding-left: 1.2rem;">
                <li><strong>Balanced Modeling:</strong> Logistic Regression & Random Forest with class-imbalance compensation.</li>
                <li><strong>Leakage Prevention:</strong> Scikit-learn Pipelines and ColumnTransformers.</li>
                <li><strong>Explainable Predictions:</strong> Identifies top model risk factors and protective signals per customer.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        """
        <div class="content-card">
            <h4>🛡️ 3. Retention Center</h4>
            <p style="color: #64748B; font-size: 0.9rem;">
                Prescriptive action engine:
            </p>
            <ul style="color: #475569; font-size: 0.85rem; padding-left: 1.2rem;">
                <li><strong>Value-at-Risk Scoring:</strong> Multiplies churn probability by financial spending to prioritize intervention.</li>
                <li><strong>Transparent Rules:</strong> Tailored strategies (Win-Back, Loyalty Perks, Onboarding).</li>
                <li><strong>Session Workflow:</strong> Track action statuses (Pending, In Progress, Completed).</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")

# Quick Navigation Guide
st.markdown("### 🗺️ Application Roadmap")

nav_col1, nav_col2 = st.columns(2)

with nav_col1:
    st.markdown(
        """
        - **[Page 1: 📊 Dashboard Overview](Dashboard)**: Executive KPIs, active/inactive counts, churn distribution, and automatic business insights.
        - **[Page 2: 🔍 Customer Segmentation](Segmentation)**: Interactive RFM analysis and K-Means clustering with PCA 2D scatter plots.
        - **[Page 3: 🎯 Churn Prediction](Churn_Prediction)**: Supervised ML model training, individual risk scores, and contributing factors.
        - **[Page 4: 🛡️ Retention Center](Retention_Center)**: High-risk customer prioritization, tailored recommendations, and action tracking.
        """
    )

with nav_col2:
    st.markdown(
        """
        - **[Page 5: 📁 Data Upload](Data_Upload)**: Ingest CSV files, inspect schemas, check missing values, and review quality scores.
        - **[Page 6: 📈 Model Performance](Model_Performance)**: ROC Curves, confusion matrices, precision/recall, and feature importance.
        - **[Page 7: 📄 Reports & Exports](Reports)**: Export segmentation lists, churn predictions, and audit summaries as CSVs.
        """
    )

st.markdown("---")

# Quick Dataset Inspection
st.markdown("### 📋 Active Dataset Preview")
st.dataframe(df.head(5), use_container_width=True)

st.markdown("---")

# Project Team & Academic Credits Card
st.markdown(
    """
    <div class="content-card" style="border-top: 4px solid #1E3A8A; margin-top: 1.5rem;">
        <h4 style="margin: 0 0 0.4rem 0; color: #1E3A8A;">🎓 Project Developed By</h4>
        <p style="font-size: 0.85rem; color: #64748B; margin-bottom: 0.8rem;">
            Academic Capstone Project: <em>Customer360 — Customer Segmentation & Retention Intelligence Platform</em>
        </p>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 1rem;">
            <div style="background: #F8FAFC; padding: 0.9rem; border-radius: 8px; border: 1px solid #E2E8F0;">
                <div style="font-weight: 700; color: #0F172A; font-size: 1rem;">👤 Zubair Warsi</div>
                <div style="color: #475569; font-size: 0.88rem; margin-top: 0.3rem;">Roll No: <code>2303410100057</code></div>
            </div>
            <div style="background: #F8FAFC; padding: 0.9rem; border-radius: 8px; border: 1px solid #E2E8F0;">
                <div style="font-weight: 700; color: #0F172A; font-size: 1rem;">👤 Sameer Mansoori</div>
                <div style="color: #475569; font-size: 0.88rem; margin-top: 0.3rem;">Roll No: <code>2303410100040</code></div>
            </div>
            <div style="background: #F8FAFC; padding: 0.9rem; border-radius: 8px; border: 1px solid #E2E8F0;">
                <div style="font-weight: 700; color: #0F172A; font-size: 1rem;">👤 Mohd Shahbaaz Khan</div>
                <div style="color: #475569; font-size: 0.88rem; margin-top: 0.3rem;">Roll No: <code>2303410100029</code></div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

