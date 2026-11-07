"""
Customer360 — Page 2: Customer Segmentation
RFM Analysis for Transaction Records & K-Means Clustering for Behavioral Feature Cohorts.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

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
from src.segmentation import calculate_rfm, perform_kmeans_clustering
from src.validation import validate_for_rfm, validate_for_clustering, infer_column_roles

st.set_page_config(page_title="Customer Segmentation — Customer360", page_icon="🔍", layout="wide")
apply_custom_css()
init_session_state()
render_sidebar()

render_page_header(
    title="Customer Segmentation Intelligence",
    subtitle="Segment your customer base using behavioral RFM analysis or unsupervised K-Means clustering.",
    badge="Unsupervised Analytics"
)

df = st.session_state.get("current_dataset")

if df is None or df.empty:
    st.error("No dataset available. Please upload a dataset in Data Upload or select a demo.")
    st.stop()

roles = infer_column_roles(df)

# Tabs: Method Selection
tab_kmeans, tab_rfm = st.tabs(["🧩 K-Means Feature Clustering", "🛒 Transactional RFM Analysis"])

# =========================================================================
# TAB 1: K-MEANS CLUSTERING
# =========================================================================
with tab_kmeans:
    st.markdown("### 🧩 Unsupervised K-Means Behavioral Clustering")
    st.caption("Group customers based on multi-dimensional numerical features (e.g., tenure, monthly spending, total charges).")

    # Detect numerical columns
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 2]
    
    # Filter out potential index/ID columns
    valid_features = [c for c in numeric_cols if not c.lower().endswith("id")]

    if len(valid_features) < 2:
        st.warning("Current dataset contains fewer than 2 continuous numerical features required for K-Means clustering.")
    else:
        col_ctrl1, col_ctrl2 = st.columns([2, 1])

        with col_ctrl1:
            default_feats = valid_features[:3] if len(valid_features) >= 3 else valid_features
            selected_features = st.multiselect(
                "Select Numerical Features for Clustering",
                options=valid_features,
                default=default_feats,
                help="Choose continuous attributes that describe customer value or engagement."
            )

        with col_ctrl2:
            n_clusters = st.slider("Select Number of Clusters (k)", min_value=2, max_value=6, value=4)

        if len(selected_features) >= 2:
            is_valid, msg = validate_for_clustering(df, selected_features)
            if not is_valid:
                st.error(f"Validation Error: {msg}")
            else:
                with st.spinner("Executing K-Means clustering and computing silhouette diagnostics..."):
                    kmeans_result = perform_kmeans_clustering(
                        df,
                        feature_cols=selected_features,
                        n_clusters=n_clusters,
                        random_state=42
                    )
                    st.session_state["kmeans_data"] = kmeans_result

                res_df = kmeans_result["dataframe"]
                sil_score = kmeans_result["silhouette_score"]
                elbow_df = kmeans_result["elbow_data"]
                profiles_df = kmeans_result["cluster_profiles"]
                pca_var = kmeans_result["pca_variance"]

                # Metrics banner
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    render_kpi("Optimal Clusters (k)", n_clusters, "User selected")
                with m2:
                    sil_quality = "Good" if sil_score >= 0.5 else ("Moderate" if sil_score >= 0.3 else "Weak")
                    render_kpi("Silhouette Score", sil_score, f"Separation: {sil_quality}")
                with m3:
                    render_kpi("Total Clustered", f"{len(res_df):,}", "Customers")
                with m4:
                    render_kpi("PCA Variance", f"{sum(pca_var):.1f}%", f"PC1: {pca_var[0]}%, PC2: {pca_var[1]}%")

                st.markdown("<br>", unsafe_allow_html=True)

                # Visualizations
                plot_col1, plot_col2 = st.columns([3, 2])

                with plot_col1:
                    st.markdown("#### 🗺️ 2D PCA Cluster Map")
                    fig_pca = px.scatter(
                        res_df,
                        x="PCA1",
                        y="PCA2",
                        color="ClusterName",
                        hover_data=[roles["id_col"]] + selected_features if roles["id_col"] else selected_features,
                        title=f"Customer Clusters Projected on First 2 Principal Components ({sum(pca_var):.1f}% Variance)",
                        color_discrete_sequence=[COLOR_PRIMARY, COLOR_SUCCESS, COLOR_WARNING, COLOR_DANGER, COLOR_ACCENT, COLOR_SECONDARY]
                    )
                    fig_pca.update_layout(get_plotly_layout(height=420))
                    st.plotly_chart(fig_pca, use_container_width=True)

                with plot_col2:
                    st.markdown("#### 📉 Elbow & Silhouette Diagnostics")
                    fig_elbow = go.Figure()
                    fig_elbow.add_trace(go.Scatter(
                        x=elbow_df["k"],
                        y=elbow_df["inertia"],
                        name="Inertia (Elbow)",
                        line=dict(color=COLOR_PRIMARY, width=2),
                        yaxis="y1"
                    ))
                    fig_elbow.add_trace(go.Scatter(
                        x=elbow_df["k"],
                        y=elbow_df["silhouette"],
                        name="Silhouette Score",
                        line=dict(color=COLOR_SUCCESS, width=2, dash="dash"),
                        yaxis="y2"
                    ))
                    fig_elbow.update_layout(
                        get_plotly_layout(height=420),
                        yaxis=dict(
                            title=dict(text="Inertia", font=dict(color=COLOR_PRIMARY)),
                            tickfont=dict(color=COLOR_PRIMARY)
                        ),
                        yaxis2=dict(
                            title=dict(text="Silhouette Score", font=dict(color=COLOR_SUCCESS)),
                            tickfont=dict(color=COLOR_SUCCESS),
                            overlaying="y",
                            side="right"
                        ),
                        legend=dict(x=0.1, y=1.1, orientation="h")
                    )
                    st.plotly_chart(fig_elbow, use_container_width=True)

                # Cluster Profiles Table
                st.markdown("### 📊 Cluster Profile Characteristics")
                st.dataframe(profiles_df, use_container_width=True)

                # Drill down customer table
                st.markdown("### 📋 Segmented Customers")
                selected_cluster = st.selectbox("Filter by Cluster", ["All"] + list(profiles_df["ClusterName"].unique()))
                view_df = res_df if selected_cluster == "All" else res_df[res_df["ClusterName"] == selected_cluster]
                
                show_cols = ([roles["id_col"]] if roles["id_col"] else []) + ["ClusterName"] + selected_features
                if roles["target_col"] and roles["target_col"] in res_df.columns:
                    show_cols.append(roles["target_col"])
                st.dataframe(view_df[show_cols].head(100), use_container_width=True)
        else:
            st.info("Please select at least 2 features to run clustering.")

# =========================================================================
# TAB 2: RFM ANALYSIS
# =========================================================================
with tab_rfm:
    st.markdown("### 🛒 Transaction-Level RFM (Recency, Frequency, Monetary) Analysis")
    st.caption("Computes behavioral value from purchase timestamps and order totals.")

    # Check suitability of current dataset for RFM
    id_candidates = [c for c in df.columns if "id" in c.lower() or "cust" in c.lower()]
    date_candidates = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
    monetary_candidates = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    if not date_candidates:
        st.info("💡 **Notice:** The active dataset does not contain transactional timestamp columns required for RFM.")
        st.markdown(
            """
            The **IBM Telco dataset** is a subscription-level dataset containing tenure and monthly recurring charges, suitable for **K-Means Clustering** and **Churn Prediction**.
            
            To test **RFM Analysis**, select **'Retail Transactions (RFM)'** from the sidebar switcher or upload your order log CSV.
            """
        )
        if st.button("🔄 Switch to Retail Transactions Demo for RFM"):
            from src.data_loader import get_sample_retail_data
            df_ret, is_syn = get_sample_retail_data()
            st.session_state["current_dataset"] = df_ret
            st.session_state["dataset_name"] = "Retail Transactions"
            st.session_state["is_synthetic"] = is_syn
            st.session_state["dataset_type"] = "transaction"
            st.rerun()

    else:
        rfm_col1, rfm_col2, rfm_col3 = st.columns(3)
        with rfm_col1:
            sel_id = st.selectbox("Customer ID Column", id_candidates or df.columns, index=0)
        with rfm_col2:
            sel_date = st.selectbox("Transaction Date Column", date_candidates or df.columns, index=0)
        with rfm_col3:
            sel_monetary = st.selectbox("Order Amount Column", monetary_candidates or df.columns, index=0)

        is_rfm_valid, rfm_msg = validate_for_rfm(df, sel_id, sel_date, sel_monetary)

        if not is_rfm_valid:
            st.error(f"Cannot perform RFM: {rfm_msg}")
        else:
            with st.spinner("Calculating Recency, Frequency, and Monetary scores..."):
                rfm_df, rfm_stats = calculate_rfm(df, sel_id, sel_date, sel_monetary)
                st.session_state["rfm_data"] = rfm_df

            # Metrics
            rkpi1, rkpi2, rkpi3, rkpi4 = st.columns(4)
            with rkpi1:
                render_kpi("Unique Customers", f"{rfm_stats['total_customers']:,}", "From transaction log")
            with rkpi2:
                render_kpi("Total Portfolio Revenue", f"${rfm_stats['total_revenue']:,.2f}", "Cumulative spending")
            with rkpi3:
                render_kpi("Average Recency", f"{rfm_stats['avg_recency_days']} days", "Since last purchase")
            with rkpi4:
                render_kpi("Average Frequency", f"{rfm_stats['avg_frequency']} orders", "Per customer")

            st.markdown("<br>", unsafe_allow_html=True)

            # Charts
            c_col1, c_col2 = st.columns([1, 1])

            with c_col1:
                st.markdown("#### 🏷️ RFM Segment Distribution")
                seg_counts = rfm_df["Segment"].value_counts().reset_index()
                seg_counts.columns = ["Segment", "Count"]
                fig_rfm_bar = px.bar(
                    seg_counts,
                    x="Count",
                    y="Segment",
                    orientation="h",
                    color="Segment",
                    color_discrete_sequence=px.colors.qualitative.Safe
                )
                fig_rfm_bar.update_layout(get_plotly_layout(height=380))
                st.plotly_chart(fig_rfm_bar, use_container_width=True)

            with c_col2:
                st.markdown("#### 🎯 Recency vs Frequency (Sized by Monetary)")
                fig_rfm_scatter = px.scatter(
                    rfm_df,
                    x="Recency",
                    y="Frequency",
                    size="Monetary",
                    color="Segment",
                    hover_name=sel_id,
                    labels={"Recency": "Days Since Last Order", "Frequency": "Order Count"},
                    color_discrete_sequence=px.colors.qualitative.Safe
                )
                fig_rfm_scatter.update_layout(get_plotly_layout(height=380))
                st.plotly_chart(fig_rfm_scatter, use_container_width=True)

            # Segment Summary Table
            st.markdown("### 📋 RFM Segment Summary")
            st.dataframe(rfm_stats["segment_summary"], use_container_width=True)

            # Customer drill down
            st.markdown("### 🔍 Customer RFM Scores")
            search_cust = st.text_input("Search Customer ID", placeholder="e.g. RET-1005")
            view_rfm = rfm_df
            if search_cust:
                view_rfm = view_rfm[view_rfm[sel_id].astype(str).str.contains(search_cust, case=False)]
            st.dataframe(view_rfm.head(100), use_container_width=True)
