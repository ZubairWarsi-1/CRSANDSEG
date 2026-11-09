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
from src.validation import validate_for_rfm, validate_for_clustering, infer_column_roles, check_feature_redundancy

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
target_col = roles["target_col"]

# Tabs: Method Selection
tab_kmeans, tab_rfm = st.tabs(["🧩 K-Means Feature Clustering", "🛒 Transactional RFM Analysis"])

# =========================================================================
# TAB 1: K-MEANS CLUSTERING
# =========================================================================
with tab_kmeans:
    st.markdown("### 🧩 Unsupervised K-Means Behavioral Clustering")
    st.caption("Group customers based on multi-dimensional continuous features (e.g. tenure, monthly charges, total spending).")

    # Detect numerical columns
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 2]
    valid_features = [c for c in numeric_cols if not c.lower().endswith("id") and c != roles["id_col"]]

    if len(valid_features) < 2:
        st.warning("Current dataset contains fewer than 2 continuous numerical features required for K-Means clustering.")
    else:
        col_ctrl1, col_ctrl2 = st.columns([2, 1])

        with col_ctrl1:
            # Default to tenure and monthly charges if available
            default_feats = []
            for preferred in ["tenure", "MonthlyCharges", "TotalCharges"]:
                matched = next((c for c in valid_features if preferred.lower() in c.lower()), None)
                if matched and matched not in default_feats:
                    default_feats.append(matched)
            if len(default_feats) < 2:
                default_feats = valid_features[:3]

            selected_features = st.multiselect(
                "Select Numerical Features for Clustering",
                options=valid_features,
                default=default_feats,
                help="Choose continuous attributes describing customer relationship and spending."
            )

        # Check for collinearity / feature redundancy
        if len(selected_features) >= 2:
            redundancies = check_feature_redundancy(df, selected_features, threshold=0.80)
            if redundancies:
                with st.expander("⚠️ Multicollinearity & Feature Redundancy Warnings", expanded=False):
                    for r in redundancies:
                        st.warning(f"**{r['feature_1']}** ↔ **{r['feature_2']}** (r = {r['correlation']}): {r['warning']}")

        with col_ctrl2:
            k_mode = st.radio("K Selection Mode", ["Automatic Recommendation", "Manual Slider"], horizontal=True)
            if k_mode == "Manual Slider":
                n_clusters_input = st.slider("Select Number of Clusters (k)", min_value=2, max_value=8, value=3)
            else:
                n_clusters_input = None  # Signals auto-selection in perform_kmeans_clustering

        if len(selected_features) >= 2:
            is_valid, msg = validate_for_clustering(df, selected_features)
            if not is_valid:
                st.error(f"Validation Error: {msg}")
            else:
                with st.spinner("Executing K-Means clustering, calculating inertia, and evaluating silhouette cohesion..."):
                    kmeans_result = perform_kmeans_clustering(
                        df,
                        feature_cols=selected_features,
                        n_clusters=n_clusters_input,
                        target_col=target_col,
                        random_state=42
                    )
                    st.session_state["kmeans_data"] = kmeans_result

                res_df = kmeans_result["dataframe"]
                chosen_k = kmeans_result["n_clusters"]
                auto_k = kmeans_result["auto_k"]
                k_reason = kmeans_result["k_selection_reason"]
                sil_score = kmeans_result["silhouette_score"]
                elbow_df = kmeans_result["elbow_data"]
                profiles_df = kmeans_result["cluster_profiles"]
                pca_var = kmeans_result["pca_variance"]

                # Explain K Selection
                st.info(f"💡 **Cluster Selection Rationale (k={chosen_k}):** {k_reason}")

                # Metrics banner
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    render_kpi("Configured Clusters (k)", chosen_k, f"Auto Recommendation: k={auto_k}")
                with m2:
                    sil_quality = "Good (≥0.50)" if sil_score >= 0.5 else ("Moderate (0.35–0.49)" if sil_score >= 0.35 else "Weak (<0.35)")
                    render_kpi("Silhouette Score", sil_score, f"Separation: {sil_quality}")
                with m3:
                    render_kpi("Total Clustered", f"{len(res_df):,}", "Customer accounts")
                with m4:
                    render_kpi("PCA Explained Variance", f"{sum(pca_var):.1f}%", f"PC1: {pca_var[0]}%, PC2: {pca_var[1]}%")

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
                        title=f"Customer Clusters Projected on First 2 Principal Components ({sum(pca_var):.1f}% Total Variance)",
                        color_discrete_sequence=[COLOR_PRIMARY, COLOR_SUCCESS, COLOR_WARNING, COLOR_DANGER, COLOR_ACCENT, COLOR_SECONDARY]
                    )
                    fig_pca.update_layout(get_plotly_layout(height=420))
                    st.plotly_chart(fig_pca, use_container_width=True)

                with plot_col2:
                    st.markdown("#### 📉 Elbow (Inertia) & Silhouette Diagnostics")
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
                        legend=dict(x=0.05, y=1.12, orientation="h")
                    )
                    st.plotly_chart(fig_elbow, use_container_width=True)

                # Cluster-Wise Churn Analysis Section
                st.markdown("### 📊 Cluster-Level Churn Analysis & Risk Profiling")
                if "HistoricalChurnRate" in profiles_df.columns:
                    c_chart_col1, c_chart_col2 = st.columns([1, 1])
                    with c_chart_col1:
                        fig_cchurn = px.bar(
                            profiles_df,
                            x="ClusterName",
                            y="HistoricalChurnRate",
                            text="HistoricalChurnRate",
                            color="HistoricalChurnRate",
                            color_continuous_scale="Reds",
                            labels={"HistoricalChurnRate": "Observed Churn Rate (%)", "ClusterName": "Cluster Profile"},
                            title="Observed Attrition Rate by Cluster (Within Group %)"
                        )
                        fig_cchurn.update_traces(texttemplate="%{y:.1f}%", textposition="outside")
                        fig_cchurn.update_layout(get_plotly_layout(height=350))
                        st.plotly_chart(fig_cchurn, use_container_width=True)

                    with c_chart_col2:
                        fig_csize = px.pie(
                            profiles_df,
                            names="ClusterName",
                            values="CustomerCount",
                            hole=0.45,
                            title="Cluster Population Share",
                            color_discrete_sequence=[COLOR_PRIMARY, COLOR_SUCCESS, COLOR_WARNING, COLOR_DANGER, COLOR_ACCENT]
                        )
                        fig_csize.update_layout(get_plotly_layout(height=350))
                        st.plotly_chart(fig_csize, use_container_width=True)

                # Actionable Segment Strategy Cards
                st.markdown("### 🎯 Strategic Recommendations per Customer Cohort")
                for _, p_row in profiles_df.iterrows():
                    c_name = p_row["ClusterName"]
                    c_cnt = p_row["CustomerCount"]
                    c_pct = p_row["CustomerPercentage"]
                    c_strat = p_row.get("RecommendedStrategy", "N/A")
                    c_rat = p_row.get("StrategyRationale", "N/A")
                    c_churn = f"{p_row.get('HistoricalChurnRate', 0.0):.1f}%" if "HistoricalChurnRate" in p_row else "N/A"

                    st.markdown(
                        f"""
                        <div class="content-card" style="margin-bottom: 0.8rem; border-left: 4px solid {COLOR_ACCENT};">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <h5 style="margin: 0; color: #1E293B;">🏷️ {c_name}</h5>
                                <span class="badge badge-info">{c_cnt:,} customers ({c_pct}%) | Historical Churn: {c_churn}</span>
                            </div>
                            <p style="margin: 0.4rem 0 0.2rem 0; font-size: 0.88rem; color: #059669;"><strong>Recommended Action:</strong> {c_strat}</p>
                            <p style="margin: 0; font-size: 0.84rem; color: #64748B;"><strong>Business Rationale:</strong> {c_rat}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # Cluster Profiles Full Table
                st.markdown("### 📋 Cluster Statistical Summary")
                st.dataframe(profiles_df, use_container_width=True, hide_index=True)

                # Drill down customer table
                st.markdown("### 🔎 Customer Record Drill-Down")
                selected_cluster = st.selectbox("Filter Customer List by Cluster", ["All"] + list(profiles_df["ClusterName"].unique()))
                view_df = res_df if selected_cluster == "All" else res_df[res_df["ClusterName"] == selected_cluster]
                
                show_cols = ([roles["id_col"]] if roles["id_col"] else []) + ["ClusterName"] + selected_features
                if roles["target_col"] and roles["target_col"] in res_df.columns:
                    show_cols.append(roles["target_col"])
                st.dataframe(view_df[show_cols].head(100), use_container_width=True)
        else:
            st.info("Please select at least 2 numerical features to run clustering.")

# =========================================================================
# TAB 2: RFM ANALYSIS
# =========================================================================
with tab_rfm:
    st.markdown("### 🛒 Transaction-Level RFM (Recency, Frequency, Monetary) Analysis")
    st.caption("Quantifies behavioral loyalty and spending velocity from granular order timestamps and transaction totals.")

    # Check suitability of current dataset for RFM
    id_candidates = [c for c in df.columns if "id" in c.lower() or "cust" in c.lower()]
    date_candidates = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
    monetary_candidates = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    if not date_candidates:
        st.info("💡 **Methodological Note:** The active dataset does not contain transaction timestamp columns required for RFM.")
        st.markdown(
            """
            The **IBM Telco dataset** is a subscription-level account dataset containing tenure and recurring monthly charges. 
            It is ideally structured for **Behavioral K-Means Clustering** and **Supervised Churn Prediction**.
            
            **RFM Analysis** strictly requires transaction logs with distinct order dates and transaction amounts to calculate:
            - **Recency:** Days since latest transaction date.
            - **Frequency:** Count of unique order transactions per customer.
            - **Monetary:** Sum of all order transactions for each customer.
            
            To explore RFM Analysis, switch to the built-in **Retail Transactions** demo dataset or upload an order log CSV in **Data Upload**.
            """
        )
        if st.button("🔄 Switch to Retail Transactions Demo for RFM Analysis", type="primary"):
            from src.data_loader import get_sample_retail_data
            df_ret, is_syn = get_sample_retail_data()
            st.session_state["current_dataset"] = df_ret
            st.session_state["dataset_name"] = "Retail Transactions"
            st.session_state["is_synthetic"] = is_syn
            st.session_state["dataset_type"] = "transaction"
            st.session_state["predictions_df"] = None
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
            with st.spinner("Aggregating purchase history and computing Recency, Frequency, and Monetary scores..."):
                rfm_df, rfm_stats = calculate_rfm(df, sel_id, sel_date, sel_monetary)
                st.session_state["rfm_data"] = rfm_df

            # Metrics
            rkpi1, rkpi2, rkpi3, rkpi4 = st.columns(4)
            with rkpi1:
                render_kpi("Unique Customers", f"{rfm_stats['total_customers']:,}", "Aggregated buyer accounts")
            with rkpi2:
                render_kpi("Total Portfolio Revenue", f"${rfm_stats['total_revenue']:,.2f}", "Cumulative transaction sum")
            with rkpi3:
                render_kpi("Average Recency", f"{rfm_stats['avg_recency_days']} days", "Since last customer purchase")
            with rkpi4:
                render_kpi("Average Frequency", f"{rfm_stats['avg_frequency']} orders", "Purchases per customer")

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
            st.markdown("### 📋 RFM Segment Statistical Profiles")
            st.dataframe(rfm_stats["segment_summary"], use_container_width=True)

            # Customer drill down
            st.markdown("### 🔍 Customer RFM Individual Records")
            search_cust = st.text_input("Search Customer ID", placeholder="e.g. RET-1005")
            view_rfm = rfm_df
            if search_cust:
                view_rfm = view_rfm[view_rfm[sel_id].astype(str).str.contains(search_cust, case=False)]
            st.dataframe(view_rfm.head(100), use_container_width=True)
