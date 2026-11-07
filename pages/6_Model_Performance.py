"""
Customer360 — Page 6: Model Performance & Comparative Diagnostics
Side-by-Side Model Evaluation, Confusion Matrices, ROC Curves, and Clustering Diagnostics.
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
    COLOR_WARNING
)
from src.churn_prediction import train_churn_model
from src.model_evaluation import evaluate_classification_model, compare_models
from src.validation import infer_column_roles

st.set_page_config(page_title="Model Performance — Customer360", page_icon="📈", layout="wide")
apply_custom_css()
init_session_state()
render_sidebar()

render_page_header(
    title="Model Diagnostics & Performance Evaluation",
    subtitle="Rigorous statistical validation of supervised classifiers and unsupervised clustering algorithms.",
    badge="Diagnostic Center"
)

df = st.session_state.get("current_dataset")

if df is None or df.empty:
    st.error("No dataset available. Please upload or load a dataset.")
    st.stop()

roles = infer_column_roles(df)
target_col = roles["target_col"]
id_col = roles["id_col"]
selected_ids = [id_col] if id_col else []

trained_models = st.session_state.get("trained_models", {})

# Quick button to train both models for comprehensive comparison
if target_col and (len(trained_models) < 2):
    st.info("💡 Run both Logistic Regression and Random Forest models to view side-by-side performance comparison.")
    if st.button("⚡ Train & Compare All Benchmark Models", type="primary"):
        with st.spinner("Training Logistic Regression and Random Forest pipelines..."):
            for m_type in ["Logistic Regression", "Random Forest"]:
                meta = train_churn_model(
                    df,
                    target_col=target_col,
                    model_type=m_type,
                    id_cols=selected_ids,
                    test_size=0.2,
                    random_state=42
                )
                trained_models[m_type] = meta
            st.session_state["trained_models"] = trained_models
            st.session_state["active_model_name"] = "Random Forest"
            st.success("Both models successfully trained and evaluated on stratified hold-out data!")
            st.rerun()

if not trained_models:
    st.warning("No supervised models have been trained yet. Please train a model in **Churn Prediction** or click above.")
    st.stop()

# 1. Comparative Leaderboard
st.markdown("### 🏆 Benchmark Comparison Table")
comp_df = compare_models(list(trained_models.values()))
st.dataframe(comp_df, use_container_width=True)

# Model Selector for Deep Dive
selected_model_name = st.selectbox("Select Model for Diagnostic Deep-Dive", list(trained_models.keys()))
active_meta = trained_models[selected_model_name]

# Compute Train and Test evaluations
train_eval = evaluate_classification_model(
    active_meta["y_train"], active_meta["y_train_pred"], active_meta["y_train_prob"], dataset_label="Train Set"
)
test_eval = evaluate_classification_model(
    active_meta["y_test"], active_meta["y_test_pred"], active_meta["y_test_prob"], dataset_label="Test Set"
)

st.markdown(f"### 🎯 Deep-Dive: `{selected_model_name}`")

# Overfitting / Generalization comparison cards
o1, o2, o3, o4 = st.columns(4)
with o1:
    render_kpi("Test Accuracy", f"{test_eval['accuracy']*100:.1f}%", f"Train: {train_eval['accuracy']*100:.1f}%")
with o2:
    render_kpi("Test Precision", f"{test_eval['precision']*100:.1f}%", f"Train: {train_eval['precision']*100:.1f}%")
with o3:
    render_kpi("Test Recall", f"{test_eval['recall']*100:.1f}%", f"Train: {train_eval['recall']*100:.1f}%")
with o4:
    render_kpi("Test F1-Score", f"{test_eval['f1_score']:.3f}", f"Train: {train_eval['f1_score']:.3f}")

st.markdown("<br>", unsafe_allow_html=True)

# Confusion Matrix and ROC Curve side by side
col_diag1, col_diag2 = st.columns([1, 1])

with col_diag1:
    st.markdown("#### 🔲 Test Set Confusion Matrix")
    cm = test_eval["confusion_matrix"]
    cm_matrix = np.array(cm["matrix"])
    labels = ["Retained (0)", "Churned (1)"]

    annotations = [
        [f"True Negative (TN)<br><b>{cm['tn']:,}</b>", f"False Positive (FP)<br><b>{cm['fp']:,}</b>"],
        [f"False Negative (FN)<br><b>{cm['fn']:,}</b>", f"True Positive (TP)<br><b>{cm['tp']:,}</b>"]
    ]

    fig_cm = ff_cm = go.Figure(data=go.Heatmap(
        z=cm_matrix,
        x=labels,
        y=labels,
        text=annotations,
        texttemplate="%{text}",
        textfont={"size": 13, "color": "white"},
        colorscale="Blues",
        showscale=False
    ))
    fig_cm.update_layout(
        get_plotly_layout(height=380),
        xaxis_title="Predicted Outcome",
        yaxis_title="Actual Outcome",
        yaxis_autorange="reversed"
    )
    st.plotly_chart(fig_cm, use_container_width=True)

with col_diag2:
    st.markdown("#### 📈 Receiver Operating Characteristic (ROC Curve)")
    roc_data = test_eval["roc_curve"]
    if roc_data["fpr"] and roc_data["tpr"]:
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=roc_data["fpr"],
            y=roc_data["tpr"],
            name=f"{selected_model_name} (AUC = {test_eval['roc_auc']})",
            line=dict(color=COLOR_PRIMARY, width=2.5)
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            name="Random Chance (AUC = 0.50)",
            line=dict(color="#94A3B8", width=1.5, dash="dash")
        ))
        fig_roc.update_layout(
            get_plotly_layout(height=380),
            xaxis_title="False Positive Rate (1 - Specificity)",
            yaxis_title="True Positive Rate (Recall / Sensitivity)",
            legend=dict(x=0.45, y=0.15)
        )
        st.plotly_chart(fig_roc, use_container_width=True)
    else:
        st.info("Probability estimates unavailable for ROC calculation.")

# Classification Report Breakdown
st.markdown("### 📋 Per-Class Classification Report (Test Set)")
cr_dict = test_eval["classification_report"]
cr_df = pd.DataFrame(cr_dict).transpose()
st.dataframe(cr_df.round(3), use_container_width=True)

# K-Means Diagnostic Section if available
kmeans_res = st.session_state.get("kmeans_data")
if kmeans_res is not None:
    st.markdown("---")
    st.markdown("### 🧩 Unsupervised Clustering Diagnostics (K-Means)")
    k1, k2, k3 = st.columns(3)
    with k1:
        render_kpi("Configured k", kmeans_res["n_clusters"], "Cluster count")
    with k2:
        render_kpi("Silhouette Score", kmeans_res["silhouette_score"], "Cluster cohesion & separation")
    with k3:
        render_kpi("Active Features", len(kmeans_res["feature_cols"]), ", ".join(kmeans_res["feature_cols"][:3]))
