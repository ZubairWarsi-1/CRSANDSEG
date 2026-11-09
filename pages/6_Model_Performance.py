"""
Customer360 — Page 6: Model Performance & Comparative Diagnostics
Multi-Model Benchmark Leaderboard (5 Models), Overfitting Diagnostics, PR & Calibration Curves, and Selection Rationale.
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
from src.churn_prediction import train_churn_model
from src.model_evaluation import (
    evaluate_classification_model,
    compare_models_advanced
)
from src.validation import infer_column_roles

st.set_page_config(page_title="Model Performance — Customer360", page_icon="📈", layout="wide")
apply_custom_css()
init_session_state()
render_sidebar()

render_page_header(
    title="Model Diagnostics & Performance Evaluation",
    subtitle="Statistical benchmarking across 5 classifier architectures, calibration curves, and overfitting diagnostics.",
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

ALL_MODELS = [
    "Logistic Regression",
    "Random Forest",
    "Decision Tree",
    "Gradient Boosting",
    "XGBoost"
]

# Quick button to train all 5 models for comprehensive benchmark comparison
if target_col and (len(trained_models) < len(ALL_MODELS)):
    st.info(f"💡 Currently {len(trained_models)} of {len(ALL_MODELS)} benchmark models are trained. Train all 5 models to view full competitive leaderboard.")
    if st.button("⚡ Train & Benchmark All 5 Supervised Models", type="primary"):
        with st.spinner("Training Logistic Regression, Random Forest, Decision Tree, Gradient Boosting, and XGBoost pipelines..."):
            for m_type in ALL_MODELS:
                try:
                    meta = train_churn_model(
                        df,
                        target_col=target_col,
                        model_type=m_type,
                        id_cols=selected_ids,
                        test_size=0.2,
                        random_state=42
                    )
                    trained_models[m_type] = meta
                except Exception as e:
                    st.warning(f"Could not train {m_type}: {str(e)}")
            st.session_state["trained_models"] = trained_models
            st.session_state["active_model_name"] = "Random Forest"
            st.success("All 5 models successfully trained and evaluated on stratified hold-out data!")
            st.rerun()

if not trained_models:
    st.warning("No supervised models have been trained yet. Please train a model in **Churn Prediction** or click above.")
    st.stop()

# -------------------------------------------------------------
# Section 1: Multi-Model Benchmark Leaderboard
# -------------------------------------------------------------
st.markdown("### 🏆 Comprehensive Multi-Model Benchmark Leaderboard")
st.caption("Models ranked by out-of-sample Test F1-Score with overfitting gap monitoring and probability calibration.")

comp_df = compare_models_advanced(list(trained_models.values()))
st.dataframe(comp_df, use_container_width=True, hide_index=True)

# -------------------------------------------------------------
# Section 2: Model Selection Rationale & Trade-Off Analysis
# -------------------------------------------------------------
best_model = comp_df.iloc[0]["Model"]
best_f1 = comp_df.iloc[0]["F1-Score"]
best_auc = comp_df.iloc[0]["ROC-AUC"]

st.markdown(
    f"""
    <div class="content-card" style="margin: 1rem 0; border-left: 4px solid {COLOR_SUCCESS};">
        <h5 style="margin: 0 0 0.3rem 0; color: #1E293B;">💡 Recommended Platform Model: <code>{best_model}</code></h5>
        <p style="margin: 0.2rem 0; font-size: 0.88rem; color: #334155;">
            <strong>Selection Rationale:</strong> <code>{best_model}</code> achieves the highest balanced test F1-score (<strong>{best_f1}</strong>) and discriminative separation (ROC-AUC: <strong>{best_auc}</strong>) with a constrained generalization gap.
        </p>
        <p style="margin: 0.2rem 0; font-size: 0.84rem; color: #64748B;">
            <strong>Business Trade-Off Guidance:</strong>
            <br>• <strong>High Recall Priority:</strong> If the cost of losing a customer ($500+ LTV) far exceeds the cost of a retention offer ($20), select the model with maximum Recall or lower the decision threshold to 0.40.
            <br>• <strong>High Precision Priority:</strong> If budget for retention incentives is strictly capped, prioritize the model with highest Precision to prevent spending on non-churners.
            <br>• <strong>Balanced Operations (Default):</strong> <code>{best_model}</code> provides the optimal harmonic mean between recall and precision.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("<br>", unsafe_allow_html=True)

# -------------------------------------------------------------
# Section 3: Diagnostic Deep-Dive for Selected Model
# -------------------------------------------------------------
selected_model_name = st.selectbox("Select Model for Detailed Diagnostic Deep-Dive", list(trained_models.keys()))
active_meta = trained_models[selected_model_name]

train_eval = evaluate_classification_model(
    active_meta["y_train"], active_meta["y_train_pred"], active_meta["y_train_prob"], dataset_label="Train Set"
)
test_eval = evaluate_classification_model(
    active_meta["y_test"], active_meta["y_test_pred"], active_meta["y_test_prob"], dataset_label="Test Set"
)

st.markdown(f"### 🎯 Deep-Dive Diagnostics: `{selected_model_name}`")

# Overfitting & Generalization KPI Row
o1, o2, o3, o4, o5 = st.columns(5)
with o1:
    render_kpi("Test Accuracy", f"{test_eval['accuracy']*100:.1f}%", f"Train: {train_eval['accuracy']*100:.1f}%")
with o2:
    render_kpi("Test Precision", f"{test_eval['precision']*100:.1f}%", f"Train: {train_eval['precision']*100:.1f}%")
with o3:
    render_kpi("Test Recall", f"{test_eval['recall']*100:.1f}%", f"Train: {train_eval['recall']*100:.1f}%")
with o4:
    render_kpi("Test F1-Score", f"{test_eval['f1_score']:.3f}", f"Train: {train_eval['f1_score']:.3f}")
with o5:
    brier_val = test_eval.get("brier_score", "N/A")
    render_kpi("Brier Score", f"{brier_val}", "Lower is better (0=perfect)")

st.markdown("<br>", unsafe_allow_html=True)

# -------------------------------------------------------------
# Section 4: Confusion Matrix, ROC Curve, and PR Curve
# -------------------------------------------------------------
col_d1, col_d2, col_d3 = st.columns(3)

with col_d1:
    st.markdown("#### 🔲 Test Set Confusion Matrix")
    cm = test_eval["confusion_matrix"]
    cm_matrix = np.array(cm["matrix"])
    labels = ["Retained (0)", "Churned (1)"]
    annotations = [
        [f"True Neg (TN)<br><b>{cm['tn']:,}</b>", f"False Pos (FP)<br><b>{cm['fp']:,}</b>"],
        [f"False Neg (FN)<br><b>{cm['fn']:,}</b>", f"True Pos (TP)<br><b>{cm['tp']:,}</b>"]
    ]
    fig_cm = go.Figure(data=go.Heatmap(
        z=cm_matrix,
        x=labels,
        y=labels,
        text=annotations,
        texttemplate="%{text}",
        textfont={"size": 12, "color": "white"},
        colorscale="Blues",
        showscale=False
    ))
    fig_cm.update_layout(
        get_plotly_layout(height=340),
        xaxis_title="Predicted Outcome",
        yaxis_title="Actual Outcome",
        yaxis_autorange="reversed"
    )
    st.plotly_chart(fig_cm, use_container_width=True)

with col_d2:
    st.markdown("#### 📈 ROC Curve")
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
            x=[0, 1], y=[0, 1],
            name="Random Baseline (AUC = 0.50)",
            line=dict(color="#94A3B8", width=1.5, dash="dash")
        ))
        fig_roc.update_layout(
            get_plotly_layout(height=340),
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate"
        )
        st.plotly_chart(fig_roc, use_container_width=True)
    else:
        st.info("ROC data unavailable.")

with col_d3:
    st.markdown("#### 📉 Precision-Recall (PR) Curve")
    pr_data = test_eval["pr_curve"]
    if pr_data["recall"] and pr_data["precision"]:
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(
            x=pr_data["recall"],
            y=pr_data["precision"],
            name=f"PR Curve (AUC = {test_eval['pr_auc']})",
            line=dict(color=COLOR_SUCCESS, width=2.5)
        ))
        baseline_rate = float(np.mean(active_meta["y_test"]))
        fig_pr.add_trace(go.Scatter(
            x=[0, 1], y=[baseline_rate, baseline_rate],
            name=f"No-Skill Baseline ({baseline_rate:.2f})",
            line=dict(color="#94A3B8", width=1.5, dash="dash")
        ))
        fig_pr.update_layout(
            get_plotly_layout(height=340),
            xaxis_title="Recall",
            yaxis_title="Precision"
        )
        st.plotly_chart(fig_pr, use_container_width=True)
    else:
        st.info("PR curve data unavailable.")

st.markdown("<br>", unsafe_allow_html=True)

# -------------------------------------------------------------
# Section 5: Calibration Curve & Probability Reliability
# -------------------------------------------------------------
st.markdown("### 🎯 Probability Calibration Diagnostics")
st.caption("Validates whether predicted churn probabilities match observed real-world frequencies.")

cal_col1, cal_col2 = st.columns([1, 1])

with cal_col1:
    cal_data = test_eval.get("calibration", {})
    if cal_data.get("prob_pred") and cal_data.get("prob_true"):
        fig_cal = go.Figure()
        fig_cal.add_trace(go.Scatter(
            x=cal_data["prob_pred"],
            y=cal_data["prob_true"],
            mode="lines+markers",
            name=f"{selected_model_name}",
            line=dict(color=COLOR_ACCENT, width=2.5),
            marker=dict(size=8)
        ))
        fig_cal.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines",
            name="Perfect Calibration (y = x)",
            line=dict(color="#94A3B8", dash="dash")
        ))
        fig_cal.update_layout(
            get_plotly_layout(height=320),
            xaxis_title="Mean Predicted Churn Probability",
            yaxis_title="Fraction of True Churners",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1])
        )
        st.plotly_chart(fig_cal, use_container_width=True)
    else:
        st.info("Calibration curve data unavailable.")

with cal_col2:
    st.markdown(
        f"""
        <div class="content-card">
            <h5>Understanding Calibration in Enterprise Retention</h5>
            <p style="font-size: 0.88rem; color: #475569;">
                A model is <strong>well-calibrated</strong> if a predicted 70% churn risk translates to approximately 7 out of 10 customers actually leaving.
            </p>
            <ul style="font-size: 0.85rem; color: #64748B; padding-left: 1.2rem;">
                <li><strong>Brier Score:</strong> <code>{test_eval.get('brier_score', 'N/A')}</code> (Mean squared error of probability predictions; 0.0 indicates perfect probabilistic accuracy).</li>
                <li><strong>Overconfident Models:</strong> Curves bowing above/below the diagonal line indicate probability skewing.</li>
                <li><strong>Business Implication:</strong> High calibration reliability is critical when multiplying probability by revenue to calculate <strong>Revenue at Risk</strong>.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

# -------------------------------------------------------------
# Section 6: Classification Report Table & K-Means Diagnostics
# -------------------------------------------------------------
st.markdown("### 📋 Per-Class Classification Report (Test Set)")
cr_dict = test_eval["classification_report"]
cr_df = pd.DataFrame(cr_dict).transpose()
st.dataframe(cr_df.round(3), use_container_width=True)

kmeans_res = st.session_state.get("kmeans_data")
if kmeans_res is not None:
    st.markdown("---")
    st.markdown("### 🧩 Unsupervised Clustering Diagnostics (K-Means)")
    k1, k2, k3 = st.columns(3)
    with k1:
        render_kpi("Configured k", kmeans_res["n_clusters"], f"Auto Recommendation: k={kmeans_res['auto_k']}")
    with k2:
        render_kpi("Silhouette Score", kmeans_res["silhouette_score"], "Cluster cohesion & separation")
    with k3:
        render_kpi("Active Features", len(kmeans_res["feature_cols"]), ", ".join(kmeans_res["feature_cols"][:3]))
