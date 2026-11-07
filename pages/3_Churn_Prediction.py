"""
Customer360 — Page 3: Churn Prediction
Supervised Machine Learning Pipelines (Logistic Regression & Random Forest) with Explainable Risk Scoring.
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
from src.churn_prediction import train_churn_model, predict_churn, explain_customer_risk
from src.model_evaluation import evaluate_classification_model
from src.recommendations import generate_customer_recommendation
from src.validation import infer_column_roles, validate_for_churn_prediction

st.set_page_config(page_title="Churn Prediction — Customer360", page_icon="🎯", layout="wide")
apply_custom_css()
init_session_state()
render_sidebar()

render_page_header(
    title="Predictive Churn Intelligence",
    subtitle="Train supervised machine learning pipelines to detect attrition risk and explain customer vulnerability.",
    badge="Supervised ML"
)

st.info(
    "💡 **Academic & Statistical Disclaimer:** Predicted churn probabilities are statistical model estimates, not deterministic guarantees. "
    "Identified contributing factors highlight correlational weights and do not automatically establish direct causation."
)

df = st.session_state.get("current_dataset")

if df is None or df.empty:
    st.error("No active dataset. Please load or upload a dataset.")
    st.stop()

roles = infer_column_roles(df)
candidate_targets = [c for c in df.columns if df[c].nunique() == 2]
default_target = roles["target_col"] if roles["target_col"] in candidate_targets else (candidate_targets[0] if candidate_targets else None)

if not candidate_targets and not default_target:
    st.warning("Current dataset does not contain a binary historical target column (e.g. Churn: Yes/No). Supervised classification requires labelled outcomes.")
    st.stop()

# Training Configuration Panel
with st.expander("⚙️ Model Training & Pipeline Configuration", expanded=True):
    col_t1, col_t2, col_t3 = st.columns(3)

    with col_t1:
        target_col = st.selectbox(
            "Select Churn Target Column",
            options=candidate_targets if candidate_targets else df.columns,
            index=candidate_targets.index(default_target) if default_target in candidate_targets else 0
        )

    with col_t2:
        id_candidates = [c for c in df.columns if "id" in c.lower() or "cust" in c.lower()]
        id_col = st.selectbox(
            "Customer ID Column (Excluded from training)",
            options=["None"] + list(df.columns),
            index=(list(df.columns).index(roles["id_col"]) + 1) if roles["id_col"] in df.columns else 0
        )
        selected_ids = [id_col] if id_col != "None" else []

    with col_t3:
        model_choice = st.selectbox(
            "Algorithm",
            ["Random Forest", "Logistic Regression"],
            help="Both algorithms incorporate class-weight balancing and scikit-learn ColumnTransformer pipelines."
        )

    train_btn = st.button("🚀 Train Supervised Churn Pipeline", type="primary")

# Validate target
is_valid, v_msg, label_map = validate_for_churn_prediction(df, target_col)
if not is_valid:
    st.error(f"Target Column Error: {v_msg}")
    st.stop()

# Check if model is trained or train on click
trained_models = st.session_state.get("trained_models", {})

if train_btn or (model_choice not in trained_models and "default_model_trained" not in st.session_state):
    with st.spinner(f"Training {model_choice} with stratified split and feature preprocessing..."):
        meta = train_churn_model(
            df,
            target_col=target_col,
            model_type=model_choice,
            id_cols=selected_ids,
            test_size=0.2,
            random_state=42
        )
        trained_models[model_choice] = meta
        st.session_state["trained_models"] = trained_models
        st.session_state["active_model_name"] = model_choice
        st.session_state["default_model_trained"] = True

        # Generate predictions for the current dataset
        pred_df = predict_churn(meta["pipeline"], df, id_cols=selected_ids, target_col=target_col)
        st.session_state["predictions_df"] = pred_df

        st.success(f"Successfully trained and serialized **{model_choice}** pipeline to `models/`!")

# Display Active Model Metrics
if model_choice in trained_models:
    meta = trained_models[model_choice]
    pipeline = meta["pipeline"]

    test_eval = evaluate_classification_model(
        meta["y_test"], meta["y_test_pred"], meta["y_test_prob"], dataset_label="Test Set"
    )

    st.markdown(f"### 📊 Test Evaluation Performance: `{model_choice}`")
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        render_kpi("Test Accuracy", f"{test_eval['accuracy']*100:.1f}%", "Out-of-sample accuracy")
    with m2:
        render_kpi("Precision", f"{test_eval['precision']*100:.1f}%", "True churn / Predicted churn")
    with m3:
        render_kpi("Recall (Sensitivity)", f"{test_eval['recall']*100:.1f}%", "Captured churners")
    with m4:
        render_kpi("F1-Score", f"{test_eval['f1_score']:.3f}", "Harmonic mean")
    with m5:
        roc_str = f"{test_eval['roc_auc']:.3f}" if isinstance(test_eval['roc_auc'], (int, float)) else "N/A"
        render_kpi("ROC-AUC Score", roc_str, "Discriminative power")

    st.markdown("<br>", unsafe_allow_html=True)

    # Feature Importance Plot
    imp_df = meta["feature_importance_df"]
    if not imp_df.empty:
        st.markdown("#### 🔑 Top Feature Drivers of Churn Model")
        top_k = imp_df.head(10)
        val_col = "Importance" if "Importance" in top_k.columns else "Coefficient"
        fig_imp = px.bar(
            top_k,
            x=val_col,
            y="Feature",
            orientation="h",
            color=val_col,
            color_continuous_scale="Blues"
        )
        fig_imp.update_layout(get_plotly_layout(height=350))
        fig_imp.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_imp, use_container_width=True)

    # Predictions Table & Filters
    st.markdown("### 👥 Customer Predictions & Risk Scoring")
    pred_df = st.session_state.get("predictions_df")
    if pred_df is None:
        pred_df = predict_churn(pipeline, df, id_cols=selected_ids, target_col=target_col)
        st.session_state["predictions_df"] = pred_df

    col_f1, col_f2 = st.columns([1, 2])
    with col_f1:
        risk_filter = st.selectbox("Filter Risk Category", ["All", "High", "Medium", "Low"])
    with col_f2:
        id_search = st.text_input("Filter by Customer Identifier", placeholder="Search ID...")

    filtered_pred = pred_df
    if risk_filter != "All":
        filtered_pred = filtered_pred[filtered_pred["Risk_Category"] == risk_filter]
    if id_search and selected_ids:
        filtered_pred = filtered_pred[filtered_pred[selected_ids[0]].astype(str).str.contains(id_search, case=False)]

    # Display columns
    display_cols = list(selected_ids) + ["Churn_Probability", "Predicted_Churn", "Risk_Category"]
    feature_extras = [c for c in ["tenure", "MonthlyCharges", "TotalCharges", "Contract"] if c in df.columns]
    display_cols += feature_extras
    if target_col in filtered_pred.columns:
        display_cols.append(target_col)

    st.dataframe(filtered_pred[display_cols].head(100), use_container_width=True)

    # Customer Risk Explainer Modal / Section
    st.markdown("---")
    st.markdown("### 🔎 Individual Customer Explainability & What-If Simulator")
    
    if selected_ids:
        sample_ids = pred_df[selected_ids[0]].astype(str).head(50).tolist()
        inspect_id = st.selectbox("Select Customer to Explain", sample_ids)
        cust_row = pred_df[pred_df[selected_ids[0]].astype(str) == inspect_id].iloc[0]

        # Calculate explanation
        feat_names = meta["feature_names"]
        explanation = explain_customer_risk(pipeline, cust_row, feat_names)
        rec = generate_customer_recommendation(cust_row)

        c_exp1, c_exp2 = st.columns([1, 1])

        with c_exp1:
            st.markdown(
                f"""
                <div class="content-card">
                    <h4>Customer Profile: <code>{inspect_id}</code></h4>
                    <p><strong>Predicted Churn Probability:</strong> {cust_row['Churn_Probability']*100:.1f}%</p>
                    <p><strong>Risk Category:</strong> <span class="badge badge-{'danger' if cust_row['Risk_Category']=='High' else ('warning' if cust_row['Risk_Category']=='Medium' else 'success')}">{cust_row['Risk_Category']}</span></p>
                    <p><strong>Monthly Spending:</strong> ${cust_row.get('MonthlyCharges', 0):.2f}</p>
                    <p><strong>Tenure:</strong> {cust_row.get('tenure', 'N/A')} months</p>
                    <p><strong>Contract:</strong> {cust_row.get('Contract', 'N/A')}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        with c_exp2:
            top_drivers = "<br>• ".join(explanation["top_risk_drivers"][:3])
            top_protective = "<br>• ".join(explanation["top_protective_factors"][:3])
            st.markdown(
                f"""
                <div class="content-card">
                    <h4>Model Risk Explanation & Action</h4>
                    <p style="color: #EF4444;"><strong>Top Risk Drivers:</strong><br>• {top_drivers}</p>
                    <p style="color: #10B981;"><strong>Protective Signals:</strong><br>• {top_protective}</p>
                    <hr>
                    <p><strong>Suggested Action:</strong> <code>{rec['Recommended_Action']}</code></p>
                    <p style="font-size: 0.85rem; color: #64748B;">{rec['Recommendation_Reason']}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
else:
    st.info("Click 'Train Supervised Churn Pipeline' to train and evaluate the selected model.")
