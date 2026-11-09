"""
Customer360 — Page 3: Churn Prediction
Supervised ML Pipelines (5 Algorithms), 5-Fold Cross-Validation, Overfitting Diagnostics, Threshold Tuning, and What-If Simulator.
"""

import os
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
from src.churn_prediction import (
    train_churn_model,
    predict_churn,
    explain_customer_risk,
    simulate_customer_churn
)
from src.model_evaluation import (
    evaluate_classification_model,
    evaluate_at_threshold
)
from src.recommendations import generate_customer_recommendation
from src.validation import infer_column_roles, validate_for_churn_prediction

st.set_page_config(page_title="Churn Prediction — Customer360", page_icon="🎯", layout="wide")
apply_custom_css()
init_session_state()
render_sidebar()

render_page_header(
    title="Predictive Churn Intelligence",
    subtitle="Train supervised machine learning pipelines, tune decision thresholds, and simulate counterfactual retention interventions.",
    badge="Supervised ML"
)

st.info(
    "💡 **Academic & Statistical Disclaimer:** Predicted churn probabilities are empirical statistical estimates, not deterministic guarantees. "
    "Identified contributing factors highlight correlational weights and do not automatically establish direct causality."
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

# -------------------------------------------------------------
# Model Training & Pipeline Configuration Panel
# -------------------------------------------------------------
with st.expander("⚙️ Supervised Model Configuration & Training Options", expanded=True):
    col_t1, col_t2, col_t3 = st.columns(3)

    with col_t1:
        target_col = st.selectbox(
            "Select Churn Target Column",
            options=candidate_targets if candidate_targets else df.columns,
            index=candidate_targets.index(default_target) if default_target in candidate_targets else 0
        )

    with col_t2:
        id_col = st.selectbox(
            "Customer ID Column (Excluded from training)",
            options=["None"] + list(df.columns),
            index=(list(df.columns).index(roles["id_col"]) + 1) if roles["id_col"] in df.columns else 0
        )
        selected_ids = [id_col] if id_col != "None" else []

    with col_t3:
        model_options = [
            "Logistic Regression",
            "Random Forest",
            "Decision Tree",
            "Gradient Boosting",
            "XGBoost"
        ]
        model_choice = st.selectbox(
            "Select Classifier Algorithm",
            model_options,
            index=0,
            help="Regularized models with class balancing and stratified 5-fold cross-validation inside ColumnTransformer pipelines."
        )

    t_btn_col1, t_btn_col2 = st.columns([1, 1])
    with t_btn_col1:
        train_btn = st.button("🚀 Train Selected Model Pipeline", type="primary")
    with t_btn_col2:
        train_all_btn = st.button("⚡ Train & Benchmark All 5 Models")

# Validate target column
is_valid, v_msg, label_map = validate_for_churn_prediction(df, target_col)
if not is_valid:
    st.error(f"Target Column Error: {v_msg}")
    st.stop()

trained_models = st.session_state.get("trained_models", {})

# Handle Training
if train_all_btn:
    with st.spinner("Training all 5 models with stratified 5-fold cross validation..."):
        for m_name in model_options:
            try:
                meta = train_churn_model(
                    df,
                    target_col=target_col,
                    model_type=m_name,
                    id_cols=selected_ids,
                    test_size=0.2,
                    random_state=42
                )
                trained_models[m_name] = meta
            except Exception as e:
                st.warning(f"Could not train {m_name}: {str(e)}")
        st.session_state["trained_models"] = trained_models
        st.session_state["active_model_name"] = model_choice
        st.session_state["default_model_trained"] = True
        st.success(f"Successfully trained and serialized all 5 models!")

elif train_btn or (model_choice not in trained_models and "default_model_trained" not in st.session_state):
    with st.spinner(f"Training {model_choice} with stratified split and 5-fold cross-validation..."):
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

        # Generate predictions for the full current dataset
        pred_df = predict_churn(meta["pipeline"], df, id_cols=selected_ids, target_col=target_col)
        st.session_state["predictions_df"] = pred_df
        st.success(f"Successfully trained and serialized **{model_choice}** pipeline to `models/`!")

# -------------------------------------------------------------
# Display Active Model Performance
# -------------------------------------------------------------
if model_choice in trained_models:
    meta = trained_models[model_choice]
    pipeline = meta["pipeline"]

    test_eval = evaluate_classification_model(
        meta["y_test"], meta["y_test_pred"], meta["y_test_prob"], dataset_label="Test Set"
    )
    train_eval = evaluate_classification_model(
        meta["y_train"], meta["y_train_pred"], meta["y_train_prob"], dataset_label="Train Set"
    )

    st.markdown(f"### 📊 Performance Summary: `{model_choice}`")
    
    # Primary Test KPIs
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
        render_kpi("ROC-AUC Score", roc_str, "Discriminative separation")

    st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------------------------------------------
    # Generalization & Overfitting Diagnostics + 5-Fold CV
    # -------------------------------------------------------------
    diag_col1, diag_col2 = st.columns([1, 1])

    with diag_col1:
        st.markdown("#### 🛡️ Overfitting & Generalization Diagnostics")
        train_f1 = train_eval["f1_score"]
        test_f1 = test_eval["f1_score"]
        f1_gap = train_f1 - test_f1

        train_acc = train_eval["accuracy"]
        test_acc = test_eval["accuracy"]
        acc_gap = train_acc - test_acc

        gap_df = pd.DataFrame([
            {"Split": "Training Set (In-Sample)", "Accuracy": f"{train_acc*100:.1f}%", "F1-Score": f"{train_f1:.3f}"},
            {"Split": "Test Set (Out-of-Sample)", "Accuracy": f"{test_acc*100:.1f}%", "F1-Score": f"{test_f1:.3f}"},
            {"Split": "Generalization Gap (Δ)", "Accuracy": f"{acc_gap*100:+.1f}%", "F1-Score": f"{f1_gap:+.3f}"}
        ])
        st.dataframe(gap_df, use_container_width=True, hide_index=True)

        if f1_gap > 0.15:
            st.warning(f"⚠️ **High Overfitting Risk Detected:** In-sample training F1 exceeds test F1 by {f1_gap:.3f}. Tree depth and leaf parameters should be constrained.")
        elif f1_gap > 0.08:
            st.info(f"ℹ️ **Mild Generalization Gap:** {f1_gap:.3f} difference between train and test F1. Typical for non-linear ensembles.")
        else:
            st.success(f"✅ **Healthy Generalization:** Low generalization gap ({f1_gap:.3f}). Model is not memorizing training data.")

    with diag_col2:
        st.markdown("#### 🔁 5-Fold Stratified Cross-Validation")
        cv_roc = meta.get("cv_mean_roc_auc")
        cv_roc_std = meta.get("cv_std_roc_auc")
        cv_acc = meta.get("cv_mean_accuracy")
        cv_acc_std = meta.get("cv_std_accuracy")

        c1, c2 = st.columns(2)
        with c1:
            roc_display = f"{cv_roc:.3f} ± {cv_roc_std:.3f}" if cv_roc is not None else "N/A"
            st.metric("5-Fold Mean ROC-AUC", roc_display)
        with c2:
            acc_display = f"{cv_acc*100:.1f}% ± {cv_acc_std*100:.1f}%" if cv_acc is not None else "N/A"
            st.metric("5-Fold Mean Accuracy", acc_display)

        cv_folds_df = pd.DataFrame({
            "Fold": [f"Fold {i+1}" for i in range(5)],
            "ROC-AUC": [f"{v:.4f}" for v in meta.get("cv_scores_roc_auc", [0]*5)],
            "Accuracy": [f"{v*100:.2f}%" for v in meta.get("cv_scores_accuracy", [0]*5)]
        })
        st.dataframe(cv_folds_df, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------------------------------------------
    # Interactive Decision Threshold Tuning
    # -------------------------------------------------------------
    st.markdown("### 🎚️ Interactive Classification Decision Threshold")
    st.caption("Adjust the decision threshold to balance Sensitivity (Recall) vs False Positive alerts (Precision).")

    thresh_val = st.slider("Probability Threshold (Cutoff for Churn = 1)", min_value=0.10, max_value=0.90, value=0.50, step=0.05)
    
    thresh_eval = evaluate_at_threshold(meta["y_test"], meta["y_test_prob"], threshold=thresh_val)
    cm_t = thresh_eval["confusion_matrix"]

    tc1, tc2, tc3, tc4 = st.columns(4)
    with tc1:
        st.metric("Threshold Accuracy", f"{thresh_eval['accuracy']*100:.1f}%")
    with tc2:
        st.metric("Threshold Precision", f"{thresh_eval['precision']*100:.1f}%", help="Higher threshold minimizes false retention alerts.")
    with tc3:
        st.metric("Threshold Recall", f"{thresh_eval['recall']*100:.1f}%", help="Lower threshold captures more vulnerable churners.")
    with tc4:
        st.metric("Threshold F1-Score", f"{thresh_eval['f1_score']:.3f}")

    # Confusion matrix display
    cm_cols1, cm_cols2 = st.columns([1, 1])
    with cm_cols1:
        cm_data = pd.DataFrame(
            [[cm_t["tn"], cm_t["fp"]], [cm_t["fn"], cm_t["tp"]]],
            index=["Actual Retained (0)", "Actual Churned (1)"],
            columns=["Predicted Retained (0)", "Predicted Churned (1)"]
        )
        st.write("**Confusion Matrix at Selected Threshold:**")
        st.dataframe(cm_data, use_container_width=True)

    with cm_cols2:
        if thresh_val < 0.40:
            st.warning("⚠️ **Low Threshold (< 0.40):** High Sensitivity/Recall. Identifies almost all potential churners, but generates higher false positives (contacting customers who were not actually leaving).")
        elif thresh_val > 0.60:
            st.info("ℹ️ **High Threshold (> 0.60):** High Precision. Minimizes wasted retention expenditure, but misses moderately vulnerable customers who quietly defect.")
        else:
            st.success("✅ **Balanced Threshold (0.40–0.60):** Optimal balance for general enterprise retention workflows.")

    st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------------------------------------------
    # Visualizations: Feature Drivers, ROC Curve, and PR Curve
    # -------------------------------------------------------------
    v_c1, v_c2, v_c3 = st.columns(3)

    with v_c1:
        st.markdown("#### 🔑 Top Feature Drivers")
        imp_df = meta["feature_importance_df"]
        if not imp_df.empty:
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
        else:
            st.info("Feature importance not extractable for this model.")

    with v_c2:
        st.markdown("#### 📈 ROC Curve")
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(meta["y_test"], meta["y_test_prob"])
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f"ROC (AUC = {test_eval['roc_auc']})", line=dict(color=COLOR_PRIMARY, width=2.5)))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random Baseline", line=dict(color="grey", dash="dash")))
        fig_roc.update_layout(get_plotly_layout(height=350), xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(fig_roc, use_container_width=True)

    with v_c3:
        st.markdown("#### 📉 Precision-Recall (PR) Curve")
        from sklearn.metrics import precision_recall_curve
        pr_p, pr_r, _ = precision_recall_curve(meta["y_test"], meta["y_test_prob"])
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(x=pr_r, y=pr_p, name=f"PR (AUC = {test_eval['pr_auc']})", line=dict(color=COLOR_SUCCESS, width=2.5)))
        baseline_rate = float(np.mean(meta["y_test"]))
        fig_pr.add_trace(go.Scatter(x=[0, 1], y=[baseline_rate, baseline_rate], name=f"Baseline ({baseline_rate:.2f})", line=dict(color="grey", dash="dash")))
        fig_pr.update_layout(get_plotly_layout(height=350), xaxis_title="Recall", yaxis_title="Precision")
        st.plotly_chart(fig_pr, use_container_width=True)

    # -------------------------------------------------------------
    # Predictions Table & Filter Section
    # -------------------------------------------------------------
    st.markdown("### 👥 Customer Predictions & Risk Scoring Table")
    pred_df = st.session_state.get("predictions_df")
    if pred_df is None:
        pred_df = predict_churn(pipeline, df, id_cols=selected_ids, target_col=target_col, threshold=thresh_val)
        st.session_state["predictions_df"] = pred_df

    col_f1, col_f2 = st.columns([1, 2])
    with col_f1:
        risk_filter = st.selectbox("Filter Risk Category", ["All", "High", "Medium", "Low"])
    with col_f2:
        id_search = st.text_input("Filter by Customer Identifier", placeholder="Search ID...")

    filtered_pred = pred_df.copy()
    if risk_filter != "All":
        filtered_pred = filtered_pred[filtered_pred["Risk_Category"] == risk_filter]
    if id_search and selected_ids:
        filtered_pred = filtered_pred[filtered_pred[selected_ids[0]].astype(str).str.contains(id_search, case=False)]

    display_cols = list(selected_ids) + ["Churn_Probability", "Predicted_Churn", "Risk_Category"]
    feature_extras = [c for c in ["tenure", "MonthlyCharges", "TotalCharges", "Contract"] if c in df.columns]
    display_cols += feature_extras
    if target_col in filtered_pred.columns:
        display_cols.append(target_col)

    st.dataframe(filtered_pred[display_cols].head(100), use_container_width=True)

    # -------------------------------------------------------------
    # What-If Counterfactual Simulator & Explainability Section
    # -------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 🔮 Interactive Customer What-If Simulator & Counterfactual Explainer")
    st.caption("Select a customer and modify contract, spending, or service features to simulate how interventions impact predicted attrition risk.")

    if selected_ids and selected_ids[0] in pred_df.columns:
        sample_ids = pred_df[selected_ids[0]].astype(str).head(50).tolist()
        inspect_id = st.selectbox("Select Customer for Simulation", sample_ids)
        cust_row = pred_df[pred_df[selected_ids[0]].astype(str) == inspect_id].iloc[0]

        # Calculate baseline explanation
        feat_names = meta["feature_names"]
        explanation = explain_customer_risk(pipeline, cust_row, feat_names)
        rec = generate_customer_recommendation(cust_row)

        c_exp1, c_exp2 = st.columns([1, 1])

        with c_exp1:
            st.markdown(
                f"""
                <div class="content-card">
                    <h4>Current Customer Profile: <code>{inspect_id}</code></h4>
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
                    <h4>Model Factor Attribution</h4>
                    <p style="color: #EF4444;"><strong>Top Risk Drivers:</strong><br>• {top_drivers}</p>
                    <p style="color: #10B981;"><strong>Protective Signals:</strong><br>• {top_protective}</p>
                    <p><strong>Prescriptive Action:</strong> <code>{rec['Recommended_Action']}</code></p>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Simulator Input Form
        st.markdown("#### 🧪 What-If Parameter Modifications")
        with st.form("what_if_form"):
            w_c1, w_c2, w_c3 = st.columns(3)
            
            with w_c1:
                cur_contract = str(cust_row.get("Contract", "Month-to-month"))
                contract_opts = ["Month-to-month", "One year", "Two year"]
                new_contract = st.selectbox("Simulate New Contract", contract_opts, index=contract_opts.index(cur_contract) if cur_contract in contract_opts else 0)

            with w_c2:
                cur_spend = float(cust_row.get("MonthlyCharges", 70.0))
                new_spend = st.slider("Simulate Monthly Charges ($)", min_value=15.0, max_value=150.0, value=cur_spend, step=5.0)

            with w_c3:
                cur_support = str(cust_row.get("TechSupport", "No"))
                support_opts = ["Yes", "No", "No internet service"] if "No internet service" in str(df.get("TechSupport", "")) else ["Yes", "No"]
                new_support = st.selectbox("Simulate Tech Support", support_opts, index=support_opts.index(cur_support) if cur_support in support_opts else 0)

            sim_submit = st.form_submit_button("⚡ Run What-If Simulation")

        if sim_submit:
            mods = {
                "Contract": new_contract,
                "MonthlyCharges": new_spend,
                "TechSupport": new_support
            }
            sim_res = simulate_customer_churn(
                pipeline,
                cust_row,
                mods,
                id_cols=selected_ids,
                target_col=target_col
            )

            p_orig = sim_res["original_probability"] * 100
            p_mod = sim_res["modified_probability"] * 100
            diff = sim_res["probability_diff"] * 100

            st.markdown("#### 🎯 Simulation Outcome")
            s1, s2, s3 = st.columns(3)
            with s1:
                st.metric("Baseline Churn Risk", f"{p_orig:.1f}%", cust_row['Risk_Category'])
            with s2:
                st.metric("Simulated Churn Risk", f"{p_mod:.1f}%", f"{diff:+.1f}% change", delta_color="inverse")
            with s3:
                st.metric("New Risk Tier", sim_res["modified_risk"], f"Shift from {sim_res['original_risk']}")

            if diff < -10.0:
                st.success(f"🎉 **Positive Retention Impact:** Proposed modifications reduce predicted attrition probability by **{abs(diff):.1f}%**, successfully de-escalating customer risk.")
            elif diff > 5.0:
                st.error(f"⚠️ **Adverse Impact:** Proposed modifications increase predicted attrition risk by **{diff:.1f}%**.")
            else:
                st.info("ℹ️ Minimal change in risk profile for the tested configuration.")
    else:
        st.info("Customer ID column required to select individual customers for What-If simulation.")
else:
    st.info("Click 'Train Selected Model Pipeline' to train and evaluate the selected classifier.")
