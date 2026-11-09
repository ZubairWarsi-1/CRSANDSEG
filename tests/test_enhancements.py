"""
Customer360 — Master Test Suite for System Enhancements
Tests all 17 advanced capabilities: SQLite persistence, Data cleaning, Transparent quality audit,
Auto-K evaluation, Unique cluster naming, 5 ML models, 5-Fold CV, Threshold tuning, What-If simulator,
Multi-sheet Excel, PDF generation, Contact logging, and Retention outcomes.
"""

import os
import pytest
import numpy as np
import pandas as pd
from datetime import datetime

from src.database import (
    init_db,
    upsert_intervention,
    get_all_interventions,
    get_intervention,
    log_contact_event,
    get_contact_history,
    add_retention_note,
    get_retention_notes,
    record_retention_outcome,
    get_retention_outcomes,
    get_observed_retention_rate,
    log_dataset_upload,
    get_upload_history,
    log_report_generation,
    get_report_history
)
from src.validation import (
    calculate_transparent_quality_score,
    clean_dataset,
    check_analytical_readiness,
    check_feature_redundancy,
    infer_column_roles
)
from src.segmentation import (
    evaluate_cluster_range,
    generate_unique_cluster_name,
    perform_kmeans_clustering
)
from src.churn_prediction import (
    train_churn_model,
    predict_churn,
    explain_customer_risk,
    simulate_customer_churn
)
from src.model_evaluation import (
    evaluate_at_threshold,
    evaluate_classification_model,
    compare_models_advanced
)
from src.recommendations import (
    calculate_priority_score_and_breakdown,
    preview_communication_template
)
from src.reporting import (
    export_multi_sheet_excel,
    generate_pdf_report,
    generate_one_page_executive_summary_pdf
)


@pytest.fixture
def sample_telco_df():
    """Generates a standard 80-row telco customer dataset for unit testing."""
    np.random.seed(42)
    n = 80
    data = {
        "customerID": [f"CUST-{i:04d}" for i in range(n)],
        "gender": np.random.choice(["Male", "Female"], size=n),
        "tenure": np.random.randint(1, 72, size=n),
        "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], size=n),
        "PaymentMethod": np.random.choice(["Electronic check", "Mailed check", "Bank transfer"], size=n),
        "MonthlyCharges": np.round(np.random.uniform(20.0, 110.0, size=n), 2),
        "TotalCharges": np.round(np.random.uniform(50.0, 4000.0, size=n), 2),
        "TechSupport": np.random.choice(["Yes", "No"], size=n),
        "InternetService": np.random.choice(["DSL", "Fiber optic", "No"], size=n),
        "Churn": np.random.choice(["Yes", "No"], size=n, p=[0.3, 0.7])
    }
    return pd.DataFrame(data)


# -------------------------------------------------------------
# 1. Database Operations & SQLite Persistence Tests
# -------------------------------------------------------------
def test_sqlite_database_crud():
    init_db()
    cid = f"TEST-{datetime.now().strftime('%f')}"

    # Upsert intervention
    upsert_intervention(
        cid,
        action_status="Pending",
        recommended_action="VIP Loyalty Perks",
        recommendation_reason="Initial note",
        priority_score=55.5,
        monthly_charges=120.0
    )
    inter = get_intervention(cid)
    assert inter is not None
    assert inter["action_status"] == "Pending"
    assert inter["priority_score"] == 55.5

    # Update intervention
    upsert_intervention(
        cid,
        action_status="In Progress",
        recommended_action="VIP Loyalty Perks",
        recommendation_reason="Updated note",
        priority_score=60.0,
        monthly_charges=130.0
    )
    inter_up = get_intervention(cid)
    assert inter_up["action_status"] == "In Progress"
    assert inter_up["priority_score"] == 60.0

    # Contact history
    log_contact_event(cid, "2026-11-09", "Phone", "Reached", "Customer agreed to review offer", "Agent Neo")
    hist = get_contact_history(cid)
    assert len(hist) > 0
    assert hist[0]["contact_method"] == "Phone"

    # Retention notes
    add_retention_note(cid, "Agent Neo", "Customer requested discount confirmation.")
    notes = get_retention_notes(cid)
    assert len(notes) > 0
    assert "discount" in notes[0]["note_text"]

    # Retention outcome
    record_retention_outcome(cid, actual_outcome="Retained", notes="Saved account with 1-yr contract")
    outcomes = get_retention_outcomes()
    assert len(outcomes) > 0
    assert any(o["actual_outcome"] == "Retained" for o in outcomes)

    # Observed retention rate
    rate_stats = get_observed_retention_rate()
    assert rate_stats["total_evaluated"] >= 1
    assert rate_stats["retention_rate"] >= 0.0

    # Dataset upload and report logging
    log_dataset_upload("test.csv", 100, 10, 95.0, "test", False)
    u_hist = get_upload_history()
    assert len(u_hist) > 0

    log_report_generation("Test Report", "test_dataset", "PDF")
    r_hist = get_report_history()
    assert not r_hist.empty



# -------------------------------------------------------------
# 2. Transparent Quality Score & Data Cleaning Tests
# -------------------------------------------------------------
def test_transparent_quality_score():
    dirty_data = pd.DataFrame({
        "id": ["1", "2", "3", "3"],  # Duplicate row
        "tenure": [12, -5, 24, 24],   # Negative tenure
        "MonthlyCharges": [" $50.0 ", "60.0", " ", "60.0"],  # Whitespace & string
        "TotalCharges": [100.0, -20.0, 300.0, 300.0],  # Negative financial
        "constant_col": [1, 1, 1, 1]  # Zero variance
    })

    score_res = calculate_transparent_quality_score(dirty_data)
    assert score_res["quality_score"] < 100
    assert len(score_res["deductions"]) > 0
    assert "grade" in score_res


def test_clean_dataset():
    dirty_data = pd.DataFrame({
        "id": ["1", "2", "3", "3"],
        "tenure": [12, 18, 24, 24],
        "TotalCharges": [" 100.5 ", " 200.0 ", " 300.0 ", " 300.0 "]
    })

    cleaned, summary = clean_dataset(dirty_data, remove_duplicates=True, correct_types=True)
    assert len(cleaned) == 3
    assert summary["duplicates_removed"] == 1
    assert pd.api.types.is_numeric_dtype(cleaned["TotalCharges"])


def test_analytical_readiness():
    telco_sample = pd.DataFrame({
        "customerID": ["C1", "C2", "C3"],
        "tenure": [10, 20, 30],
        "MonthlyCharges": [50.0, 60.0, 70.0],
        "Churn": ["Yes", "No", "No"]
    })
    roles = infer_column_roles(telco_sample)
    readiness = check_analytical_readiness(telco_sample, roles)

    assert "churn_prediction" in readiness
    assert "kmeans_clustering" in readiness
    assert "rfm_analysis" in readiness


def test_feature_redundancy():
    df_collinear = pd.DataFrame({
        "tenure": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "tenure_dup": [1.01, 2.02, 3.01, 3.99, 5.02, 6.01, 7.03, 8.01, 9.02, 10.01],
        "spending": [50, 45, 60, 55, 70, 65, 80, 75, 90, 85]
    })
    warnings = check_feature_redundancy(df_collinear, ["tenure", "tenure_dup", "spending"], threshold=0.85)
    assert len(warnings) >= 1
    assert warnings[0]["correlation"] >= 0.85


# -------------------------------------------------------------
# 3. K-Means Auto-K, Silhouette & Unique Naming Tests
# -------------------------------------------------------------
def test_kmeans_auto_k_and_unique_naming(sample_telco_df):
    km_result = perform_kmeans_clustering(
        sample_telco_df,
        feature_cols=["tenure", "MonthlyCharges", "TotalCharges"],
        n_clusters=None,  # Triggers auto-K selection
        target_col="Churn",
        random_state=42
    )

    assert km_result["auto_k"] in [2, 3, 4, 5, 6]
    assert "silhouette_score" in km_result
    assert "k_selection_reason" in km_result

    # Verify unique cluster names
    profiles = km_result["cluster_profiles"]
    names = profiles["ClusterName"].tolist()
    assert len(names) == len(set(names)), "Cluster names must be strictly unique!"

    # Verify cluster-level churn rate is present
    assert "HistoricalChurnRate" in profiles.columns
    assert "RecommendedStrategy" in profiles.columns


# -------------------------------------------------------------
# 4. Supervised ML: 5 Models, 5-Fold CV, Threshold & Simulator
# -------------------------------------------------------------
@pytest.mark.parametrize("model_name", [
    "Logistic Regression",
    "Random Forest",
    "Decision Tree",
    "Gradient Boosting",
    "XGBoost"
])
def test_all_five_classifiers(sample_telco_df, model_name):
    meta = train_churn_model(
        sample_telco_df,
        target_col="Churn",
        model_type=model_name,
        id_cols=["customerID"],
        test_size=0.25,
        random_state=42
    )

    assert meta["model_type"] == model_name
    assert meta["pipeline"] is not None
    assert "cv_mean_roc_auc" in meta
    assert len(meta["cv_scores_roc_auc"]) == 5

    # Test predict_churn
    pred_df = predict_churn(meta["pipeline"], sample_telco_df, id_cols=["customerID"], target_col="Churn")
    assert "Churn_Probability" in pred_df.columns
    assert "Risk_Category" in pred_df.columns
    assert set(pred_df["Risk_Category"].unique()).issubset({"High", "Medium", "Low"})


def test_threshold_tuning():
    y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    y_prob = np.array([0.9, 0.7, 0.2, 0.4, 0.65, 0.1, 0.8, 0.35])

    t_eval_low = evaluate_at_threshold(y_true, y_prob, threshold=0.30)
    t_eval_high = evaluate_at_threshold(y_true, y_prob, threshold=0.85)

    assert t_eval_low["recall"] >= t_eval_high["recall"]
    assert "confusion_matrix" in t_eval_low


def test_what_if_simulator(sample_telco_df):
    meta = train_churn_model(
        sample_telco_df,
        target_col="Churn",
        model_type="Logistic Regression",
        id_cols=["customerID"],
        test_size=0.25,
        random_state=42
    )

    cust = sample_telco_df.iloc[0]
    mods = {"Contract": "Two year", "MonthlyCharges": 25.0}

    sim = simulate_customer_churn(
        meta["pipeline"],
        cust,
        mods,
        id_cols=["customerID"],
        target_col="Churn"
    )

    assert "original_probability" in sim
    assert "modified_probability" in sim
    assert "probability_diff" in sim
    assert sim["original_risk"] in ["High", "Medium", "Low"]


def test_compare_models_advanced(sample_telco_df):
    m1 = train_churn_model(sample_telco_df, "Churn", "Logistic Regression", ["customerID"], random_state=42)
    m2 = train_churn_model(sample_telco_df, "Churn", "Random Forest", ["customerID"], random_state=42)

    comp_df = compare_models_advanced([m1, m2])
    assert len(comp_df) == 2
    assert "Rank" in comp_df.columns
    assert "Generalization Gap (Δ)" in comp_df.columns
    assert "Overfitting Flag" in comp_df.columns


# -------------------------------------------------------------
# 5. Priority Score Breakdown & Communication Preview
# -------------------------------------------------------------
def test_priority_score_breakdown():
    row = pd.Series({
        "MonthlyCharges": 85.0,
        "tenure": 6,
        "Churn_Probability": 0.80,
        "Risk_Category": "High"
    })
    breakdown = calculate_priority_score_and_breakdown(row)

    assert breakdown["risk_points"] == 40.0  # 0.80 * 50
    assert breakdown["tenure_points"] > 10.0  # (1 - 6/72)*15
    assert breakdown["priority_score"] > 50.0

    # Test communication template
    email_tpl = preview_communication_template(row, channel="Email", discount_pct=20)
    assert email_tpl["channel"] == "Email"
    assert "20%" in email_tpl["body"]

    sms_tpl = preview_communication_template(row, channel="SMS", discount_pct=15)
    assert sms_tpl["channel"] == "SMS"
    assert "15%" in sms_tpl["body"]


# -------------------------------------------------------------
# 6. Reporting Deliverables: Multi-Sheet Excel & PDF Generation
# -------------------------------------------------------------
def test_export_multi_sheet_excel(sample_telco_df):
    sheets = {
        "Executive Summary": pd.DataFrame({"Metric": ["Total"], "Value": [len(sample_telco_df)]}),
        "Customer Records": sample_telco_df.head(20),
        "High Risk Churn": sample_telco_df.head(5)
    }
    excel_bytes = export_multi_sheet_excel(sheets)
    assert len(excel_bytes) > 1000
    assert excel_bytes[:2] == b"PK"  # Zip archive header for .xlsx


def test_pdf_report_generation(sample_telco_df):
    report_data = {
        "dataset_name": "Unit_Test_Telco",
        "dataset_source": "Synthetic Benchmark",
        "row_count": len(sample_telco_df),
        "col_count": len(sample_telco_df.columns),
        "quality_score": 92,
        "selected_model": "Random Forest",
        "active_customers": "60",
        "avg_spending": 65.50,
        "churn_rate": "25.0%",
        "high_risk_count": "20",
        "revenue_at_risk": 1310.0,
        "models_summary": pd.DataFrame({"Model": ["Random Forest"], "F1-Score": [0.75]}),
        "clusters_summary": pd.DataFrame({"ClusterName": ["Cluster 1"], "CustomerCount": [80]}),
        "top_customers": sample_telco_df.head(5)
    }

    pdf_bytes = generate_pdf_report(report_data)
    assert len(pdf_bytes) > 2000
    assert pdf_bytes[:4] == b"%PDF"

    one_page_bytes = generate_one_page_executive_summary_pdf(report_data)
    assert len(one_page_bytes) > 1000
    assert one_page_bytes[:4] == b"%PDF"
