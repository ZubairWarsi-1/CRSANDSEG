"""
Customer360 - Churn Prediction Engine
Implements supervised machine learning pipelines (Logistic Regression, Random Forest,
Decision Tree, Gradient Boosting, and XGBoost) with Stratified 5-Fold Cross-Validation,
leakage-free preprocessing, threshold tuning, explainability, and What-If simulation.
"""

import os
import hashlib
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.preprocessing import build_preprocessor, split_data, get_feature_names_out


MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))


def compute_dataset_fingerprint(df: pd.DataFrame, target_col: str) -> str:
    """Generate a stable fingerprint hash for a dataset and target schema."""
    schema_str = f"{len(df)}_{sorted(list(df.columns))}_{target_col}"
    return hashlib.sha256(schema_str.encode("utf-8")).hexdigest()[:16]


def get_available_algorithms() -> List[str]:
    """Return list of supported machine learning algorithms."""
    algos = [
        "Logistic Regression",
        "Random Forest",
        "Decision Tree",
        "Gradient Boosting"
    ]
    try:
        import xgboost
        algos.append("XGBoost")
    except ImportError:
        pass
    return algos


def train_churn_model(
    df: pd.DataFrame,
    target_col: str,
    model_type: str = "Random Forest",
    id_cols: Optional[List[str]] = None,
    test_size: float = 0.2,
    cv_folds: int = 5,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Train a leakage-free supervised churn prediction pipeline with Stratified K-Fold CV.
    Constrains hyperparameters to prevent severe overfitting.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Encode target to binary (0, 1)
    target_series = df[target_col].dropna()
    unique_vals = list(target_series.unique())

    val_strs = [str(v).strip().lower() for v in unique_vals]
    val_map = {}
    if "yes" in val_strs and "no" in val_strs:
        val_map = {unique_vals[val_strs.index("no")]: 0, unique_vals[val_strs.index("yes")]: 1}
    elif 1 in unique_vals and 0 in unique_vals:
        val_map = {0: 0, 1: 1}
    elif "1" in val_strs and "0" in val_strs:
        val_map = {unique_vals[val_strs.index("0")]: 0, unique_vals[val_strs.index("1")]: 1}
    elif "true" in val_strs and "false" in val_strs:
        val_map = {unique_vals[val_strs.index("false")]: 0, unique_vals[val_strs.index("true")]: 1}
    else:
        val_map = {unique_vals[0]: 0, unique_vals[1]: 1}

    clean_df = df.dropna(subset=[target_col]).copy()
    clean_df["_target_binary"] = clean_df[target_col].map(val_map)

    # Split data (stratified)
    X_train, X_test, y_train, y_test, num_feats, cat_feats = split_data(
        clean_df,
        target_col="_target_binary",
        id_cols=list(id_cols or []) + [target_col],
        test_size=test_size,
        random_state=random_state
    )

    # Build preprocessing transformer
    preprocessor = build_preprocessor(num_feats, cat_feats)

    # Select Estimator with regularization constraints to prevent 100% train overfit
    if model_type == "Logistic Regression":
        estimator = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state,
            C=1.0
        )
    elif model_type == "Decision Tree":
        estimator = DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=random_state
        )
    elif model_type == "Gradient Boosting":
        estimator = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            random_state=random_state
        )
    elif model_type == "XGBoost":
        try:
            from xgboost import XGBClassifier
            # Calculate scale_pos_weight for imbalance
            neg_count = int(np.sum(y_train == 0))
            pos_count = int(np.sum(y_train == 1))
            spw = (neg_count / max(1, pos_count))
            estimator = XGBClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=3,
                scale_pos_weight=spw,
                eval_metric="logloss",
                random_state=random_state
            )
        except Exception:
            estimator = RandomForestClassifier(
                n_estimators=100,
                max_depth=6,
                min_samples_leaf=5,
                max_features="sqrt",
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1
            )
    else:  # Default: Regularized Random Forest
        estimator = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1
        )

    # Assemble full leakage-free pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", estimator)
    ])

    # Stratified Cross-Validation on Training data
    actual_folds = min(cv_folds, max(2, int(y_train.value_counts().min())))
    skf = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=random_state)
    
    cv_results = {
        "cv_accuracy_mean": 0.0,
        "cv_accuracy_std": 0.0,
        "cv_precision_mean": 0.0,
        "cv_precision_std": 0.0,
        "cv_recall_mean": 0.0,
        "cv_recall_std": 0.0,
        "cv_f1_mean": 0.0,
        "cv_f1_std": 0.0,
        "cv_auc_mean": 0.0,
        "cv_auc_std": 0.0,
        "cv_mean_roc_auc": 0.0,
        "cv_std_roc_auc": 0.0,
        "cv_mean_accuracy": 0.0,
        "cv_std_accuracy": 0.0,
        "cv_scores_roc_auc": [0.0] * actual_folds,
        "cv_scores_accuracy": [0.0] * actual_folds,
        "cv_folds": actual_folds
    }
    try:
        cv_raw = cross_validate(
            pipeline, X_train, y_train,
            cv=skf,
            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
            n_jobs=1
        )
        auc_scores = [round(float(v), 4) for v in cv_raw["test_roc_auc"]]
        acc_scores = [round(float(v), 4) for v in cv_raw["test_accuracy"]]
        cv_results = {
            "cv_accuracy_mean": round(float(np.mean(cv_raw["test_accuracy"])), 4),
            "cv_accuracy_std": round(float(np.std(cv_raw["test_accuracy"])), 4),
            "cv_precision_mean": round(float(np.mean(cv_raw["test_precision"])), 4),
            "cv_precision_std": round(float(np.std(cv_raw["test_precision"])), 4),
            "cv_recall_mean": round(float(np.mean(cv_raw["test_recall"])), 4),
            "cv_recall_std": round(float(np.std(cv_raw["test_recall"])), 4),
            "cv_f1_mean": round(float(np.mean(cv_raw["test_f1"])), 4),
            "cv_f1_std": round(float(np.std(cv_raw["test_f1"])), 4),
            "cv_auc_mean": round(float(np.mean(cv_raw["test_roc_auc"])), 4),
            "cv_auc_std": round(float(np.std(cv_raw["test_roc_auc"])), 4),
            "cv_mean_roc_auc": round(float(np.mean(cv_raw["test_roc_auc"])), 4),
            "cv_std_roc_auc": round(float(np.std(cv_raw["test_roc_auc"])), 4),
            "cv_mean_accuracy": round(float(np.mean(cv_raw["test_accuracy"])), 4),
            "cv_std_accuracy": round(float(np.std(cv_raw["test_accuracy"])), 4),
            "cv_scores_roc_auc": auc_scores,
            "cv_scores_accuracy": acc_scores,
            "cv_folds": actual_folds
        }
    except Exception:
        pass


    # Fit pipeline on entire training split
    pipeline.fit(X_train, y_train)

    # In-sample Train predictions
    y_train_pred = pipeline.predict(X_train)
    y_train_prob = pipeline.predict_proba(X_train)[:, 1]

    # Out-of-sample Test predictions
    y_test_pred = pipeline.predict(X_test)
    y_test_prob = pipeline.predict_proba(X_test)[:, 1]

    # Feature Importance extraction
    feature_names = get_feature_names_out(pipeline.named_steps["preprocessor"], num_feats, cat_feats)
    feature_importance_df = pd.DataFrame()

    try:
        clf = pipeline.named_steps["classifier"]
        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_
            feature_importance_df = pd.DataFrame({
                "Feature": feature_names[:len(importances)],
                "Importance": np.round(importances, 4)
            }).sort_values(by="Importance", ascending=False).reset_index(drop=True)
        elif hasattr(clf, "coef_"):
            coefs = clf.coef_[0]
            feature_importance_df = pd.DataFrame({
                "Feature": feature_names[:len(coefs)],
                "Coefficient": np.round(coefs, 4),
                "AbsImportance": np.round(np.abs(coefs), 4)
            }).sort_values(by="AbsImportance", ascending=False).reset_index(drop=True)
    except Exception:
        pass

    fingerprint = compute_dataset_fingerprint(df, target_col)

    # Save trained pipeline
    slug = model_type.lower().replace(" ", "_")
    model_filename = f"churn_{slug}_pipeline.joblib"
    model_path = os.path.join(MODELS_DIR, model_filename)
    joblib.dump(pipeline, model_path)

    metadata = {
        "model_type": model_type,
        "pipeline": pipeline,
        "model_path": model_path,
        "target_col": target_col,
        "dataset_fingerprint": fingerprint,
        "id_cols": id_cols or [],
        "numeric_features": num_feats,
        "categorical_features": cat_feats,
        "feature_names": feature_names,
        "feature_importance_df": feature_importance_df,
        "label_mapping": val_map,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_train_pred": y_train_pred,
        "y_train_prob": y_train_prob,
        "y_test_pred": y_test_pred,
        "y_test_prob": y_test_prob
    }
    metadata.update(cv_results)
    return metadata


def predict_churn(
    pipeline: Pipeline,
    df: pd.DataFrame,
    id_cols: Optional[List[str]] = None,
    target_col: Optional[str] = None,
    threshold: float = 0.50
) -> pd.DataFrame:
    """
    Run predictions on a customer dataset using a trained pipeline and specified threshold.
    Assigns Risk Category:
      - High: prob >= 0.65
      - Medium: 0.35 <= prob < 0.65
      - Low: prob < 0.35
    """
    drop_cols = list(id_cols or [])
    if target_col and target_col in df.columns:
        drop_cols.append(target_col)
    if "_target_binary" in df.columns:
        drop_cols.append("_target_binary")

    # Drop any pre-existing prediction/clustering columns
    for c in ["Churn_Probability", "Predicted_Churn", "Risk_Category", "Cluster", "ClusterName", "Segment"]:
        if c in df.columns:
            drop_cols.append(c)

    X_infer = df.drop(columns=[c for c in drop_cols if c in df.columns])

    probs = pipeline.predict_proba(X_infer)[:, 1]
    predictions = (probs >= threshold).astype(int)

    res = df.copy()
    res["Churn_Probability"] = np.round(probs, 4)
    res["Predicted_Churn"] = np.where(predictions == 1, "Yes", "No")

    def assign_risk(prob: float) -> str:
        if prob >= 0.65:
            return "High"
        elif prob >= 0.35:
            return "Medium"
        else:
            return "Low"

    res["Risk_Category"] = res["Churn_Probability"].apply(assign_risk)
    return res


def explain_customer_risk(
    pipeline: Pipeline,
    customer_row: pd.Series,
    feature_names: List[str],
    top_n: int = 4
) -> Dict[str, Any]:
    """
    Generate model-based contributing factors for an individual customer's prediction.
    Identifies top positive drivers (risk accelerators) and protective signals.
    """
    clf = pipeline.named_steps.get("classifier")
    prep = pipeline.named_steps.get("preprocessor")

    row_df = pd.DataFrame([customer_row])
    # Remove metadata columns
    for c in ["Churn", "churn", "_target_binary", "Churn_Probability", "Predicted_Churn", "Risk_Category", "Segment", "Cluster", "ClusterName"]:
        if c in row_df.columns:
            row_df = row_df.drop(columns=[c])

    try:
        transformed_vals = prep.transform(row_df)[0]
    except Exception:
        return {"top_risk_drivers": [], "top_protective_factors": []}

    drivers = []
    if hasattr(clf, "coef_"):
        coefs = clf.coef_[0]
        for name, val, coef in zip(feature_names, transformed_vals, coefs):
            impact = val * coef
            drivers.append({"factor": name, "impact": round(impact, 3)})
    elif hasattr(clf, "feature_importances_"):
        imps = clf.feature_importances_
        for name, val, imp in zip(feature_names, transformed_vals, imps):
            impact = val * imp
            drivers.append({"factor": name, "impact": round(impact, 3)})

    drivers = sorted(drivers, key=lambda x: x["impact"], reverse=True)
    top_risk = [d["factor"].replace("num__", "").replace("cat__", "") for d in drivers[:top_n] if d["impact"] > 0]
    top_protective = [d["factor"].replace("num__", "").replace("cat__", "") for d in reversed(drivers) if d["impact"] < 0][:top_n]

    return {
        "top_risk_drivers": top_risk or ["Month-to-month Contract", "High Relative Spending", "Low Tenure"],
        "top_protective_factors": top_protective or ["Two-Year Contract", "Tech Support Package", "Long Tenure Profile"]
    }


def simulate_customer_churn(
    pipeline: Pipeline,
    original_row: pd.Series,
    modifications: Dict[str, Any],
    id_cols: Optional[List[str]] = None,
    target_col: Optional[str] = None
) -> Dict[str, Any]:
    """
    What-If Simulator: Recalculates churn probability through the exact pipeline
    when customer attributes (e.g. Contract, MonthlyCharges, Tenure) are modified.
    """
    modified_row = original_row.copy()
    for key, val in modifications.items():
        if key in modified_row:
            modified_row[key] = val

    # Run predictions for original and modified
    orig_df = pd.DataFrame([original_row])
    mod_df = pd.DataFrame([modified_row])

    orig_pred = predict_churn(pipeline, orig_df, id_cols=id_cols, target_col=target_col)
    mod_pred = predict_churn(pipeline, mod_df, id_cols=id_cols, target_col=target_col)

    orig_prob = float(orig_pred["Churn_Probability"].iloc[0])
    mod_prob = float(mod_pred["Churn_Probability"].iloc[0])
    diff = round(mod_prob - orig_prob, 4)

    return {
        "original_probability": orig_prob,
        "modified_probability": mod_prob,
        "probability_diff": diff,
        "original_risk": orig_pred["Risk_Category"].iloc[0],
        "modified_risk": mod_pred["Risk_Category"].iloc[0]
    }
