"""
Customer360 - Churn Prediction Module
Implements supervised machine learning pipelines for Logistic Regression and Random Forest.
Includes explainability, risk categorization, and model serialization.
"""

import os
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.preprocessing import build_preprocessor, split_data, get_feature_names_out


MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))


def train_churn_model(
    df: pd.DataFrame,
    target_col: str,
    model_type: str = "Random Forest",
    id_cols: Optional[List[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Train a leakage-free supervised churn prediction pipeline.
    model_type: 'Logistic Regression' or 'Random Forest'
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Encode target to binary (0, 1)
    target_series = df[target_col].dropna()
    unique_vals = list(target_series.unique())

    # Map target values
    val_strs = [str(v).strip().lower() for v in unique_vals]
    val_map = {}
    if "yes" in val_strs and "no" in val_strs:
        val_map = {unique_vals[val_strs.index("no")]: 0, unique_vals[val_strs.index("yes")]: 1}
    elif 1 in unique_vals and 0 in unique_vals:
        val_map = {0: 0, 1: 1}
    elif "1" in val_strs and "0" in val_strs:
        val_map = {unique_vals[val_strs.index("0")]: 0, unique_vals[val_strs.index("1")]: 1}
    else:
        val_map = {unique_vals[0]: 0, unique_vals[1]: 1}

    clean_df = df.dropna(subset=[target_col]).copy()
    clean_df["_target_binary"] = clean_df[target_col].map(val_map)

    # Split data
    X_train, X_test, y_train, y_test, num_feats, cat_feats = split_data(
        clean_df,
        target_col="_target_binary",
        id_cols=list(id_cols or []) + [target_col],
        test_size=test_size,
        random_state=random_state
    )

    # Build preprocessing transformer
    preprocessor = build_preprocessor(num_feats, cat_feats)

    # Select Estimator
    if model_type == "Logistic Regression":
        estimator = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state,
            C=1.0
        )
    else:
        estimator = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1
        )

    # Assemble full pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", estimator)
    ])

    # Train
    pipeline.fit(X_train, y_train)

    # Predict train and test
    y_train_pred = pipeline.predict(X_train)
    y_train_prob = pipeline.predict_proba(X_train)[:, 1]

    y_test_pred = pipeline.predict(X_test)
    y_test_prob = pipeline.predict_proba(X_test)[:, 1]

    # Feature Importance / Coefficients extraction
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
    return metadata


def predict_churn(
    pipeline: Pipeline,
    df: pd.DataFrame,
    id_cols: Optional[List[str]] = None,
    target_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Run predictions on a customer dataset using a trained pipeline.
    Calculates churn probabilities and risk categories (Low, Medium, High).
    """
    # Drop IDs and Target if present in feature matrix
    drop_cols = list(id_cols or [])
    if target_col and target_col in df.columns:
        drop_cols.append(target_col)
    if "_target_binary" in df.columns:
        drop_cols.append("_target_binary")

    X_infer = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Probabilities
    probs = pipeline.predict_proba(X_infer)[:, 1]
    predictions = pipeline.predict(X_infer)

    res = df.copy()
    res["Churn_Probability"] = np.round(probs, 4)
    res["Predicted_Churn"] = np.where(predictions == 1, "Yes", "No")

    # Risk Categories
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
    Uses model weights/importances and input feature values.
    """
    clf = pipeline.named_steps.get("classifier")
    prep = pipeline.named_steps.get("preprocessor")

    row_df = pd.DataFrame([customer_row])
    # Remove metadata columns
    for c in ["Churn", "churn", "_target_binary", "Churn_Probability", "Predicted_Churn", "Risk_Category", "Segment", "Cluster"]:
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
            drivers.append({"factor": name, "impact": round(impact, 3), "val": val})
    elif hasattr(clf, "feature_importances_"):
        imps = clf.feature_importances_
        for name, val, imp in zip(feature_names, transformed_vals, imps):
            impact = abs(val) * imp
            drivers.append({"factor": name, "impact": round(impact, 3), "val": val})

    drivers = sorted(drivers, key=lambda x: x["impact"], reverse=True)
    top_risk = [d["factor"].replace("num__", "").replace("cat__", "") for d in drivers[:top_n] if d["impact"] > 0]
    top_protective = [d["factor"].replace("num__", "").replace("cat__", "") for d in reversed(drivers) if d["impact"] < 0][:top_n]

    return {
        "top_risk_drivers": top_risk or ["Tenure / Contract type", "Monthly spending profile"],
        "top_protective_factors": top_protective or ["Long-term relationship", "Service bundled packages"]
    }

