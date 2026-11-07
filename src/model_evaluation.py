"""
Customer360 - Model Evaluation Module
Calculates performance metrics, confusion matrices, ROC curves, and side-by-side model comparisons.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)


def evaluate_classification_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    dataset_label: str = "Test"
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics for a model on a given dataset (Train or Test).
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    roc_auc = None
    fpr, tpr, thresholds = None, None, None
    if y_prob is not None:
        try:
            roc_auc = roc_auc_score(y_true, y_prob)
            fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        except Exception:
            pass

    cm = confusion_matrix(y_true, y_pred)
    # cm format: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    return {
        "dataset": dataset_label,
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4) if roc_auc is not None else "N/A",
        "confusion_matrix": {
            "matrix": cm.tolist(),
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp
        },
        "roc_curve": {
            "fpr": fpr.tolist() if fpr is not None else [],
            "tpr": tpr.tolist() if tpr is not None else [],
            "thresholds": thresholds.tolist() if thresholds is not None else []
        },
        "classification_report": report_dict
    }


def compare_models(
    models_metadata: List[Dict[str, Any]]
) -> pd.DataFrame:
    """
    Generate a formatted side-by-side comparison table of trained models.
    """
    rows = []
    for meta in models_metadata:
        m_name = meta["model_type"]
        # Train metrics
        train_eval = evaluate_classification_model(
            meta["y_train"], meta["y_train_pred"], meta["y_train_prob"], dataset_label="Train"
        )
        # Test metrics
        test_eval = evaluate_classification_model(
            meta["y_test"], meta["y_test_pred"], meta["y_test_prob"], dataset_label="Test"
        )

        rows.append({
            "Model": m_name,
            "Test Accuracy": test_eval["accuracy"],
            "Test Precision": test_eval["precision"],
            "Test Recall": test_eval["recall"],
            "Test F1-Score": test_eval["f1_score"],
            "Test ROC-AUC": test_eval["roc_auc"],
            "Train Accuracy": train_eval["accuracy"],
            "Train F1-Score": train_eval["f1_score"]
        })

    return pd.DataFrame(rows)

