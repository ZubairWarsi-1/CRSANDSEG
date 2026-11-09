"""
Customer360 - Model Evaluation and Diagnostics Module
Computes comprehensive classification metrics, cross-validation scores,
overfitting detection, ROC & PR curves, calibration diagnostics, and threshold tuning.
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)
from sklearn.calibration import calibration_curve


def evaluate_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.50
) -> Dict[str, Any]:
    """
    Evaluate classification performance at a specific probability threshold.
    """
    y_pred = (y_prob >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    return {
        "threshold": round(threshold, 2),
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "confusion_matrix": {
            "matrix": cm.tolist(),
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp
        }
    }


def evaluate_classification_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    dataset_label: str = "Test"
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics for a model on a given dataset (Train, Val, or Test).
    Includes ROC, Precision-Recall Curve, and Calibration Brier Score.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    roc_auc = None
    pr_auc = None
    brier = None
    fpr, tpr, roc_thresh = None, None, None
    pr_prec, pr_rec, pr_thresh = None, None, None
    cal_prob_true, cal_prob_pred = None, None

    if y_prob is not None:
        try:
            roc_auc = roc_auc_score(y_true, y_prob)
            fpr, tpr, roc_thresh = roc_curve(y_true, y_prob)
        except Exception:
            pass

        try:
            pr_auc = average_precision_score(y_true, y_prob)
            pr_prec, pr_rec, pr_thresh = precision_recall_curve(y_true, y_prob)
        except Exception:
            pass

        try:
            brier = brier_score_loss(y_true, y_prob)
            cal_prob_true, cal_prob_pred = calibration_curve(y_true, y_prob, n_bins=5, strategy="uniform")
        except Exception:
            pass

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    return {
        "dataset": dataset_label,
        "sample_count": len(y_true),
        "churn_count": int(np.sum(y_true == 1)),
        "retained_count": int(np.sum(y_true == 0)),
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4) if roc_auc is not None else "N/A",
        "pr_auc": round(float(pr_auc), 4) if pr_auc is not None else "N/A",
        "brier_score": round(float(brier), 4) if brier is not None else "N/A",
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
            "thresholds": roc_thresh.tolist() if roc_thresh is not None else []
        },
        "pr_curve": {
            "precision": pr_prec.tolist() if pr_prec is not None else [],
            "recall": pr_rec.tolist() if pr_rec is not None else [],
            "thresholds": pr_thresh.tolist() if pr_thresh is not None else []
        },
        "calibration": {
            "prob_true": cal_prob_true.tolist() if cal_prob_true is not None else [],
            "prob_pred": cal_prob_pred.tolist() if cal_prob_pred is not None else []
        },
        "classification_report": report_dict
    }


def compare_models(models_metadata: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Generate standard comparison table of trained models.
    """
    rows = []
    for meta in models_metadata:
        m_name = meta["model_type"]
        train_eval = evaluate_classification_model(
            meta["y_train"], meta["y_train_pred"], meta["y_train_prob"], dataset_label="Train"
        )
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


def compare_models_advanced(
    models_metadata: List[Dict[str, Any]],
    return_tuple: bool = False
) -> Any:
    """
    Comprehensive diagnostic comparison table including Cross-Validation,
    Overfitting Detection (Generalization Gap), and Multi-Metric Selection Score.
    Returns: comparison_df (or (comparison_df, best_model, best_reason) if return_tuple=True).
    """
    rows = []
    best_model = None
    best_score = -1.0
    best_reason = ""

    for meta in models_metadata:
        m_name = meta["model_type"]
        train_eval = evaluate_classification_model(
            meta["y_train"], meta["y_train_pred"], meta["y_train_prob"], dataset_label="Train"
        )
        test_eval = evaluate_classification_model(
            meta["y_test"], meta["y_test_pred"], meta["y_test_prob"], dataset_label="Test"
        )

        cv_f1 = meta.get("cv_f1_mean", test_eval["f1_score"])
        cv_f1_std = meta.get("cv_f1_std", 0.0)
        cv_recall = meta.get("cv_recall_mean", test_eval["recall"])
        cv_auc = meta.get("cv_auc_mean", test_eval["roc_auc"] if isinstance(test_eval["roc_auc"], (int, float)) else 0.5)

        # Generalization gap
        train_f1 = train_eval["f1_score"]
        test_f1 = test_eval["f1_score"]
        gen_gap = round(train_f1 - test_f1, 3)

        # Overfitting check
        is_overfit = False
        overfit_msg = "Stable"
        if train_eval["accuracy"] >= 0.98 and test_eval["accuracy"] < 0.75:
            is_overfit = True
            overfit_msg = "Severe Overfit (Train ~100% vs Test Low)"
        elif gen_gap > 0.15:
            is_overfit = True
            overfit_msg = f"Moderate Overfit (Gap: +{gen_gap:.2f})"

        # Multi-metric selection score (Prioritize Recall 35%, F1 30%, ROC-AUC 25%, Precision 10%)
        t_auc = float(test_eval["roc_auc"]) if isinstance(test_eval["roc_auc"], (int, float)) else 0.5
        selection_score = (
            0.35 * test_eval["recall"] +
            0.30 * test_eval["f1_score"] +
            0.25 * t_auc +
            0.10 * test_eval["precision"]
        )
        # Penalize severe overfitting in selection score
        if is_overfit:
            selection_score *= 0.85

        selection_score = round(selection_score, 4)

        if selection_score > best_score:
            best_score = selection_score
            best_model = m_name
            best_reason = (
                f"{m_name} was selected because it achieves the highest balanced retention performance: "
                f"Recall: {test_eval['recall']*100:.1f}% (capturing churners), F1-Score: {test_eval['f1_score']:.3f}, "
                f"and ROC-AUC: {t_auc:.3f} without severe generalization gap (gap: {gen_gap:+.2f})."
            )

        rows.append({
            "Model": m_name,
            "Algorithm": m_name,
            "Test Accuracy": test_eval["accuracy"],
            "Test Recall (Sensitivity)": test_eval["recall"],
            "Test Precision": test_eval["precision"],
            "F1-Score": test_eval["f1_score"],
            "Test F1-Score": test_eval["f1_score"],
            "ROC-AUC": test_eval["roc_auc"],
            "Test ROC-AUC": test_eval["roc_auc"],
            "CV F1 (5-Fold)": f"{cv_f1:.3f} ± {cv_f1_std:.3f}" if cv_f1_std > 0 else f"{cv_f1:.3f}",
            "Train Accuracy": train_eval["accuracy"],
            "Train F1": train_f1,
            "Generalization Gap (Δ)": gen_gap,
            "Generalization Gap": gen_gap,
            "Overfitting Flag": overfit_msg,
            "Overfitting Status": overfit_msg,
            "Selection Score": selection_score
        })

    comp_df = pd.DataFrame(rows).sort_values(by="Selection Score", ascending=False).reset_index(drop=True)
    comp_df["Rank"] = range(1, len(comp_df) + 1)
    
    selected_name = best_model or "Random Forest"
    comp_df.attrs["best_model"] = selected_name
    comp_df.attrs["best_reason"] = best_reason

    if return_tuple:
        return comp_df, selected_name, best_reason
    return comp_df

