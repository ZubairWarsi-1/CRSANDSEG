import pytest
import pandas as pd
import numpy as np
from src.model_evaluation import evaluate_classification_model, compare_models


def test_evaluate_classification_model():
    y_true = np.array([0, 1, 0, 1, 1, 0, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1, 0, 1, 1])
    y_prob = np.array([0.1, 0.8, 0.2, 0.4, 0.9, 0.3, 0.7, 0.85])

    eval_res = evaluate_classification_model(y_true, y_pred, y_prob, dataset_label="Test")
    assert eval_res["accuracy"] > 0
    assert eval_res["precision"] > 0
    assert eval_res["recall"] > 0
    assert eval_res["f1_score"] > 0
    assert eval_res["roc_auc"] > 0.5
    assert "confusion_matrix" in eval_res
    assert eval_res["confusion_matrix"]["tp"] >= 1


def test_compare_models():
    meta1 = {
        "model_type": "Logistic Regression",
        "y_train": np.array([0, 1] * 20),
        "y_train_pred": np.array([0, 1] * 20),
        "y_train_prob": np.array([0.1, 0.9] * 20),
        "y_test": np.array([0, 1] * 10),
        "y_test_pred": np.array([0, 1] * 10),
        "y_test_prob": np.array([0.2, 0.8] * 10)
    }
    comp_df = compare_models([meta1])
    assert len(comp_df) == 1
    assert "Test Accuracy" in comp_df.columns
    assert "Test F1-Score" in comp_df.columns

