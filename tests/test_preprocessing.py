import pytest
import pandas as pd
from src.preprocessing import get_feature_types, build_preprocessor, split_data


def test_get_feature_types():
    df = pd.DataFrame({
        "id": ["1", "2"],
        "num1": [1.0, 2.0],
        "cat1": ["A", "B"],
        "target": [0, 1]
    })
    num_cols, cat_cols = get_feature_types(df, target_col="target", exclude_cols=["id"])
    assert num_cols == ["num1"]
    assert cat_cols == ["cat1"]


def test_build_preprocessor_and_transform():
    df = pd.DataFrame({
        "num1": [1.0, 2.0, None, 4.0],
        "cat1": ["Low", "High", "Low", "Medium"]
    })
    preprocessor = build_preprocessor(numeric_features=["num1"], categorical_features=["cat1"])
    transformed = preprocessor.fit_transform(df)
    assert transformed.shape[0] == 4
    # No NaNs should remain
    assert not pd.isna(transformed).any()


def test_split_data():
    df = pd.DataFrame({
        "id": [f"C{i}" for i in range(40)],
        "num1": list(range(40)),
        "cat1": ["A", "B"] * 20,
        "target": [0] * 20 + [1] * 20
    })
    X_train, X_test, y_train, y_test, num_feats, cat_feats = split_data(
        df, target_col="target", id_cols=["id"], test_size=0.25
    )
    assert len(X_train) == 30
    assert len(X_test) == 10
    assert "id" not in X_train.columns
    assert "target" not in X_train.columns

