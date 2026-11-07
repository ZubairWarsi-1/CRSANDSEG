"""
Customer360 - Preprocessing Module
Constructs leakage-free scikit-learn preprocessing pipelines with ColumnTransformer.
"""

from typing import List, Optional, Tuple, Dict, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def get_feature_types(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    exclude_cols: Optional[List[str]] = None
) -> Tuple[List[str], List[str]]:
    """
    Separate features into numeric and categorical lists, excluding target and identifier columns.
    """
    excluded = set(exclude_cols or [])
    if target_col:
        excluded.add(target_col)

    numeric_cols: List[str] = []
    categorical_cols: List[str] = []

    for col in df.columns:
        if col in excluded:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    return numeric_cols, categorical_cols


def build_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str]
) -> ColumnTransformer:
    """
    Build a ColumnTransformer that imputes and scales numeric features,
    and imputes and one-hot encodes categorical features without data leakage.
    """
    transformers = []

    if numeric_features:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        transformers.append(("num", numeric_pipeline, numeric_features))

    if categorical_features:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        transformers.append(("cat", categorical_pipeline, categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor


def split_data(
    df: pd.DataFrame,
    target_col: str,
    id_cols: Optional[List[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str], List[str]]:
    """
    Split the dataset into training and testing sets with stratification.
    Excludes identifiers and target from feature matrix X.
    Returns: (X_train, X_test, y_train, y_test, numeric_features, categorical_features)
    """
    excluded = list(id_cols or [])
    excluded.append(target_col)

    X = df.drop(columns=[c for c in excluded if c in df.columns])
    y = df[target_col]

    numeric_features, categorical_features = get_feature_types(df, target_col=target_col, exclude_cols=id_cols)

    # Check stratification feasibility
    stratify = y if (y.nunique() == 2 and y.value_counts().min() >= 2) else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    return X_train, X_test, y_train, y_test, numeric_features, categorical_features


def get_feature_names_out(preprocessor: ColumnTransformer, numeric_features: List[str], categorical_features: List[str]) -> List[str]:
    """
    Extract transformed feature names from a fitted ColumnTransformer.
    """
    feature_names = []
    try:
        # scikit-learn 1.0+ get_feature_names_out
        feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        # Fallback if get_feature_names_out is not available
        feature_names.extend(numeric_features)
        if categorical_features and "cat" in preprocessor.named_transformers_:
            ohe = preprocessor.named_transformers_["cat"].named_steps.get("onehot")
            if ohe and hasattr(ohe, "get_feature_names_out"):
                feature_names.extend(list(ohe.get_feature_names_out(categorical_features)))
            else:
                feature_names.extend(categorical_features)
    return feature_names

