"""
Customer360 - Data Validation and Quality Auditing Module
Provides dataset inspection, column role inference, data quality scoring, and pipeline validations.
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np


def inspect_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform a comprehensive health check and structural inspection of a DataFrame.
    """
    row_count, col_count = df.shape
    duplicate_count = int(df.duplicated().sum())
    duplicate_pct = round((duplicate_count / max(1, row_count)) * 100, 2)
    memory_mb = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)

    columns_info = []
    numeric_cols = []
    categorical_cols = []
    date_cols = []

    total_missing_cells = 0
    total_cells = row_count * col_count

    for col in df.columns:
        series = df[col]
        missing_count = int(series.isna().sum())
        total_missing_cells += missing_count
        missing_pct = round((missing_count / max(1, row_count)) * 100, 2)
        unique_count = int(series.nunique(dropna=True))
        
        # Check date
        is_date = False
        if pd.api.types.is_datetime64_any_dtype(series):
            is_date = True
        elif series.dtype == object and ("date" in col.lower() or "time" in col.lower()):
            try:
                sample = series.dropna().head(20)
                pd.to_datetime(sample, errors="raise")
                is_date = True
            except Exception:
                is_date = False

        if is_date:
            date_cols.append(col)
            dtype_label = "datetime"
        elif pd.api.types.is_numeric_dtype(series):
            numeric_cols.append(col)
            dtype_label = "numeric"
        else:
            categorical_cols.append(col)
            dtype_label = "categorical"

        samples = series.dropna().unique()[:3].tolist()
        columns_info.append({
            "name": col,
            "type": dtype_label,
            "raw_dtype": str(series.dtype),
            "missing_count": missing_count,
            "missing_pct": missing_pct,
            "unique_count": unique_count,
            "samples": [str(s) for s in samples]
        })

    # Data Quality Score calculation (0 - 100)
    missing_ratio = (total_missing_cells / max(1, total_cells)) if total_cells > 0 else 0
    dup_ratio = (duplicate_count / max(1, row_count)) if row_count > 0 else 0
    
    score = 100.0 - (missing_ratio * 50.0) - (dup_ratio * 30.0)
    if col_count < 2 or row_count < 10:
        score -= 20.0
    quality_score = max(0, min(100, round(score, 1)))

    if quality_score >= 85:
        grade = "Excellent"
        grade_color = "green"
    elif quality_score >= 70:
        grade = "Good"
        grade_color = "blue"
    elif quality_score >= 50:
        grade = "Fair"
        grade_color = "orange"
    else:
        grade = "Poor"
        grade_color = "red"

    return {
        "row_count": row_count,
        "col_count": col_count,
        "memory_mb": memory_mb,
        "duplicate_count": duplicate_count,
        "duplicate_pct": duplicate_pct,
        "total_missing_cells": total_missing_cells,
        "columns_info": columns_info,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "date_cols": date_cols,
        "quality_score": quality_score,
        "grade": grade,
        "grade_color": grade_color
    }


def infer_column_roles(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """
    Intelligently infer probable roles for columns (ID, Target Churn, Tenure, Spending, Date).
    """
    inferred: Dict[str, Optional[str]] = {
        "id_col": None,
        "target_col": None,
        "tenure_col": None,
        "monetary_col": None,
        "date_col": None
    }
    col_names = list(df.columns)
    lower_map = {c.lower(): c for c in col_names}

    # 1. Target Churn column
    for key in ["churn", "churned", "ischurn", "target", "status", "exited", "attrition"]:
        for col_l, orig in lower_map.items():
            if key == col_l or key in col_l.split("_"):
                # verify it has few unique values (binary/status)
                if df[orig].nunique() <= 5:
                    inferred["target_col"] = orig
                    break
        if inferred["target_col"]:
            break

    # 2. Customer ID column
    for key in ["customerid", "customer_id", "cust_id", "client_id", "userid", "user_id", "id"]:
        for col_l, orig in lower_map.items():
            if orig == inferred["target_col"]:
                continue
            if key == col_l or key in col_l.split("_"):
                inferred["id_col"] = orig
                break
        if inferred["id_col"]:
            break

    # If still not found, check high cardinality unique text columns
    if not inferred["id_col"]:
        for col in col_names:
            if col == inferred["target_col"]:
                continue
            if df[col].dtype == object and df[col].nunique() == len(df):
                inferred["id_col"] = col
                break

    # 3. Tenure column
    for key in ["tenure", "tenure_months", "months_active", "membership_months", "account_age"]:
        for col_l, orig in lower_map.items():
            if key in col_l and pd.api.types.is_numeric_dtype(df[orig]):
                inferred["tenure_col"] = orig
                break
        if inferred["tenure_col"]:
            break

    # 4. Monetary / Spending column
    for key in ["monthlycharges", "totalcharges", "totalamount", "amount", "monetary", "spending", "revenue", "price"]:
        for col_l, orig in lower_map.items():
            if key in col_l.replace("_", "") and pd.api.types.is_numeric_dtype(df[orig]):
                inferred["monetary_col"] = orig
                break
        if inferred["monetary_col"]:
            break

    # 5. Date column
    for col in col_names:
        if "date" in col.lower() or "time" in col.lower():
            inferred["date_col"] = col
            break

    return inferred


def validate_for_churn_prediction(
    df: pd.DataFrame, target_col: str, id_col: Optional[str] = None
) -> Tuple[bool, str, Optional[Dict[str, int]]]:
    """
    Validate whether the dataset and chosen target column are suitable for supervised churn classification.
    Returns: (is_valid, message, label_mapping)
    """
    if target_col not in df.columns:
        return False, f"Target column '{target_col}' not found in dataset.", None

    clean_target = df[target_col].dropna()
    unique_vals = list(clean_target.unique())

    if len(unique_vals) != 2:
        return False, f"Target column '{target_col}' must have exactly 2 distinct classes for binary churn prediction. Found {len(unique_vals)}: {unique_vals[:5]}.", None

    # Determine positive (Churn) and negative (Retain) label
    val_strs = [str(v).strip().lower() for v in unique_vals]
    val_map = {}
    
    # Priority matching for positive class (Churn = 1)
    if "yes" in val_strs and "no" in val_strs:
        yes_idx = val_strs.index("yes")
        no_idx = val_strs.index("no")
        val_map = {unique_vals[no_idx]: 0, unique_vals[yes_idx]: 1}
    elif 1 in unique_vals and 0 in unique_vals:
        val_map = {0: 0, 1: 1}
    elif "1" in val_strs and "0" in val_strs:
        idx_1 = val_strs.index("1")
        idx_0 = val_strs.index("0")
        val_map = {unique_vals[idx_0]: 0, unique_vals[idx_1]: 1}
    elif "true" in val_strs and "false" in val_strs:
        t_idx = val_strs.index("true")
        f_idx = val_strs.index("false")
        val_map = {unique_vals[f_idx]: 0, unique_vals[t_idx]: 1}
    elif "churn" in val_strs[0] or "churned" in val_strs[0]:
        val_map = {unique_vals[1]: 0, unique_vals[0]: 1}
    elif "churn" in val_strs[1] or "churned" in val_strs[1]:
        val_map = {unique_vals[0]: 0, unique_vals[1]: 1}
    else:
        # Default second value is positive class
        val_map = {unique_vals[0]: 0, unique_vals[1]: 1}

    # Verify counts
    mapped_counts = clean_target.map(val_map).value_counts().to_dict()
    min_count = min(mapped_counts.values()) if mapped_counts else 0
    if min_count < 5:
        return False, f"Minority class in '{target_col}' has fewer than 5 records ({min_count}), insufficient for training.", None

    if len(df) < 30:
        return False, f"Dataset has only {len(df)} rows. A minimum of 30 records is recommended for training and validation.", None

    return True, "Dataset is valid for supervised churn prediction.", val_map


def validate_for_rfm(
    df: pd.DataFrame, customer_id_col: str, date_col: str, monetary_col: str
) -> Tuple[bool, str]:
    """
    Validate whether a dataset has the transaction-level columns required for RFM segmentation.
    """
    for col, name in [(customer_id_col, "Customer ID"), (date_col, "Date"), (monetary_col, "Monetary/Amount")]:
        if not col or col not in df.columns:
            return False, f"Required {name} column '{col}' is missing."

    # Validate monetary column can be converted to float
    try:
        non_null = df[monetary_col].dropna()
        pd.to_numeric(non_null, errors="raise")
    except Exception:
        return False, f"Monetary column '{monetary_col}' contains non-numeric values."

    # Validate date column can be parsed
    try:
        non_null_date = df[date_col].dropna().head(30)
        pd.to_datetime(non_null_date, errors="raise")
    except Exception:
        return False, f"Date column '{date_col}' cannot be parsed as datetime."

    return True, "Dataset is valid for RFM analysis."


def validate_for_clustering(
    df: pd.DataFrame, feature_cols: List[str]
) -> Tuple[bool, str]:
    """
    Validate whether chosen features are suitable for K-Means clustering.
    """
    if len(feature_cols) < 2:
        return False, "K-Means clustering requires at least 2 numerical features."

    for col in feature_cols:
        if col not in df.columns:
            return False, f"Feature column '{col}' is missing from the dataset."
        if not pd.api.types.is_numeric_dtype(df[col]):
            return False, f"Feature column '{col}' is not numeric."
        if df[col].nunique(dropna=True) <= 1:
            return False, f"Feature column '{col}' has zero variance (all values are identical)."

    return True, "Selected features are valid for K-Means clustering."

