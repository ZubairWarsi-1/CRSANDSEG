"""
Customer360 - Data Validation, Cleaning and Quality Auditing Module
Provides advanced dataset inspection, rule validation, automated cleaning,
analytical readiness diagnostics, and intelligent column role mapping.
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np


def inspect_dataset(df: pd.DataFrame, id_col: Optional[str] = None) -> Dict[str, Any]:
    """
    Perform a comprehensive health check and structural inspection of a DataFrame.
    Includes duplicate IDs, negative monetary checks, impossible tenure, constant columns,
    and outlier detection.
    """
    row_count, col_count = df.shape
    duplicate_rows = int(df.duplicated().sum())
    duplicate_pct = round((duplicate_rows / max(1, row_count)) * 100, 2)
    memory_mb = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)

    columns_info = []
    numeric_cols = []
    categorical_cols = []
    date_cols = []
    constant_cols = []
    outlier_counts = {}

    total_missing_cells = 0
    total_cells = row_count * col_count

    # Check duplicate customer IDs if an ID column is specified or detected
    duplicate_ids = 0
    inferred_id = id_col
    if not inferred_id:
        for c in df.columns:
            if "id" in c.lower() or "cust" in c.lower():
                inferred_id = c
                break
    if inferred_id and inferred_id in df.columns:
        duplicate_ids = int(df[inferred_id].duplicated().sum())

    negative_value_cols = []
    impossible_tenure_count = 0

    for col in df.columns:
        series = df[col]
        missing_count = int(series.isna().sum())
        total_missing_cells += missing_count
        missing_pct = round((missing_count / max(1, row_count)) * 100, 2)
        unique_count = int(series.nunique(dropna=True))

        # Constant column check
        if unique_count <= 1 and row_count > 1:
            constant_cols.append(col)

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
            
            # Check negative monetary values
            if any(term in col.lower() for term in ["charge", "spend", "amount", "price", "revenue", "monetary"]):
                neg_count = int((series < 0).sum())
                if neg_count > 0:
                    negative_value_cols.append({"column": col, "negative_count": neg_count})

            # Check impossible tenure (< 0 or > 120 months)
            if "tenure" in col.lower():
                impossible_tenure_count += int(((series < 0) | (series > 120)).sum())

            # Outlier detection using IQR (values beyond Q1 - 3*IQR or Q3 + 3*IQR)
            non_null = series.dropna()
            if len(non_null) > 10:
                q25 = non_null.quantile(0.25)
                q75 = non_null.quantile(0.75)
                iqr = q75 - q25
                if iqr > 0:
                    outliers = int(((non_null < (q25 - 3 * iqr)) | (non_null > (q75 + 3 * iqr))).sum())
                    if outliers > 0:
                        outlier_counts[col] = outliers

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

    # Transparent Data Quality Score calculation (0 - 100)
    score = 100.0
    deductions = []

    missing_ratio = (total_missing_cells / max(1, total_cells)) if total_cells > 0 else 0
    missing_deduction = round(missing_ratio * 40.0, 1)
    if missing_deduction > 0:
        score -= missing_deduction
        deductions.append(f"Missing Values: -{missing_deduction} pts ({missing_ratio*100:.1f}% empty cells)")

    dup_ratio = (duplicate_rows / max(1, row_count)) if row_count > 0 else 0
    dup_deduction = round(dup_ratio * 25.0, 1)
    if dup_deduction > 0:
        score -= dup_deduction
        deductions.append(f"Duplicate Rows: -{dup_deduction} pts ({duplicate_rows} duplicate rows)")

    if duplicate_ids > 0:
        id_dup_deduction = round(min(15.0, (duplicate_ids / max(1, row_count)) * 25.0), 1)
        score -= id_dup_deduction
        deductions.append(f"Duplicate Customer IDs: -{id_dup_deduction} pts ({duplicate_ids} non-unique IDs)")

    if constant_cols:
        score -= min(10.0, len(constant_cols) * 2.0)
        deductions.append(f"Zero-Variance Columns: -{min(10.0, len(constant_cols) * 2.0)} pts ({len(constant_cols)} columns)")

    if negative_value_cols:
        score -= 5.0
        deductions.append("Negative Financial Values: -5.0 pts")

    if impossible_tenure_count > 0:
        score -= 5.0
        deductions.append(f"Impossible Tenure (>120 or <0): -5.0 pts ({impossible_tenure_count} records)")

    quality_score = max(0.0, min(100.0, round(score, 1)))

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
        "duplicate_count": duplicate_rows,
        "duplicate_pct": duplicate_pct,
        "duplicate_ids": duplicate_ids,
        "total_missing_cells": total_missing_cells,
        "columns_info": columns_info,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "date_cols": date_cols,
        "constant_cols": constant_cols,
        "negative_value_cols": negative_value_cols,
        "impossible_tenure_count": impossible_tenure_count,
        "outlier_counts": outlier_counts,
        "quality_score": quality_score,
        "grade": grade,
        "grade_color": grade_color,
        "deductions": deductions
    }


def calculate_transparent_quality_score(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate transparent data quality score with deductions breakdown and grade.
    """
    res = inspect_dataset(df)
    formatted_deductions = []
    for d in res["deductions"]:
        parts = d.split(":")
        title = parts[0].strip() if len(parts) > 0 else "Audit Deduction"
        detail = ":".join(parts[1:]).strip() if len(parts) > 1 else d
        formatted_deductions.append({
            "penalty": title,
            "reason": detail
        })
    return {
        "quality_score": res["quality_score"],
        "grade": res["grade"],
        "grade_color": res["grade_color"],
        "deductions": formatted_deductions,
        "raw_deductions": res["deductions"],
        "inspection": res
    }



def clean_dataset(
    df: pd.DataFrame,
    remove_duplicates: bool = True,
    impute_missing: bool = True,
    correct_types: bool = True,
    cap_outliers: bool = False,
    normalize_categories: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Clean dataset according to user configurations while preserving the original.
    Returns: (cleaned_df, cleaning_summary)
    """
    cleaned = df.copy()
    initial_rows = len(cleaned)
    initial_missing = int(cleaned.isna().sum().sum())

    summary = {
        "initial_rows": initial_rows,
        "initial_missing": initial_missing,
        "duplicates_removed": 0,
        "missing_imputed": 0,
        "types_corrected": 0,
        "outliers_capped": 0,
        "categories_normalized": 0
    }

    # 1. Remove duplicate rows
    if remove_duplicates:
        dups = int(cleaned.duplicated().sum())
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)
        summary["duplicates_removed"] = dups

    # 2. Correct data types (numeric strings with spaces or currency signs to float)
    if correct_types:
        for col in cleaned.columns:
            if cleaned[col].dtype == object or pd.api.types.is_string_dtype(cleaned[col]):
                stripped = cleaned[col].astype(str).str.strip().str.replace("$", "", regex=False).str.replace(",", "", regex=False)
                # Ignore pure booleans or binary labels
                unique_lower = set(stripped.str.lower().unique())
                if unique_lower.issubset({"yes", "no", "true", "false", "0", "1", "churn", "churned"}):
                    continue
                numeric_conv = pd.to_numeric(stripped, errors="coerce")
                non_null_ratio = (numeric_conv.notna().sum() / max(1, len(cleaned)))
                if non_null_ratio > 0.70:
                    cleaned[col] = numeric_conv
                    summary["types_corrected"] += 1


    # 3. Normalize categories
    if normalize_categories:
        for col in cleaned.columns:
            if cleaned[col].dtype == object:
                cleaned[col] = cleaned[col].astype(str).str.strip()
                summary["categories_normalized"] += 1

    # 4. Impute missing values
    if impute_missing:
        for col in cleaned.columns:
            miss_count = int(cleaned[col].isna().sum())
            if miss_count > 0:
                if pd.api.types.is_numeric_dtype(cleaned[col]):
                    median_val = cleaned[col].median()
                    cleaned[col] = cleaned[col].fillna(median_val)
                    summary["missing_imputed"] += miss_count
                else:
                    mode_val = cleaned[col].mode()
                    if not mode_val.empty:
                        cleaned[col] = cleaned[col].fillna(mode_val[0])
                        summary["missing_imputed"] += miss_count

    # 5. Cap extreme outliers (IQR method)
    if cap_outliers:
        for col in cleaned.select_dtypes(include=[np.number]).columns:
            q25 = cleaned[col].quantile(0.25)
            q75 = cleaned[col].quantile(0.75)
            iqr = q75 - q25
            if iqr > 0:
                lower_bound = q25 - 3.0 * iqr
                upper_bound = q75 + 3.0 * iqr
                capped_mask = (cleaned[col] < lower_bound) | (cleaned[col] > upper_bound)
                summary["outliers_capped"] += int(capped_mask.sum())
                cleaned[col] = cleaned[col].clip(lower=lower_bound, upper=upper_bound)

    summary["final_rows"] = len(cleaned)
    summary["final_missing"] = int(cleaned.isna().sum().sum())
    return cleaned, summary


def infer_column_roles(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """
    Intelligently infer probable roles for columns (ID, Target Churn, Tenure, Spending, Date, Contract).
    """
    inferred: Dict[str, Optional[str]] = {
        "id_col": None,
        "target_col": None,
        "tenure_col": None,
        "monetary_col": None,
        "date_col": None,
        "contract_col": None,
        "internet_col": None
    }
    col_names = list(df.columns)
    lower_map = {c.lower(): c for c in col_names}

    # 1. Target Churn column
    for key in ["churn", "churned", "ischurn", "target", "status", "exited", "attrition"]:
        for col_l, orig in lower_map.items():
            if key == col_l or key in col_l.split("_"):
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

    # 6. Contract column
    for col in col_names:
        if "contract" in col.lower():
            inferred["contract_col"] = col
            break

    # 7. Internet service
    for col in col_names:
        if "internet" in col.lower():
            inferred["internet_col"] = col
            break

    return inferred


def check_analytical_readiness(df: pd.DataFrame, roles: Dict[str, Optional[str]]) -> Dict[str, Dict[str, Any]]:
    """
    Check readiness for Churn Prediction, K-Means Clustering, and RFM Analysis.
    Returns status (Ready/Not Ready) and missing requirements.
    """
    readiness = {}

    # 1. Churn Prediction Readiness
    churn_target = roles.get("target_col")
    churn_missing = []
    if not churn_target:
        churn_missing.append("Binary historical churn target column (e.g. 'Churn' with Yes/No or 1/0)")
    else:
        u_count = df[churn_target].nunique()
        if u_count != 2:
            churn_missing.append(f"Target '{churn_target}' must have exactly 2 distinct classes (found {u_count})")
    if len(df) < 30:
        churn_missing.append("Dataset has fewer than 30 records")

    churn_ready = len(churn_missing) == 0
    churn_status = "Ready" if churn_ready else ("Needs Preparation" if churn_target else "Incompatible")
    churn_msg = "Eligible for supervised ML classification." if churn_ready else "; ".join(churn_missing)

    churn_entry = {
        "name": "Supervised Churn Prediction",
        "ready": churn_ready,
        "status": churn_status,
        "message": churn_msg,
        "missing": churn_missing
    }
    readiness["churn"] = churn_entry
    readiness["churn_prediction"] = churn_entry

    # 2. K-Means Clustering Readiness
    kmeans_missing = []
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if not c.lower().endswith("id")]
    valid_numeric = [c for c in numeric_cols if df[c].nunique() > 1]
    if len(valid_numeric) < 2:
        kmeans_missing.append("At least 2 numerical features with non-zero variance")
    if len(df) < 10:
        kmeans_missing.append("Dataset has fewer than 10 records")

    km_ready = len(kmeans_missing) == 0
    km_status = "Ready" if km_ready else "Needs Preparation"
    km_msg = f"{len(valid_numeric)} valid numeric features available for clustering." if km_ready else "; ".join(kmeans_missing)

    km_entry = {
        "name": "K-Means Behavioral Clustering",
        "ready": km_ready,
        "status": km_status,
        "message": km_msg,
        "missing": kmeans_missing
    }
    readiness["kmeans"] = km_entry
    readiness["kmeans_clustering"] = km_entry

    # 3. Transactional RFM Readiness
    rfm_missing = []
    date_col = roles.get("date_col")
    monetary_col = roles.get("monetary_col")
    id_col = roles.get("id_col")

    if not id_col:
        rfm_missing.append("Customer identifier column")
    if not date_col:
        rfm_missing.append("Transaction order timestamp/date column")
    if not monetary_col:
        rfm_missing.append("Transaction monetary value column")

    rfm_ready = len(rfm_missing) == 0
    rfm_status = "Ready" if rfm_ready else "Incompatible"
    rfm_msg = "Contains transactional order history." if rfm_ready else "Missing transaction timestamp / amount columns."

    rfm_entry = {
        "name": "Transactional RFM Analysis",
        "ready": rfm_ready,
        "status": rfm_status,
        "message": rfm_msg,
        "missing": rfm_missing
    }
    readiness["rfm"] = rfm_entry
    readiness["rfm_analysis"] = rfm_entry


    return readiness


def check_feature_redundancy(df: pd.DataFrame, feature_cols: List[str]) -> List[Dict[str, Any]]:
    """
    Detect multicollinearity and redundancy among selected numerical features.
    Particularly checks Tenure, MonthlyCharges, and TotalCharges.
    """
    warnings = []
    num_df = df[[c for c in feature_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]]
    if len(num_df.columns) < 2:
        return warnings

    corr_matrix = num_df.corr().abs()
    cols = list(num_df.columns)

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c1, c2 = cols[i], cols[j]
            corr_val = corr_matrix.loc[c1, c2]
            if corr_val > 0.80:
                warnings.append({
                    "feature_1": c1,
                    "feature_2": c2,
                    "correlation": round(float(corr_val), 3),
                    "note": f"High collinearity between '{c1}' and '{c2}' (r={corr_val:.2f}). Consider keeping only one to avoid redundant weighting."
                })

    return warnings


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
        return False, f"Target column '{target_col}' must have exactly 2 distinct classes. Found {len(unique_vals)}: {unique_vals[:5]}.", None

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
    elif "churn" in val_strs[0] or "churned" in val_strs[0]:
        val_map = {unique_vals[1]: 0, unique_vals[0]: 1}
    elif "churn" in val_strs[1] or "churned" in val_strs[1]:
        val_map = {unique_vals[0]: 0, unique_vals[1]: 1}
    else:
        val_map = {unique_vals[0]: 0, unique_vals[1]: 1}

    mapped_counts = clean_target.map(val_map).value_counts().to_dict()
    min_count = min(mapped_counts.values()) if mapped_counts else 0
    if min_count < 5:
        return False, f"Minority class in '{target_col}' has fewer than 5 records ({min_count}), insufficient for modeling.", None

    if len(df) < 30:
        return False, f"Dataset has only {len(df)} rows. Minimum 30 records required for train/test split.", None

    return True, "Dataset is valid for supervised churn prediction.", val_map


def validate_for_rfm(
    df: pd.DataFrame, customer_id_col: str, date_col: str, monetary_col: str
) -> Tuple[bool, str]:
    """Validate whether dataset has transaction-level columns required for RFM."""
    for col, name in [(customer_id_col, "Customer ID"), (date_col, "Date"), (monetary_col, "Monetary/Amount")]:
        if not col or col not in df.columns:
            return False, f"Required {name} column '{col}' is missing."

    try:
        non_null = df[monetary_col].dropna()
        pd.to_numeric(non_null, errors="raise")
    except Exception:
        return False, f"Monetary column '{monetary_col}' contains non-numeric values."

    try:
        non_null_date = df[date_col].dropna().head(30)
        pd.to_datetime(non_null_date, errors="raise")
    except Exception:
        return False, f"Date column '{date_col}' cannot be parsed as datetime."

    return True, "Dataset is valid for RFM analysis."


def validate_for_clustering(
    df: pd.DataFrame, feature_cols: List[str]
) -> Tuple[bool, str]:
    """Validate whether chosen features are suitable for K-Means clustering."""
    if len(feature_cols) < 2:
        return False, "K-Means clustering requires at least 2 numerical features."

    for col in feature_cols:
        if col not in df.columns:
            return False, f"Feature column '{col}' is missing from the dataset."
        if not pd.api.types.is_numeric_dtype(df[col]):
            return False, f"Feature column '{col}' is not numeric."
        if df[col].nunique(dropna=True) <= 1:
            return False, f"Feature column '{col}' has zero variance (all values identical)."

    return True, "Selected features are valid for K-Means clustering."


def check_feature_redundancy(
    df: pd.DataFrame, feature_cols: List[str], threshold: float = 0.80
) -> List[Dict[str, Any]]:
    """
    Detect pairs of numerical features with high correlation (> threshold).
    In Euclidean K-Means clustering, redundant/collinear features artificially double-weight dimensions.
    """
    redundancies = []
    if len(feature_cols) < 2:
        return redundancies

    valid_sub = df[feature_cols].dropna()
    if len(valid_sub) < 5:
        return redundancies

    corr = valid_sub.corr().abs()
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            col1 = cols[i]
            col2 = cols[j]
            r_val = float(corr.loc[col1, col2])
            if r_val >= threshold:
                redundancies.append({
                    "feature_1": col1,
                    "feature_2": col2,
                    "correlation": round(r_val, 3),
                    "warning": (
                        f"Features '{col1}' and '{col2}' exhibit high collinearity (r = {r_val:.2f}). "
                        "Because K-Means uses Euclidean distance, including both gives double weight to this shared variance. "
                        "Consider selecting one or relying on the 2D PCA projection."
                    )
                })

    return redundancies

