"""
Customer360 - Data Loading Module
Handles CSV ingestion, sample dataset retrieval, and synthetic dataset generation.
"""

import os
from typing import Optional, Union, Tuple
import pandas as pd
import numpy as np


DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
TELCO_DATA_PATH = os.path.join(DATA_DIR, "telco_customer_churn.csv")
RETAIL_DATA_PATH = os.path.join(DATA_DIR, "retail_transactions.csv")


def load_dataset(file_or_path: Union[str, os.PathLike, object]) -> pd.DataFrame:
    """
    Safely load a CSV dataset from a file path or uploaded buffer.
    Attempts multiple encodings and handles whitespace/delimiter variations.
    """
    encodings = ["utf-8", "latin1", "iso-8859-1", "cp1252"]
    df = None
    last_error = None

    for enc in encodings:
        try:
            if hasattr(file_or_path, "seek"):
                file_or_path.seek(0)
            df = pd.read_csv(file_or_path, encoding=enc)
            break
        except Exception as e:
            last_error = e
            continue

    if df is None:
        raise ValueError(f"Failed to read CSV dataset. Error: {str(last_error)}")

    # Standardize column headers: strip whitespace
    df.columns = [str(col).strip() for col in df.columns]

    # Clean known edge cases (e.g. Telco TotalCharges whitespace strings)
    for col in df.columns:
        if df[col].dtype == object:
            # Check if column contains mostly numbers with empty spaces
            cleaned_num = pd.to_numeric(df[col].astype(str).str.strip(), errors="coerce")
            # If over 80% non-null numeric after conversion and original had nulls/spaces
            if cleaned_num.notna().sum() > 0.8 * len(df) and df[col].nunique() > 10:
                df[col] = cleaned_num

    return df


def generate_synthetic_telco_data(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate a realistic synthetic Telco customer churn dataset.
    Clearly labeled as synthetic when generated.
    """
    rng = np.random.default_rng(random_state)
    customer_ids = [f"CUST-{1000 + i}" for i in range(n_samples)]
    genders = rng.choice(["Male", "Female"], size=n_samples)
    senior_citizens = rng.choice([0, 1], size=n_samples, p=[0.84, 0.16])
    partners = rng.choice(["Yes", "No"], size=n_samples, p=[0.48, 0.52])
    dependents = rng.choice(["Yes", "No"], size=n_samples, p=[0.30, 0.70])
    
    tenures = rng.integers(1, 73, size=n_samples)
    phone_services = rng.choice(["Yes", "No"], size=n_samples, p=[0.90, 0.10])
    multiple_lines = np.where(
        phone_services == "No",
        "No phone service",
        rng.choice(["Yes", "No"], size=n_samples, p=[0.42, 0.58])
    )
    internet_services = rng.choice(["DSL", "Fiber optic", "No"], size=n_samples, p=[0.34, 0.44, 0.22])
    
    online_security = np.where(internet_services == "No", "No internet service", rng.choice(["Yes", "No"], size=n_samples, p=[0.38, 0.62]))
    online_backup = np.where(internet_services == "No", "No internet service", rng.choice(["Yes", "No"], size=n_samples, p=[0.44, 0.56]))
    device_protection = np.where(internet_services == "No", "No internet service", rng.choice(["Yes", "No"], size=n_samples, p=[0.43, 0.57]))
    tech_support = np.where(internet_services == "No", "No internet service", rng.choice(["Yes", "No"], size=n_samples, p=[0.39, 0.61]))
    streaming_tv = np.where(internet_services == "No", "No internet service", rng.choice(["Yes", "No"], size=n_samples, p=[0.49, 0.51]))
    streaming_movies = np.where(internet_services == "No", "No internet service", rng.choice(["Yes", "No"], size=n_samples, p=[0.50, 0.50]))

    contracts = rng.choice(["Month-to-month", "One year", "Two year"], size=n_samples, p=[0.55, 0.21, 0.24])
    paperless_billing = rng.choice(["Yes", "No"], size=n_samples, p=[0.59, 0.41])
    payment_methods = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        size=n_samples,
        p=[0.33, 0.23, 0.22, 0.22]
    )

    # Monthly charges based on services
    base_charge = 20.0
    fiber_charge = np.where(internet_services == "Fiber optic", 40.0, np.where(internet_services == "DSL", 20.0, 0.0))
    services_add = (
        (online_security == "Yes") * 8.0 +
        (online_backup == "Yes") * 8.0 +
        (device_protection == "Yes") * 8.0 +
        (tech_support == "Yes") * 8.0 +
        (streaming_tv == "Yes") * 12.0 +
        (streaming_movies == "Yes") * 12.0
    )
    monthly_charges = np.round(base_charge + fiber_charge + services_add + rng.uniform(0.5, 6.0, size=n_samples), 2)
    total_charges = np.round(monthly_charges * tenures + rng.normal(0, 10, size=n_samples), 2)
    total_charges = np.maximum(total_charges, monthly_charges)

    # Churn probability higher for month-to-month, fiber without tech support, high monthly charge, low tenure
    churn_log_odds = (
        -1.5
        + (contracts == "Month-to-month") * 1.6
        - (contracts == "Two year") * 1.5
        + (payment_methods == "Electronic check") * 0.7
        + (internet_services == "Fiber optic") * 0.8
        - (tech_support == "Yes") * 0.6
        - (tenures / 72.0) * 2.2
        + (monthly_charges / 100.0) * 1.0
        + rng.normal(0, 0.4, size=n_samples)
    )
    churn_prob = 1.0 / (1.0 + np.exp(-churn_log_odds))
    churns = np.where(churn_prob > 0.48, "Yes", "No")

    df = pd.DataFrame({
        "customerID": customer_ids,
        "gender": genders,
        "SeniorCitizen": senior_citizens,
        "Partner": partners,
        "Dependents": dependents,
        "tenure": tenures,
        "PhoneService": phone_services,
        "MultipleLines": multiple_lines,
        "InternetService": internet_services,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contracts,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_methods,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Churn": churns
    })
    return df


def generate_synthetic_retail_transactions(n_customers: int = 300, n_transactions: int = 2500, random_state: int = 42) -> pd.DataFrame:
    """
    Generate synthetic transaction-level retail data for RFM segmentation.
    Columns: TransactionID, CustomerID, TransactionDate, OrderQuantity, UnitPrice, TotalAmount
    """
    rng = np.random.default_rng(random_state)
    customer_ids = [f"RET-{1000 + i}" for i in range(n_customers)]
    
    # Assign customer weights so some are frequent buyers and some infrequent
    customer_weights = rng.exponential(scale=1.0, size=n_customers)
    customer_weights /= customer_weights.sum()

    tx_customers = rng.choice(customer_ids, size=n_transactions, p=customer_weights)
    
    # Dates over the last 365 days
    ref_date = pd.Timestamp("2026-10-01")
    days_back = rng.integers(0, 365, size=n_transactions)
    tx_dates = [ref_date - pd.Timedelta(days=int(d)) for d in days_back]

    quantities = rng.integers(1, 10, size=n_transactions)
    unit_prices = np.round(rng.uniform(10.0, 150.0, size=n_transactions), 2)
    totals = np.round(quantities * unit_prices, 2)

    df = pd.DataFrame({
        "TransactionID": [f"TX-{20000 + i}" for i in range(n_transactions)],
        "CustomerID": tx_customers,
        "TransactionDate": tx_dates,
        "Quantity": quantities,
        "UnitPrice": unit_prices,
        "TotalAmount": totals
    })
    df = df.sort_values("TransactionDate").reset_index(drop=True)
    return df


def get_sample_telco_data() -> Tuple[pd.DataFrame, bool]:
    """
    Returns the Telco Churn dataset.
    Returns (DataFrame, is_synthetic_flag)
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(TELCO_DATA_PATH):
        try:
            df = load_dataset(TELCO_DATA_PATH)
            return df, False
        except Exception:
            pass

    # Generate synthetic fallback and save
    df = generate_synthetic_telco_data(n_samples=1200)
    df.to_csv(TELCO_DATA_PATH, index=False)
    return df, True


def get_sample_retail_data() -> Tuple[pd.DataFrame, bool]:
    """
    Returns the transaction-level Retail dataset for RFM analysis.
    Returns (DataFrame, is_synthetic_flag)
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(RETAIL_DATA_PATH):
        try:
            df = load_dataset(RETAIL_DATA_PATH)
            return df, False
        except Exception:
            pass

    df = generate_synthetic_retail_transactions(n_customers=350, n_transactions=3000)
    df.to_csv(RETAIL_DATA_PATH, index=False)
    return df, True

