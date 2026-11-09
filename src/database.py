"""
Customer360 - Database and Persistence Layer (SQLite)
Provides persistent storage for customer interventions, contact history,
notes, retention outcomes, dataset upload history, and report generation logs.
"""

import os
import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd

DB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
DB_PATH = os.path.join(DB_DIR, "customer360.db")


def get_db_connection() -> sqlite3.Connection:
    """Creates a thread-safe connection to the SQLite database."""
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_database() -> None:
    """Initializes all necessary database tables if they do not exist."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. Customer Interventions Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS customer_interventions (
            customer_id TEXT PRIMARY KEY,
            action_status TEXT NOT NULL DEFAULT 'Pending',
            recommended_action TEXT,
            recommendation_reason TEXT,
            priority_score REAL DEFAULT 0.0,
            churn_probability REAL DEFAULT 0.0,
            risk_category TEXT,
            monthly_charges REAL DEFAULT 0.0,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # 2. Contact History Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS contact_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id TEXT NOT NULL,
            contact_date TEXT NOT NULL,
            contact_method TEXT NOT NULL,
            offer_given TEXT,
            employee_name TEXT NOT NULL,
            customer_response TEXT,
            follow_up_date TEXT,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # 3. Customer Retention Notes Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS retention_notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id TEXT NOT NULL,
            employee_name TEXT NOT NULL,
            note_text TEXT NOT NULL,
            action_status TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # 4. Retention Outcomes Table (For tracking genuine retention success)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS retention_outcomes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id TEXT NOT NULL,
            intervention_date TEXT NOT NULL,
            intervention_type TEXT NOT NULL,
            previous_risk_probability REAL,
            observation_date TEXT NOT NULL,
            actual_outcome TEXT NOT NULL, -- 'Retained' or 'Churned'
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # 5. Dataset Upload History Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS upload_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_name TEXT NOT NULL,
            upload_timestamp TEXT NOT NULL,
            row_count INTEGER,
            col_count INTEGER,
            dataset_source TEXT NOT NULL,
            validation_status TEXT NOT NULL,
            quality_score REAL,
            file_path TEXT
        )
    """)

    # 6. Report Generation History Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS report_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_name TEXT NOT NULL,
            generation_timestamp TEXT NOT NULL,
            dataset_name TEXT NOT NULL,
            dataset_fingerprint TEXT,
            selected_model TEXT,
            report_type TEXT NOT NULL, -- 'CSV', 'PDF', 'Excel', 'Executive Summary'
            evaluation_metrics TEXT
        )
    """)

    # 7. Model Metadata / Fingerprint Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_registry (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL,
            dataset_fingerprint TEXT NOT NULL,
            target_column TEXT NOT NULL,
            training_timestamp TEXT NOT NULL,
            metrics_json TEXT NOT NULL,
            feature_names_json TEXT NOT NULL,
            model_path TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


# Ensure DB is created on import
init_database()


init_db = init_database

# -------------------------------------------------------------------------
# INTERVENTIONS CRUD & HELPERS
# -------------------------------------------------------------------------


def upsert_intervention(
    customer_id: str,
    action_status: str = "Pending",
    recommended_action: Optional[str] = None,
    recommendation_reason: Optional[str] = None,
    priority_score: float = 0.0,
    churn_probability: float = 0.0,
    risk_category: Optional[str] = None,
    monthly_charges: float = 0.0,
    assigned_to: Optional[str] = None,
    notes: Optional[str] = None,
    revenue_at_risk: Optional[float] = None
) -> None:
    """Insert or update customer intervention status."""
    conn = get_db_connection()
    cursor = conn.cursor()
    now_str = datetime.now().isoformat()
    # Check if extra columns exist, otherwise use standard schema
    cursor.execute("""
        INSERT INTO customer_interventions (
            customer_id, action_status, recommended_action, recommendation_reason,
            priority_score, churn_probability, risk_category, monthly_charges, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(customer_id) DO UPDATE SET
            action_status = excluded.action_status,
            recommended_action = COALESCE(excluded.recommended_action, customer_interventions.recommended_action),
            recommendation_reason = COALESCE(excluded.recommendation_reason, customer_interventions.recommendation_reason),
            priority_score = CASE WHEN excluded.priority_score > 0 THEN excluded.priority_score ELSE customer_interventions.priority_score END,
            churn_probability = CASE WHEN excluded.churn_probability > 0 THEN excluded.churn_probability ELSE customer_interventions.churn_probability END,
            risk_category = COALESCE(excluded.risk_category, customer_interventions.risk_category),
            monthly_charges = CASE WHEN excluded.monthly_charges > 0 THEN excluded.monthly_charges ELSE customer_interventions.monthly_charges END,
            updated_at = excluded.updated_at
    """, (
        customer_id,
        action_status,
        recommended_action,
        recommendation_reason or notes,
        priority_score,
        churn_probability,
        risk_category,
        monthly_charges if monthly_charges > 0 else (revenue_at_risk or 0.0),
        now_str
    ))
    conn.commit()
    conn.close()


def get_all_interventions() -> pd.DataFrame:
    """Retrieve all saved customer intervention records as a DataFrame."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM customer_interventions")
    rows = cursor.fetchall()
    conn.close()
    if not rows:
        return pd.DataFrame(columns=[
            "customer_id", "action_status", "recommended_action", "recommendation_reason",
            "priority_score", "churn_probability", "risk_category", "monthly_charges", "updated_at"
        ])
    return pd.DataFrame([dict(r) for r in rows])


def get_intervention(customer_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve a single intervention record by customer_id."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM customer_interventions WHERE customer_id = ?", (customer_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None



# -------------------------------------------------------------------------
# CONTACT HISTORY CRUD
# -------------------------------------------------------------------------

def add_contact_history(
    customer_id: str,
    contact_method: str,
    employee_name: str,
    offer_given: str = "",
    customer_response: str = "",
    follow_up_date: str = "",
    notes: str = "",
    contact_date: Optional[str] = None
) -> int:
    """Record customer contact log."""
    conn = get_db_connection()
    cursor = conn.cursor()
    c_date = contact_date or datetime.now().strftime("%Y-%m-%d %H:%M")
    cursor.execute("""
        INSERT INTO contact_history (
            customer_id, contact_date, contact_method, offer_given,
            employee_name, customer_response, follow_up_date, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (customer_id, c_date, contact_method, offer_given, employee_name, customer_response, follow_up_date, notes))
    rec_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return rec_id


def get_contact_history(customer_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get contact history for a specific customer or all customers."""
    conn = get_db_connection()
    cursor = conn.cursor()
    if customer_id:
        cursor.execute("SELECT * FROM contact_history WHERE customer_id = ? ORDER BY id DESC", (customer_id,))
    else:
        cursor.execute("SELECT * FROM contact_history ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


# -------------------------------------------------------------------------
# RETENTION NOTES CRUD
# -------------------------------------------------------------------------

def add_retention_note(
    customer_id: str,
    employee_name: str,
    note_text: str,
    action_status: Optional[str] = None
) -> int:
    """Add an employee note for a customer account."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO retention_notes (customer_id, employee_name, note_text, action_status)
        VALUES (?, ?, ?, ?)
    """, (customer_id, employee_name, note_text, action_status))
    rec_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return rec_id


def get_retention_notes(customer_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get retention notes for a customer or all records."""
    conn = get_db_connection()
    cursor = conn.cursor()
    if customer_id:
        cursor.execute("SELECT * FROM retention_notes WHERE customer_id = ? ORDER BY id DESC", (customer_id,))
    else:
        cursor.execute("SELECT * FROM retention_notes ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


# -------------------------------------------------------------------------
# RETENTION OUTCOMES (TRUE RETENTION SUCCESS METRICS)
# -------------------------------------------------------------------------

def record_retention_outcome(
    customer_id: str,
    intervention_date: Optional[str] = None,
    intervention_type: Optional[str] = None,
    actual_outcome: str = "Retained",
    previous_risk_probability: Optional[float] = None,
    observation_date: Optional[str] = None,
    notes: str = "",
    final_outcome: Optional[str] = None,
    days_since_intervention: Optional[int] = None,
    revenue_saved: Optional[float] = None
) -> int:
    """Record verified customer outcome post-intervention."""
    conn = get_db_connection()
    cursor = conn.cursor()
    int_date = intervention_date or datetime.now().strftime("%Y-%m-%d")
    int_type = intervention_type or "Retention Offer"
    outcome_val = final_outcome or actual_outcome
    obs_date = observation_date or datetime.now().strftime("%Y-%m-%d")
    note_val = notes or (f"Saved revenue: ${revenue_saved:.2f}" if revenue_saved is not None else "")
    cursor.execute("""
        INSERT INTO retention_outcomes (
            customer_id, intervention_date, intervention_type, previous_risk_probability,
            observation_date, actual_outcome, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (customer_id, int_date, int_type, previous_risk_probability, obs_date, outcome_val, note_val))
    rec_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return rec_id



def get_retention_outcomes() -> List[Dict[str, Any]]:
    """Retrieve all verified retention outcomes."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM retention_outcomes ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_retention_success_metrics() -> Dict[str, Any]:
    """Calculate true observed retention metrics from verified outcomes."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM customer_interventions WHERE action_status = 'Completed'")
    completed_interventions = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM retention_outcomes")
    total_observed = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM retention_outcomes WHERE actual_outcome = 'Retained'")
    retained_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM retention_outcomes WHERE actual_outcome = 'Churned'")
    churned_count = cursor.fetchone()[0]

    conn.close()

    retention_rate = round((retained_count / total_observed * 100), 1) if total_observed > 0 else None

    return {
        "completed_interventions": completed_interventions,
        "total_observed_outcomes": total_observed,
        "retained_count": retained_count,
        "churned_count": churned_count,
        "retention_success_rate": retention_rate
    }


# -------------------------------------------------------------------------
# UPLOAD HISTORY CRUD
# -------------------------------------------------------------------------

def log_upload_history(
    dataset_name: str,
    row_count: int,
    col_count: int,
    dataset_source: str,
    validation_status: str,
    quality_score: Optional[float] = None,
    file_path: Optional[str] = None
) -> int:
    """Log an uploaded dataset record."""
    conn = get_db_connection()
    cursor = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
        INSERT INTO upload_history (
            dataset_name, upload_timestamp, row_count, col_count,
            dataset_source, validation_status, quality_score, file_path
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (dataset_name, now_str, row_count, col_count, dataset_source, validation_status, quality_score, file_path))
    rec_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return rec_id


def get_upload_history() -> List[Dict[str, Any]]:
    """Retrieve history of uploaded datasets."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM upload_history ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


# -------------------------------------------------------------------------
# REPORT GENERATION HISTORY CRUD
# -------------------------------------------------------------------------

def log_report_generation(
    report_name: str,
    dataset_name: str = "Customer360 Dataset",
    report_type: str = "CSV",
    selected_model: Optional[str] = None,
    dataset_fingerprint: Optional[str] = None,
    evaluation_metrics: Optional[Dict[str, Any]] = None
) -> int:
    """Log generated reports."""
    conn = get_db_connection()
    cursor = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metrics_str = json.dumps(evaluation_metrics) if isinstance(evaluation_metrics, dict) else str(evaluation_metrics or "")
    cursor.execute("""
        INSERT INTO report_history (
            report_name, generation_timestamp, dataset_name, dataset_fingerprint,
            selected_model, report_type, evaluation_metrics
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (report_name, now_str, dataset_name, dataset_fingerprint, selected_model, str(report_type), metrics_str))
    rec_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return rec_id



def get_report_history() -> pd.DataFrame:
    """Retrieve history of generated reports as a DataFrame."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM report_history ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    if not rows:
        return pd.DataFrame(columns=["id", "report_name", "generation_timestamp", "dataset_name", "dataset_fingerprint", "selected_model", "report_type", "evaluation_metrics", "file_size_bytes", "parameters", "generated_at", "format"])
    df = pd.DataFrame([dict(r) for r in rows])
    if "generation_timestamp" in df.columns and "generated_at" not in df.columns:
        df["generated_at"] = df["generation_timestamp"]
    if "report_name" in df.columns and "report_type" not in df.columns:
        df["report_type"] = df["report_name"]
    if "file_size_bytes" not in df.columns:
        df["file_size_bytes"] = 2048
    if "parameters" not in df.columns:
        df["parameters"] = df.get("selected_model", "Standard")
    if "format" not in df.columns:
        df["format"] = df.get("report_type", "CSV")
    return df


# -------------------------------------------------------------------------
# CONVENIENCE WRAPPERS & ALIASES FOR UI PAGES
# -------------------------------------------------------------------------

def log_contact_event(
    customer_id: str,
    contact_date: str,
    channel: str,
    outcome: str,
    notes: str = "",
    agent_name: str = "Customer Success"
) -> int:
    """Convenience wrapper for contact history logging."""
    return add_contact_history(
        customer_id=customer_id,
        contact_method=channel,
        employee_name=agent_name,
        offer_given=notes,
        customer_response=outcome,
        notes=notes,
        contact_date=contact_date
    )


def get_contact_history_df(customer_id: Optional[str] = None) -> pd.DataFrame:
    """Get contact history as a pandas DataFrame."""
    logs = get_contact_history(customer_id)
    if not logs:
        return pd.DataFrame(columns=["id", "customer_id", "contact_date", "contact_method", "channel", "employee_name", "agent_name", "customer_response", "outcome", "notes"])
    df = pd.DataFrame(logs)
    if "contact_method" in df.columns:
        df["channel"] = df["contact_method"]
    if "employee_name" in df.columns:
        df["agent_name"] = df["employee_name"]
    if "customer_response" in df.columns:
        df["outcome"] = df["customer_response"]
    return df


def get_retention_notes_df(customer_id: Optional[str] = None) -> pd.DataFrame:
    """Get retention notes as a pandas DataFrame."""
    notes = get_retention_notes(customer_id)
    if not notes:
        return pd.DataFrame(columns=["id", "customer_id", "employee_name", "author", "note_text", "action_status", "created_at"])
    df = pd.DataFrame(notes)
    if "employee_name" in df.columns:
        df["author"] = df["employee_name"]
    if "created_at" not in df.columns:
        df["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    return df


def get_observed_retention_rate() -> Dict[str, Any]:
    """Return observed retention rate statistics."""
    metrics = get_retention_success_metrics()
    rate = metrics.get("retention_success_rate")
    return {
        "retention_rate": rate if rate is not None else 0.0,
        "total_evaluated": metrics.get("total_observed_outcomes", 0),
        "retained_count": metrics.get("retained_count", 0),
        "churned_count": metrics.get("churned_count", 0)
    }


def log_dataset_upload(
    filename: str,
    row_count: int,
    col_count: int,
    quality_score: float = 100.0,
    dataset_type: str = "upload",
    is_synthetic: bool = False
) -> int:
    """Convenience wrapper for upload history logging."""
    return log_upload_history(
        dataset_name=filename,
        row_count=row_count,
        col_count=col_count,
        dataset_source="Synthetic Test Data" if is_synthetic else "User Upload",
        validation_status="Valid",
        quality_score=quality_score
    )


def get_upload_history_df() -> pd.DataFrame:
    """Retrieve history of uploaded datasets as a DataFrame."""
    uploads = get_upload_history()
    if not uploads:
        return pd.DataFrame(columns=["uploaded_at", "filename", "row_count", "col_count", "quality_score", "dataset_type"])
    df = pd.DataFrame(uploads)
    if "upload_timestamp" in df.columns and "uploaded_at" not in df.columns:
        df["uploaded_at"] = df["upload_timestamp"]
    if "dataset_name" in df.columns and "filename" not in df.columns:
        df["filename"] = df["dataset_name"]
    if "dataset_source" in df.columns and "dataset_type" not in df.columns:
        df["dataset_type"] = df["dataset_source"]
    return df


