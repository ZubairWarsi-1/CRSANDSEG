import pytest
import pandas as pd
from src.reporting import export_dataframe_to_csv, generate_data_quality_markdown, get_data_quality_df


def test_export_dataframe_to_csv():
    df = pd.DataFrame({"col1": [1, 2], "col2": ["A", "B"]})
    csv_bytes = export_dataframe_to_csv(df)
    assert isinstance(csv_bytes, bytes)
    assert b"col1,col2" in csv_bytes


def test_generate_data_quality_markdown():
    inspection = {
        "quality_score": 92.5,
        "grade": "Excellent",
        "row_count": 500,
        "col_count": 10,
        "memory_mb": 0.45,
        "duplicate_count": 0,
        "duplicate_pct": 0.0,
        "total_missing_cells": 5,
        "columns_info": [
            {"name": "tenure", "type": "numeric", "missing_count": 0, "missing_pct": 0.0, "unique_count": 72}
        ]
    }
    md = generate_data_quality_markdown(inspection)
    assert "Customer360 - Data Quality Audit Report" in md
    assert "92.5/100" in md
    assert "tenure" in md

    cols_df = get_data_quality_df(inspection)
    assert len(cols_df) == 1
    assert "name" in cols_df.columns

