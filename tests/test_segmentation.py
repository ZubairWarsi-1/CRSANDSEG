import pytest
import pandas as pd
from src.segmentation import calculate_rfm, perform_kmeans_clustering


def test_calculate_rfm():
    tx_df = pd.DataFrame({
        "CustomerID": ["C1", "C1", "C2", "C3", "C3", "C3"],
        "Date": ["2026-09-01", "2026-09-10", "2026-08-01", "2026-09-20", "2026-09-25", "2026-09-28"],
        "Amount": [50.0, 75.0, 30.0, 100.0, 120.0, 80.0]
    })
    rfm, stats = calculate_rfm(tx_df, "CustomerID", "Date", "Amount")
    assert len(rfm) == 3
    assert "Recency" in rfm.columns
    assert "Frequency" in rfm.columns
    assert "Monetary" in rfm.columns
    assert "Segment" in rfm.columns
    assert stats["total_customers"] == 3
    assert stats["total_revenue"] == 455.0


def test_perform_kmeans():
    df = pd.DataFrame({
        "feat1": [10, 12, 11, 80, 85, 90, 150, 160, 155],
        "feat2": [5, 6, 4, 40, 42, 39, 95, 90, 92]
    })
    res = perform_kmeans_clustering(df, ["feat1", "feat2"], n_clusters=3, random_state=42)
    assert "Cluster" in res["dataframe"].columns
    assert "ClusterName" in res["dataframe"].columns
    assert res["n_clusters"] == 3
    assert res["silhouette_score"] > 0.4
    assert len(res["elbow_data"]) >= 2
    assert "PCA1" in res["dataframe"].columns

