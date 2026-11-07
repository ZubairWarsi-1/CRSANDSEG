"""
Customer360 - Customer Segmentation Module
Implements RFM Analysis and K-Means Clustering with interpretable segments.
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


def calculate_rfm(
    df: pd.DataFrame,
    customer_id_col: str,
    date_col: str,
    monetary_col: str,
    reference_date: Optional[pd.Timestamp] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Compute Recency, Frequency, and Monetary metrics from transaction records.
    Assigns quartile scores (1-5) and human-interpretable segment names.
    """
    clean_df = df.copy()
    clean_df[date_col] = pd.to_datetime(clean_df[date_col], errors="coerce")
    clean_df[monetary_col] = pd.to_numeric(clean_df[monetary_col], errors="coerce")
    clean_df = clean_df.dropna(subset=[customer_id_col, date_col, monetary_col])

    if reference_date is None:
        reference_date = clean_df[date_col].max() + pd.Timedelta(days=1)

    # Aggregate by customer
    rfm = clean_df.groupby(customer_id_col).agg(
        Recency=(date_col, lambda dates: max(0, int((reference_date - dates.max()).days))),
        Frequency=(date_col, "count"),
        Monetary=(monetary_col, "sum")
    ).reset_index()

    rfm["Monetary"] = rfm["Monetary"].round(2)

    # Quartile / Quintile scoring (1 to 5)
    # Higher recency days means worse (less recent), so invert R score
    # Use rank(method='first') to ensure robust binning even with duplicate values
    for col, invert in [("Recency", True), ("Frequency", False), ("Monetary", False)]:
        score_col = f"{col[0]}_Score"
        try:
            if invert:
                rfm[score_col] = pd.qcut(rfm[col].rank(method="first"), q=5, labels=[5, 4, 3, 2, 1]).astype(int)
            else:
                rfm[score_col] = pd.qcut(rfm[col].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
        except Exception:
            # Fallback if too few records
            rfm[score_col] = 3

    rfm["RFM_Score"] = (
        rfm["R_Score"].astype(str) + rfm["F_Score"].astype(str) + rfm["M_Score"].astype(str)
    )

    def assign_rfm_segment(row) -> str:
        r, f, m = row["R_Score"], row["F_Score"], row["M_Score"]
        if r >= 4 and f >= 4 and m >= 4:
            return "Champions"
        elif f >= 3 and m >= 3 and r >= 2:
            return "Loyal Customers"
        elif r >= 4 and (f >= 2 or m >= 2):
            return "Potential Loyalists"
        elif r <= 2 and (f >= 3 or m >= 3):
            return "At-Risk Customers"
        elif r <= 2 and f <= 2 and m <= 2:
            return "Lost / Inactive Customers"
        elif r >= 3 and f <= 2:
            return "Promising / Recent Customers"
        else:
            return "Regular Customers"

    rfm["Segment"] = rfm.apply(assign_rfm_segment, axis=1)

    # Segment summary statistics
    summary = rfm.groupby("Segment").agg(
        CustomerCount=(customer_id_col, "count"),
        AvgRecency=("Recency", "mean"),
        AvgFrequency=("Frequency", "mean"),
        AvgMonetary=("Monetary", "mean"),
        TotalRevenue=("Monetary", "sum")
    ).reset_index()

    summary["CustomerPct"] = ((summary["CustomerCount"] / len(rfm)) * 100).round(1)
    summary["AvgRecency"] = summary["AvgRecency"].round(1)
    summary["AvgFrequency"] = summary["AvgFrequency"].round(1)
    summary["AvgMonetary"] = summary["AvgMonetary"].round(2)
    summary["TotalRevenue"] = summary["TotalRevenue"].round(2)

    stats = {
        "total_customers": len(rfm),
        "total_revenue": round(rfm["Monetary"].sum(), 2),
        "avg_recency_days": round(rfm["Recency"].mean(), 1),
        "avg_frequency": round(rfm["Frequency"].mean(), 1),
        "avg_monetary": round(rfm["Monetary"].mean(), 2),
        "segment_summary": summary
    }

    return rfm, stats


def perform_kmeans_clustering(
    df: pd.DataFrame,
    feature_cols: List[str],
    n_clusters: int = 4,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Perform K-Means clustering on specified numerical features.
    Computes silhouette scores, elbow curve data, PCA 2D coordinates,
    and descriptive segment names.
    """
    X_raw = df[feature_cols].copy()
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_imputed = imputer.fit_transform(X_raw)
    X_scaled = scaler.fit_transform(X_imputed)

    # Elbow and Silhouette evaluation
    max_k = min(8, max(3, len(df) // 10))
    elbow_data = []
    
    for k in range(2, max_k + 1):
        km_test = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels_test = km_test.fit_predict(X_scaled)
        inertia = float(km_test.inertia_)
        sil = float(silhouette_score(X_scaled, labels_test)) if len(np.unique(labels_test)) > 1 else 0.0
        elbow_data.append({"k": k, "inertia": round(inertia, 2), "silhouette": round(sil, 3)})

    # Fit selected K
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    chosen_silhouette = float(silhouette_score(X_scaled, cluster_labels)) if len(np.unique(cluster_labels)) > 1 else 0.0

    # PCA for 2D visualization
    pca = PCA(n_components=2, random_state=random_state)
    coords_2d = pca.fit_transform(X_scaled)
    pca_variance = [round(float(v) * 100, 1) for v in pca.explained_variance_ratio_]

    # Cluster profiles
    res_df = df.copy()
    res_df["Cluster"] = cluster_labels
    res_df["PCA1"] = coords_2d[:, 0]
    res_df["PCA2"] = coords_2d[:, 1]

    # Compute means per cluster to generate dynamic, human-interpretable cluster names
    overall_means = X_raw.mean()
    cluster_profiles = []
    cluster_name_map = {}

    for c in range(n_clusters):
        c_sub = res_df[res_df["Cluster"] == c]
        c_means = c_sub[feature_cols].mean()
        c_count = len(c_sub)
        c_pct = round((c_count / len(res_df)) * 100, 1)

        # Determine distinguishing traits
        rel_diff = (c_means - overall_means) / (overall_means.abs() + 1e-6)
        top_high = rel_diff.idxmax()
        top_low = rel_diff.idxmin()

        # Generate descriptive label
        clean_high = top_high.replace("_", " ").title()
        clean_low = top_low.replace("_", " ").title()
        if rel_diff[top_high] > 0.15 and rel_diff[top_low] < -0.15:
            name = f"High {clean_high} / Low {clean_low}"
        elif rel_diff[top_high] > 0.15:
            name = f"High {clean_high} Group"
        elif rel_diff[top_low] < -0.15:
            name = f"Low {clean_low} Group"
        else:
            name = f"Moderate Segment {c + 1}"

        cluster_name_map[c] = f"Cluster {c + 1}: {name}"
        
        prof_entry = {"Cluster": c, "ClusterName": cluster_name_map[c], "Count": c_count, "Percentage": c_pct}
        for f in feature_cols:
            prof_entry[f"{f}_mean"] = round(float(c_means[f]), 2)
        cluster_profiles.append(prof_entry)

    res_df["ClusterName"] = res_df["Cluster"].map(cluster_name_map)

    return {
        "dataframe": res_df,
        "kmeans_model": kmeans,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "n_clusters": n_clusters,
        "silhouette_score": round(chosen_silhouette, 3),
        "elbow_data": pd.DataFrame(elbow_data),
        "cluster_profiles": pd.DataFrame(cluster_profiles),
        "pca_variance": pca_variance
    }

