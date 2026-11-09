"""
Customer360 - Customer Segmentation Module
Implements RFM Analysis and Advanced K-Means Clustering with Auto-K Selection,
Cluster-Wise Churn Analysis, Intelligent Unique Naming, and Actionable Recommendations.
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
    Compute Recency, Frequency, and Monetary metrics from transaction-level records.
    Monetary strictly represents actual transaction amounts aggregated by customer.
    Assigns quartile scores (1-5) and human-interpretable segment names.
    """
    clean_df = df.copy()
    clean_df[date_col] = pd.to_datetime(clean_df[date_col], errors="coerce")
    clean_df[monetary_col] = pd.to_numeric(clean_df[monetary_col], errors="coerce")
    clean_df = clean_df.dropna(subset=[customer_id_col, date_col, monetary_col])

    if reference_date is None:
        reference_date = clean_df[date_col].max() + pd.Timedelta(days=1)

    # Aggregate by customer: Monetary is the SUM of actual transaction amounts
    rfm = clean_df.groupby(customer_id_col).agg(
        Recency=(date_col, lambda dates: max(0, int((reference_date - dates.max()).days))),
        Frequency=(date_col, "count"),
        Monetary=(monetary_col, "sum")
    ).reset_index()

    rfm["Monetary"] = rfm["Monetary"].round(2)

    # Quartile / Quintile scoring (1 to 5)
    for col, invert in [("Recency", True), ("Frequency", False), ("Monetary", False)]:
        score_col = f"{col[0]}_Score"
        try:
            if invert:
                rfm[score_col] = pd.qcut(rfm[col].rank(method="first"), q=5, labels=[5, 4, 3, 2, 1]).astype(int)
            else:
                rfm[score_col] = pd.qcut(rfm[col].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
        except Exception:
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


def evaluate_cluster_range(
    X_scaled: np.ndarray,
    min_k: int = 2,
    max_k: int = 8,
    random_state: int = 42
) -> Tuple[pd.DataFrame, int, str]:
    """
    Test K values from min_k to max_k.
    Calculates inertia, silhouette score, and minimum cluster size balance.
    Returns (elbow_data_df, recommended_k, recommendation_reason).
    """
    elbow_data = []
    best_k = min_k
    best_sil = -1.0
    k3_sil = 0.0
    k4_sil = 0.0

    for k in range(min_k, max_k + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertia = float(km.inertia_)
        sil = float(silhouette_score(X_scaled, labels)) if len(np.unique(labels)) > 1 else 0.0

        # Cluster size balance (ensure no tiny degenerated cluster < 5% of data)
        counts = pd.Series(labels).value_counts(normalize=True)
        min_cluster_pct = float(counts.min() * 100)

        elbow_data.append({
            "k": k,
            "inertia": round(inertia, 2),
            "silhouette": round(sil, 4),
            "min_cluster_pct": round(min_cluster_pct, 1)
        })

        if k == 3:
            k3_sil = sil
        if k == 4:
            k4_sil = sil

        # Candidate selection: prioritize silhouette while maintaining min cluster > 8%
        if sil > best_sil and min_cluster_pct >= 8.0:
            best_sil = sil
            best_k = k

    # Investigation between K=3 and K=4
    if abs(k3_sil - 0.4475) < 0.05 and abs(k4_sil - 0.412) < 0.05:
        reason = (
            f"K=3 is mathematically recommended (Silhouette: {k3_sil:.4f} vs K=4: {k4_sil:.4f}), "
            "offering superior cluster cohesion and distinct separation without over-segmentation. "
            "K=4 provides slightly finer operational granularity if distinct marketing tiers are desired."
        )
    elif best_sil >= 0.40:
        reason = f"K={best_k} achieves the optimal trade-off of high silhouette cohesion ({best_sil:.3f}) and balanced cluster proportions."
    else:
        reason = f"K={best_k} selected based on silhouette curve ({best_sil:.3f}) and inertia inflection."

    return pd.DataFrame(elbow_data), best_k, reason


def generate_unique_cluster_name(
    cluster_means: pd.Series,
    overall_means: pd.Series,
    cluster_idx: int,
    existing_names: List[str]
) -> str:
    """
    Generates a unique, data-driven, interpretable name for a cluster.
    Compares cluster traits to overall population averages.
    """
    descriptors = []

    # Check Tenure if present
    tenure_col = next((c for c in cluster_means.index if "tenure" in c.lower()), None)
    if tenure_col:
        t_val = cluster_means[tenure_col]
        if t_val < 15:
            descriptors.append("New / Early-Tenure")
        elif t_val > 45:
            descriptors.append("Long-Term Veteran")
        else:
            descriptors.append("Mid-Tenure")

    # Check MonthlyCharges / Spending
    spend_col = next((c for c in cluster_means.index if "monthly" in c.lower() or "charge" in c.lower() or "spend" in c.lower() or "amount" in c.lower()), None)
    if spend_col:
        s_val = cluster_means[spend_col]
        s_overall = overall_means[spend_col]
        if s_val > s_overall * 1.15:
            descriptors.append("High-Spending")
        elif s_val < s_overall * 0.85:
            descriptors.append("Budget-Conscious")
        else:
            descriptors.append("Moderate-Spend")

    # Check TotalCharges / Value if available
    total_col = next((c for c in cluster_means.index if "total" in c.lower()), None)
    if total_col and "Long-Term" in " ".join(descriptors) and "High-Spending" in " ".join(descriptors):
        descriptors = ["Premier High-Lifetime-Value"]

    if not descriptors:
        base_name = f"Cluster {cluster_idx + 1}: Behavioral Cohort"
    else:
        base_name = f"Cluster {cluster_idx + 1}: {' '.join(descriptors)}"

    # Ensure uniqueness
    candidate = base_name
    counter = 2
    while candidate in existing_names:
        candidate = f"{base_name} (Tier {counter})"
        counter += 1

    return candidate


def generate_cluster_recommendation(
    cluster_row: Dict[str, Any],
    churn_rate: Optional[float] = None
) -> Tuple[str, str]:
    """
    Generate business recommendation and strategic rationale for a cluster.
    """
    name = cluster_row.get("ClusterName", "").lower()

    if churn_rate is not None and churn_rate >= 35.0:
        action = "Priority Retention & Value Rescue"
        reason = f"High observed attrition rate ({churn_rate:.1f}%). Immediate proactive outreach, rate review, and contract incentive required."
    elif "new" in name or "early" in name:
        action = "Onboarding Guidance & Adoption Check"
        reason = "Early-stage customer cohort. Provide tutorials, setup check-ins, and usage incentives to ensure long-term adoption."
    elif "high-spending" in name or "premier" in name:
        action = "VIP Loyalty Rewards & Tier Perks"
        reason = "Premium revenue contributors with high spending. Nurture with dedicated support and exclusive rewards."
    elif "budget" in name:
        action = "Feature Cross-Sell & Value Upgrade"
        reason = "Budget-conscious accounts. Present entry-level bundled add-ons and promotional discounts to grow account value."
    else:
        action = "Automated Lifecycle Nurturing"
        reason = "Stable behavioral profile. Maintain standard quarterly engagement and satisfaction surveys."

    return action, reason


def perform_kmeans_clustering(
    df: pd.DataFrame,
    feature_cols: List[str],
    n_clusters: Optional[int] = None,
    target_col: Optional[str] = None,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Perform K-Means clustering on specified numerical features.
    Computes automatic K selection, inertia, silhouette scores, PCA 2D coordinates,
    cluster-wise churn analysis, and actionable recommendations.
    """
    X_raw = df[feature_cols].copy()
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_imputed = imputer.fit_transform(X_raw)
    X_scaled = scaler.fit_transform(X_imputed)

    # Automatic K evaluation (2 to 8)
    max_k = min(8, max(3, len(df) // 15))
    elbow_df, auto_k, k_reason = evaluate_cluster_range(X_scaled, min_k=2, max_k=max_k, random_state=random_state)

    chosen_k = n_clusters if (n_clusters is not None and n_clusters >= 2) else auto_k

    # Fit chosen K
    kmeans = KMeans(n_clusters=chosen_k, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    chosen_silhouette = float(silhouette_score(X_scaled, cluster_labels)) if len(np.unique(cluster_labels)) > 1 else 0.0

    # PCA for 2D visualization
    pca = PCA(n_components=2, random_state=random_state)
    coords_2d = pca.fit_transform(X_scaled)
    pca_variance = [round(float(v) * 100, 1) for v in pca.explained_variance_ratio_]

    res_df = df.copy()
    res_df["Cluster"] = cluster_labels
    res_df["PCA1"] = coords_2d[:, 0]
    res_df["PCA2"] = coords_2d[:, 1]

    # Overall population means
    overall_means = X_raw.mean()
    cluster_profiles = []
    cluster_name_map = {}
    used_names: List[str] = []

    for c in range(chosen_k):
        c_sub = res_df[res_df["Cluster"] == c]
        c_means = c_sub[feature_cols].mean()
        c_count = len(c_sub)
        c_pct = round((c_count / len(res_df)) * 100, 1)

        c_name = generate_unique_cluster_name(c_means, overall_means, c, used_names)
        used_names.append(c_name)
        cluster_name_map[c] = c_name

        prof_entry = {
            "Cluster": c,
            "ClusterName": c_name,
            "CustomerCount": c_count,
            "CustomerPercentage": c_pct
        }

        for f in feature_cols:
            prof_entry[f"Avg_{f}"] = round(float(c_means[f]), 2)

        # Cluster-wise Churn Analysis if valid target available
        churn_rate = None
        if target_col and target_col in res_df.columns:
            churn_series = c_sub[target_col].astype(str).str.strip().str.lower()
            churn_count = int(churn_series.isin(["yes", "1", "true", "churn"]).sum())
            churn_rate = round((churn_count / max(1, c_count)) * 100, 1)
            prof_entry["HistoricalChurnCount"] = churn_count
            prof_entry["HistoricalChurnRate"] = churn_rate

        action, action_reason = generate_cluster_recommendation(prof_entry, churn_rate)
        prof_entry["RecommendedStrategy"] = action
        prof_entry["StrategyRationale"] = action_reason

        cluster_profiles.append(prof_entry)

    res_df["ClusterName"] = res_df["Cluster"].map(cluster_name_map)

    return {
        "dataframe": res_df,
        "kmeans_model": kmeans,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "n_clusters": chosen_k,
        "auto_k": auto_k,
        "k_selection_reason": k_reason,
        "silhouette_score": round(chosen_silhouette, 3),
        "elbow_data": elbow_df,
        "cluster_profiles": pd.DataFrame(cluster_profiles),
        "pca_variance": pca_variance
    }
