"""
FocusLens — clusterer.py
=========================
Implements KMeans and DBSCAN clustering on hourly behavior feature vectors.

Design decisions:
  - KMeans (k=4): finds the 4 recurring behavior archetypes
    (Deep Focus / Distracted / Mixed / Offline) — good for well-separated clusters
  - DBSCAN: finds anomalous sessions that don't fit any pattern — no need to
    specify k, naturally handles noise points (label = -1)
  - StandardScaler: all features must be scaled before clustering
  - PCA (2D): for visualization only — not used during clustering
  - Cluster labels are derived from centroid characteristics, not hardcoded
"""

import sys
import logging
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)


# ─── Scaling ──────────────────────────────────────────────────────────────────

def scale_features(X: np.ndarray) -> Tuple[np.ndarray, StandardScaler]:
    """
    Standardize features to zero mean, unit variance.
    
    KMeans is distance-based — features on different scales will dominate
    distance calculations. Scaling ensures equal contribution from all features.
    
    Returns: (scaled array, fitted scaler for inverse-transform later)
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    logger.info(f"Features scaled: {X_scaled.shape}")
    return X_scaled, scaler


# ─── Elbow Method ─────────────────────────────────────────────────────────────

def elbow_analysis(X_scaled: np.ndarray, k_range: range = range(2, 10)) -> pd.DataFrame:
    """
    Compute inertia and silhouette score for a range of k values.
    Used to justify the choice of k=4 via the elbow method.
    
    Returns DataFrame with k, inertia, and silhouette_score for plotting.
    """
    results = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=config.RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_scaled)
        
        inertia = km.inertia_
        sil = silhouette_score(X_scaled, labels) if k > 1 else 0.0
        
        results.append({"k": k, "inertia": inertia, "silhouette_score": round(sil, 4)})
        logger.debug(f"k={k}: inertia={inertia:.1f}, silhouette={sil:.4f}")
    
    df = pd.DataFrame(results)
    logger.info(f"Elbow analysis complete. Best silhouette at k={df.loc[df['silhouette_score'].idxmax(), 'k']}")
    return df


# ─── KMeans Clustering ────────────────────────────────────────────────────────

def run_kmeans(
    X_scaled: np.ndarray,
    n_clusters: int = None,
) -> Tuple[np.ndarray, KMeans]:
    """
    Fit KMeans on scaled feature matrix.
    
    n_init=10: run with 10 different centroid seeds, keep best result.
    This prevents landing on a poor local minimum.
    
    Returns:
        labels: integer cluster assignment per sample
        model: fitted KMeans object (for centroid inspection)
    """
    if n_clusters is None:
        n_clusters = config.KMEANS_CLUSTERS
    
    km = KMeans(
        n_clusters=n_clusters,
        random_state=config.RANDOM_STATE,
        n_init=10,
        max_iter=300,
    )
    labels = km.fit_predict(X_scaled)
    
    sil = silhouette_score(X_scaled, labels)
    logger.info(
        f"KMeans (k={n_clusters}): "
        f"inertia={km.inertia_:.1f}, "
        f"silhouette={sil:.4f}"
    )
    
    # Log cluster sizes
    unique, counts = np.unique(labels, return_counts=True)
    for cluster, count in zip(unique, counts):
        logger.info(f"  Cluster {cluster}: {count} samples ({100*count/len(labels):.1f}%)")
    
    return labels, km


# ─── Auto-Label Clusters ──────────────────────────────────────────────────────

def auto_label_clusters(
    km_model: KMeans,
    scaler: StandardScaler,
    feature_names: list,
) -> Dict[int, str]:
    """
    Derive human-readable cluster labels from centroid characteristics.
    
    We inverse-transform the scaled centroids back to original feature space,
    then apply rules based on productive_ratio, distraction_ratio, and
    context_switch_rate to assign meaningful labels.
    
    This is data-driven — not hardcoded — so it adapts to different users.
    
    Rules (applied in priority order):
      1. Low total activity → "Low Activity / Offline"
      2. High productive_ratio (>= 0.5) and low distraction → "Deep Focus"
      3. High distraction_ratio (>= 0.4) → "Distracted"
      4. Default → "Mixed / Moderate Work"
    """
    centroids_scaled = km_model.cluster_centers_
    centroids = scaler.inverse_transform(centroids_scaled)
    centroid_df = pd.DataFrame(centroids, columns=feature_names)
    
    labels = {}
    for cluster_id, row in centroid_df.iterrows():
        prod = row.get("productive_ratio", 0)
        dist = row.get("distraction_ratio", 0)
        switches = row.get("context_switch_rate", 0)
        session = row.get("avg_session_minutes", 0)
        
        # Rule-based labeling
        if session < 1.0 and prod < 0.2 and dist < 0.2:
            label = config.CLUSTER_LABELS.get(3, "💤 Low Activity")
        elif prod >= 0.45 and dist <= 0.25 and switches <= 0.3:
            label = config.CLUSTER_LABELS.get(0, "🎯 Deep Focus")
        elif dist >= 0.38 or (dist > prod and switches >= 0.3):
            label = config.CLUSTER_LABELS.get(1, "😵 Distracted")
        else:
            label = config.CLUSTER_LABELS.get(2, "⚖️ Mixed Mode")
        
        labels[cluster_id] = label
        logger.info(
            f"  Cluster {cluster_id} → {label} "
            f"(prod={prod:.2f}, dist={dist:.2f}, switches={switches:.2f})"
        )
    
    return labels


def get_cluster_summary(
    feature_df: pd.DataFrame,
    labels: np.ndarray,
    cluster_names: Dict[int, str],
) -> pd.DataFrame:
    """
    Build a summary table of each cluster's mean characteristics.
    Used in the dashboard's ML Clusters page.
    """
    df = feature_df.copy()
    df["cluster"] = labels
    df["cluster_name"] = df["cluster"].map(cluster_names)
    
    numeric_cols = [
        "productive_ratio", "distraction_ratio", "context_switch_rate",
        "avg_session_minutes", "domain_diversity", "streak_ratio"
    ]
    available_cols = [c for c in numeric_cols if c in df.columns]
    
    summary = (
        df.groupby(["cluster", "cluster_name"])[available_cols]
        .mean()
        .round(3)
        .reset_index()
    )
    
    return summary


# ─── DBSCAN Clustering ────────────────────────────────────────────────────────

def run_dbscan(
    X_scaled: np.ndarray,
    eps: float = None,
    min_samples: int = None,
) -> np.ndarray:
    """
    Run DBSCAN to detect anomalous browsing sessions (noise points = -1).
    
    DBSCAN differs from KMeans:
    - Doesn't require specifying number of clusters
    - Points in low-density regions are labeled -1 (noise/anomaly)
    - Ideal for detecting unusual days — extreme overwork, all-distraction days, etc.
    
    Parameter guidance:
      eps: maximum distance between two points to be neighbors
           (in standardized feature space, 0.5–1.0 is a good starting range)
      min_samples: minimum neighbors to form a core point
                   (3–5 works for hourly data)
    """
    if eps is None:
        eps = config.DBSCAN_EPS
    if min_samples is None:
        min_samples = config.DBSCAN_MIN_SAMPLES
    
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
    labels = db.fit_predict(X_scaled)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    
    logger.info(
        f"DBSCAN (eps={eps}, min_samples={min_samples}): "
        f"{n_clusters} clusters, {n_noise} anomalies ({100*n_noise/len(labels):.1f}%)"
    )
    
    return labels


def get_anomaly_days(
    feature_df: pd.DataFrame,
    dbscan_labels: np.ndarray,
) -> pd.DataFrame:
    """
    Return the specific dates flagged as anomalies by DBSCAN (label == -1).
    These are days/hours that don't fit any normal behavior pattern.
    """
    df = feature_df.copy()
    df["dbscan_label"] = dbscan_labels
    
    anomalies = df[df["dbscan_label"] == -1].copy()
    
    if anomalies.empty:
        logger.info("No anomaly days detected by DBSCAN.")
        return pd.DataFrame()
    
    # Aggregate anomaly info by date
    anomaly_dates = (
        anomalies.groupby("date")
        .agg(
            anomaly_hours=("hour", "count"),
            avg_productive_ratio=("productive_ratio", "mean"),
            avg_distraction_ratio=("distraction_ratio", "mean"),
        )
        .round(3)
        .reset_index()
    )
    
    logger.info(f"Anomaly dates:\n{anomaly_dates[['date', 'anomaly_hours']].to_string()}")
    return anomaly_dates


# ─── PCA for Visualization ────────────────────────────────────────────────────

def run_pca(X_scaled: np.ndarray, n_components: int = None) -> Tuple[np.ndarray, PCA]:
    """
    Reduce feature matrix to 2D for scatter plot visualization.
    
    PCA is applied AFTER scaling and is used ONLY for visualization.
    Clustering is performed on the full feature space for accuracy.
    
    Returns:
        X_pca: (n_samples, 2) array of principal components
        pca: fitted PCA object (for explained variance inspection)
    """
    if n_components is None:
        n_components = config.PCA_COMPONENTS
    
    pca = PCA(n_components=n_components, random_state=config.RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    
    explained = pca.explained_variance_ratio_
    logger.info(
        f"PCA: {explained[0]*100:.1f}% + {explained[1]*100:.1f}% = "
        f"{sum(explained)*100:.1f}% variance explained"
    )
    
    return X_pca, pca


# ─── Main Pipeline ────────────────────────────────────────────────────────────

def run_full_clustering(
    X: np.ndarray,
    feature_df: pd.DataFrame,
) -> dict:
    """
    Run the complete clustering pipeline:
      1. Scale features
      2. KMeans clustering
      3. Auto-label clusters
      4. DBSCAN anomaly detection
      5. PCA for visualization
    
    Returns:
        dict with all results needed by dashboard and database storage
    """
    feature_names = [c for c in feature_df.columns if c not in ("date", "hour")]
    
    # Step 1: Scale
    X_scaled, scaler = scale_features(X)
    
    # Step 2: Elbow analysis (for dashboard display)
    elbow_df = elbow_analysis(X_scaled, k_range=range(2, 8))
    
    # Step 3: KMeans
    km_labels, km_model = run_kmeans(X_scaled, n_clusters=config.KMEANS_CLUSTERS)
    
    # Step 4: Auto-label clusters
    cluster_names = auto_label_clusters(km_model, scaler, feature_names)
    
    # Step 5: Cluster summary table
    cluster_summary = get_cluster_summary(feature_df, km_labels, cluster_names)
    
    # Step 6: DBSCAN
    dbscan_labels = run_dbscan(X_scaled)
    anomaly_days = get_anomaly_days(feature_df, dbscan_labels)
    
    # Step 7: PCA for 2D visualization
    X_pca, pca_model = run_pca(X_scaled)
    
    return {
        "km_labels": km_labels,
        "km_model": km_model,
        "cluster_names": cluster_names,
        "cluster_summary": cluster_summary,
        "dbscan_labels": dbscan_labels,
        "anomaly_days": anomaly_days,
        "X_pca": X_pca,
        "pca_model": pca_model,
        "scaler": scaler,
        "elbow_df": elbow_df,
        "silhouette": silhouette_score(X_scaled, km_labels),
    }


if __name__ == "__main__":
    from src.history_reader import generate_demo_history
    from src.categorizer import categorize_dataframe
    from src.feature_engineer import assign_sessions, build_hourly_features, build_feature_matrix
    
    df = generate_demo_history(days=14)
    df = categorize_dataframe(df)
    df = assign_sessions(df)
    hourly = build_hourly_features(df)
    X, feat_df = build_feature_matrix(hourly)
    
    results = run_full_clustering(X, feat_df)
    
    print("\n=== Clustering Results ===")
    print(f"Silhouette score: {results['silhouette']:.4f}")
    print(f"\nCluster names: {results['cluster_names']}")
    print(f"\nCluster summary:\n{results['cluster_summary']}")
    print(f"\nAnomalous days: {len(results['anomaly_days'])}")
    print(f"\nPCA shape: {results['X_pca'].shape}")
