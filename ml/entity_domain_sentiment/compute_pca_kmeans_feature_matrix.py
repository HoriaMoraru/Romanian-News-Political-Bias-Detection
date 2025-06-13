import logging
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ───────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ───────────────────────────────────────────────────────────────────────────────
INPUT_FEATURE_MATRIX     = "dataset/ml/entity_source_sentiment_features.csv"
OUTPUT_SILHOUETTE_CSV    = "dataset/ml/entity_source_pca_silhouette.csv"
OUTPUT_CLUSTERS          = "dataset/ml/entity_source_pca_clusters.csv"
OUTPUT_SUMMARY           = "dataset/ml/entity_source_pca_clusters_summary.csv"

K_MIN, K_MAX = 2, 7
N_COMPONENTS = 2
RANDOM_STATE = 42
# ───────────────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_and_project(path: str) -> pd.DataFrame:
    """Load feature matrix, standardize, and reduce to 2 PCA components."""
    df = pd.read_csv(path)
    features = [
        "pos_rate", "neg_rate", "entropy",
        "pos_tfidf", "neg_tfidf",
        "polarity", "total_mentions"
    ]
    X = df[features].fillna(0).values

    logging.info("Standardizing features...")
    X_scaled = StandardScaler().fit_transform(X)

    logging.info(f"Running PCA (n_components={N_COMPONENTS})...")
    pca = PCA(n_components=N_COMPONENTS, random_state=RANDOM_STATE)
    pcs = pca.fit_transform(X_scaled)
    df["PC1"], df["PC2"] = pcs[:, 0], pcs[:, 1]
    logging.info(f"PCA explained variance ratios: {pca.explained_variance_ratio_}")
    return df


def evaluate_k_range(df: pd.DataFrame, k_min: int, k_max: int) -> pd.DataFrame:
    """
    For each k in [k_min..k_max], fit KMeans on the PCA coords
    and compute the silhouette score.
    Returns a DataFrame with columns ['k','silhouette_score'].
    """
    X_pca = df[["PC1","PC2"]].values
    records = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE)
        labels = km.fit_predict(X_pca)
        score = silhouette_score(X_pca, labels)
        logging.info(f"k={k}: silhouette_score={score:.4f}")
        records.append({"k": k, "silhouette_score": score})
    return pd.DataFrame.from_records(records)


def fit_best_k(df: pd.DataFrame, silhouette_df: pd.DataFrame) -> pd.DataFrame:
    """
    Select the k with highest silhouette_score, refit KMeans,
    and attach the 'Cluster' column to df.
    """
    best_row = silhouette_df.sort_values("silhouette_score", ascending=False).iloc[0]
    best_k = int(best_row["k"])
    logging.info(f"Best k by silhouette: {best_k}")
    km = KMeans(n_clusters=best_k, random_state=RANDOM_STATE)
    df["Cluster"] = km.fit_predict(df[["PC1","PC2"]].values)
    return df


def save_cluster_summary(df: pd.DataFrame, path: str):
    """Compute and save mean feature values per cluster."""
    summary = (
        df.groupby("Cluster")
          .agg({
              "pos_rate":      "mean",
              "neg_rate":      "mean",
              "entropy":       "mean",
              "pos_tfidf":     "mean",
              "neg_tfidf":     "mean",
              "polarity":      "mean",
              "total_mentions":"mean"
          })
          .round(3)
    )
    summary.to_csv(path)
    logging.info(f"Saved cluster summary to {path}")


def main():
    logging.info(f"Loading & projecting features from {INPUT_FEATURE_MATRIX} …")
    df = load_and_project(INPUT_FEATURE_MATRIX)

    logging.info(f"Evaluating silhouette scores for k={K_MIN}…{K_MAX}")
    sil_df = evaluate_k_range(df, K_MIN, K_MAX)
    sil_df.to_csv(OUTPUT_SILHOUETTE_CSV, index=False)
    logging.info(f"Saved silhouette scores to {OUTPUT_SILHOUETTE_CSV}")

    logging.info("Refitting KMeans with best k …")
    df_clustered = fit_best_k(df, sil_df)
    df_clustered.to_csv(OUTPUT_CLUSTERS, index=False)
    logging.info(f"Saved clustered data to {OUTPUT_CLUSTERS}")

    save_cluster_summary(df_clustered, OUTPUT_SUMMARY)


if __name__ == "__main__":
    main()
