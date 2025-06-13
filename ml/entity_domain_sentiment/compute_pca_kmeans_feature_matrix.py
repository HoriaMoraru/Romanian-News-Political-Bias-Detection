import logging

import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# ───────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ───────────────────────────────────────────────────────────────────────────────
INPUT_FEATURE_MATRIX = "dataset/ml/entity_source_sentiment_features.csv"
OUTPUT_CLUSTERS       = "dataset/ml/entity_source_pca_clusters.csv"
OUTPUT_SUMMARY        = "dataset/ml/entity_source_pca_clusters_summary.csv"

N_COMPONENTS = 2
N_CLUSTERS   = 3
# ───────────────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def run_pca_kmeans(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize, project to PCA space, and assign KMeans clusters."""
    features = [
        "pos_rate", "neg_rate", "entropy",
        "pos_tfidf", "neg_tfidf",
        "polarity", "total_mentions"
    ]
    X = df[features].fillna(0).values

    logging.info("Standardizing features...")
    X_scaled = StandardScaler().fit_transform(X)

    logging.info(f"Running PCA (n_components={N_COMPONENTS})...")
    pca = PCA(n_components=N_COMPONENTS, random_state=42)
    pcs = pca.fit_transform(X_scaled)
    df["PC1"], df["PC2"] = pcs[:, 0], pcs[:, 1]
    logging.info(f"PCA explained variance ratios: {pca.explained_variance_ratio_}")

    logging.info(f"Clustering into {N_CLUSTERS} groups with KMeans...")
    km = KMeans(n_clusters=N_CLUSTERS, random_state=42)
    df["Cluster"] = km.fit_predict(pcs)

    return df


def main():
    logging.info(f"Loading domain features from {INPUT_FEATURE_MATRIX}...")
    df = pd.read_csv(INPUT_FEATURE_MATRIX)

    df = run_pca_kmeans(df)

    logging.info(f"Saving PCA+clustered data to {OUTPUT_CLUSTERS}...")
    df.to_csv(OUTPUT_CLUSTERS, index=False)

    logging.info("Computing cluster summary statistics...")
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
    summary.to_csv(OUTPUT_SUMMARY)
    logging.info(f"Saved cluster summary to {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()
