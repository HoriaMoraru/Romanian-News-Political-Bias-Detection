import pandas as pd
import hdbscan
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

INPUT_CSV = "dataset/nlp/bert_article_embeddings_umap.csv"
OUTPUT_CSV = "dataset/nlp/bert_article_embeddings_umap_hdbscan_clusters.csv"

MIN_CLUSTER_SIZE = 30
MIN_SAMPLES = 15

if __name__ == "__main__":
    logging.info("Reading input...")
    df = pd.read_csv(INPUT_CSV)

    umap_cols = [col for col in df.columns if col.startswith("UMAP")]
    X_umap = df[umap_cols].values

    logging.info(f"X_umap shape is: {X_umap.shape}")

    logging.info("Running HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE,
                                min_samples=MIN_SAMPLES)
    cluster_labels = clusterer.fit_predict(X_umap)                  # shape: (N,)

    df["cluster"] = cluster_labels

    cluster_counts = df["cluster"].value_counts().sort_index()
    for label, count in cluster_counts.items():
        if label == -1:
            logging.info(f"Cluster -1 (noise): {count} articles")
        else:
            logging.info(f"Cluster {label}: {count} articles")
    logging.info(f"Cluster persistance is: {clusterer.cluster_persistence_}")

    df.to_csv(OUTPUT_CSV, index=False)
    logging.info(f"Saved clustered articles to {OUTPUT_CSV}")
