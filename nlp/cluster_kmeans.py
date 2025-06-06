import pandas as pd
import logging
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

INPUT_CSV = "dataset/nlp/bert_article_embeddings_umap.csv"
OUTPUT_CSV = "dataset/nlp/bert_article_embeddings_umap_kmeans_clusters.csv"

if __name__ == "__main__":
    logging.info("Reading input...")
    df = pd.read_csv(INPUT_CSV)

    umap_cols = [col for col in df.columns if col.startswith("UMAP")]
    X_umap = df[umap_cols].values
    X_umap = normalize(X_umap, norm='l2')
    logging.info(f"X_umap shape is: {X_umap.shape}")

    best_k = None
    best_score = -1
    best_labels = None

    logging.info("Running KMeans and computing silhouette scores from k=2 to k=6...")
    for k in range(2, 7):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(X_umap)
        score = silhouette_score(X_umap, labels)
        logging.info(f"k = {k}: Silhouette Score = {score:.4f}")

        if score > best_score:
            best_k = k
            best_score = score
            best_labels = labels

    logging.info(f"✅ Best k: {best_k} with Silhouette Score: {best_score:.4f}")
    df["cluster"] = best_labels

    cluster_counts = df["cluster"].value_counts().sort_index()
    for label, count in cluster_counts.items():
        logging.info(f"Cluster {label}: {count} articles")

    df.to_csv(OUTPUT_CSV, index=False)
    logging.info(f"Saved clustered articles to {OUTPUT_CSV}")
