import logging
import numpy as np
import pandas as pd
import umap.umap_ as umap

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PCA_FILE = "dataset/nlp/bert_article_embeddings.npy"
PCA_FILE_CSV = "dataset/nlp/bert_article_embeddings.csv"
OUTPUT_UMAP_EMBEDDINGS = "dataset/nlp/bert_article_embeddings_umap.npy"
OUTPUT_UMAP_EMBEDDINGS_CSV = "dataset/nlp/bert_article_embeddings_umap.csv"

N_COMPONENTS = 15
N_NEIGHBORS = 15
MIN_DIST = 0.3
METRIC = 'cosine'

if __name__ == "__main__":
    logging.info(f"Loading PCA embeddings from {PCA_FILE}…")
    pca_embeddings = np.load(PCA_FILE)
    n_articles, _ = pca_embeddings.shape
    logging.info(f"Loaded PCA embeddings of shape {pca_embeddings.shape}")

    reducer = umap.UMAP(
        random_state=42,
        n_components= N_COMPONENTS,
        n_neighbors=N_NEIGHBORS,
        min_dist=MIN_DIST,
        metric=METRIC)

    umap_embeddings = reducer.fit_transform(pca_embeddings)
    logging.info(f"UMAP reduced embeddings to shape {umap_embeddings.shape}")

    np.save(OUTPUT_UMAP_EMBEDDINGS, umap_embeddings)
    logging.info(f"UMAP embeddings saved successfully to {OUTPUT_UMAP_EMBEDDINGS}.")

    logging.info(f"Loading metadata (url, source) from {PCA_FILE_CSV}…")
    meta_df = pd.read_csv(PCA_FILE_CSV, usecols=["url", "source"])
    if len(meta_df) != n_articles:
        raise ValueError(
            f"Number of rows in {PCA_FILE_CSV} ({len(meta_df)}) "
            f"does not match number of PCA embeddings ({n_articles})."
        )
    logging.info(f"Loaded metadata for {len(meta_df)} articles")

    umap_cols = [f"UMAP{i}" for i in range(umap_embeddings.shape[1])]
    umap_df = pd.DataFrame(data=umap_embeddings, columns=umap_cols)

    final_df = pd.concat([meta_df.reset_index(drop=True), umap_df.reset_index(drop=True)], axis=1)
    final_df.to_csv(OUTPUT_UMAP_EMBEDDINGS_CSV, index=False)
    logging.info(f"Saved CSV with URL, source, and UMAP columns to: {OUTPUT_UMAP_EMBEDDINGS_CSV}")

