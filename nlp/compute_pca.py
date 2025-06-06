import logging
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PCA_COMPONENTS_FROM_ELBOW = 80
INPUT_EMB_NPY = "dataset/nlp/bert_article_embeddings.npy"
INPUT_EMB_CSV = "dataset/nlp/bert_article_embeddings.csv"
OUTPUT_PCA = "dataset/nlp/bert_article_embeddings_pca.npy"
OUTPUT_PCA_CSV = "dataset/nlp/bert_article_embeddings_pca.csv"

if __name__ == "__main__":
    logging.info(f"Loading embeddings from {INPUT_EMB_NPY}…")
    embeddings = np.load(INPUT_EMB_NPY)
    n_articles, _ = embeddings.shape
    logging.info(f"Loaded BERT embeddings of shape {embeddings.shape}")

    # logging.info(f"Normalizing embeddings...")
    # normalized_embeddings = normalize(embeddings)

    pca = PCA(PCA_COMPONENTS_FROM_ELBOW, random_state=42)
    reduced_embeddings = pca.fit_transform(embeddings)
    logging.info(f"PCA reduced embeddings to shape {reduced_embeddings.shape}")

    np.save(OUTPUT_PCA, reduced_embeddings)
    logging.info(f"PCA embeddings saved successfully to {OUTPUT_PCA}.")

    logging.info(f"Loading metadata (url, source) from {INPUT_EMB_CSV}…")
    meta_df = pd.read_csv(INPUT_EMB_CSV, usecols=["url", "source"])
    if len(meta_df) != n_articles:
        raise ValueError(
            f"Number of rows in {INPUT_EMB_CSV} ({len(meta_df)}) "
            f"does not match number of embeddings ({n_articles})."
        )
    logging.info(f"Loaded metadata for {len(meta_df)} articles")

    pca_cols = [f"PC{i}" for i in range(PCA_COMPONENTS_FROM_ELBOW)]
    pca_df = pd.DataFrame(data=reduced_embeddings, columns=pca_cols)

    final_df = pd.concat([meta_df.reset_index(drop=True), pca_df.reset_index(drop=True)], axis=1)
    final_df.to_csv(OUTPUT_PCA_CSV, index=False)
    logging.info(f"Saved CSV with URL, source, and PCA columns to: {OUTPUT_PCA_CSV}")
