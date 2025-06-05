import logging
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

INPUT_EMB_NPY = "dataset/nlp/bert_article_embeddings.npy"
OUTPUT_PCA = "dataset/nlp/bert_article_embeddings_pca.npy"
OUTUT_PCA_SUMMARY_CSV = "dataset/nlp/bert_article_embeddings_pca_summary.csv"

if __name__ == "__main__":
    logging.info(f"Loading embeddings from {INPUT_EMB_NPY}…")
    embeddings = np.load(INPUT_EMB_NPY)
    logging.info(f"Loaded {embeddings.shape[0]} embeddings of shape {embeddings.shape[1]}")

    pca_components = min(embeddings.shape[0], embeddings.shape[1])
    pca = PCA(pca_components, random_state=42)
    reduced_embeddings = pca.fit_transform(embeddings)
    logging.info(f"PCA reduced embeddings to shape {reduced_embeddings.shape}")

    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    pca_df = pd.DataFrame({
    'Component': np.arange(1, len(cumulative_variance) + 1),
    'Explained Variance Ratio': explained_variance,
    'Cumulative Variance': cumulative_variance
    })

    pca_df.to_csv(OUTUT_PCA_SUMMARY_CSV, index=False)
    logging.info(f"PCA summary saved to {OUTUT_PCA_SUMMARY_CSV}")
