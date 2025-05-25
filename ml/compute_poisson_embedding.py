import logging
from ml.svd_postprocess import svd_postprocess
from ml.PoissonBiasEmbedder import PoissonBiasEmbedder
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PHRASE_FREQUENCY_MATRIX_FILE = "dataset/ml/phrase_domain_frequency_matrix_v2.csv"
PHRASE_EMBEDDINGS_FILE = "dataset/ml/phrase_embeddings.csv"
DOMAIN_EMBEDDINGS_FILE = "dataset/ml/domain_embeddings.csv"
SVD_FILE = "dataset/ml/svd.csv"

if __name__ == "__main__":
    logging.info(f"Loading phrase frequency matrix from {PHRASE_FREQUENCY_MATRIX_FILE}...")
    phrase_frequency_matrix = pd.read_csv(PHRASE_FREQUENCY_MATRIX_FILE, index_col=0)

    logging.info("Training Poisson Bias Embedder...")
    model = PoissonBiasEmbedder(nij_matrix=phrase_frequency_matrix, rank=3, seed=42)
    model.fit()

    logging.info("Post-processing SVD...")
    U_svd, V_svd, s = svd_postprocess(model.U, model.V, model.w)

    phrase_embeddings = pd.DataFrame(U_svd, index=phrase_frequency_matrix.index, columns=[f"bias_dim_{i+1}" for i in range(U_svd.shape[1])])
    domain_embeddings = pd.DataFrame(V_svd, index=phrase_frequency_matrix.columns, columns=[f"bias_dim_{i+1}" for i in range(V_svd.shape[1])])

    phrase_embeddings.to_csv(PHRASE_EMBEDDINGS_FILE)
    logging.info(f"Phrase embeddings saved to {PHRASE_EMBEDDINGS_FILE}")

    domain_embeddings.to_csv(DOMAIN_EMBEDDINGS_FILE)
    logging.info(f"Domain embeddings saved to {DOMAIN_EMBEDDINGS_FILE}")

    pd.Series(s, name="singular_value").to_csv(SVD_FILE, index_label="bias_dim")
    logging.info(f"Singular values saved to {SVD_FILE}")
