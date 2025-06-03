import logging
from ml.svd_postprocess import svd_postprocess
from ml.PoissonBiasEmbedder import PoissonBiasEmbedder
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

ENTITY_FREQUENCY_MATRIX_FILE = "dataset/ml/entity_domain_frequency_matrix.csv"
ENTITY_EMBEDDINGS_FILE = "dataset/ml/entity_embeddings.csv"
DOMAIN_EMBEDDINGS_FILE = "dataset/ml/domain_entity_embeddings.csv"
SVD_FILE = "dataset/ml/svd_entity.csv"

if __name__ == "__main__":
    logging.info(f"Loading entity frequency matrix from {ENTITY_FREQUENCY_MATRIX_FILE}...")
    entity_frequency_matrix = pd.read_csv(ENTITY_FREQUENCY_MATRIX_FILE, index_col=0)

    logging.info("Training Poisson Bias Embedder...")
    model = PoissonBiasEmbedder(nij_matrix=entity_frequency_matrix, rank=3, seed=42)
    model.fit()

    logging.info("Post-processing SVD...")
    U_svd, V_svd, s = svd_postprocess(model.U, model.V, model.w)

    entity_embeddings = pd.DataFrame(U_svd,
                                     index=pd.Index(entity_frequency_matrix.index, name ="entity"),
                                     columns=[f"bias_dim_{i}" for i in range(U_svd.shape[1])])
    domain_embeddings = pd.DataFrame(V_svd,
                                     index=pd.Index(entity_frequency_matrix.columns, name="domain"),
                                     columns=[f"bias_dim_{i}" for i in range(V_svd.shape[1])])

    entity_embeddings.to_csv(ENTITY_EMBEDDINGS_FILE)
    logging.info(f"Phrase embeddings saved to {ENTITY_EMBEDDINGS_FILE}")

    domain_embeddings.to_csv(DOMAIN_EMBEDDINGS_FILE)
    logging.info(f"Domain embeddings saved to {DOMAIN_EMBEDDINGS_FILE}")

    pd.Series(s, name="singular_value").to_csv(SVD_FILE, index_label="bias_dim")
    logging.info(f"Singular values saved to {SVD_FILE}")
