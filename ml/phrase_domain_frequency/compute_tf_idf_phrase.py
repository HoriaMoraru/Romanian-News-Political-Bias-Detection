import logging
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PHRASE_FREQUENCY_MATRIX_FILE = "dataset/ml/phrase_domain_frequency_matrix.csv"
TF_IDF_MATRIX_FILE = "dataset/ml/phrase_domain_tf_idf_matrix.csv"

if __name__ == "__main__":
    logging.info(f"Loading phrase frequency matrix from {PHRASE_FREQUENCY_MATRIX_FILE}...")
    phrase_frequency_matrix = pd.read_csv(PHRASE_FREQUENCY_MATRIX_FILE, index_col=0)

    tf_idf_phrase_frequency_matrix = phrase_frequency_matrix.T

    logging.info("Converting matrix to TF-IDF form...")
    tfidf_transformer = TfidfTransformer(norm="l2", use_idf=True, smooth_idf=True)
    tfidf_matrix = tfidf_transformer.fit_transform(tf_idf_phrase_frequency_matrix)

    df_tfidf = pd.DataFrame(
    tfidf_matrix.toarray().T,
    index=phrase_frequency_matrix.index,
    columns=phrase_frequency_matrix.columns
    )

    df_tfidf.to_csv(TF_IDF_MATRIX_FILE)
    logging.info(f"TF-IDF Matrix saved to {TF_IDF_MATRIX_FILE}.")
