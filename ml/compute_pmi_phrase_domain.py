import logging
import pandas as pd
from ml.PMICalculator import PMICalculator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PHRASE_FREQUENCY_MATRIX_FILE = "dataset/ml/phrase_domain_frequency_matrix_v2.csv"
PMI_FILE = "dataset/ml/pointwise_mutual_information_scores.csv"

if __name__ == "__main__":
    logging.info(f"Loading phrase frequency matrix from {PHRASE_FREQUENCY_MATRIX_FILE}...")
    phrase_frequency_matrix = pd.read_csv(PHRASE_FREQUENCY_MATRIX_FILE, index_col=0)

    calculator = PMICalculator(phrase_frequency_matrix)

    logging.info("Computing pointwise mutual information scores...")
    pmi_scores = calculator.compute_pmi(export_path=PMI_FILE, row_label="phrase", column_label="domain")
