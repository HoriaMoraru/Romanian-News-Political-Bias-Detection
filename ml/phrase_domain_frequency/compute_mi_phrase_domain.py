import logging
import pandas as pd
from ml.PMICalculator import PMICalculator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PHRASE_FREQUENCY_MATRIX_FILE = "dataset/ml/phrase_domain_frequency_matrix.csv"
MI_FILE = "dataset/ml/phrase_mutual_information_scores.csv"

if __name__ == "__main__":
    logging.info(f"Loading phrase frequency matrix from {PHRASE_FREQUENCY_MATRIX_FILE}...")
    phrase_frequency_matrix = pd.read_csv(PHRASE_FREQUENCY_MATRIX_FILE, index_col=0)

    calculator = PMICalculator(phrase_frequency_matrix)

    logging.info("Computing mutual information scores...")
    mutual_info_scores = calculator.compute_mi(export_path=MI_FILE, row_label="phrase")
