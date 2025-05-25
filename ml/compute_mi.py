import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PHRASE_FREQUENCY_MATRIX_FILE = "dataset/ml/phrase_domain_frequency_matrix_v2.csv"
MI_FILE = "dataset/ml/mutual_information_scores_v2.csv"

def compute_mutual_information_scores(nij_matrix, export_path):
    """
    Computes mutual information scores for each phrase in a phrase-domain frequency matrix.

    The score reflects how informative each phrase is in distinguishing between domains.
    Higher scores indicate phrases that are disproportionately associated with specific domains.

    Returns:
        Sorted series of MI scores indexed by phrase, descending.
    """
    n_total = nij_matrix.values.sum()
    if n_total == 0:
        raise ValueError("The total count in nij_matrix is zero. Cannot compute probabilities.")

    # Joint probability P(i, j)
    pij_matrix = nij_matrix / n_total

    # Marginal probabilities
    pi_matrix = pij_matrix.sum(axis=1)
    pj_matrix = pij_matrix.sum(axis=0)

    scores = {}

    for phrase in pij_matrix.index:
        score = 0.0
        for domain in pij_matrix.columns:
            pij = pij_matrix.at[phrase, domain]
            if pij > 0:
                pi = pi_matrix.at[phrase]
                pj = pj_matrix.at[domain]
                score += pij * np.log2(pij / (pi * pj))
        scores[phrase] = score

    info_scores = pd.Series(scores).sort_values(ascending=False)

    info_scores.to_csv(export_path, sep="\t", header=["MI_score"], index_label="phrase")

    logging.info(f"Mutual information scores exported to {export_path}")

    return info_scores

if __name__ == "__main__":
    logging.info(f"Loading phrase frequency matrix from {PHRASE_FREQUENCY_MATRIX_FILE}...")
    phrase_frequency_matrix = pd.read_csv(PHRASE_FREQUENCY_MATRIX_FILE, index_col=0)

    logging.info("Computing mutual information scores...")
    mutual_info_scores = compute_mutual_information_scores(phrase_frequency_matrix,
                                                           export_path=MI_FILE)
