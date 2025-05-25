import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PHRASE_FREQUENCY_MATRIX_FILE = "dataset/ml/phrase_domain_frequency_matrix_v2.csv"
PMI_FILE = "dataset/ml/pointwise_mutual_information_scores.csv"

def compute_pointwise_mutual_information(nij_matrix: pd.DataFrame, export_path: str):
    """
    Computes Pointwise Mutual Information (PMI) for each (phrase, domain) pair.

    PMI(phrase, domain) = log2( P(phrase, domain) / (P(phrase) * P(domain)) )

    Returns:
        pd.DataFrame with columns: ['phrase', 'domain', 'pmi'], sorted by PMI descending.
    """
    n_total = nij_matrix.values.sum()
    if n_total == 0:
        raise ValueError("The total count in nij_matrix is zero. Cannot compute probabilities.")

    pij_matrix = nij_matrix / n_total
    pi_matrix = pij_matrix.sum(axis=1)
    pj_matrix = pij_matrix.sum(axis=0)

    rows = []

    for phrase in pij_matrix.index:
        for domain in pij_matrix.columns:
            pij = pij_matrix.at[phrase, domain]
            if pij > 0:
                pi = pi_matrix.at[phrase]
                pj = pj_matrix.at[domain]
                pmi = np.log2(pij / (pi * pj))
                rows.append((phrase, domain, pmi))

    pmi_df = pd.DataFrame(rows, columns=["phrase", "domain", "pmi"]).sort_values(by="pmi", ascending=False)
    pmi_df.to_csv(export_path, sep="\t", index=False)

    logging.info(f"Pointwise Mutual Information scores exported to {export_path}")
    return pmi_df

if __name__ == "__main__":
    logging.info(f"Loading phrase frequency matrix from {PHRASE_FREQUENCY_MATRIX_FILE}...")
    phrase_frequency_matrix = pd.read_csv(PHRASE_FREQUENCY_MATRIX_FILE, index_col=0)

    logging.info("Computing pointwise mutual information scores...")
    pmi_scores = compute_pointwise_mutual_information(phrase_frequency_matrix,
                                                      export_path=PMI_FILE)
