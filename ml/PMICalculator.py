import numpy as np
import pandas as pd
import logging

class PMICalculator:
    def __init__(self, matrix: pd.DataFrame):
        """
        :param matrix: A DataFrame with counts, rows = observations (phrases/entities), columns = features (domains)
        """
        self.matrix = matrix
        self.n_total = matrix.values.sum()
        if self.n_total == 0:
            error = "Total count in matrix is zero. Cannot compute PMI."
            logging.error(error)
            raise ValueError(error)

        self.pij_matrix = self.matrix / self.n_total
        self.pi_matrix = self.pij_matrix.sum(axis=1)
        self.pj_matrix = self.pij_matrix.sum(axis=0)

    def compute_pmi(self, export_path: str = None, row_label: str = "row", column_label: str = "column") -> pd.DataFrame:
        """Compute PMI for all nonzero (row, column) pairs."""
        rows = []

        for i in self.pij_matrix.index:
            for j in self.pij_matrix.columns:
                pij = self.pij_matrix.at[i, j]
                if pij > 0:
                    pi = self.pi_matrix.at[i]
                    pj = self.pj_matrix.at[j]
                    pmi = np.log2(pij / (pi * pj))
                    rows.append((i, j, pmi))

        pmi_df = pd.DataFrame(rows, columns=[row_label, column_label, "pmi"]).sort_values(by="pmi", ascending=False)

        if export_path:
            pmi_df.to_csv(export_path, index=False)
            logging.info(f"Pointwise Mutual Information scores exported to {export_path}")

        return pmi_df

    def compute_total_mi(self) -> float:
        """Compute total Mutual Information (MI) between row and column variables."""
        pij = self.pij_matrix
        pi = self.pi_matrix
        pj = self.pj_matrix

        P_expected = pi.values[:, None] * pj.values[None, :]
        with np.errstate(divide='ignore', invalid='ignore'):
            log_term = np.log2(pij.values / P_expected)
            log_term[np.isnan(log_term)] = 0
            log_term[np.isinf(log_term)] = 0

        mi = np.sum(pij.values * log_term)
        return float(mi)

    def compute_mi(self, export_path: str = None, row_label: str = "row") -> pd.Series:
        """
        Computes Mutual Information (MI) score for each row
        indicating how informative that row is in distinguishing between columns (e.g. domains).

        Returns:
            pd.Series of MI scores indexed by row, sorted in descending order.
        """
        pij_matrix = self.pij_matrix
        pi_matrix = self.pi_matrix
        pj_matrix = self.pj_matrix

        scores = {}

        for i in pij_matrix.index:
            score = 0.0
            for j in pij_matrix.columns:
                pij = pij_matrix.at[i, j]
                if pij > 0:
                    pi = pi_matrix.at[i]
                    pj = pj_matrix.at[j]
                    score += pij * np.log2(pij / (pi * pj))
            scores[i] = score

        info_scores = pd.Series(scores).sort_values(ascending=False)

        if export_path:
            info_scores.to_csv(export_path, header=["mi"], index_label=row_label)
            logging.info(f"Mutual Information scores exported to {export_path}")

        return info_scores
