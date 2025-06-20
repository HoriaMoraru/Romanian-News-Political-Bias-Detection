import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import TruncatedSVD

class PoissonBiasEmbedder:

    def __init__(self, nij_matrix: pd.DataFrame, rank: int = 3, seed: int = 42, l2_regularization: float = 1e-4):
        """
        :param nij_matrix: DataFrame with phrases as rows and domains as columns
        :param rank: latent dimension r
        """
        self.nij_matrix = nij_matrix
        self.rank = rank
        self.m, self.n = nij_matrix.shape
        self.phrases = nij_matrix.index.tolist()
        self.domains = nij_matrix.columns.tolist()
        self.n_total = nij_matrix.values.sum()
        self.seed = seed
        self.l2_regularization = l2_regularization

    def _init_params(self):
        """Initialize U, V using TruncatedSVD for more informed start"""
        svd = TruncatedSVD(n_components=self.rank, random_state=self.seed)
        U_init = svd.fit_transform(self.nij_matrix.values)
        V_init = svd.components_.T
        w_init = np.ones(self.rank)
        return np.concatenate([U_init.ravel(), V_init.ravel(), w_init])

    def _unpack_params(self, x):
        m, n, r = self.m, self.n, self.rank
        U = x[:m * r].reshape(m, r)
        V = x[m * r:m * r + n * r].reshape(n, r)
        w = x[-r:]
        return U, V, w

    def _loss_and_grad(self, x):
        U, V, w = self._unpack_params(x)

        W = np.diag(w)
        logits = U @ W @ V.T
        logits = np.clip(logits, -20, 20)

        N = self.nij_matrix.values
        N_hat = np.exp(logits)

        loss = np.sum(N_hat - N * logits)

        # L2 regularization
        loss += self.l2_regularization * (np.sum(U**2) + np.sum(V**2) + np.sum(w**2))

        # Gradient
        D = N_hat - N  # Shape (m, n)

        grad_U = D @ V @ W.T + 2 * self.l2_regularization * U
        grad_V = D.T @ U @ W + 2 * self.l2_regularization * V

        grad_w = np.array([
            np.sum(U[:, k][:, None] * D * V[:, k][None, :])
            for k in range(self.rank)
        ])
        grad_w += 2 * self.l2_regularization * w

        grad = np.concatenate([grad_U.ravel(), grad_V.ravel(), grad_w])
        return loss, grad


    def fit(self, max_iter: int = 500):
        x0 = self._init_params()

        bound_limit = 10.0
        bounds = [(None, None)] * (self.m * self.rank + self.n * self.rank) + \
                [(-bound_limit, bound_limit)] * self.rank

        result = minimize(
            fun=self._loss_and_grad,
            x0=x0,
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxiter": max_iter,
                "disp": True,
                "gtol": 1e-5
            }
        )

        self.U, self.V, self.w = self._unpack_params(result.x)
        self.N_hat = np.exp(np.clip(self.U @ np.diag(self.w) @ self.V.T, -20, 20))
        return result

    def get_phrase_embeddings(self):
        return pd.DataFrame(self.U, index=self.phrases)

    def get_domain_embeddings(self):
        return pd.DataFrame(self.V, index=self.domains)

    def get_reconstructed_matrix(self):
        return pd.DataFrame(self.N_hat, index=self.phrases, columns=self.domains)

    def get_weights(self):
        return self.w
