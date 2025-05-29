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
        N = self.nij_matrix.values
        N_hat = np.exp(U @ W @ V.T)


        logits = U @ W @ V.T
        logits = np.clip(logits, -20, 20)
        N_hat = np.exp(logits)

        # Loss
        eps = 1e-10
        loss = np.sum(N_hat - N * np.log(N_hat + eps))

        # Add L2 regularization
        loss += self.l2_regularization * (np.sum(U ** 2) + np.sum(V ** 2) + np.sum(w ** 2))

        # Gradient
        D = N_hat - N  # (m, n)
        grad_U = D @ V @ W.T + 2 * self.l2_regularization *  U # (m, r)
        grad_V = D.T @ U @ W + 2 * self.l2_regularization * V # (n, r)
        grad_w = np.array([(U[:, k][:, None] * D) @ V[:, k] for k in range(self.rank)])
        grad_w = grad_w.sum(axis=1) + 2 * self.l2_regularization * w

        grad = np.concatenate([grad_U.ravel(), grad_V.ravel(), grad_w.ravel()])
        return loss, grad

    def fit(self, max_iter: int = 100):
        x0 = self._init_params()
        result = minimize(
            fun=lambda x: self._loss_and_grad(x),
            x0=x0,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": max_iter, "disp": True}
        )
        self.U, self.V, self.w = self._unpack_params(result.x)
        self.N_hat = np.exp(self.U @ np.diag(self.w) @ self.V.T)
        return result

    def get_phrase_embeddings(self):
        return pd.DataFrame(self.U, index=self.phrases)

    def get_domain_embeddings(self):
        return pd.DataFrame(self.V, index=self.domains)

    def get_reconstructed_matrix(self):
        return pd.DataFrame(self.N_hat, index=self.phrases, columns=self.domains)

    def get_weights(self):
        return self.w


