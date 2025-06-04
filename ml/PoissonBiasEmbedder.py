import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import TruncatedSVD

class PoissonBiasEmbedder:
    def __init__(
        self,
        nij_matrix: pd.DataFrame,
        rank: int = 3,
        seed: int = 42,
        l2_regularization: float = 1e-4,
        clip_value: float = 20.0,
        chunk_size: int = 500
    ):
        """
        :param nij_matrix: DataFrame with phrases as rows and domains as columns
        :param rank: latent dimension r
        :param clip_value: max absolute value for logits before exp (to prevent overflow)
        :param chunk_size: how many rows of N to process at once (to save memory)
        """
        self.nij_matrix = nij_matrix
        self.rank = rank
        self.m, self.n = nij_matrix.shape
        self.phrases = nij_matrix.index.tolist()
        self.domains = nij_matrix.columns.tolist()
        self.n_total = nij_matrix.values.sum()
        self.seed = seed
        self.l2 = l2_regularization
        self.clip_value = clip_value
        self.chunk_size = chunk_size

    def _init_params(self):
        """Initialize U, V via TruncatedSVD (and set w=1)."""
        svd = TruncatedSVD(n_components=self.rank, random_state=self.seed)
        U_init = svd.fit_transform(self.nij_matrix.values)         # shape (m, r)
        V_init = svd.components_.T                                  # shape (n, r)
        w_init = np.ones(self.rank, dtype=float)                    # shape (r,)
        return np.concatenate([U_init.ravel(), V_init.ravel(), w_init])

    def _unpack_params(self, x: np.ndarray):
        m, n, r = self.m, self.n, self.rank
        U_flat = x[: m * r]
        V_flat = x[m * r : m * r + n * r]
        w_vec = x[-r:]
        U = U_flat.reshape(m, r)
        V = V_flat.reshape(n, r)
        return U, V, w_vec

    def _loss_and_grad(self, x: np.ndarray):
        """
        Compute:
          Loss = Σ_{i,j} [ exp((U W Vᵀ)_{i,j}) - N_{i,j} * (U W Vᵀ)_{i,j} ]
               + λ (‖U‖² + ‖V‖² + ‖w‖²)
        plus its gradient w.r.t. U, V, and w. All done in row‐chunks of size self.chunk_size.
        """
        U, V, w = self._unpack_params(x)
        r = self.rank
        W = np.diag(w)                   # shape (r, r)
        N_full = self.nij_matrix.values  # shape (m, n)

        # Precompute V @ W  (shape: (n×r) = (n×r) @ (r×r))
        VW = V @ W                       # shape (n, r)

        total_loss = 0.0
        grad_U = np.zeros_like(U)       # shape (m, r)
        grad_V = np.zeros_like(V)       # shape (n, r)
        grad_w = np.zeros(r, dtype=float)

        m = self.m
        for start in range(0, m, self.chunk_size):
            end = min(start + self.chunk_size, m)
            U_chunk = U[start:end, :]               # shape (chunk_size, r)
            N_chunk = N_full[start:end, :]          # shape (chunk_size, n)

            # 1) logits_chunk = U_chunk @ W @ Vᵀ  → (chunk_size, n)
            #    but since VW = V @ W (n×r), we do logits_chunk = U_chunk @ VW.T
            logits_chunk = U_chunk.dot(VW.T)        # shape (chunk_size, n)

            # 2) clip logits before exp
            np.clip(logits_chunk, -self.clip_value, self.clip_value, out=logits_chunk)

            # 3) N_hat_chunk = exp(logits_chunk)
            N_hat_chunk = np.exp(logits_chunk, dtype=np.float64)  # shape (chunk_size, n)

            # 4) loss_chunk = Σ[ N_hat_chunk - N_chunk * logits_chunk ]
            loss_chunk = np.sum(N_hat_chunk - N_chunk * logits_chunk)
            total_loss += loss_chunk

            # 5) grad_U_chunk = (N_hat_chunk - N_chunk) @ (V W)  + 2λ U_chunk
            D_chunk = N_hat_chunk - N_chunk  # shape (chunk_size, n)
            grad_U_chunk = D_chunk.dot(VW) + 2 * self.l2 * U_chunk  # (chunk_size, r)
            grad_U[start:end, :] = grad_U_chunk

            # 6) grad_V accumulates Dᵀ_chunk @ (U_chunk W) over all chunks
            UW_chunk = U_chunk.dot(W)      # shape (chunk_size, r)
            grad_V += D_chunk.T.dot(UW_chunk)  # (n, r) += (n×chunk) @ (chunk×r)

            # 7) grad_w: for each k, add Σ_{i in chunk, j} D[i,j] * U[i,k] * V[j,k]
            for k_idx in range(r):
                U_col = U_chunk[:, k_idx]    # shape (chunk_size,)
                V_col = V[:, k_idx]          # shape (n,)
                # (U_col[:,None] * D_chunk) has shape (chunk_size, n)
                # Then @ V_col gives a vector of length chunk_size.
                # We need the sum over all i in this chunk:
                scalar_contribution = ((U_col[:, None] * D_chunk) @ V_col).sum()
                grad_w[k_idx] += scalar_contribution

        # 8) Add L2 penalties
        total_loss += self.l2 * (np.sum(U * U) + np.sum(V * V) + np.sum(w * w))
        grad_V += 2 * self.l2 * V
        grad_w += 2 * self.l2 * w

        # 9) Flatten gradients into one vector
        grad = np.concatenate([grad_U.ravel(), grad_V.ravel(), grad_w.ravel()])
        return total_loss, grad

    def fit(self, max_iter: int = 100):
        """
        Run L-BFGS-B to optimize U, V, w. Returns the optimization result.
        """
        x0 = self._init_params()
        result = minimize(
            fun=lambda x: self._loss_and_grad(x),
            x0=x0,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": max_iter, "disp": True}
        )
        # Unpack the final U, V, w
        self.U, self.V, self.w = self._unpack_params(result.x)

        # If you still want the full N_hat in memory, recompute here (clipped)
        logits_full = self.U.dot(np.diag(self.w)).dot(self.V.T)
        np.clip(logits_full, -self.clip_value, self.clip_value, out=logits_full)
        self.N_hat = np.exp(logits_full)

        return result

    def get_phrase_embeddings(self) -> pd.DataFrame:
        return pd.DataFrame(self.U, index=self.phrases)

    def get_domain_embeddings(self) -> pd.DataFrame:
        return pd.DataFrame(self.V, index=self.domains)

    def get_reconstructed_matrix(self) -> pd.DataFrame:
        return pd.DataFrame(self.N_hat, index=self.phrases, columns=self.domains)

    def get_weights(self) -> np.ndarray:
        return self.w
