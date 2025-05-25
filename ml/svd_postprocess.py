import numpy as np

def svd_postprocess(U, V, w):
    """
    Performs SVD-based post-processing to re-orthogonalize U, V and absorb weights w.

    Args:
        U: m x r matrix of phrase embeddings
        V: n x r matrix of source embeddings
        w: r-dimensional weights vector

    Returns:
        U_svd: orthogonalized phrase embeddings (m x r)
        V_svd: orthogonalized source embeddings (n x r)
        s: singular values representing importance of each dimension (r,)
    """
    # Compute the low-rank approximation matrix A = U * W * V^T
    A = U @ np.diag(w) @ V.T  # shape (m, n)

    # Perform SVD on A
    U_svd, s, Vt_svd = np.linalg.svd(A, full_matrices=False)

    # Update U and V using the new orthogonal basis
    V_svd = Vt_svd.T  # shape (n, r)

    return U_svd, V_svd, s
