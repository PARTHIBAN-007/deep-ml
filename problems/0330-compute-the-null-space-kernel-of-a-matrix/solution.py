import numpy as np

def compute_null_space(A: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    u,s,vh = np.linalg.svd(A)
    null_mask = s < tol
    num_null_vectors = np.sum(null_mask)
    num_extra_cols = A.shape[1] - len(s)
    if num_extra_cols:
        null_mask = np.concatenate([null_mask, np.ones(num_extra_cols, dtype=bool)])
    null_space = vh[null_mask].T
    return null_space