import numpy as np

def qr_decomposition(A: list[list[float]]) -> tuple[list[list[float]], list[list[float]]]:
    A_np = np.array(A, dtype=float)
    m, n = A_np.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    
    for j in range(n):
        v = A_np[:, j]
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A_np[:, j])
            v = v - R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]
        
    return Q.tolist(), R.tolist()