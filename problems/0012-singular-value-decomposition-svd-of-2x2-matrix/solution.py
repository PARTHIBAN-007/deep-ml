import numpy as np

def svd_2x2_singular_values(A: np.ndarray) -> tuple:
    G = A.T @ A
    g11 = G[0, 0]
    g12 = G[0, 1]
    g22 = G[1, 1]

    if abs(g12) < 1e-15:
        theta = 0.0
    else:
        theta = 0.5 * np.arctan2(2.0 * g12, g11 - g22)
    
    c = np.cos(theta)
    s = np.sin(theta)
    V = np.array([[c, -s],
                  [s,  c]])
    Vt = V.T

    B = A @ V
    s1 = np.linalg.norm(B[:, 0])
    s2 = np.linalg.norm(B[:, 1])
    u1 = B[:, 0] / s1 if s1 > 1e-15 else np.array([1.0, 0.0])
    u2 = B[:, 1] / s2 if s2 > 1e-15 else np.array([0.0, 1.0])
    U = np.column_stack((u1, u2))

    S = np.array([s1, s2])
    if S[0] < S[1]:
        S = S[::-1]
        U = U[:, ::-1]
        Vt = Vt[::-1, :]
        
    return U, S, Vt