import numpy as np

def orthonormal_basis(vectors: list[list[float]], tol: float = 1e-10) -> list[np.ndarray]:
    orthonormal_vectors = []
    
    for v in vectors:
        v = np.array(v, dtype=float)
        for u in orthonormal_vectors:
            v -= np.dot(v, u) * u
        norm = np.linalg.norm(v)
        if norm > tol:
            orthonormal_vectors.append(v / norm)     
    return orthonormal_vectors