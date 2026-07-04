import numpy as np

def is_linearly_independent(vectors: list[list[float]]) -> bool:
    matrix = np.array(vectors)
    n = len(matrix)
    rank = np.linalg.matrix_rank(matrix)
    return n==rank