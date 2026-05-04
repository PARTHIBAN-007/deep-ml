import numpy as np
def inverse_2x2(matrix: list[list[float]]) -> list[list[float]] | None:
    matrix = np.array(matrix, dtype=float)
    
    try:
        inv = np.linalg.inv(matrix)
        return inv.tolist()
    except np.linalg.LinAlgError:
        return None