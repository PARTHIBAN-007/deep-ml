import numpy as np

def solve_jacobi(A: np.ndarray, b: np.ndarray, n: int) -> list:
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    num_equations = len(b)
    x = np.zeros(num_equations)
    D = np.diag(A)
    R = A - np.diag(D)
    
    for _ in range(n):
        x_new = (b - np.dot(R, x)) / D
        x = np.round(x_new, 4)
        
    return x.tolist()