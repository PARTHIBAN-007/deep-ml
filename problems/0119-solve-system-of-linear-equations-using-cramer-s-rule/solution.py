import numpy as np

def cramers_rule(A, b):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    
    det_A = np.linalg.det(A)
    if np.isclose(det_A, 0.0):
        return -1
        
    n = A.shape[0]
    x = np.zeros(n)
    
    for i in range(n):
        Ai = A.copy()
        Ai[:, i] = b
        x[i] = np.linalg.det(Ai) / det_A
        
    return x