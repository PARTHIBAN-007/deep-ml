import numpy as np

def check_positive_definite(matrix: list) -> dict:
    A = np.array(matrix,dtype = float)
    if A.shape[0]!=A.shape[1]:
        return 
    
    is_symmetric = np.allclose(A,A.T,atol = 1e-8)

    if is_symmetric:
        eigenvalues = np.linalg.eigvalsh(A)
    else:
        eigenvalues = np.linalg.eigvals(A)
    
    if np.any(np.iscomplex(eigenvalues)):
        is_pos_def = False
        eigenvalues = np.real(eigenvalues)
    else:
        is_pos_def = is_symmetric and np.all(eigenvalues> 1e-10)
    
    sorted_eigenvalues = np.sort(eigenvalues)
    rounded_eigenvalues = [round(float(val),4) for val in sorted_eigenvalues]

    return {
        "is_positive_definite": bool(is_pos_def),
        "eigenvalues": rounded_eigenvalues
    }
