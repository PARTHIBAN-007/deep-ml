import numpy as np

def matrix_rank(A: np.ndarray, tol: float = 1e-10) -> int:
    matrix = np.array(A,dtype = float)
    m,n = matrix.shape
    rank = 0
    col = 0
    row = 0

    while row<m and col<n:
        pivot_row = row + np.argmax(np.abs(matrix[row:m,col]))
        if np.abs(matrix[pivot_row,col])<tol:
            col+=1
            continue
        if pivot_row!=row:
            matrix[[row,pivot_row]] = matrix[[pivot_row,row]]
        
        for r in range(row+1,m):
            factor = matrix[r,col] / matrix[row,col]
            matrix[r,col:] -= factor * matrix[row,col:]
        rank += 1
        row += 1
        col += 1
    return rank