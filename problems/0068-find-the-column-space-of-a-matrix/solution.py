
import numpy as np

def matrix_image(A):
    matrix = A.astype(float).copy()
    n,m = matrix.shape

    pivot_columns = []
    curr_r = 0

    for c in range(m):
        if curr_r>=n:
            break
        max_row_idx = np.argmax(np.abs(matrix[curr_r:,c])) + curr_r
        if np.isclose(matrix[max_row_idx,c],0):
            continue
        matrix[[curr_r,max_row_idx]] = matrix[[max_row_idx,curr_r]]

        for r in range(curr_r+1,n):
            factor = matrix[r,c] / matrix[curr_r,c]
            matrix[r,c:] -= factor * matrix[curr_r,c:]
        
        pivot_columns.append(c)
        curr_r += 1
    return A[:,pivot_columns]