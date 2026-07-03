import numpy as np

def rref(matrix):
	A = np.array(matrix, dtype=float)
	rows, cols = A.shape

	pivot_row = 0
	for col in range(cols):
		if pivot_row >= rows:
			break
		max_row = np.argmax(np.abs(A[pivot_row:rows, col])) + pivot_row
		if np.isclose(A[max_row, col], 0):
			continue
		A[[pivot_row, max_row]] = A[[max_row, pivot_row]]
		pivot_val = A[pivot_row, col]
		A[pivot_row] = A[pivot_row] / pivot_val
		for r in range(rows):
			if r != pivot_row:
				factor = A[r, col]
				A[r] = A[r] - factor * A[pivot_row]
		pivot_row += 1
	A[np.isclose(A, 0)] = 0.0

	return A
