import numpy as np
def matrix_determinant_and_trace(matrix: list[list[float]]) -> tuple[float, float]:
	matrix = np.array(matrix)
	determinant = np.linalg.det(matrix)
	trace = 0
	n = len(matrix)
	for i in range(n):
		trace += matrix[i][i]
	return (determinant,trace)	