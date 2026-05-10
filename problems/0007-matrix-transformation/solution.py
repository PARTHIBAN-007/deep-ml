import numpy as np

def transform_matrix(A: list[list[int|float]], T: list[list[int|float]], S: list[list[int|float]]) -> list[list[int|float]]:
	det_t = np.linalg.det(T)
	det_s = np.linalg.det(S)
	if det_t == 0 or det_s ==0:
		return -1
	t_inverse = np.linalg.inv(T)
	res = np.matmul(t_inverse, A)
	return np.matmul(res, S)