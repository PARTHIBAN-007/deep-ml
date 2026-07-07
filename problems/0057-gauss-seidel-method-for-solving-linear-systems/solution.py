import numpy as np

def gauss_seidel(A, b, n, x_ini=None):
	A = np.asarray(A, dtype=float)
	b = np.asarray(b, dtype=float)
	m = A.shape[0]

	if x_ini is None:
		x = np.zeros(m, dtype=float)
	else:
		x = np.array(x_ini, dtype=float)

	for _ in range(n):
		for i in range(m):
			row_sum = np.dot(A[i, :], x) - A[i, i] * x[i]
			x[i] = (b[i] - row_sum) / A[i, i]    
	return x