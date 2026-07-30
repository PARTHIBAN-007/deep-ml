import numpy as np

def jacobian_matrix(f, x: list[float], h: float = 1e-5) -> list[list[float]]:
	n = len(x)
	fx = f(x)
	m = len(fx)

	jacobian = [[0.0 for _ in range(n)] for _ in range(m)]

	for j in range(n):
		x_forward = list(x)
		x_backward = list(x)
		
		x_forward[j] += h
		x_backward[j] -= h
		
		f_forward = f(x_forward)
		f_backward = f(x_backward)
		
		for i in range(m):
			jacobian[i][j] = (f_forward[i] - f_backward[i]) / (2 * h)
			
	return jacobian