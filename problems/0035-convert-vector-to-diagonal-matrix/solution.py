import numpy as np

def make_diagonal(x):
	n = len(x)
	res = np.zeros((n,n))
	for i in range(n):
		res[i][i] = x[i]
	return res