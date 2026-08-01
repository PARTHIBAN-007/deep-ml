import numpy as np
def softmax_derivative(x: list[float]) -> list[list[float]]:
	n = len(x)
	softmax = []
	summ = sum(np.exp(num) for num in x)
	for num in x:
		softmax.append(np.exp(num)/summ)

	derivative = [[0]*n for _ in range(n)]
	for i in range(n):
		for j in range(n):
			if i==j:
				derivative[i][i] = softmax[i] * ( 1- softmax[i])
			else:
				derivative[i][j] = -softmax[i] * softmax[j]
	return derivative