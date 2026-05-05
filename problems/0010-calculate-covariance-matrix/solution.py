import numpy as np
def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
	n,m = len(vectors), len(vectors[0])

	def covariance(x_k,y_k):
		x_mean = sum(x_k)//len(x_k)
		y_mean = sum(y_k)//len(y_k)
		summ = 0
		for x,y in zip(x_k,y_k):
			summ += (x-x_mean)*(y-y_mean)
		return summ/(m-1)
	
	cov = [[0]*n for _ in range(n)]
	for i in range(n):
		for j in range(n):
			cov[i][j] = covariance(vectors[i],vectors[j])
	return cov