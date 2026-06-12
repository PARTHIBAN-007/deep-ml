import numpy as np

def calculate_correlation_matrix(X, Y=None):
	if Y is None:
		Y = X
	n = X.shape[0]
	x_centered = X - X.mean(axis = 0)
	y_centered = Y - Y.mean(axis = 0)

	covariance = x_centered.T.dot(y_centered) / n

	std_x = x_centered.std(axis = 0,keepdims = True).T
	std_y = y_centered.std(axis = 0,keepdims = True)

	return covariance / (std_x * std_y)

	 