import numpy as np
def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def predict_logistic(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
	y_pred = X @ weights.T + bias
	res = []
	for num in y_pred:
		pred = sigmoid(num)
		if pred>=0.5:
			res.append(1)
		else:
			res.append(0)
	return res
	
	