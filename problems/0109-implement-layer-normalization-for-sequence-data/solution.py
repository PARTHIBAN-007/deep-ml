import numpy as np

def layer_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
	res = []
	for batch in X:
		batches = []
		for token in batch:
			mean = np.mean(token)
			var = np.var(token)
			norm = (token - mean)/ ((var + epsilon)**0.5)
			norm = (norm*gamma[-1][-1]) + beta[-1][-1]
			batches.append(norm)
		res.append(batches)
	return res
