import numpy as np

def log_softmax(scores: list) -> np.ndarray:
	summ = np.sum(np.exp(scores))
	softmax = []
	for num in scores:
		softmax.append(np.exp(num)/summ)
	return np.round(np.log(softmax),4)