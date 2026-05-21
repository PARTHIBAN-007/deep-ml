import numpy as np

def GeLU(x: np.ndarray) -> np.ndarray:
	scores  = []
	for num in x:
		gelu = 0.5 * num *( 1 + np.tanh(np.sqrt(2/3.14) * (num+0.044715 * (num**3))))
		scores.append(gelu)
	return scores