import numpy as np
def relu(x):
	res = []
	for num in x:
		res.append(max(0,num))
	return res

def residual_block(x: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
	fc1 = w1 @ x
	fc1 = relu(fc1)
	fc2 = w2 @ fc1
	x = x + fc2
	return relu(x)
