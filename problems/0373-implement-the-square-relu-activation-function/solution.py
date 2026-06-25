import numpy as np

def square_relu(x: np.ndarray) -> dict:
	mask = x > 0
	activation = np.where(mask, x**2, 0.0)
	derivative = np.where(mask, 2 * x, 0.0)
	return {"output": np.round(activation,2), "derivative": np.round(derivative,2)}
