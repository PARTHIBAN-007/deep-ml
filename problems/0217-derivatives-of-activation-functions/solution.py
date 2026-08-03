import numpy as np

def sigmoid(x):
	return np.exp(x) / (1 + np.exp(x))

def tanh(x):
	return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def activation_derivatives(x: float) -> dict[str, float]:
	sigmoid_ = sigmoid(x) * ( 1 - sigmoid(x))
	tanh_ = 1 - (tanh(x)**2)
	relu = 1 if x>0 else 0
	return {
		"sigmoid": sigmoid_,
		"tanh": tanh_,
		"relu": relu
	}
	