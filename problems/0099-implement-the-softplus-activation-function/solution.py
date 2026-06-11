import numpy as np
def softplus(x: float) -> float:
	return np.log(1 + np.exp(x))