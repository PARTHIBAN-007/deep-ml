
import numpy as np
def tanh(x: float) -> float:
	return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))