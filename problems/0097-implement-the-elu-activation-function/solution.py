import numpy as np
def elu(x: float, alpha: float = 1.0) -> float:
	if x<0:
		val = alpha * (np.exp(x)-1)
	else:
		val = x
	return round(val,4)