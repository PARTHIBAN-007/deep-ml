def leaky_relu(z: float, alpha: float = 0.01) -> float|int:
	sign = 1
	if z<0:
		sign = -1
	return sign*max(z,alpha)
