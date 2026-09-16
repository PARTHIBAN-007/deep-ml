import math

def mish(x: float) -> float:
	softplus = math.log(1 + math.exp(x))
	return x * math.tanh(softplus)