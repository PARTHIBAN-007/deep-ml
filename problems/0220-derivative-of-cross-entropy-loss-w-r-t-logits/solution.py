import numpy as np
def cross_entropy_derivative(logits: list[float], target: int) -> list[float]:
	softmax = []
	base = sum(np.exp(num) for num in logits)
	for num in logits:
		softmax.append(np.exp(num)/base)
	
	res = []
	for i,num in enumerate(softmax):
		pred = 0
		if target==i:
			pred = 1
		gradient = num - pred
		res.append(gradient)
	return res