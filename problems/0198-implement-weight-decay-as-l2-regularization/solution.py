def apply_weight_decay(parameters: list[list[float]], gradients: list[list[float]], 
                       lr: float, weight_decay: float, apply_to_all: list[bool]) -> list[list[float]]:
	res = []
	n = len(parameters)
	for i in range(n):
		ans = []
		for param,grad in zip(parameters[i],gradients[i]):
			w = param - (lr * grad) - lr * weight_decay*param
			ans.append(w)
		res.append(ans)
	return res