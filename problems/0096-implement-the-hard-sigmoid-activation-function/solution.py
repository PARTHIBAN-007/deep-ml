def hard_sigmoid(x: float) -> float:
	if x<=-2.5:
		return 0
	elif -2.5<x<2.5:
		return 0.2*x + 0.5
	else:
		return 1