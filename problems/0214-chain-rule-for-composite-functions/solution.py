import numpy as np

def compute_chain_rule_gradient(functions: list[str], x: float) -> float:
	func_map = {
		'square': lambda val: val**2,
		'sin': np.sin,
		'exp': np.exp,
		'log': np.log
	}

	deriv_map = {
		'square': lambda val: 2 * val,
		'sin': np.cos,
		'exp': np.exp,
		'log': lambda val: 1.0 / val
	}

	val = x
	intermediate_values = []
	for fn in reversed(functions):
		intermediate_values.append(val)
		val = func_map[fn](val)

	grad = 1.0
	for fn, inp in zip(functions, reversed(intermediate_values)):
		grad *= deriv_map[fn](inp)
		
	return float(grad)