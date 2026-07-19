import numpy as np

def compute_partial_derivatives(func_name: str, point: tuple[float, ...]) -> tuple[float, ...]:
	def poly2d(x, y):
		return x**2 * y + x * y**2

	def exp_sum(x, y):
		return np.exp(x + y)

	def product_sin(x, y):
		return x * np.sin(y)

	def poly3d(x, y, z):
		return x**2 * y + y * z**2

	def squared_error(x, y):
		return (x - y)**2

	funcs = {
		'poly2d': poly2d,
		'exp_sum': exp_sum,
		'product_sin': product_sin,
		'poly3d': poly3d,
		'squared_error': squared_error
	}

	f = funcs[func_name]
	h = 1e-6
	grad = []
	point_lst = list(point)

	for i in range(len(point_lst)):
		point_plus = point_lst.copy()
		point_minus = point_lst.copy()
		
		point_plus[i] += h
		point_minus[i] -= h
		
		df = (f(*point_plus) - f(*point_minus)) / (2 * h)
		grad.append(round(df, 4))
		
	return tuple(grad)