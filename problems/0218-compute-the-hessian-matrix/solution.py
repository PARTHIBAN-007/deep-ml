from typing import Callable

def compute_hessian(f: Callable[[list[float]], float], point: list[float], h: float = 1e-5) -> list[list[float]]:
	n = len(point)
	hessian = [[0.0] * n for _ in range(n)]

	for i in range(n):
		point_plus = list(point)
		point_minus = list(point)

		point_plus[i] += h
		point_minus[i] -= h

		f_plus = f(point_plus)
		f_center = f(point)
		f_minus = f(point_minus)

		hessian[i][i] = (f_plus - 2 * f_center + f_minus) / (h**2)

	for i in range(n):
		for j in range(i + 1, n):
			point_pp = list(point)
			point_pm = list(point)
			point_mp = list(point)
			point_mm = list(point)

			point_pp[i] += h
			point_pp[j] += h

			point_pm[i] += h
			point_pm[j] -= h

			point_mp[i] -= h
			point_mp[j] += h

			point_mm[i] -= h
			point_mm[j] -= h

			f_pp = f(point_pp)
			f_pm = f(point_pm)
			f_mp = f(point_mp)
			f_mm = f(point_mm)

			val = (f_pp - f_pm - f_mp + f_mm) / (4 * h**2)
			hessian[i][j] = val
			hessian[j][i] = val

	return hessian