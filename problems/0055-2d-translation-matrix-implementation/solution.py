import numpy as np
def translate_object(points, tx, ty):
	n,m = len(points), len(points[0])
	for i in range(n):
		points[i][0] += tx
		points[i][1] += ty
	return points
